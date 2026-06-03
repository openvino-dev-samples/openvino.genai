// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include "image_generation/diffusion_pipeline.hpp"
#include "image_generation/numpy_utils.hpp"
#include "image_generation/threaded_callback.hpp"

#include "openvino/genai/image_generation/autoencoder_kl.hpp"
#include "openvino/genai/image_generation/flux_transformer_2d_model.hpp"
#include "openvino/genai/tokenizer.hpp"

#include "json_utils.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

// Minimal reader for a single float32 array stored in an *uncompressed* .npz
// (np.savez, not savez_compressed). Returns the values for the requested entry name.
// Robust against ZIP data descriptors (np.savez sets general-purpose bit 3, so the
// compressed size in the local header may be 0); the element count is derived from
// the embedded .npy header shape, and all accesses are bounds-checked.
inline bool read_npz_float_array(const std::filesystem::path& npz_path,
                                 const std::string& entry_name,
                                 std::vector<float>& out_values) {
    std::ifstream file(npz_path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    std::vector<char> buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const std::string target = entry_name + ".npy";

    auto read_u16 = [&](size_t off) -> uint16_t {
        return static_cast<uint16_t>(static_cast<unsigned char>(buffer[off]) |
                                     (static_cast<unsigned char>(buffer[off + 1]) << 8));
    };

    // Scan for local file header signatures: PK\x03\x04 (0x04034b50).
    for (size_t pos = 0; pos + 30 <= buffer.size(); ++pos) {
        if (!(static_cast<unsigned char>(buffer[pos]) == 0x50 &&
              static_cast<unsigned char>(buffer[pos + 1]) == 0x4b &&
              static_cast<unsigned char>(buffer[pos + 2]) == 0x03 &&
              static_cast<unsigned char>(buffer[pos + 3]) == 0x04)) {
            continue;
        }

        const uint16_t compression = read_u16(pos + 8);
        const uint16_t name_len = read_u16(pos + 26);
        const uint16_t extra_len = read_u16(pos + 28);

        const size_t name_off = pos + 30;
        if (name_off + name_len > buffer.size()) {
            return false;
        }
        std::string name(buffer.begin() + name_off, buffer.begin() + name_off + name_len);
        const size_t data_off = name_off + name_len + extra_len;

        if (name != target || compression != 0) {
            continue;
        }

        // Parse the embedded .npy header: magic(6) + version(2) + header_len(2) + header.
        if (data_off + 10 > buffer.size()) {
            return false;
        }
        const size_t header_len = read_u16(data_off + 8);
        const size_t header_off = data_off + 10;
        const size_t payload_off = header_off + header_len;
        if (payload_off > buffer.size()) {
            return false;
        }

        // Extract the shape tuple from the header dict to compute the element count.
        std::string header(buffer.begin() + header_off, buffer.begin() + payload_off);
        const size_t shape_pos = header.find("'shape':");
        if (shape_pos == std::string::npos) {
            return false;
        }
        const size_t open_paren = header.find('(', shape_pos);
        const size_t close_paren = header.find(')', open_paren);
        if (open_paren == std::string::npos || close_paren == std::string::npos) {
            return false;
        }
        size_t count = 1;
        bool has_dim = false;
        std::string dims = header.substr(open_paren + 1, close_paren - open_paren - 1);
        size_t i = 0;
        while (i < dims.size()) {
            if (std::isdigit(static_cast<unsigned char>(dims[i]))) {
                size_t value = 0;
                while (i < dims.size() && std::isdigit(static_cast<unsigned char>(dims[i]))) {
                    value = value * 10 + static_cast<size_t>(dims[i] - '0');
                    ++i;
                }
                count *= value;
                has_dim = true;
            } else {
                ++i;
            }
        }
        if (!has_dim) {
            return false;
        }

        const size_t payload_bytes = count * sizeof(float);
        if (payload_off + payload_bytes > buffer.size()) {
            return false;
        }

        out_values.resize(count);
        std::memcpy(out_values.data(), buffer.data() + payload_off, payload_bytes);
        return true;
    }

    return false;
}

inline ov::Tensor pack_ref_latents(const ov::Tensor& latents) {
    // (B, C, H, W) -> (B, H * W, C)
    const auto& shape = latents.get_shape();
    OPENVINO_ASSERT(shape.size() == 4, "Expected 4D tensor for ref latents");
    const size_t b = shape[0];
    const size_t c = shape[1];
    const size_t h = shape[2];
    const size_t w = shape[3];

    ov::Tensor packed(latents.get_element_type(), {b, h * w, c});
    const float* src = latents.data<const float>();
    float* dst = packed.data<float>();

    for (size_t bi = 0; bi < b; ++bi) {
        for (size_t hi = 0; hi < h; ++hi) {
            for (size_t wi = 0; wi < w; ++wi) {
                const size_t dst_base = (bi * h * w + hi * w + wi) * c;
                for (size_t ci = 0; ci < c; ++ci) {
                    const size_t src_idx = ((bi * c + ci) * h + hi) * w + wi;
                    dst[dst_base + ci] = src[src_idx];
                }
            }
        }
    }

    return packed;
}

inline ov::Tensor patchify_latents(const ov::Tensor& latents) {
    // (B, 32, H, W) -> (B, 128, H/2, W/2)
    const auto& shape = latents.get_shape();
    OPENVINO_ASSERT(shape.size() == 4, "Expected 4D latents");
    const size_t b = shape[0];
    const size_t c = shape[1];
    const size_t h = shape[2];
    const size_t w = shape[3];
    OPENVINO_ASSERT(h % 2 == 0 && w % 2 == 0, "Latent H/W must be divisible by 2");

    const size_t h2 = h / 2;
    const size_t w2 = w / 2;

    ov::Tensor out(latents.get_element_type(), {b, c * 4, h2, w2});
    const float* src = latents.data<const float>();
    float* dst = out.data<float>();

    for (size_t bi = 0; bi < b; ++bi) {
        for (size_t ci = 0; ci < c; ++ci) {
            for (size_t hi = 0; hi < h2; ++hi) {
                for (size_t wi = 0; wi < w2; ++wi) {
                    const size_t src_base = ((bi * c + ci) * h + hi * 2) * w + wi * 2;
                    const size_t dst_base = ((bi * (c * 4) + ci * 4) * h2 + hi) * w2 + wi;
                    dst[dst_base] = src[src_base];
                    dst[dst_base + h2 * w2] = src[src_base + 1];
                    dst[dst_base + 2 * h2 * w2] = src[src_base + w];
                    dst[dst_base + 3 * h2 * w2] = src[src_base + w + 1];
                }
            }
        }
    }

    return out;
}

inline ov::Tensor unpatchify_latents(const ov::Tensor& latents) {
    // (B, 128, H/2, W/2) -> (B, 32, H, W)
    const auto& shape = latents.get_shape();
    OPENVINO_ASSERT(shape.size() == 4, "Expected 4D patched latents");
    const size_t b = shape[0];
    const size_t c4 = shape[1];
    const size_t h2 = shape[2];
    const size_t w2 = shape[3];
    OPENVINO_ASSERT(c4 % 4 == 0, "Patched channel dim must be divisible by 4");

    const size_t c = c4 / 4;
    const size_t h = h2 * 2;
    const size_t w = w2 * 2;

    ov::Tensor out(latents.get_element_type(), {b, c, h, w});
    const float* src = latents.data<const float>();
    float* dst = out.data<float>();

    for (size_t bi = 0; bi < b; ++bi) {
        for (size_t ci = 0; ci < c; ++ci) {
            for (size_t hi = 0; hi < h2; ++hi) {
                for (size_t wi = 0; wi < w2; ++wi) {
                    const size_t src_base = ((bi * c4 + ci * 4) * h2 + hi) * w2 + wi;
                    const size_t dst_base = ((bi * c + ci) * h + hi * 2) * w + wi * 2;
                    dst[dst_base] = src[src_base];
                    dst[dst_base + 1] = src[src_base + h2 * w2];
                    dst[dst_base + w] = src[src_base + 2 * h2 * w2];
                    dst[dst_base + w + 1] = src[src_base + 3 * h2 * w2];
                }
            }
        }
    }

    return out;
}

inline ov::Tensor unpack_patched_latents(const ov::Tensor& latents, size_t height, size_t width, size_t vae_scale_factor) {
    // (B, H*W, 128) -> (B, 128, H/2, W/2), where H/W are latent H/W before patchify.
    const auto& in_shape = latents.get_shape();
    OPENVINO_ASSERT(in_shape.size() == 3, "Expected 3D packed latents");

    const size_t batch = in_shape[0];
    const size_t seq = in_shape[1];
    const size_t channels = in_shape[2];

    const size_t latent_h = height / vae_scale_factor;
    const size_t latent_w = width / vae_scale_factor;
    const size_t h2 = latent_h / 2;
    const size_t w2 = latent_w / 2;

    OPENVINO_ASSERT(h2 * w2 == seq, "Packed sequence length does not match expected latent grid");

    ov::Tensor out(latents.get_element_type(), {batch, channels, h2, w2});
    const float* src = latents.data<const float>();
    float* dst = out.data<float>();

    for (size_t b = 0; b < batch; ++b) {
        for (size_t i = 0; i < seq; ++i) {
            const size_t h = i / w2;
            const size_t w = i % w2;
            for (size_t c = 0; c < channels; ++c) {
                const size_t src_idx = (b * seq + i) * channels + c;
                const size_t dst_idx = ((b * channels + c) * h2 + h) * w2 + w;
                dst[dst_idx] = src[src_idx];
            }
        }
    }

    return out;
}

inline ov::Tensor prepare_flux2_text_ids(size_t seq_len) {
    ov::Tensor text_ids(ov::element::f32, {seq_len, 4});
    float* data = text_ids.data<float>();
    for (size_t i = 0; i < seq_len; ++i) {
        data[i * 4 + 0] = 0.0f;
        data[i * 4 + 1] = 0.0f;
        data[i * 4 + 2] = 0.0f;
        data[i * 4 + 3] = static_cast<float>(i);
    }
    return text_ids;
}

inline ov::Tensor prepare_flux2_latent_ids(size_t h, size_t w) {
    ov::Tensor latent_ids(ov::element::f32, {h * w, 4});
    float* data = latent_ids.data<float>();
    for (size_t i = 0; i < h; ++i) {
        for (size_t j = 0; j < w; ++j) {
            const size_t idx = (i * w + j) * 4;
            data[idx + 0] = 0.0f;
            data[idx + 1] = static_cast<float>(i);
            data[idx + 2] = static_cast<float>(j);
            data[idx + 3] = 0.0f;
        }
    }
    return latent_ids;
}

inline ov::Tensor prepare_flux2_ref_ids(size_t num_refs, size_t h, size_t w, size_t scale = 10) {
    ov::Tensor ref_ids(ov::element::f32, {num_refs * h * w, 4});
    float* data = ref_ids.data<float>();
    size_t out_idx = 0;
    for (size_t r = 0; r < num_refs; ++r) {
        const float t = static_cast<float>(scale + scale * r);
        for (size_t i = 0; i < h; ++i) {
            for (size_t j = 0; j < w; ++j) {
                data[out_idx * 4 + 0] = t;
                data[out_idx * 4 + 1] = static_cast<float>(i);
                data[out_idx * 4 + 2] = static_cast<float>(j);
                data[out_idx * 4 + 3] = 0.0f;
                ++out_idx;
            }
        }
    }
    return ref_ids;
}

inline ov::Tensor concat_seq2d(const ov::Tensor& a, const ov::Tensor& b) {
    // a: [N1, D], b: [N2, D]
    const auto& sa = a.get_shape();
    const auto& sb = b.get_shape();
    OPENVINO_ASSERT(sa.size() == 2 && sb.size() == 2 && sa[1] == sb[1], "Expected [N, D] tensors with same D");
    ov::Tensor out(a.get_element_type(), {sa[0] + sb[0], sa[1]});
    const float* ad = a.data<const float>();
    const float* bd = b.data<const float>();
    float* od = out.data<float>();
    std::copy(ad, ad + a.get_size(), od);
    std::copy(bd, bd + b.get_size(), od + a.get_size());
    return out;
}

inline ov::Tensor concat_seq3d(const ov::Tensor& a, const ov::Tensor& b) {
    // a: [B, N1, C], b: [B, N2, C]
    const auto& sa = a.get_shape();
    const auto& sb = b.get_shape();
    OPENVINO_ASSERT(sa.size() == 3 && sb.size() == 3, "Expected [B, N, C] tensors");
    OPENVINO_ASSERT(sa[0] == sb[0] && sa[2] == sb[2], "Batch/channel must match for concatenation");

    const size_t bsz = sa[0];
    const size_t n1 = sa[1];
    const size_t n2 = sb[1];
    const size_t c = sa[2];

    ov::Tensor out(a.get_element_type(), {bsz, n1 + n2, c});
    const float* ad = a.data<const float>();
    const float* bd = b.data<const float>();
    float* od = out.data<float>();

    for (size_t bidx = 0; bidx < bsz; ++bidx) {
        const size_t a_off = bidx * n1 * c;
        const size_t b_off = bidx * n2 * c;
        const size_t o_off = bidx * (n1 + n2) * c;
        std::copy(ad + a_off, ad + a_off + n1 * c, od + o_off);
        std::copy(bd + b_off, bd + b_off + n2 * c, od + o_off + n1 * c);
    }

    return out;
}

class Flux2KleinPipeline : public DiffusionPipeline {
public:
    Flux2KleinPipeline(PipelineType pipeline_type, const std::filesystem::path& root_dir) : Flux2KleinPipeline(pipeline_type) {
        m_root_dir = root_dir;
        std::ifstream file(root_dir / "model_index.json");
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", root_dir / "model_index.json");
        nlohmann::json data = nlohmann::json::parse(file);

        set_scheduler(Scheduler::from_config(root_dir / "scheduler" / "scheduler_config.json"));

        OPENVINO_ASSERT(data["text_encoder"][1].get<std::string>() == "Qwen3ForCausalLM",
                        "Flux2KleinPipeline expects Qwen3ForCausalLM text encoder");
        OPENVINO_ASSERT(data["transformer"][1].get<std::string>() == "Flux2Transformer2DModel",
                        "Flux2KleinPipeline expects Flux2Transformer2DModel");
        OPENVINO_ASSERT(data["vae"][1].get<std::string>() == "AutoencoderKLFlux2",
                        "Flux2KleinPipeline expects AutoencoderKLFlux2");

        m_is_distilled = data.value("is_distilled", true);

        m_tokenizer = Tokenizer(root_dir / "tokenizer");
        m_text_encoder_model = utils::singleton_core().read_model(root_dir / "text_encoder" / "openvino_model.xml");
        m_vae_encoder_model = utils::singleton_core().read_model(root_dir / "vae_encoder" / "openvino_model.xml");
        m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_decoder");
        m_transformer = std::make_shared<FluxTransformer2DModel>(root_dir / "transformer");

        load_bn_stats(root_dir / "vae_bn_stats.npz");

        const std::string class_name = data["_class_name"].get<std::string>();
        initialize_generation_config(class_name);
    }

    Flux2KleinPipeline(PipelineType pipeline_type,
                       const std::filesystem::path& root_dir,
                       const std::string& device,
                       const ov::AnyMap& properties)
        : Flux2KleinPipeline(pipeline_type, root_dir) {
        compile(device, properties);
    }

    void reshape(const int num_images_per_prompt, const int height, const int width, const float guidance_scale) override {
        check_image_size(height, width);

        const int max_seq_len = m_generation_config.max_sequence_length;
        m_text_encoder_model->reshape({
            {"input_ids", {1, max_seq_len}},
            {"attention_mask", {1, max_seq_len}},
        });

        m_transformer->reshape(num_images_per_prompt, height, width, max_seq_len);
        m_vae_encoder_model->reshape({{"sample", {num_images_per_prompt, 3, height, width}}});
        m_vae->reshape(num_images_per_prompt, height, width);
    }

    void compile(const std::string& text_encode_device,
                 const std::string& denoise_device,
                 const std::string& vae_device,
                 const ov::AnyMap& properties) override {
        update_adapters_from_properties(properties, m_generation_config.adapters);

        auto updated_properties = update_adapters_in_properties(properties, &Flux2KleinPipeline::derived_adapters);

        auto te = utils::singleton_core().compile_model(m_text_encoder_model, text_encode_device, *updated_properties);
        m_text_encoder_request = te.create_infer_request();
        m_text_encoder_model.reset();

        auto ve = utils::singleton_core().compile_model(m_vae_encoder_model, vae_device, *updated_properties);
        m_vae_encoder_request = ve.create_infer_request();
        m_vae_encoder_model.reset();

        m_transformer->compile(denoise_device, *updated_properties);
        m_vae->compile(vae_device, *updated_properties);
    }

    void compile(const std::string& device, const ov::AnyMap& properties) override {
        compile(device, device, device, properties);
    }

    std::shared_ptr<DiffusionPipeline> clone() override {
        OPENVINO_THROW("Flux2KleinPipeline::clone is not implemented yet");
    }

    ov::Tensor decode(const ov::Tensor latent) override {
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        ov::Tensor patched = unpack_patched_latents(
            latent,
            static_cast<size_t>(m_custom_generation_config.height),
            static_cast<size_t>(m_custom_generation_config.width),
            vae_scale_factor);
        apply_bn_denorm(patched);
        ov::Tensor unpatched = unpatchify_latents(patched);
        return m_vae->decode(unpatched);
    }

    std::tuple<ov::Tensor, ov::Tensor, ov::Tensor, ov::Tensor> prepare_latents(ov::Tensor initial_image, const ImageGenerationConfig& generation_config) override {
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        const size_t channels = m_transformer->get_config().in_channels / 4;
        const size_t height = generation_config.height / vae_scale_factor;
        const size_t width = generation_config.width / vae_scale_factor;

        ov::Shape latent_shape{generation_config.num_images_per_prompt, channels, height, width};
        ov::Tensor noise = generation_config.generator->randn_tensor(latent_shape);
        ov::Tensor latent = pack_latents(noise, generation_config.num_images_per_prompt, channels, height, width);
        return std::make_tuple(latent, ov::Tensor{}, ov::Tensor{}, noise);
    }

    std::tuple<ov::Tensor, ov::Tensor> prepare_mask_latents(ov::Tensor, ov::Tensor, const ImageGenerationConfig&, const size_t) override {
        OPENVINO_THROW("Mask latents are not supported for Flux2KleinPipeline");
    }

    void blend_latents(ov::Tensor, const ov::Tensor, const ov::Tensor, const ov::Tensor, size_t) override {
        OPENVINO_THROW("Inpainting is not supported for Flux2KleinPipeline");
    }

    ov::Tensor generate(const std::string& positive_prompt,
                        ov::Tensor initial_image,
                        ov::Tensor,
                        const ov::AnyMap& properties) override {
        const auto gen_start = std::chrono::steady_clock::now();
        m_perf_metrics.clean_up();

        m_custom_generation_config = m_generation_config;
        m_custom_generation_config.update_generation_config(properties);

        if (m_custom_generation_config.height < 0)
            compute_dim(m_custom_generation_config.height, initial_image, 1);
        if (m_custom_generation_config.width < 0)
            compute_dim(m_custom_generation_config.width, initial_image, 2);

        check_inputs(m_custom_generation_config, initial_image);
        set_lora_adapters(m_custom_generation_config.adapters);

        std::shared_ptr<ThreadedCallbackWrapper> callback_ptr = nullptr;
        auto callback_iter = properties.find(ov::genai::callback.name());
        if (callback_iter != properties.end()) {
            callback_ptr = std::make_shared<ThreadedCallbackWrapper>(
                callback_iter->second.as<std::function<bool(size_t, size_t, ov::Tensor&)>>());
            callback_ptr->start();
        }

        ov::Tensor prompt_embeds = encode_prompt(positive_prompt);
        ov::Tensor text_ids = prepare_flux2_text_ids(prompt_embeds.get_shape()[1]);

        ov::Tensor negative_prompt_embeds;
        ov::Tensor negative_text_ids;
        const bool do_cfg = (m_custom_generation_config.guidance_scale > 1.0f) && !m_is_distilled;
        if (do_cfg) {
            negative_prompt_embeds = encode_prompt("");
            negative_text_ids = prepare_flux2_text_ids(negative_prompt_embeds.get_shape()[1]);
        }

        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        const size_t latent_h = static_cast<size_t>(m_custom_generation_config.height) / vae_scale_factor / 2;
        const size_t latent_w = static_cast<size_t>(m_custom_generation_config.width) / vae_scale_factor / 2;
        ov::Tensor latent_ids = prepare_flux2_latent_ids(latent_h, latent_w);

        ov::Tensor image_latents;
        ov::Tensor image_latent_ids;
        if (initial_image) {
            ov::Tensor processed = m_image_resizer->execute(initial_image,
                                                            m_custom_generation_config.height,
                                                            m_custom_generation_config.width);
            processed = m_image_processor->execute(processed);
            image_latents = encode_image_latents(processed);
            image_latent_ids = prepare_flux2_ref_ids(1, latent_h, latent_w);
        }

        size_t image_seq_len = (m_custom_generation_config.height / static_cast<int64_t>(vae_scale_factor) / 2) *
                               (m_custom_generation_config.width / static_cast<int64_t>(vae_scale_factor) / 2);
        m_scheduler->set_timesteps(image_seq_len, m_custom_generation_config.num_inference_steps, m_custom_generation_config.strength);

        std::vector<float> timesteps = m_scheduler->get_float_timesteps();

        ov::Tensor latents, processed_image, image_latent, noise;
        std::tie(latents, processed_image, image_latent, noise) = prepare_latents({}, m_custom_generation_config);

        ov::Tensor timestep(ov::element::f32, {latents.get_shape()[0]});
        float* timestep_data = timestep.data<float>();

        for (size_t step = 0; step < timesteps.size(); ++step) {
            const float t = timesteps[step] / 1000.0f;
            std::fill_n(timestep_data, timestep.get_size(), t);

            ov::Tensor model_input = latents;
            ov::Tensor model_img_ids = latent_ids;
            if (image_latents) {
                model_input = concat_seq3d(latents, image_latents);
                model_img_ids = concat_seq2d(latent_ids, image_latent_ids);
            }

            m_transformer->set_hidden_states("encoder_hidden_states", prompt_embeds);
            m_transformer->set_hidden_states("txt_ids", text_ids);
            m_transformer->set_hidden_states("img_ids", model_img_ids);
            ov::Tensor noise_pred = m_transformer->infer(model_input, timestep);

            if (noise_pred.get_shape()[1] != latents.get_shape()[1]) {
                const auto ns = noise_pred.get_shape();
                const auto ls = latents.get_shape();
                ov::Tensor cropped(noise_pred.get_element_type(), {ns[0], ls[1], ns[2]});
                const float* src = noise_pred.data<const float>();
                float* dst = cropped.data<float>();
                std::copy(src, src + cropped.get_size(), dst);
                noise_pred = cropped;
            }

            if (do_cfg) {
                m_transformer->set_hidden_states("encoder_hidden_states", negative_prompt_embeds);
                m_transformer->set_hidden_states("txt_ids", negative_text_ids);
                m_transformer->set_hidden_states("img_ids", model_img_ids);
                ov::Tensor neg_noise_pred = m_transformer->infer(model_input, timestep);

                if (neg_noise_pred.get_shape()[1] != latents.get_shape()[1]) {
                    const auto ns = neg_noise_pred.get_shape();
                    const auto ls = latents.get_shape();
                    ov::Tensor cropped(neg_noise_pred.get_element_type(), {ns[0], ls[1], ns[2]});
                    const float* src = neg_noise_pred.data<const float>();
                    float* dst = cropped.data<float>();
                    std::copy(src, src + cropped.get_size(), dst);
                    neg_noise_pred = cropped;
                }

                float* np = noise_pred.data<float>();
                const float* nn = neg_noise_pred.data<const float>();
                for (size_t i = 0; i < noise_pred.get_size(); ++i) {
                    np[i] = nn[i] + static_cast<float>(m_custom_generation_config.guidance_scale) * (np[i] - nn[i]);
                }
            }

            auto scheduler_step_result = m_scheduler->step(noise_pred, latents, step, m_custom_generation_config.generator);
            latents = scheduler_step_result["latent"];

            if (callback_ptr && callback_ptr->has_callback() && callback_ptr->write(step, timesteps.size(), latents) == CallbackStatus::STOP) {
                callback_ptr->end();
                m_perf_metrics.generate_duration =
                    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - gen_start).count();
                return ov::Tensor(ov::element::u8, {});
            }
        }

        if (callback_ptr) {
            callback_ptr->end();
        }

        const auto decode_start = std::chrono::steady_clock::now();
        auto image = decode(latents);
        m_perf_metrics.vae_decoder_inference_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - decode_start).count();
        m_perf_metrics.generate_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - gen_start).count();

        return image;
    }

    ImageGenerationPerfMetrics get_performance_metrics() override {
        m_perf_metrics.load_time = m_load_time_ms;
        return m_perf_metrics;
    }

protected:
    explicit Flux2KleinPipeline(PipelineType pipeline_type)
        : DiffusionPipeline(pipeline_type) {}

    void initialize_generation_config(const std::string& class_name) override {
        OPENVINO_ASSERT(m_transformer != nullptr);
        OPENVINO_ASSERT(m_vae != nullptr);

        const auto& transformer_config = m_transformer->get_config();
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();

        m_generation_config = ImageGenerationConfig();
        m_generation_config.height = transformer_config.m_default_sample_size * vae_scale_factor;
        m_generation_config.width = transformer_config.m_default_sample_size * vae_scale_factor;

        if (class_name == "Flux2KleinPipeline") {
            m_generation_config.guidance_scale = 1.0f;
            m_generation_config.num_inference_steps = 4;
            m_generation_config.strength = 1.0f;
            m_generation_config.max_sequence_length = 512;
        } else {
            OPENVINO_THROW("Unsupported class_name '", class_name, "' for Flux2KleinPipeline");
        }
    }

    void check_image_size(const int height, const int width) const override {
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        OPENVINO_ASSERT((height % (vae_scale_factor * 2) == 0 || height < 0) &&
                        (width % (vae_scale_factor * 2) == 0 || width < 0),
                        "Both 'width' and 'height' must be divisible by ", vae_scale_factor * 2);
    }

    void check_inputs(const ImageGenerationConfig& generation_config, ov::Tensor initial_image) const override {
        check_image_size(generation_config.height, generation_config.width);
        OPENVINO_ASSERT(generation_config.max_sequence_length <= 512,
                        "Flux2Klein max_sequence_length must be <= 512");
        OPENVINO_ASSERT(generation_config.negative_prompt_2 == std::nullopt,
                        "negative_prompt_2 is not used by Flux2KleinPipeline");
        OPENVINO_ASSERT(generation_config.negative_prompt_3 == std::nullopt,
                        "negative_prompt_3 is not used by Flux2KleinPipeline");
        OPENVINO_ASSERT(generation_config.prompt_2 == std::nullopt,
                        "prompt_2 is not used by Flux2KleinPipeline");
        OPENVINO_ASSERT(generation_config.prompt_3 == std::nullopt,
                        "prompt_3 is not used by Flux2KleinPipeline");

        if (m_pipeline_type == PipelineType::TEXT_2_IMAGE) {
            OPENVINO_ASSERT(!initial_image, "initial_image must be empty for text2image");
            OPENVINO_ASSERT(generation_config.strength == 1.0f,
                            "strength must be 1.0 for text2image");
        } else if (m_pipeline_type == PipelineType::IMAGE_2_IMAGE) {
            OPENVINO_ASSERT(initial_image, "initial_image must be provided for image2image");
            OPENVINO_ASSERT(generation_config.strength >= 0.0f && generation_config.strength <= 1.0f,
                            "strength must be in [0,1] for image2image");
        } else {
            OPENVINO_THROW("Flux2KleinPipeline supports only text2image and image2image");
        }
    }

    size_t get_config_in_channels() const override {
        return m_transformer->get_config().in_channels;
    }

    void compute_dim(int64_t& generation_config_value, ov::Tensor, int) {
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        const auto& transformer_config = m_transformer->get_config();
        if (generation_config_value < 0)
            generation_config_value = transformer_config.m_default_sample_size * vae_scale_factor;
    }

    void compute_hidden_states(const std::string& positive_prompt, const ImageGenerationConfig& generation_config) override {
        m_prompt_embeds = encode_prompt(positive_prompt);

        const bool do_cfg = (generation_config.guidance_scale > 1.0f) && !m_is_distilled;
        if (do_cfg) {
            m_negative_prompt_embeds = encode_prompt("");
        } else {
            m_negative_prompt_embeds = ov::Tensor();
        }
    }

    ov::Tensor encode_prompt(const std::string& prompt) {
        OPENVINO_ASSERT(m_text_encoder_request, "Text encoder must be compiled first");

        ChatHistory history{{{{"role", "user"}, {"content", prompt}}}};
        JsonContainer extra = JsonContainer::object();
        extra["enable_thinking"] = false;
        history.set_extra_context(extra);

        std::string rendered = m_tokenizer.apply_chat_template(history, true);
        ov::AnyMap tok_params = {
            {"max_length", static_cast<int64_t>(m_custom_generation_config.max_sequence_length)},
            {ov::genai::pad_to_max_length.name(), true},
            {ov::genai::add_special_tokens.name(), true},
        };
        auto tokenized = m_tokenizer.encode(rendered, tok_params);

        m_text_encoder_request.set_tensor("input_ids", tokenized.input_ids);
        m_text_encoder_request.set_tensor("attention_mask", tokenized.attention_mask);
        m_text_encoder_request.infer();
        return m_text_encoder_request.get_output_tensor();
    }

    ov::Tensor encode_image_latents(const ov::Tensor& image) {
        OPENVINO_ASSERT(m_vae_encoder_request, "VAE encoder must be compiled first");

        m_vae_encoder_request.set_tensor("sample", image);
        m_vae_encoder_request.infer();

        ov::Tensor params = m_vae_encoder_request.get_output_tensor();  // [B,64,H,W]
        const auto& ps = params.get_shape();
        OPENVINO_ASSERT(ps.size() == 4 && ps[1] % 2 == 0,
                        "Unexpected vae_encoder output shape for latent_parameters");

        ov::Tensor mean(params.get_element_type(), {ps[0], ps[1] / 2, ps[2], ps[3]});
        const float* src = params.data<const float>();
        float* dst = mean.data<float>();
        std::copy(src, src + mean.get_size(), dst);

        ov::Tensor patched = patchify_latents(mean);
        apply_bn_norm(patched);
        return pack_ref_latents(patched);
    }

    void load_bn_stats(const std::filesystem::path& npz_path) {
        m_bn_mean.clear();
        m_bn_std.clear();
        if (!std::filesystem::exists(npz_path)) {
            return;
        }
        std::vector<float> running_mean, running_var;
        const bool ok_mean = read_npz_float_array(npz_path, "running_mean", running_mean);
        const bool ok_var = read_npz_float_array(npz_path, "running_var", running_var);
        if (!ok_mean || !ok_var || running_mean.size() != running_var.size()) {
            return;
        }
        const float eps = 1e-4f;
        m_bn_mean = std::move(running_mean);
        m_bn_std.resize(running_var.size());
        for (size_t i = 0; i < running_var.size(); ++i) {
            m_bn_std[i] = std::sqrt(running_var[i] + eps);
        }
    }

    void apply_bn_norm(ov::Tensor& patched_latents) const {
        if (m_bn_mean.size() != patched_latents.get_shape()[1] || m_bn_std.size() != patched_latents.get_shape()[1]) {
            return;
        }

        const auto& s = patched_latents.get_shape();
        const size_t b = s[0], c = s[1], h = s[2], w = s[3];
        float* data = patched_latents.data<float>();

        for (size_t bi = 0; bi < b; ++bi) {
            for (size_t ci = 0; ci < c; ++ci) {
                const float mean = m_bn_mean[ci];
                const float std = m_bn_std[ci];
                for (size_t hi = 0; hi < h; ++hi) {
                    for (size_t wi = 0; wi < w; ++wi) {
                        const size_t idx = ((bi * c + ci) * h + hi) * w + wi;
                        data[idx] = (data[idx] - mean) / std;
                    }
                }
            }
        }
    }

    void apply_bn_denorm(ov::Tensor& patched_latents) const {
        if (m_bn_mean.size() != patched_latents.get_shape()[1] || m_bn_std.size() != patched_latents.get_shape()[1]) {
            return;
        }

        const auto& s = patched_latents.get_shape();
        const size_t b = s[0], c = s[1], h = s[2], w = s[3];
        float* data = patched_latents.data<float>();

        for (size_t bi = 0; bi < b; ++bi) {
            for (size_t ci = 0; ci < c; ++ci) {
                const float mean = m_bn_mean[ci];
                const float std = m_bn_std[ci];
                for (size_t hi = 0; hi < h; ++hi) {
                    for (size_t wi = 0; wi < w; ++wi) {
                        const size_t idx = ((bi * c + ci) * h + hi) * w + wi;
                        data[idx] = data[idx] * std + mean;
                    }
                }
            }
        }
    }

    void set_lora_adapters(std::optional<AdapterConfig> adapters) {
        if (adapters) {
            if (auto updated_adapters = derived_adapters(*adapters)) {
                adapters = updated_adapters;
            }
            m_transformer->set_adapters(adapters);
        }
    }

    static std::optional<AdapterConfig> derived_adapters(const AdapterConfig& adapters) {
        return ov::genai::derived_adapters(adapters, flux_adapter_normalization);
    }

private:
    std::shared_ptr<FluxTransformer2DModel> m_transformer = nullptr;
    std::shared_ptr<AutoencoderKL> m_vae = nullptr;

    Tokenizer m_tokenizer;
    std::shared_ptr<ov::Model> m_text_encoder_model;
    ov::InferRequest m_text_encoder_request;

    std::shared_ptr<ov::Model> m_vae_encoder_model;
    ov::InferRequest m_vae_encoder_request;

    std::vector<float> m_bn_mean;
    std::vector<float> m_bn_std;

    bool m_is_distilled = true;

    ov::Tensor m_prompt_embeds;
    ov::Tensor m_negative_prompt_embeds;

    ImageGenerationConfig m_custom_generation_config;
};

}  // namespace genai
}  // namespace ov
