// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cassert>

#include "image_generation/diffusion_pipeline.hpp"
#include "image_generation/numpy_utils.hpp"
#include "image_generation/threaded_callback.hpp"

#include "openvino/genai/image_generation/autoencoder_kl.hpp"
#include "openvino/genai/image_generation/z_image_transformer_2d_model.hpp"
#include "openvino/genai/image_generation/z_image_text_encoder.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

// Pipeline for the Z-Image / Z-Image-Turbo text-to-image model.
//
// Z-Image uses a Single-Stream Diffusion Transformer (S3-DiT) trained with
// flow-matching. Compared to Flux:
//   * the text encoder is an instruction-tuned LLM (Qwen3-style) and the prompt
//     embeddings are the penultimate hidden state, trimmed to the valid tokens;
//   * latents are NOT packed - the transformer consumes them as
//     [batch, in_channels, 1, height, width];
//   * the transformer timestep input is `(1000 - t) / 1000` and the predicted
//     flow is negated before the scheduler step;
//   * Z-Image-Turbo runs without classifier-free guidance (guidance_scale = 0).
class ZImagePipeline : public DiffusionPipeline {
public:
    ZImagePipeline(PipelineType pipeline_type, const std::filesystem::path& root_dir)
        : ZImagePipeline(pipeline_type) {
        m_root_dir = root_dir;
        const std::filesystem::path model_index_path = root_dir / "model_index.json";
        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

        nlohmann::json data = nlohmann::json::parse(file);
        using utils::read_json_param;

        set_scheduler(Scheduler::from_config(root_dir / "scheduler/scheduler_config.json"));

        const std::string text_encoder = data["text_encoder"][1].get<std::string>();
        m_text_encoder = std::make_shared<ZImageTextEncoder>(root_dir / "text_encoder");

        const std::string vae = data["vae"][1].get<std::string>();
        OPENVINO_ASSERT(vae == "AutoencoderKL", "Unsupported '", vae, "' VAE type");
        m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_decoder");

        const std::string transformer = data["transformer"][1].get<std::string>();
        m_transformer = std::make_shared<ZImageTransformer2DModel>(root_dir / "transformer");

        const std::string class_name = data["_class_name"].get<std::string>();
        initialize_generation_config(class_name);
    }

    ZImagePipeline(PipelineType pipeline_type,
                   const std::filesystem::path& root_dir,
                   const std::string& device,
                   const ov::AnyMap& properties)
        : ZImagePipeline(pipeline_type) {
        m_root_dir = root_dir;
        const std::filesystem::path model_index_path = root_dir / "model_index.json";
        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

        nlohmann::json data = nlohmann::json::parse(file);
        using utils::read_json_param;

        set_scheduler(Scheduler::from_config(root_dir / "scheduler/scheduler_config.json"));

        const std::string text_encoder = data["text_encoder"][1].get<std::string>();
        m_text_encoder = std::make_shared<ZImageTextEncoder>(root_dir / "text_encoder", device, properties);

        const std::string vae = data["vae"][1].get<std::string>();
        OPENVINO_ASSERT(vae == "AutoencoderKL", "Unsupported '", vae, "' VAE type");
        m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_decoder", device, properties);

        const std::string transformer = data["transformer"][1].get<std::string>();
        m_transformer = std::make_shared<ZImageTransformer2DModel>(root_dir / "transformer", device, properties);

        const std::string class_name = data["_class_name"].get<std::string>();
        initialize_generation_config(class_name);
    }

    ZImagePipeline(PipelineType pipeline_type,
                   const ZImageTextEncoder& text_encoder,
                   const ZImageTransformer2DModel& transformer,
                   const AutoencoderKL& vae)
        : ZImagePipeline(pipeline_type) {
        m_text_encoder = std::make_shared<ZImageTextEncoder>(text_encoder);
        m_transformer = std::make_shared<ZImageTransformer2DModel>(transformer);
        m_vae = std::make_shared<AutoencoderKL>(vae);
        initialize_generation_config("ZImagePipeline");
    }

    void reshape(const int num_images_per_prompt,
                 const int height,
                 const int width,
                 const float guidance_scale) override {
        check_image_size(height, width);

        m_text_encoder->reshape(1, m_generation_config.max_sequence_length);
        m_transformer->reshape(num_images_per_prompt, height, width, m_generation_config.max_sequence_length);
        m_vae->reshape(num_images_per_prompt, height, width);
    }

    void compile(const std::string& text_encode_device,
                 const std::string& denoise_device,
                 const std::string& vae_device,
                 const ov::AnyMap& properties) override {
        m_text_encoder->compile(text_encode_device, properties);
        m_transformer->compile(denoise_device, properties);
        m_vae->compile(vae_device, properties);
    }

    std::shared_ptr<DiffusionPipeline> clone() override {
        OPENVINO_ASSERT(!m_root_dir.empty(), "Cannot clone pipeline without root directory");

        std::shared_ptr<AutoencoderKL> vae = std::make_shared<AutoencoderKL>(m_vae->clone());
        std::shared_ptr<ZImageTextEncoder> text_encoder = m_text_encoder->clone();
        std::shared_ptr<ZImageTransformer2DModel> transformer = std::make_shared<ZImageTransformer2DModel>(m_transformer->clone());

        std::shared_ptr<ZImagePipeline> pipeline = std::make_shared<ZImagePipeline>(m_pipeline_type, *text_encoder, *transformer, *vae);

        pipeline->m_root_dir = m_root_dir;
        pipeline->set_scheduler(Scheduler::from_config(m_root_dir / "scheduler/scheduler_config.json"));
        pipeline->set_generation_config(m_generation_config);
        return pipeline;
    }

    void compute_hidden_states(const std::string& positive_prompt, const ImageGenerationConfig& generation_config) override {
        auto infer_start = std::chrono::steady_clock::now();
        ov::Tensor prompt_embeds = m_text_encoder->infer(positive_prompt, generation_config.max_sequence_length);
        auto infer_duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - infer_start).count();
        m_perf_metrics.encoder_inference_duration["text_encoder"] = infer_duration;

        ov::Tensor attention_mask = m_text_encoder->get_prompt_attention_mask();

        // trim the padded positions: keep only tokens where attention_mask == 1
        ov::Tensor trimmed = trim_prompt_embeds(prompt_embeds, attention_mask);

        // repeat for num_images_per_prompt
        trimmed = numpy_utils::repeat(trimmed, generation_config.num_images_per_prompt);

        m_transformer->set_hidden_states("encoder_hidden_states", trimmed);
    }

    std::tuple<ov::Tensor, ov::Tensor, ov::Tensor, ov::Tensor> prepare_latents(ov::Tensor initial_image, const ImageGenerationConfig& generation_config) override {
        OPENVINO_ASSERT(m_pipeline_type == PipelineType::TEXT_2_IMAGE,
                        "ZImagePipeline currently supports text-to-image generation only");

        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        const size_t num_channels_latents = m_transformer->get_config().in_channels;
        const size_t height = generation_config.height / vae_scale_factor;
        const size_t width = generation_config.width / vae_scale_factor;

        ov::Shape latent_shape{generation_config.num_images_per_prompt, num_channels_latents, height, width};
        ov::Tensor latent = generation_config.generator->randn_tensor(latent_shape);

        ov::Tensor processed_image, image_latents, noise;
        return std::make_tuple(latent, processed_image, image_latents, noise);
    }

    void set_lora_adapters(std::optional<AdapterConfig> adapters) override {
        if (adapters) {
            m_transformer->set_adapters(adapters);
        }
    }

    std::tuple<ov::Tensor, ov::Tensor> prepare_mask_latents(ov::Tensor mask_image,
                                                            ov::Tensor processed_image,
                                                            const ImageGenerationConfig& generation_config,
                                                            const size_t batch_size_multiplier = 1) override {
        OPENVINO_THROW("Inpainting is not supported by ZImagePipeline");
    }

    ov::Tensor generate(const std::string& positive_prompt,
                        ov::Tensor initial_image,
                        ov::Tensor mask_image,
                        const ov::AnyMap& properties) override {
        const auto gen_start = std::chrono::steady_clock::now();
        m_perf_metrics.clean_up();
        m_custom_generation_config = m_generation_config;
        m_custom_generation_config.update_generation_config(properties);

        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();

        if (m_custom_generation_config.height < 0)
            compute_dim(m_custom_generation_config.height, initial_image, 1);
        if (m_custom_generation_config.width < 0)
            compute_dim(m_custom_generation_config.width, initial_image, 2);

        check_inputs(m_custom_generation_config, initial_image);

        set_lora_adapters(m_custom_generation_config.adapters);

        std::shared_ptr<ThreadedCallbackWrapper> callback_ptr = nullptr;
        auto callback_iter = properties.find(ov::genai::callback.name());
        if (callback_iter != properties.end()) {
            callback_ptr = std::make_shared<ThreadedCallbackWrapper>(callback_iter->second.as<std::function<bool(size_t, size_t, ov::Tensor&)>>());
            callback_ptr->start();
        }

        compute_hidden_states(positive_prompt, m_custom_generation_config);

        // Z-Image uses the standard FlowMatchEuler schedule (static shift, the
        // resolution-dependent `mu` is unused because `use_dynamic_shifting` is
        // false). This matches diffusers' default path of
        // `linspace(sigma_to_t(sigma_max), sigma_to_t(sigma_min), num_steps)`.
        m_scheduler->set_timesteps(m_custom_generation_config.num_inference_steps, m_custom_generation_config.strength);

        std::vector<float> timesteps = m_scheduler->get_float_timesteps();

        ov::Tensor latents, processed_image, image_latent, noise;
        std::tie(latents, processed_image, image_latent, noise) = prepare_latents(initial_image, m_custom_generation_config);

        const ov::Shape latents_shape = latents.get_shape();  // [B, C, H, W]
        const ov::Shape latents_shape_5d{latents_shape[0], latents_shape[1], 1, latents_shape[2], latents_shape[3]};

        ov::Tensor timestep(ov::element::f32, {latents_shape[0]});
        float* timestep_data = timestep.data<float>();

        for (size_t inference_step = 0; inference_step < timesteps.size(); ++inference_step) {
            auto step_start = std::chrono::steady_clock::now();

            // Z-Image timestep convention: (1000 - t) / 1000
            const float t_value = (1000.0f - timesteps[inference_step]) / 1000.0f;
            std::fill_n(timestep_data, timestep.get_size(), t_value);

            // add the temporal dim expected by the transformer: [B, C, 1, H, W]
            ov::Tensor latent_model_input(latents.get_element_type(), latents_shape_5d, latents.data<float>());

            auto infer_start = std::chrono::steady_clock::now();
            ov::Tensor noise_pred_5d = m_transformer->infer(latent_model_input, timestep);
            auto infer_duration = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start);
            m_perf_metrics.raw_metrics.transformer_inference_durations.emplace_back(MicroSeconds(infer_duration));

            // squeeze temporal dim and negate the predicted flow
            ov::Tensor noise_pred(ov::element::f32, latents_shape);
            const float* src = noise_pred_5d.data<float>();
            float* dst = noise_pred.data<float>();
            for (size_t i = 0; i < noise_pred.get_size(); ++i) {
                dst[i] = -src[i];
            }

            auto scheduler_step_result = m_scheduler->step(noise_pred, latents, inference_step, m_custom_generation_config.generator);
            latents = scheduler_step_result["latent"];

            if (callback_ptr && callback_ptr->has_callback() && callback_ptr->write(inference_step, timesteps.size(), latents) == CallbackStatus::STOP) {
                callback_ptr->end();
                auto image = ov::Tensor(ov::element::u8, {});
                m_perf_metrics.generate_duration =
                    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - gen_start).count();
                return image;
            }

            auto step_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - step_start);
            m_perf_metrics.raw_metrics.iteration_durations.emplace_back(MicroSeconds(step_ms));
        }

        if (callback_ptr != nullptr) {
            callback_ptr->end();
        }

        const auto decode_start = std::chrono::steady_clock::now();
        auto image = m_vae->decode(latents);
        m_perf_metrics.vae_decoder_inference_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - decode_start).count();
        m_perf_metrics.generate_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - gen_start).count();
        return image;
    }

    ov::Tensor decode(const ov::Tensor latent) override {
        return m_vae->decode(latent);
    }

    ImageGenerationPerfMetrics get_performance_metrics() override {
        m_perf_metrics.load_time = m_load_time_ms;
        return m_perf_metrics;
    }

protected:
    explicit ZImagePipeline(PipelineType pipeline_type) : DiffusionPipeline(pipeline_type) {}

    // Build [B, valid_tokens, hidden] from [1, max_len, hidden] using attention_mask.
    static ov::Tensor trim_prompt_embeds(const ov::Tensor& prompt_embeds, const ov::Tensor& attention_mask) {
        const ov::Shape embeds_shape = prompt_embeds.get_shape();  // [1, max_len, hidden]
        const size_t max_len = embeds_shape[1];
        const size_t hidden = embeds_shape[2];

        // count valid tokens
        size_t valid = 0;
        auto count_valid = [&](auto* mask_data) {
            for (size_t i = 0; i < max_len; ++i) {
                if (mask_data[i] != 0) {
                    ++valid;
                }
            }
        };
        if (attention_mask.get_element_type() == ov::element::i64) {
            count_valid(attention_mask.data<int64_t>());
        } else {
            count_valid(attention_mask.data<int32_t>());
        }

        ov::Tensor trimmed(prompt_embeds.get_element_type(), ov::Shape{1, valid, hidden});
        const float* src = prompt_embeds.data<float>();
        float* dst = trimmed.data<float>();

        size_t row = 0;
        auto copy_rows = [&](auto* mask_data) {
            for (size_t i = 0; i < max_len; ++i) {
                if (mask_data[i] != 0) {
                    std::copy_n(src + i * hidden, hidden, dst + row * hidden);
                    ++row;
                }
            }
        };
        if (attention_mask.get_element_type() == ov::element::i64) {
            copy_rows(attention_mask.data<int64_t>());
        } else {
            copy_rows(attention_mask.data<int32_t>());
        }

        return trimmed;
    }

    void compute_dim(int64_t& generation_config_value, ov::Tensor initial_image, int dim_idx) {
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        const auto& transformer_config = m_transformer->get_config();
        if (generation_config_value < 0)
            generation_config_value = transformer_config.m_default_sample_size * vae_scale_factor;
    }

    void initialize_generation_config(const std::string& class_name) override {
        OPENVINO_ASSERT(m_transformer != nullptr);
        OPENVINO_ASSERT(m_vae != nullptr);

        const auto& transformer_config = m_transformer->get_config();
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();

        m_generation_config = ImageGenerationConfig();

        m_generation_config.height = transformer_config.m_default_sample_size * vae_scale_factor;
        m_generation_config.width = transformer_config.m_default_sample_size * vae_scale_factor;

        // Z-Image-Turbo is distilled: no classifier-free guidance, few steps.
        m_generation_config.guidance_scale = 0.0f;
        m_generation_config.num_inference_steps = 8;
        m_generation_config.strength = 1.0f;
        m_generation_config.max_sequence_length = 512;
    }

    void check_image_size(const int height, const int width) const override {
        OPENVINO_ASSERT(m_transformer != nullptr);
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        const size_t patch = vae_scale_factor * 2;
        OPENVINO_ASSERT((height % patch == 0 || height < 0) && (width % patch == 0 || width < 0),
                        "Both 'width' and 'height' must be divisible by ", patch);
    }

    void check_inputs(const ImageGenerationConfig& generation_config, ov::Tensor initial_image) const override {
        check_image_size(generation_config.height, generation_config.width);

        OPENVINO_ASSERT(generation_config.max_sequence_length <= 512, "Z-Image 'max_sequence_length' must be less or equal to 512");
        OPENVINO_ASSERT(generation_config.negative_prompt == std::nullopt, "Negative prompt is not used by ZImagePipeline");
        OPENVINO_ASSERT(generation_config.negative_prompt_2 == std::nullopt, "Negative prompt 2 is not used by ZImagePipeline");
        OPENVINO_ASSERT(generation_config.negative_prompt_3 == std::nullopt, "Negative prompt 3 is not used by ZImagePipeline");
        OPENVINO_ASSERT(generation_config.prompt_2 == std::nullopt, "Prompt 2 is not used by ZImagePipeline");
        OPENVINO_ASSERT(generation_config.prompt_3 == std::nullopt, "Prompt 3 is not used by ZImagePipeline");
        OPENVINO_ASSERT(!initial_image, "Internal error: initial_image must be empty for Text 2 image pipeline");
    }

    size_t get_config_in_channels() const override {
        OPENVINO_ASSERT(m_transformer != nullptr);
        return m_transformer->get_config().in_channels;
    }

    void blend_latents(ov::Tensor image_latent,
                       ov::Tensor noise,
                       ov::Tensor mask,
                       ov::Tensor latent,
                       size_t inference_step) override {
        OPENVINO_THROW("Inpainting is not supported by ZImagePipeline");
    }

    std::shared_ptr<ZImageTransformer2DModel> m_transformer = nullptr;
    std::shared_ptr<ZImageTextEncoder> m_text_encoder = nullptr;

    ImageGenerationConfig m_custom_generation_config;
};

}  // namespace genai
}  // namespace ov
