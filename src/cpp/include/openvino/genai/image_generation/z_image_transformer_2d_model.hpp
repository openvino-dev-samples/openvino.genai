// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>

#include "openvino/core/any.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/genai/lora_adapter.hpp"

#include "openvino/genai/visibility.hpp"

namespace ov {
namespace genai {

// S3-DiT (Single-Stream Diffusion Transformer) used by Z-Image / Z-Image-Turbo.
//
// The OpenVINO export wraps the whole transformer into a single graph with the
// following I/O contract (see optimum-intel `OVModelTransformer.forward`):
//   inputs:
//     - "hidden_states"          : packed latents, shape [batch, in_channels, 1, H, W]
//     - "timestep"               : normalized timestep, shape [batch]
//     - "encoder_hidden_states"  : text embeddings from the LLM text encoder
//   output:
//     - predicted flow (same spatial layout as "hidden_states")
class OPENVINO_GENAI_EXPORTS ZImageTransformer2DModel {
public:
    struct OPENVINO_GENAI_EXPORTS Config {
        size_t in_channels = 16;
        size_t m_default_sample_size = 128;

        explicit Config(const std::filesystem::path& config_path);
    };

    explicit ZImageTransformer2DModel(const std::filesystem::path& root_dir);

    ZImageTransformer2DModel(const std::filesystem::path& root_dir,
                             const std::string& device,
                             const ov::AnyMap& properties = {});

    ZImageTransformer2DModel(const std::string& model,
                             const Tensor& weights,
                             const Config& config,
                             const size_t vae_scale_factor);

    ZImageTransformer2DModel(const std::string& model,
                             const Tensor& weights,
                             const Config& config,
                             const size_t vae_scale_factor,
                             const std::string& device,
                             const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    ZImageTransformer2DModel(const std::filesystem::path& root_dir, const std::string& device, Properties&&... properties)
        : ZImageTransformer2DModel(root_dir, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    ZImageTransformer2DModel(const std::string& model,
                             const Tensor& weights,
                             const Config& config,
                             const size_t vae_scale_factor,
                             const std::string& device,
                             Properties&&... properties)
        : ZImageTransformer2DModel(model, weights, config, vae_scale_factor, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    ZImageTransformer2DModel(const ZImageTransformer2DModel&);

    ZImageTransformer2DModel clone();

    const Config& get_config() const;

    ZImageTransformer2DModel& reshape(int batch_size, int height, int width, int max_sequence_length);

    ZImageTransformer2DModel& compile(const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<ZImageTransformer2DModel&, Properties...> compile(const std::string& device,
                                                                                     Properties&&... properties) {
        return compile(device, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    void set_hidden_states(const std::string& tensor_name, ov::Tensor encoder_hidden_states);

    void set_adapters(const std::optional<AdapterConfig>& adapters);

    ov::Tensor infer(const ov::Tensor latent, const ov::Tensor timestep);

private:
    Config m_config;
    AdapterController m_adapter_controller;
    ov::InferRequest m_request;
    std::shared_ptr<ov::Model> m_model;
    size_t m_vae_scale_factor;
};

}  // namespace genai
}  // namespace ov
