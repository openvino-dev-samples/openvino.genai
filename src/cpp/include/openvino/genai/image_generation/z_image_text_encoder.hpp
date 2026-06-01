// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "openvino/genai/visibility.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/lora_adapter.hpp"

#include "openvino/core/any.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {
namespace genai {

// Text encoder used by Z-Image / Z-Image-Turbo.
//
// Unlike CLIP/T5 encoders, Z-Image uses an instruction-tuned decoder-only LLM
// (Qwen3-style). The prompt is wrapped into a chat template before tokenization
// and the *penultimate* hidden state is used as the conditioning signal
// (mirrors diffusers `ZImagePipeline.encode_prompt`, which reads
// `hidden_states[-2]`).
class OPENVINO_GENAI_EXPORTS ZImageTextEncoder {
public:
    struct OPENVINO_GENAI_EXPORTS Config {
        // index into the exported hidden-state outputs that should be used as
        // the conditioning embedding. Negative values count from the end, so -2
        // selects the second-to-last hidden state (the Z-Image default).
        int hidden_state_index = -2;
        // whether to enable the "thinking" branch of the chat template (Qwen3).
        bool enable_thinking = true;

        explicit Config(const std::filesystem::path& config_path);
    };

    explicit ZImageTextEncoder(const std::filesystem::path& root_dir);

    ZImageTextEncoder(const std::filesystem::path& root_dir,
                      const std::string& device,
                      const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    ZImageTextEncoder(const std::filesystem::path& root_dir, const std::string& device, Properties&&... properties)
        : ZImageTextEncoder(root_dir, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    ZImageTextEncoder(const ZImageTextEncoder&);

    std::shared_ptr<ZImageTextEncoder> clone();

    const Config& get_config() const;

    ZImageTextEncoder& reshape(int batch_size, int max_sequence_length);

    ZImageTextEncoder& compile(const std::string& device, const ov::AnyMap& properties = {});

    template <typename... Properties>
    ov::util::EnableIfAllStringAny<ZImageTextEncoder&, Properties...> compile(const std::string& device,
                                                                              Properties&&... properties) {
        return compile(device, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    // Runs the encoder for a single prompt and returns the conditioning hidden
    // state with shape [1, max_sequence_length, hidden_size]. The attention mask
    // of the templated/tokenized prompt is stored and accessible via
    // `get_prompt_attention_mask()` so the pipeline can drop padded positions.
    ov::Tensor infer(const std::string& prompt, int max_sequence_length);

    ov::Tensor get_output_tensor(const size_t idx);

    ov::Tensor get_prompt_attention_mask() const;

private:
    Config m_config;
    ov::InferRequest m_request;
    std::shared_ptr<ov::Model> m_model;
    ov::Tensor m_prompt_attention_mask;
    Tokenizer m_tokenizer;
};

}  // namespace genai
}  // namespace ov
