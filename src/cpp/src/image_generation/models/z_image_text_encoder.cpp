// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/z_image_text_encoder.hpp"

#include <fstream>

#include "json_utils.hpp"
#include "lora/helper.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

std::filesystem::path get_tokenizer_path_by_text_encoder(const std::filesystem::path& text_encoder_path);

ZImageTextEncoder::Config::Config(const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;

    read_json_param(data, "hidden_state_index", hidden_state_index);
    read_json_param(data, "enable_thinking", enable_thinking);
}

ZImageTextEncoder::ZImageTextEncoder(const std::filesystem::path& root_dir)
    : m_config(root_dir / "config.json"),
      m_tokenizer(get_tokenizer_path_by_text_encoder(root_dir)) {
    m_model = utils::singleton_core().read_model(root_dir / "openvino_model.xml");
}

ZImageTextEncoder::ZImageTextEncoder(const std::filesystem::path& root_dir,
                                     const std::string& device,
                                     const ov::AnyMap& properties)
    : ZImageTextEncoder(root_dir) {
    compile(device, properties);
}

ZImageTextEncoder::ZImageTextEncoder(const ZImageTextEncoder&) = default;

std::shared_ptr<ZImageTextEncoder> ZImageTextEncoder::clone() {
    OPENVINO_ASSERT((m_model != nullptr) ^ static_cast<bool>(m_request),
                    "ZImageTextEncoder must have exactly one of m_model or m_request initialized");

    std::shared_ptr<ZImageTextEncoder> cloned = std::make_shared<ZImageTextEncoder>(*this);

    if (m_model) {
        cloned->m_model = m_model->clone();
    } else {
        cloned->m_request = m_request.get_compiled_model().create_infer_request();
    }

    return cloned;
}

const ZImageTextEncoder::Config& ZImageTextEncoder::get_config() const {
    return m_config;
}

ZImageTextEncoder& ZImageTextEncoder::reshape(int batch_size, int max_sequence_length) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot reshape already compiled model");

    std::map<std::string, ov::PartialShape> name_to_shape;
    for (auto&& input : m_model->inputs()) {
        std::string input_name = input.get_any_name();
        ov::PartialShape shape = input.get_partial_shape();
        if (shape.size() >= 2) {
            shape[0] = batch_size;
            shape[1] = max_sequence_length;
        }
        name_to_shape[input_name] = shape;
    }
    m_model->reshape(name_to_shape);

    return *this;
}

ZImageTextEncoder& ZImageTextEncoder::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot re-compile already compiled model");
    ov::CompiledModel compiled_model = utils::singleton_core().compile_model(m_model, device, *extract_adapters_from_properties(properties));
    ov::genai::utils::print_compiled_model_properties(compiled_model, "Z-Image text encoder model");
    m_request = compiled_model.create_infer_request();
    // release the original model
    m_model.reset();

    return *this;
}

ov::Tensor ZImageTextEncoder::infer(const std::string& prompt, int max_sequence_length) {
    OPENVINO_ASSERT(m_request, "Z-Image text encoder model must be compiled first. Cannot infer non-compiled model");

    // 1. Wrap the prompt into a chat template (Qwen3-style instruction format).
    ChatHistory history({{{"role", "user"}, {"content", prompt}}});
    JsonContainer extra_context({{"enable_thinking", m_config.enable_thinking}});
    const std::string templated_prompt = m_tokenizer.apply_chat_template(history, /*add_generation_prompt=*/true, /*chat_template=*/{}, /*tools=*/std::nullopt, extra_context);

    // 2. Tokenize. The chat template already inserts the required special tokens,
    //    so we disable add_special_tokens and pad/truncate to max_sequence_length.
    TokenizedInputs tokenized = m_tokenizer.encode(templated_prompt,
                                                   ov::AnyMap{{"add_special_tokens", false},
                                                              {"max_length", max_sequence_length},
                                                              {"pad_to_max_length", true}});

    ov::Tensor input_ids = tokenized.input_ids;
    ov::Tensor attention_mask = tokenized.attention_mask;

    // store the attention mask so the pipeline can drop padded positions.
    m_prompt_attention_mask = ov::Tensor(attention_mask.get_element_type(), attention_mask.get_shape());
    attention_mask.copy_to(m_prompt_attention_mask);

    // 3. Feed inputs by name (the exported graph may or may not take attention_mask).
    auto compiled_inputs = m_request.get_compiled_model().inputs();
    for (const auto& port : compiled_inputs) {
        const std::string name = port.get_any_name();
        if (name == "input_ids") {
            m_request.set_tensor(name, input_ids);
        } else if (name == "attention_mask") {
            m_request.set_tensor(name, attention_mask);
        }
    }

    m_request.infer();

    // 4. Select the conditioning hidden state. Z-Image uses the penultimate
    //    hidden state of the full `output_hidden_states` tuple (`hidden_states[-2]`).
    //
    //    The exported graph exposes the tuple as separate ports: `hidden_states.0`
    //    ... `hidden_states.{L-1}` (one per decoder layer input) plus a final
    //    `last_hidden_state` port. The reference HF tuple therefore is
    //    `[hidden_states.0, ..., hidden_states.{L-1}, last_hidden_state]` and has
    //    `L + 1` entries, so `hidden_states[-2]` maps to `hidden_states.{L-1}`
    //    (the highest-numbered `hidden_states.N` port).
    auto compiled_outputs = m_request.get_compiled_model().outputs();
    int max_hidden_state_index = -1;
    size_t selected_output = 0;
    bool found = false;
    size_t last_hidden_state_output = 0;
    bool has_last_hidden_state = false;
    for (size_t i = 0; i < compiled_outputs.size(); ++i) {
        for (const auto& nm : compiled_outputs[i].get_names()) {
            if (nm == "last_hidden_state") {
                last_hidden_state_output = i;
                has_last_hidden_state = true;
            }
            const std::string prefix = "hidden_states.";
            if (nm.rfind(prefix, 0) == 0) {
                int n = std::atoi(nm.substr(prefix.size()).c_str());
                if (n > max_hidden_state_index) {
                    max_hidden_state_index = n;
                    selected_output = i;
                    found = true;
                }
            }
        }
    }

    if (found) {
        // Build the full tuple size: numbered hidden states (0..max) + last_hidden_state.
        const int num_numbered = max_hidden_state_index + 1;
        const int tuple_size = num_numbered + (has_last_hidden_state ? 1 : 0);
        int idx = m_config.hidden_state_index;
        if (idx < 0) {
            idx += tuple_size;
        }
        idx = std::max(0, std::min(idx, tuple_size - 1));
        if (has_last_hidden_state && idx == tuple_size - 1) {
            return m_request.get_output_tensor(last_hidden_state_output);
        }
        // idx now refers to a numbered hidden_states.N port; locate it by name.
        const std::string target = "hidden_states." + std::to_string(idx);
        for (size_t i = 0; i < compiled_outputs.size(); ++i) {
            for (const auto& nm : compiled_outputs[i].get_names()) {
                if (nm == target) {
                    return m_request.get_output_tensor(i);
                }
            }
        }
        // Fall back to the highest-numbered hidden state if exact match is missing.
        return m_request.get_output_tensor(selected_output);
    }

    if (has_last_hidden_state) {
        return m_request.get_output_tensor(last_hidden_state_output);
    }

    return m_request.get_output_tensor(0);
}

ov::Tensor ZImageTextEncoder::get_output_tensor(const size_t idx) {
    return m_request.get_output_tensor(idx);
}

ov::Tensor ZImageTextEncoder::get_prompt_attention_mask() const {
    OPENVINO_ASSERT(m_prompt_attention_mask, "Prompt attention mask must be initialized before use. You must call infer.");
    return m_prompt_attention_mask;
}

}  // namespace genai
}  // namespace ov
