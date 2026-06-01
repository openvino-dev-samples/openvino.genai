# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import pytest
import openvino_genai as ov_genai

# Z-Image-Turbo is a ~6B model with no tiny-random variant published, so these
# tests run only when a converted model is provided locally via the
# ZIMAGE_MODEL_PATH environment variable. They are skipped otherwise so CI is
# not forced to download the full model.
ZIMAGE_MODEL_PATH = os.environ.get("ZIMAGE_MODEL_PATH")

requires_model = pytest.mark.skipif(
    not (ZIMAGE_MODEL_PATH and os.path.isdir(ZIMAGE_MODEL_PATH)),
    reason="ZIMAGE_MODEL_PATH is not set to a converted Z-Image model directory",
)


class TestZImageModels:
    def test_transformer_classes_are_exported(self):
        assert hasattr(ov_genai, "ZImageTransformer2DModel")
        assert hasattr(ov_genai, "ZImageTextEncoder")

    @requires_model
    def test_transformer_loads_and_reports_config(self):
        transformer = ov_genai.ZImageTransformer2DModel(
            os.path.join(ZIMAGE_MODEL_PATH, "transformer")
        )
        config = transformer.get_config()
        assert config.in_channels == 16

    @requires_model
    def test_text_encoder_loads(self):
        text_encoder = ov_genai.ZImageTextEncoder(
            os.path.join(ZIMAGE_MODEL_PATH, "text_encoder")
        )
        assert text_encoder is not None


@requires_model
class TestZImageText2Image:
    def test_pipeline_is_auto_detected(self):
        pipe = ov_genai.Text2ImagePipeline(ZIMAGE_MODEL_PATH, "CPU")
        assert pipe is not None

    def test_generate_produces_image(self):
        pipe = ov_genai.Text2ImagePipeline(ZIMAGE_MODEL_PATH, "CPU")
        image = pipe.generate(
            "a photo of a red panda",
            width=256,
            height=256,
            num_inference_steps=2,
            guidance_scale=0.0,
            generator=ov_genai.TorchGenerator(42),
        )
        arr = np.array(image.data, copy=True).reshape(image.get_shape())
        assert arr.shape == (1, 256, 256, 3)
        assert arr.dtype == np.uint8
        # A real decode produces varied pixels, not a constant frame.
        assert arr.std() > 1.0

    def test_generate_is_deterministic_with_seed(self):
        pipe = ov_genai.Text2ImagePipeline(ZIMAGE_MODEL_PATH, "CPU")
        kwargs = dict(
            width=256,
            height=256,
            num_inference_steps=2,
            guidance_scale=0.0,
        )
        first = pipe.generate("a red panda", generator=ov_genai.TorchGenerator(7), **kwargs)
        second = pipe.generate("a red panda", generator=ov_genai.TorchGenerator(7), **kwargs)
        first_arr = np.array(first.data, copy=True).reshape(first.get_shape())
        second_arr = np.array(second.data, copy=True).reshape(second.get_shape())
        assert np.array_equal(first_arr, second_arr)
