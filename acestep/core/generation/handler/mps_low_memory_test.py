"""Regression coverage for opt-in MPS initialization without duplicate backends."""

import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from acestep.core.generation.handler.init_service_test import _Host


class MpsLowMemoryTests(unittest.TestCase):
    """Exercise initialization policy and real small text checkpoint loading."""

    def test_initialization_limits_half_precision_to_opted_in_mps(self):
        """The initial model uses BF16 only for explicitly enabled MPS runs."""
        cases = (
            ("mps", "true", torch.bfloat16),
            ("mps", "false", torch.float32),
            ("cpu", "true", torch.float32),
        )
        for device, enabled, expected_dtype in cases:
            with self.subTest(device=device, enabled=enabled), ExitStack() as stack:
                host = _Host(project_root="/unused", device=device)

                def load_main(**_kwargs):
                    """Allocate small weights using the resolved initialization dtype."""
                    host.model = torch.nn.Linear(2, 2, dtype=host.dtype)
                    host.config = types.SimpleNamespace(_attn_implementation="sdpa")

                stack.enter_context(patch.dict("os.environ", {"ACESTEP_MPS_LOW_MEMORY": enabled}))
                stack.enter_context(patch.object(host, "_resolve_initialize_device", return_value=device))
                stack.enter_context(patch.object(torch.backends.mps, "is_macos_or_newer", return_value=True))
                stack.enter_context(patch.object(host, "_ensure_models_present", return_value=None))
                stack.enter_context(patch.object(host, "_sync_model_code_if_needed"))
                stack.enter_context(patch.object(host, "_load_main_model_from_checkpoint", side_effect=load_main))
                stack.enter_context(patch.object(host, "_load_vae_model", return_value="vae"))
                stack.enter_context(patch.object(host, "_load_text_encoder_and_tokenizer", return_value="text"))

                status, success = host.initialize_service("/unused", "acestep-v15-base", device=device)

                self.assertTrue(success, status)
                self.assertEqual(next(host.model.parameters()).dtype, expected_dtype)

    def test_low_memory_skips_both_mlx_backends_and_clears_previous_state(self):
        """A reused host cannot retain duplicate MLX model allocations."""
        host = _Host(project_root="/unused", device="mps")
        host.mps_low_memory = True
        host.mlx_decoder = torch.nn.Linear(2, 2)
        host.mlx_vae = torch.nn.Linear(2, 2)
        host.use_mlx_dit = host.use_mlx_vae = host.mlx_dit_compiled = True
        with patch.object(host, "_init_mlx_dit", side_effect=AssertionError("duplicate DiT")), \
                patch.object(host, "_init_mlx_vae", side_effect=AssertionError("duplicate VAE")):
            host._initialize_mlx_backends(
                device="mps", use_mlx_dit=True, mlx_compile_requested=True,
            )

        self.assertIsNone(host.mlx_decoder)
        self.assertIsNone(host.mlx_vae)
        self.assertFalse(host.use_mlx_dit)
        self.assertFalse(host.use_mlx_vae)
        self.assertFalse(host.mlx_dit_compiled)

    def test_unsupported_mps_fails_before_loading_without_float32_fallback(self):
        """Unsupported macOS fails clearly before allocating model weights."""
        host = _Host(project_root="/unused", device="mps")
        with patch.dict("os.environ", {"ACESTEP_MPS_LOW_MEMORY": "true"}), \
                patch.object(host, "_resolve_initialize_device", return_value="mps"), \
                patch.object(torch.backends.mps, "is_macos_or_newer", return_value=False), \
                patch.object(host, "_ensure_models_present", side_effect=AssertionError("load started")):
            status, success = host.initialize_service("/unused", "acestep-v15-base", device="mps")

        self.assertFalse(success)
        self.assertIn("ACESTEP_MPS_LOW_MEMORY requires macOS 14", status)
        self.assertIsNone(host.model)

    def test_text_checkpoint_stays_bfloat16_while_offloaded(self):
        """Loading a real small checkpoint does not first expand it to FP32."""
        from transformers import AutoModel, BertConfig, BertModel

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "Qwen3-Embedding-0.6B"
            model = BertModel(BertConfig(
                vocab_size=16, hidden_size=8, num_hidden_layers=1,
                num_attention_heads=2, intermediate_size=16,
            )).to(torch.bfloat16)
            model.save_pretrained(checkpoint)
            host = _Host(project_root=directory, device="mps")
            host.dtype = torch.bfloat16
            host.mps_low_memory = True
            host.offload_to_cpu = True
            loaded_dtypes = []
            original_loader = AutoModel.from_pretrained

            def load_checkpoint(*args, **kwargs):
                """Observe the actual checkpoint dtype before device/offload conversion."""
                loaded = original_loader(*args, **kwargs)
                loaded_dtypes.append(next(loaded.parameters()).dtype)
                return loaded

            with patch("transformers.AutoTokenizer.from_pretrained", return_value=Mock()), \
                    patch("transformers.AutoModel.from_pretrained", side_effect=load_checkpoint):
                host._load_text_encoder_and_tokenizer(checkpoint_dir=directory, device="mps")

            self.assertEqual(loaded_dtypes, [torch.bfloat16])
            self.assertEqual(next(host.text_encoder.parameters()).dtype, torch.bfloat16)
            self.assertEqual(next(host.text_encoder.parameters()).device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
