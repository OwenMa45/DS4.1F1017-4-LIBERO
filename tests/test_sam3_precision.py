"""Tiny real-tensor regression; no SAM3 weights, GPU, simulator or API calls."""

import contextlib
import unittest
from types import ModuleType
from unittest.mock import patch

from ds_libero.sam3_server import configure_model_precision

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "CPU PyTorch required; exercised by sam3-precision CI")
class PrecisionTests(unittest.TestCase):
    @contextlib.contextmanager
    def model(self, activation="gelu"):
        nn, functional = torch.nn, torch.nn.functional

        class Mlp(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1, self.fc2 = nn.Linear(4, 8), nn.Linear(8, 4)
                self.act = nn.GELU() if activation == "gelu" else nn.ReLU()
                self.drop1, self.drop2, self.norm = nn.Dropout(0.1), nn.Dropout(0.1), nn.Identity()

            def forward(self, value):
                # Reproduce the dtype boundary in SAM3's fused fc1 + activation.
                value = functional.linear(value.bfloat16(), self.fc1.weight.bfloat16(), self.fc1.bias.bfloat16())
                value = self.norm(self.drop1(self.act(value)))
                return self.drop2(self.fc2(value))

        vitdet = ModuleType("sam3.model.vitdet")
        vitdet.Mlp = Mlp
        torch.manual_seed(7)
        model = nn.Sequential(Mlp(), nn.Linear(4, 2)).eval()
        with patch.dict("sys.modules", {"sam3.model.vitdet": vitdet}):
            yield model

    def test_reported_dtype_failure_then_float32_forward_preserves_weights(self):
        for activation in ("gelu", "relu"):
            for shape in ((2, 3, 4), (1, 2, 3, 4)):
                with self.subTest(activation=activation, shape=shape), self.model(activation) as model:
                    image = torch.randn(*shape)
                    before = {key: value.clone() for key, value in model.state_dict().items()}
                    other_forward = model[1].forward.__func__
                    with torch.inference_mode(), torch.autocast("cpu", enabled=False):
                        with self.assertRaisesRegex(RuntimeError, "same dtype"):
                            model(image)
                        policy = configure_model_precision(model, "float32")
                        actual = model(image)
                        mlp = model[0]
                        expected = model[1](mlp.fc2(mlp.act(mlp.fc1(image))))
                    self.assertEqual(actual.dtype, torch.float32)
                    torch.testing.assert_close(actual, expected)
                    self.assertEqual(policy["float32_mlp_modules"], ["0"])
                    self.assertFalse(policy["autocast_enabled"])
                    self.assertIs(model[1].forward.__func__, other_forward)
                    for key, value in model.state_dict().items():
                        torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_float_cast_preserves_complex_rope_buffers(self):
        with self.model() as model:
            model.bfloat16()
            rope = torch.tensor([1 + 2j, 3 - 4j], dtype=torch.complex64)
            model.register_buffer("rope", rope.clone())
            configure_model_precision(model, "float32")
            self.assertTrue(all(parameter.dtype == torch.float32 for parameter in model.parameters()))
            torch.testing.assert_close(model.rope, rope, rtol=0, atol=0)

    def test_bfloat16_mode_preserves_upstream_forward(self):
        with self.model() as model:
            before = model[0].forward.__func__
            policy = configure_model_precision(model, "bfloat16")
            self.assertIs(model[0].forward.__func__, before)
            self.assertEqual(policy["vit_mlp_implementation"], "upstream")
            self.assertEqual(policy["float32_mlp_modules"], [])
            self.assertTrue(policy["autocast_enabled"])
            with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
                self.assertEqual(model(torch.randn(2, 3, 4)).dtype, torch.bfloat16)

    def test_float32_forward_inside_disabled_autocast_region(self):
        with self.model() as model:
            configure_model_precision(model, "float32")
            image = torch.randn(1, 2, 3, 4)
            with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
                with torch.autocast("cpu", enabled=False):
                    self.assertEqual(model(image).dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
