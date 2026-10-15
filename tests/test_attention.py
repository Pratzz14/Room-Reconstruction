import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

from reconstruction_v2.inference import bounded_attention


@unittest.skipIf(torch is None, "Optional CPU PyTorch is needed for attention arithmetic checks")
class AttentionTests(unittest.TestCase):
    def test_query_slicing_matches_full_attention_and_independent_formula(self):
        generator = torch.Generator().manual_seed(29)
        original = torch.nn.functional.scaled_dot_product_attention
        for dtype in (torch.float32, torch.float64):
            q = torch.randn(2, 3, 17, 8, generator=generator, dtype=dtype)
            k = torch.randn(2, 3, 23, 8, generator=generator, dtype=dtype)
            v = torch.randn(2, 3, 23, 7, generator=generator, dtype=dtype)
            reference = original(q, k, v, scale=0.3)
            formula = torch.softmax((q @ k.transpose(-2, -1)) * 0.3, dim=-1) @ v
            with torch.inference_mode(), patch.object(torch.nn.functional, "scaled_dot_product_attention", wraps=original) as calls:
                with bounded_attention(torch, query_chunk_size=5) as stats:
                    actual = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=0.3)
                self.assertEqual(len(calls.call_args_list), 4)
                for call in calls.call_args_list:
                    self.assertLessEqual(call.args[0].shape[-2], 5)
                    self.assertIs(call.args[1], k)
                    self.assertIs(call.args[2], v)
                self.assertEqual(stats["max_key_tokens"], 23)
                self.assertEqual(stats["chunked_calls"], 1)
            torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(actual, formula, rtol=1e-5, atol=1e-6)
            self.assertIs(torch.nn.functional.scaled_dot_product_attention, original)

    def test_masked_and_causal_attention_are_delegated_without_changed_offsets(self):
        original = torch.nn.functional.scaled_dot_product_attention
        q = k = v = torch.randn(1, 2, 13, 8)
        for kwargs in ({"is_causal": True}, {"attn_mask": torch.eye(13, dtype=torch.bool)}):
            expected = original(q, k, v, **kwargs)
            with torch.inference_mode(), bounded_attention(torch, 3) as stats:
                actual = torch.nn.functional.scaled_dot_product_attention(q, k, v, **kwargs)
            torch.testing.assert_close(actual, expected)
            self.assertEqual(stats["chunked_calls"], 0)

    def test_operator_restores_after_kernel_failure(self):
        q = torch.randn(1, 2, 13, 8)
        with torch.inference_mode(), patch.object(torch.nn.functional, "scaled_dot_product_attention", side_effect=RuntimeError("known kernel failure")) as original:
            with self.assertRaisesRegex(RuntimeError, "known kernel failure"):
                with bounded_attention(torch, 3):
                    torch.nn.functional.scaled_dot_product_attention(q, q, q)
            self.assertIs(torch.nn.functional.scaled_dot_product_attention, original)

    def test_grad_enabled_execution_remains_native(self):
        q = torch.randn(1, 2, 13, 8, requires_grad=True)
        with bounded_attention(torch, 3) as stats:
            result = torch.nn.functional.scaled_dot_product_attention(q, q, q)
            result.sum().backward()
        self.assertEqual(stats["chunked_calls"], 0)
        self.assertIsNotNone(q.grad)


if __name__ == "__main__":
    unittest.main()
