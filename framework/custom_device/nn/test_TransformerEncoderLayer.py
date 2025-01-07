#!/usr/bin/env python

"""
test TransformerEncoderLayer
"""

import unittest
import numpy as np
import paddle
import pytest
from paddle.nn import MultiHeadAttention, TransformerEncoderLayer, TransformerEncoder
from util import *


class Test(unittest.TestCase):
    """
    test
    """

    def test_transformer_encoder_layer(self):
        """
        test_transformer_encoder_layer
        """
        paddle.set_device("cpu")
        paddle.seed(2020)
        if paddle.framework.use_pir_api():
            with paddle.pir_utils.OldIrGuard():
                # Note: dygraph use self.main_program.global_block().create_parameter(),
                # it's need manual seed to old Program
                paddle.framework.random._manual_program_seed(2020)
            paddle.framework.random._manual_program_seed(2020)
        else:
            paddle.framework.random._manual_program_seed(2020)

        ffn_fc1_act = "relu"
        # 1.generate basic params
        [
            batch_size,
            d_model,
            n_head,
            dim_feedforward,
            dropout,
            attn_dropout,
            act_dropout,
            sequence_length,
        ] = generate_basic_params(mode="encoder_layer")
        # 2.generate input for encoder
        src = np.random.rand(batch_size, sequence_length, d_model).astype("float32")
        residual = src
        src_mask = np.zeros((batch_size, n_head, sequence_length, sequence_length)).astype("float32")
        src_mask[0][0][0][0] = -np.inf

        # paddle
        encoder_layer = TransformerEncoderLayer(
            d_model, n_head, dim_feedforward, dropout, ffn_fc1_act, attn_dropout, act_dropout
        )

        encoder_output = encoder_layer(paddle.to_tensor(src), paddle.to_tensor(src_mask))  # paddle.to_tensor(src_mask))
        # 4.numpy:
        # paddle self attention
        self_attn = MultiHeadAttention(d_model, n_head, dropout=attn_dropout)
        attn_output = self_attn(
            paddle.to_tensor(src), paddle.to_tensor(src), paddle.to_tensor(src), paddle.to_tensor(src_mask)
        ).numpy()

        src = attn_output + residual
        src_norm = layer_norm(src, d_model, encoder_layer.norm1)
        residual = src_norm

        ffn_output = ffn(src_norm, encoder_layer, ffn_fc1_act)
        src = residual + ffn_output
        src = layer_norm(src, d_model, encoder_layer.norm2)

        np.testing.assert_allclose(encoder_output.numpy(), src, rtol=1e-5, atol=1e-6)

    def test_main(self):
        """
        test
        """
        paddle.set_default_dtype("float32")
        # encoder input: [batch_size, src_len, d_model]
        enc_input = paddle.rand((2, 4, 8))
        # self attention mask: [batch_size, n_head, src_len, src_len]
        attn_mask = paddle.rand((2, 2, 4, 4))
        encoder_layer = TransformerEncoderLayer(8, 2, 16)
        enc_output = encoder_layer(enc_input, attn_mask)  # [2, 4, 8]
        assert enc_output.shape == [2, 4, 8]
        assert enc_output[0].shape == [4, 8]


if __name__ == "__main__":
    unittest.main()
