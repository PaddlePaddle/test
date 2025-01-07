#!/usr/bin/env python

"""
test TransformerDecoderLayer
"""

import unittest
import numpy as np
import pytest
import paddle
from paddle.nn import MultiHeadAttention, TransformerDecoderLayer
from util import *


class Test(unittest.TestCase):
    """
    test
    """

    @pytest.mark.api_nn_TransformerDecoderLayer_parameters
    def test_transformer_decoder_layer(self):
        """
        test_transformer_decoder_layer
        """
        paddle.seed(2020)
        if paddle.framework.use_pir_api():
            with paddle.pir_utils.OldIrGuard():
                # Note: dygraph use self.main_program.global_block().create_parameter(),
                # it's need manual seed to old Program
                paddle.framework.random._manual_program_seed(2020)
            paddle.framework.random._manual_program_seed(2020)
        else:
            paddle.framework.random._manual_program_seed(2020)
        activation = "relu"
        normalize_before = False
        batch_size, d_model, n_head, dim_feedforward, dropout = generate_basic_params(mode="decoder_layer")[:5]
        attn_dropout, act_dropout, source_length, target_length = generate_basic_params(mode="decoder_layer")[5:]
        tgt = np.random.rand(batch_size, target_length, d_model).astype("float32")
        memory = np.random.rand(batch_size, source_length, d_model).astype("float32")
        tgt_mask = np.zeros((batch_size, n_head, target_length, target_length)).astype("float32")
        tgt_mask[0][0][0][0] = -1e9
        memory_mask = np.zeros((batch_size, n_head, target_length, source_length)).astype("float32")
        memory_mask[0][0][0][0] = -1e9
        for cache in [True, False]:
            self_attn = MultiHeadAttention(d_model, n_head, dropout=attn_dropout)
            cross_attn = MultiHeadAttention(d_model, n_head, dropout=attn_dropout)

            # paddle decoderlayer:
            decoder_layer = TransformerDecoderLayer(
                d_model, n_head, dim_feedforward, dropout, activation, attn_dropout, act_dropout, normalize_before
            )
            cache_objs = None
            if cache:
                cache_objs = decoder_layer.gen_cache(paddle.to_tensor(memory))

            decoder_output = decoder_layer(
                paddle.to_tensor(tgt),
                paddle.to_tensor(memory),
                paddle.to_tensor(tgt_mask),
                paddle.to_tensor(memory_mask),
                cache_objs,
            )

            decoder_output = decoder_output[0].numpy() if cache else decoder_output.numpy()

            # numpy:
            residual = tgt
            # self-attn
            self_attn_cache = cache_objs[0] if cache_objs is not None else None
            tgt = self_attn(
                paddle.to_tensor(tgt),
                paddle.to_tensor(tgt),
                paddle.to_tensor(tgt),
                paddle.to_tensor(tgt_mask),
                self_attn_cache,
            )

            tgt = tgt[0].numpy() if cache else tgt.numpy()

            tgt = residual + tgt
            # postprocess
            tgt_norm = layer_norm(tgt, d_model, decoder_layer.norm1)
            residual = tgt_norm
            # cross-attn
            cross_attn_cache = cache_objs[1] if cache_objs is not None else None
            tgt = cross_attn(
                paddle.to_tensor(tgt_norm),
                paddle.to_tensor(memory),
                paddle.to_tensor(memory),
                paddle.to_tensor(memory_mask),
                cross_attn_cache,
            )
            tgt = tgt[0].numpy() if cache else tgt.numpy()

            # postprocess
            tgt = tgt + residual
            tgt_norm = layer_norm(tgt, d_model, decoder_layer.norm2)
            residual = tgt_norm
            # FFN
            ffn_output = ffn(tgt_norm, decoder_layer, activation)
            # post process
            tgt = residual + ffn_output
            tgt_norm = layer_norm(tgt, d_model, decoder_layer.norm3)

            np.testing.assert_allclose(decoder_output, tgt_norm, rtol=1e-5, atol=1e-6)

    @pytest.mark.api_nn_TransformerDecoderLayer_parameters
    def test(self):
        """
        test
        """
        paddle.disable_static()
        # decoder input: [batch_size, tgt_len, d_model]
        dec_input = paddle.rand((2, 4, 8))
        # encoder output: [batch_size, src_len, d_model]
        enc_output = paddle.rand((2, 6, 8))
        # self attention mask: [batch_size, n_head, tgt_len, tgt_len]
        self_attn_mask = paddle.rand((2, 2, 4, 4))
        # cross attention mask: [batch_size, n_head, tgt_len, src_len]
        cross_attn_mask = paddle.rand((2, 2, 4, 6))
        decoder_layer = TransformerDecoderLayer(8, 2, 16)
        output = decoder_layer(dec_input, enc_output, self_attn_mask, cross_attn_mask)  # [2, 4, 8]
        assert output[0].shape == [4, 8]
        assert output.shape == [2, 4, 8]


if __name__ == "__main__":
    unittest.main()
