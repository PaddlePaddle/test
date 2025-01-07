#!/usr/bin/env python

"""
test TransformerDecoder

"""

import unittest
import numpy as np
import pytest
import paddle
from paddle.base.framework import in_pir_mode
from paddle.nn import MultiHeadAttention, TransformerDecoderLayer, TransformerDecoder
from util import *


class Test(unittest.TestCase):
    """
    test
    """

    @pytest.mark.api_nn_TransformerDecoder_parameters
    def test_decoder(self):
        """
        test_decoder
        """
        paddle.set_device("cpu")
        [
            batch_size,
            d_model,
            n_head,
            dim_feedforward,
            dropout,
            _,
            _,
            source_length,
            target_length,
        ] = generate_basic_params(mode="decoder_layer")
        tgt = np.random.rand(batch_size, target_length, d_model).astype("float32")
        memory = np.random.rand(batch_size, source_length, d_model).astype("float32")
        tgt_mask = np.zeros((batch_size, n_head, target_length, target_length)).astype("float32")
        tgt_mask[0][0][0][0] = -1e9
        memory_mask = np.zeros((batch_size, n_head, target_length, source_length)).astype("float32")
        memory_mask[0][0][0][0] = -1e9
        decoder_layer = TransformerDecoderLayer(d_model, n_head, dim_feedforward, dropout)
        num_layers = 6
        decoder = TransformerDecoder(decoder_layer, num_layers)

        output = decoder(
            paddle.to_tensor(tgt),
            paddle.to_tensor(memory),
            paddle.to_tensor(tgt_mask),
            paddle.to_tensor(memory_mask),
        )
        print(output)

    @pytest.mark.api_nn_TransformerDecoder_parameters
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
        decoder = TransformerDecoder(decoder_layer, 2)
        output = decoder(dec_input, enc_output, self_attn_mask, cross_attn_mask)  # [2, 4, 8]
        assert output[0].shape == [4, 8]
        assert output.shape == [2, 4, 8]

        paddle.enable_static()
        # decoder input: [batch_size, tgt_len, d_model]
        dec_input = paddle.rand((2, 4, 8))
        # encoder output: [batch_size, src_len, d_model]
        enc_output = paddle.rand((2, 6, 8))
        # self attention mask: [batch_size, n_head, tgt_len, tgt_len]
        self_attn_mask = paddle.rand((2, 2, 4, 4))
        # cross attention mask: [batch_size, n_head, tgt_len, src_len]
        cross_attn_mask = paddle.rand((2, 2, 4, 6))
        decoder_layer = TransformerDecoderLayer(8, 2, 16)
        decoder = TransformerDecoder(decoder_layer, 2)
        output = decoder(dec_input, enc_output, self_attn_mask, cross_attn_mask)  # [2, 4, 8]
        if in_pir_mode():
            assert output[0].shape == [4, 8]
            assert output.shape == [2, 4, 8]
        else:
            assert output[0].shape == (4, 8)
            assert output.shape == (2, 4, 8)


if __name__ == "__main__":
    unittest.main()
