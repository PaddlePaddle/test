#!/usr/bin/env python

"""
test Transformer
"""

import pytest
import paddle
from paddle.nn import Transformer
from util import *

paddle.set_default_dtype("float64")
batch_size, d_model, n_head = 4, 12, 2
dim_feedforward = 3
dropout = 0.0
source_length, target_length = 20, 30  # 3, 3
num_encoder_layers, num_decoder_layers = 2, 2
np.random.seed(123)
paddle.seed(123)
if paddle.framework.use_pir_api():
    with paddle.pir_utils.OldIrGuard():
        # Note: dygraph use self.main_program.global_block().create_parameter(),
        # it's need manual seed to old Program
        paddle.framework.random._manual_program_seed(123)
    paddle.framework.random._manual_program_seed(123)
else:
    paddle.framework.random._manual_program_seed(123)


@pytest.mark.api_nn_Transformer_parameters
def test1():
    """
    test1
    """
    [batch_size, d_model, n_head, dim_feedforward, dropout, _, _, source_length, target_length] = generate_basic_params(
        mode="decoder_layer"
    )

    # batch_size, source_length, target_length, d_model, n_head = 4, 8, 8, 64, 8
    paddle.set_device("cpu")
    transformer = Transformer(d_model, n_head, dim_feedforward=dim_feedforward, dropout=dropout)
    src = paddle.to_tensor(np.random.rand(batch_size, source_length, d_model).astype("float64"))
    tgt = paddle.to_tensor(np.random.rand(batch_size, target_length, d_model).astype("float64"))
    src_mask = np.zeros((batch_size, n_head, source_length, source_length)).astype("float64")
    src_mask[0][0][0][0] = -np.inf
    src_mask = paddle.to_tensor(src_mask)
    tgt_mask = np.zeros((batch_size, n_head, target_length, target_length)).astype("float64")
    tgt_mask[0][0][0][0] = -1e9
    memory_mask = np.zeros((batch_size, n_head, target_length, source_length)).astype("float64")
    memory_mask[0][0][0][0] = -1e9
    tgt_mask, memory_mask = paddle.to_tensor(tgt_mask), paddle.to_tensor(memory_mask)
    trans_output = transformer(src, tgt, src_mask, tgt_mask, memory_mask)
    print(trans_output)


@pytest.mark.api_nn_Transformer_parameters
def test2():
    """
    test2
    """
    np.random.seed(123)
    paddle.seed(123)
    if paddle.framework.use_pir_api():
        with paddle.pir_utils.OldIrGuard():
            # Note: dygraph use self.main_program.global_block().create_parameter(),
            # it's need manual seed to old Program
            paddle.framework.random._manual_program_seed(123)
        paddle.framework.random._manual_program_seed(123)
    else:
        paddle.framework.random._manual_program_seed(123)
    paddle.disable_static()
    # src: [batch_size, tgt_len, d_model]
    enc_input = paddle.rand((2, 4, 128))
    # tgt: [batch_size, src_len, d_model]
    dec_input = paddle.rand((2, 6, 128))
    # src_mask: [batch_size, n_head, src_len, src_len]
    enc_self_attn_mask = paddle.rand((2, 2, 4, 4))
    # tgt_mask: [batch_size, n_head, tgt_len, tgt_len]
    dec_self_attn_mask = paddle.rand((2, 2, 6, 6))
    # memory_mask: [batch_size, n_head, tgt_len, src_len]
    cross_attn_mask = paddle.rand((2, 2, 6, 4))
    transformer = Transformer(128, 2, 4, 4, 512, 0.0, "relu", 0.0, 0.0, False)
    output = transformer(enc_input, dec_input, enc_self_attn_mask, dec_self_attn_mask, cross_attn_mask)  # [2, 6, 128]
    assert output[0].shape == [6, 128]
    assert output.shape == [2, 6, 128]
    transformer = Transformer(128, 2, 4, 4, 512, 0.0, "relu", 0.0, 0.0, True)
    output = transformer(enc_input, dec_input, enc_self_attn_mask, dec_self_attn_mask, cross_attn_mask)  # [2, 6, 128]
    assert output[0].shape == [6, 128]
    assert output.shape == [2, 6, 128]

    paddle.enable_static()
    # src: [batch_size, tgt_len, d_model]
    enc_input = paddle.rand((2, 4, 128))
    # tgt: [batch_size, src_len, d_model]
    dec_input = paddle.rand((2, 6, 128))
    # src_mask: [batch_size, n_head, src_len, src_len]
    enc_self_attn_mask = paddle.rand((2, 2, 4, 4))
    # tgt_mask: [batch_size, n_head, tgt_len, tgt_len]
    dec_self_attn_mask = paddle.rand((2, 2, 6, 6))
    # memory_mask: [batch_size, n_head, tgt_len, src_len]
    cross_attn_mask = paddle.rand((2, 2, 6, 4))
    transformer = Transformer(128, 2, 4, 4, 512, 0.5, "relu", 0.0, 0.0, False)
    output = transformer(enc_input, dec_input, enc_self_attn_mask, dec_self_attn_mask, cross_attn_mask)  # [2, 6, 128]
    assert output[0].shape == (6, 128)
    assert output.shape == (2, 6, 128)
    transformer = Transformer(128, 2, 4, 4, 512, 1.0, "relu", 0.0, 0.0, False)
    output = transformer(enc_input, dec_input, enc_self_attn_mask, dec_self_attn_mask, cross_attn_mask)  # [2, 6, 128]
    assert output[0].shape == (6, 128)
    assert output.shape == (2, 6, 128)


if __name__ == "__main__":
    test1()
    test2()
