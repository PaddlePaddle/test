#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
test paddle.nn.BeamSearchDecoder
"""

import random
from apibase import compare
import paddle
import pytest
import numpy as np
from paddle.nn import BeamSearchDecoder, dynamic_decode
from paddle.nn import GRUCell, Linear, Embedding, LSTMCell, SimpleRNNCell, Softmax

is_in_eager = True


np.random.seed(2)
random.seed(2)
paddle.seed(2)


class ModelGRUCell(paddle.nn.Layer):
    """
    GRUCell model
    """

    def __init__(self):
        """
        initialize
        """
        super(ModelGRUCell, self).__init__()
        self.trg_embeder = Embedding(100, 32)
        self.output_layer = Linear(32, 32)
        self.decoder_cell = GRUCell(input_size=32, hidden_size=32)
        self.decoder = BeamSearchDecoder(
            self.decoder_cell,
            start_token=0,
            end_token=1,
            beam_size=4,
            embedding_fn=self.trg_embeder,
            output_fn=self.output_layer,
        )

    def forward(self):
        """
        forward
        """
        encoder_output = paddle.ones((4, 8, 32), dtype=paddle.get_default_dtype())
        outputs = dynamic_decode(
            decoder=self.decoder, inits=self.decoder_cell.get_initial_states(encoder_output), max_step_num=10
        )
        return outputs[0]


class ModelLSTMCell(paddle.nn.Layer):
    """
    LSTMCell model
    """

    def __init__(self):
        """
        initialize
        """
        super(ModelLSTMCell, self).__init__()
        self.trg_embeder = Embedding(100, 32)
        self.output_layer = Linear(32, 32)
        self.decoder_cell = LSTMCell(input_size=32, hidden_size=32)
        self.decoder = BeamSearchDecoder(
            self.decoder_cell,
            start_token=0,
            end_token=1,
            beam_size=4,
            embedding_fn=self.trg_embeder,
            output_fn=self.output_layer,
        )

    def forward(self):
        """
        forward
        """
        encoder_output = paddle.ones((4, 8, 32), dtype=paddle.get_default_dtype())
        outputs = dynamic_decode(
            decoder=self.decoder, inits=self.decoder_cell.get_initial_states(encoder_output), max_step_num=10
        )
        return outputs[0]


class ModelSimpleRNNCell(paddle.nn.Layer):
    """
    SimpleRNNCell model
    """

    def __init__(self):
        """
        initialize
        """
        super(ModelSimpleRNNCell, self).__init__()
        self.trg_embeder = Embedding(100, 32)
        self.output_layer = Linear(32, 32)
        self.decoder_cell = SimpleRNNCell(input_size=32, hidden_size=32)
        self.decoder = BeamSearchDecoder(
            self.decoder_cell,
            start_token=0,
            end_token=1,
            beam_size=4,
            embedding_fn=self.trg_embeder,
            output_fn=self.output_layer,
        )

    def forward(self):
        """
        forward
        """
        encoder_output = paddle.ones((4, 8, 32), dtype=paddle.get_default_dtype())
        outputs = dynamic_decode(
            decoder=self.decoder, inits=self.decoder_cell.get_initial_states(encoder_output), max_step_num=10
        )
        return outputs[0]


class ModelGRUCell1(paddle.nn.Layer):
    """
    GRUCell model1
    """

    def __init__(self):
        """
        initialize
        """
        super(ModelGRUCell1, self).__init__()
        self.trg_embeder = Embedding(100, 32)
        self.output_layer = Linear(32, 32)
        self.decoder_cell = GRUCell(input_size=32, hidden_size=32)
        self.decoder = BeamSearchDecoder(
            self.decoder_cell,
            start_token=0,
            end_token=1,
            beam_size=8,
            embedding_fn=self.trg_embeder,
            output_fn=self.output_layer,
        )

    def forward(self):
        """
        forward
        """
        encoder_output = paddle.ones((4, 8, 32), dtype=paddle.get_default_dtype())
        outputs = dynamic_decode(
            decoder=self.decoder, inits=self.decoder_cell.get_initial_states(encoder_output), max_step_num=10
        )
        return outputs[0]


class ModelGRUCell2(paddle.nn.Layer):
    """
    GRUCell model2
    """

    def __init__(self):
        """
        initialize
        """
        super(ModelGRUCell2, self).__init__()
        self.trg_embeder = Embedding(100, 16)
        self.output_layer = Linear(16, 16)
        self.decoder_cell = GRUCell(input_size=16, hidden_size=16)
        self.decoder = BeamSearchDecoder(
            self.decoder_cell,
            start_token=0,
            end_token=1,
            beam_size=4,
            embedding_fn=self.trg_embeder,
            output_fn=self.output_layer,
        )

    def forward(self):
        """
        forward
        """
        encoder_output = paddle.ones((4, 8, 16), dtype=paddle.get_default_dtype())
        outputs = dynamic_decode(
            decoder=self.decoder, inits=self.decoder_cell.get_initial_states(encoder_output), max_step_num=5
        )
        return outputs[0]


class ModelGRUCell3(paddle.nn.Layer):
    """
    GRUCell model3
    """

    def __init__(self):
        """
        initialize
        """
        super(ModelGRUCell3, self).__init__()
        self.trg_embeder = Embedding(100, 16)
        self.output_layer = Softmax(-1)
        self.decoder_cell = GRUCell(input_size=16, hidden_size=16)
        self.decoder = BeamSearchDecoder(
            self.decoder_cell,
            start_token=0,
            end_token=1,
            beam_size=4,
            embedding_fn=self.trg_embeder,
            output_fn=self.output_layer,
        )

    def forward(self):
        """
        forward
        """
        encoder_output = paddle.ones((4, 8, 16), dtype=paddle.get_default_dtype())
        outputs = dynamic_decode(
            decoder=self.decoder, inits=self.decoder_cell.get_initial_states(encoder_output), max_step_num=5
        )
        return outputs[0]


@pytest.mark.api_nn_BeamSearchDecoder_parameter
def test_beamsearchdecoder0():
    """
    GRUCell
    """
    m = ModelGRUCell()
    a = paddle.load("model/model_grucell")
    m.set_state_dict(a)
    res = [
        [
            [24, 24, 24, 24],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [24, 24, 0, 0],
            [0, 0, 24, 24],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 24, 0, 24],
        ],
        [
            [24, 24, 24, 24],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [24, 24, 0, 0],
            [0, 0, 24, 24],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 24, 0, 24],
        ],
        [
            [24, 24, 24, 24],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [24, 24, 0, 0],
            [0, 0, 24, 24],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 24, 0, 24],
        ],
        [
            [24, 24, 24, 24],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [24, 24, 0, 0],
            [0, 0, 24, 24],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 24, 0, 24],
        ],
    ]

    compare(m().numpy(), res)


@pytest.mark.api_nn_BeamSearchDecoder_parameters
def test_beamsearchdecoder1():
    """
    change decoder_cell to LSTM cell
    """
    m = ModelLSTMCell()
    a = paddle.load("model/model_lstmcell")
    m.set_state_dict(a)
    res = [
        [
            [20, 20, 20, 20],
            [31, 31, 31, 31],
            [24, 24, 24, 24],
            [20, 20, 20, 20],
            [24, 24, 24, 24],
            [20, 20, 20, 20],
            [27, 27, 27, 24],
            [20, 20, 20, 20],
            [27, 24, 27, 24],
            [20, 20, 20, 20],
            [24, 24, 27, 24],
        ],
        [
            [20, 20, 20, 20],
            [31, 31, 31, 31],
            [24, 24, 24, 24],
            [20, 20, 20, 20],
            [24, 24, 24, 24],
            [20, 20, 20, 20],
            [27, 27, 27, 24],
            [20, 20, 20, 20],
            [27, 24, 27, 24],
            [20, 20, 20, 20],
            [24, 24, 27, 24],
        ],
        [
            [20, 20, 20, 20],
            [31, 31, 31, 31],
            [24, 24, 24, 24],
            [20, 20, 20, 20],
            [24, 24, 24, 24],
            [20, 20, 20, 20],
            [27, 27, 27, 24],
            [20, 20, 20, 20],
            [27, 24, 27, 24],
            [20, 20, 20, 20],
            [24, 24, 27, 24],
        ],
        [
            [20, 20, 20, 20],
            [31, 31, 31, 31],
            [24, 24, 24, 24],
            [20, 20, 20, 20],
            [24, 24, 24, 24],
            [20, 20, 20, 20],
            [27, 27, 27, 24],
            [20, 20, 20, 20],
            [27, 24, 27, 24],
            [20, 20, 20, 20],
            [24, 24, 27, 24],
        ],
    ]

    compare(m().numpy(), res)


@pytest.mark.api_nn_BeamSearchDecoder_parameters
def test_beamsearchdecoder2():
    """
    change decoder_cell to simpleRNNCell
    """
    m = ModelSimpleRNNCell()
    a = paddle.load("model/model_simplernncell")
    m.set_state_dict(a)
    res = [
        [
            [0, 0, 0, 0],
            [31, 31, 31, 31],
            [19, 19, 19, 19],
            [25, 25, 25, 25],
            [29, 29, 29, 29],
            [14, 14, 14, 14],
            [5, 12, 12, 12],
            [14, 5, 5, 5],
            [14, 14, 14, 14],
            [5, 5, 14, 14],
            [14, 14, 5, 14],
        ],
        [
            [0, 0, 0, 0],
            [31, 31, 31, 31],
            [19, 19, 19, 19],
            [25, 25, 25, 25],
            [29, 29, 29, 29],
            [14, 14, 14, 14],
            [5, 12, 12, 12],
            [14, 5, 5, 5],
            [14, 14, 14, 14],
            [5, 5, 14, 14],
            [14, 14, 5, 14],
        ],
        [
            [0, 0, 0, 0],
            [31, 31, 31, 31],
            [19, 19, 19, 19],
            [25, 25, 25, 25],
            [29, 29, 29, 29],
            [14, 14, 14, 14],
            [5, 12, 12, 12],
            [14, 5, 5, 5],
            [14, 14, 14, 14],
            [5, 5, 14, 14],
            [14, 14, 5, 14],
        ],
        [
            [0, 0, 0, 0],
            [31, 31, 31, 31],
            [19, 19, 19, 19],
            [25, 25, 25, 25],
            [29, 29, 29, 29],
            [14, 14, 14, 14],
            [5, 12, 12, 12],
            [14, 5, 5, 5],
            [14, 14, 14, 14],
            [5, 5, 14, 14],
            [14, 14, 5, 14],
        ],
    ]
    compare(m().numpy(), res)


@pytest.mark.api_nn_BeamSearchDecoder_parameters
def test_beamsearchdecoder3():
    """
    change the beam size
    """
    m = ModelGRUCell1()
    a = paddle.load("model/model_grucell1")
    m.set_state_dict(a)
    res = [
        [
            [24, 24, 24, 24, 24, 24, 24, 24],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 24],
            [24, 24, 0, 24, 0, 24, 24, 0],
            [0, 0, 24, 0, 24, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 24, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 24, 0, 0],
            [0, 0, 0, 0, 0, 0, 24, 0],
            [0, 24, 0, 0, 24, 0, 0, 0],
        ],
        [
            [24, 24, 24, 24, 24, 24, 24, 24],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 24],
            [24, 24, 0, 24, 0, 24, 24, 0],
            [0, 0, 24, 0, 24, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 24, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 24, 0, 0],
            [0, 0, 0, 0, 0, 0, 24, 0],
            [0, 24, 0, 0, 24, 0, 0, 0],
        ],
        [
            [24, 24, 24, 24, 24, 24, 24, 24],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 24],
            [24, 24, 0, 24, 0, 24, 24, 0],
            [0, 0, 24, 0, 24, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 24, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 24, 0, 0],
            [0, 0, 0, 0, 0, 0, 24, 0],
            [0, 24, 0, 0, 24, 0, 0, 0],
        ],
        [
            [24, 24, 24, 24, 24, 24, 24, 24],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 24],
            [24, 24, 0, 24, 0, 24, 24, 0],
            [0, 0, 24, 0, 24, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 24, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 24, 0, 0],
            [0, 0, 0, 0, 0, 0, 24, 0],
            [0, 24, 0, 0, 24, 0, 0, 0],
        ],
    ]

    compare(m().numpy(), res)


@pytest.mark.api_nn_BeamSearchDecoder_parameters
def test_beamsearchdecoder4():
    """
    change the size of the input sequence
    """
    m = ModelGRUCell2()
    a = paddle.load("model/model_grucell2")
    m.set_state_dict(a)
    res = [
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 13, 0], [13, 13, 0, 13], [0, 13, 13, 13], [13, 13, 13, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 13, 0], [13, 13, 0, 13], [0, 13, 13, 13], [13, 13, 13, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 13, 0], [13, 13, 0, 13], [0, 13, 13, 13], [13, 13, 13, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 13, 0], [13, 13, 0, 13], [0, 13, 13, 13], [13, 13, 13, 0]],
    ]

    compare(m().numpy(), res)


@pytest.mark.api_nn_BeamSearchDecoder_parameters
def test_beamsearchdecoder5():
    """
    change the type of the output_layer
    """
    m = ModelGRUCell3()
    a = paddle.load("model/model_grucell3")
    m.set_state_dict(a)
    res = [
        [[14, 9, 0, 9], [9, 9, 9, 0], [9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9]],
        [[14, 9, 0, 9], [9, 9, 9, 0], [9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9]],
        [[14, 9, 0, 9], [9, 9, 9, 0], [9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9]],
        [[14, 9, 0, 9], [9, 9, 9, 0], [9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9]],
    ]

    compare(m().numpy(), res)


@pytest.mark.api_nn_BeamSearchDecoder_exception
def test_beamsearchdecoder6():
    """
    error shape
    input_size mismatch hidden_size
    """
    paddle.seed(33)
    trg_embeder = Embedding(100, 16)
    output_layer = Linear(16, 16)
    decoder_cell = GRUCell(input_size=16, hidden_size=32)
    decoder = BeamSearchDecoder(
        decoder_cell, start_token=0, end_token=1, beam_size=4, embedding_fn=trg_embeder, output_fn=output_layer
    )
    encoder_output = paddle.ones((4, 8, 16), dtype=paddle.get_default_dtype())
    try:
        dynamic_decode(decoder=decoder, inits=decoder_cell.get_initial_states(encoder_output), max_step_num=5)
    except Exception as e:
        print(e)
        if is_in_eager:
            if "InvalidArgument" in e.args[0]:
                pass
            else:
                raise Exception
        else:
            if ("[operator < matmul > error]" in e.args[0]) or ("[operator < matmul_v2 > error]" in e.args[0]):
                pass
            else:
                raise Exception


@pytest.mark.api_nn_BeamSearchDecoder_exception
def test_beamsearchdecoder7():
    """
    end_token out of range
    """
    paddle.seed(33)
    trg_embeder = Embedding(100, 16)
    output_layer = Linear(16, 16)
    decoder_cell = GRUCell(input_size=16, hidden_size=16)
    try:
        decoder = BeamSearchDecoder(
            decoder_cell, start_token=0, end_token=16, beam_size=4, embedding_fn=trg_embeder, output_fn=output_layer
        )
        encoder_output = paddle.ones((4, 8, 16), dtype=paddle.get_default_dtype())
        dynamic_decode(decoder=decoder, inits=decoder_cell.get_initial_states(encoder_output), max_step_num=5)
    except Exception as e:
        # print(e)
        if "list assignment index out of range" in e.args[0]:
            pass
        else:
            raise Exception


@pytest.mark.api_nn_BeamSearchDecoder_exception
def test_beamsearchdecoder8():
    """
    Exception to the type of start_id
    """
    paddle.seed(33)
    paddle.enable_static()
    trg_embeder = Embedding(100, 16)
    output_layer = Linear(16, 16)
    decoder_cell = GRUCell(input_size=16, hidden_size=16)
    try:
        decoder = BeamSearchDecoder(
            decoder_cell, start_token="a", end_token=1, beam_size=4, embedding_fn=trg_embeder, output_fn=output_layer
        )
        encoder_output = paddle.ones((4, 8, 16), dtype=paddle.get_default_dtype())
        dynamic_decode(decoder=decoder, inits=decoder_cell.get_initial_states(encoder_output), max_step_num=5)
    except Exception as e:
        # print(e)
        if "invalid literal for int()" in e.args[0]:
            pass
        elif "must be double" in e.args[0] and paddle.framework.in_pir_mode():
            pass
        else:
            raise Exception
    paddle.disable_static()


@pytest.mark.api_nn_BeamSearchDecoder_exception
def test_beamsearchdecoder9():
    """
    the size of each embedding vector mismatch the size of GRUCell
    """
    paddle.seed(33)
    try:

        trg_embeder = Embedding(100, 32)
        output_layer = Linear(16, 16)
        decoder_cell = GRUCell(input_size=16, hidden_size=16)
        decoder = BeamSearchDecoder(
            decoder_cell, start_token=0, end_token=2, beam_size=4, embedding_fn=trg_embeder, output_fn=output_layer
        )
        encoder_output = paddle.ones((4, 8, 16), dtype=paddle.get_default_dtype())
        dynamic_decode(decoder=decoder, inits=decoder_cell.get_initial_states(encoder_output), max_step_num=5)
    except Exception as e:
        # print(e)
        if "InvalidArgument" in e.args[0]:
            pass
        else:
            raise Exception
