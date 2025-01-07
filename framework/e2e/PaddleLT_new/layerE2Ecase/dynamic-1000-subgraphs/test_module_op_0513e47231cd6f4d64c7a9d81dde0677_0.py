import os
os.environ['FLAGS_cinn_new_group_scheduler'] = '1'
os.environ['FLAGS_group_schedule_tiling_first'] = '1'
os.environ['FLAGS_enable_pir_api'] = '1'
os.environ['FLAGS_cinn_bucket_compile'] = '1'
import sys
import unittest
import numpy as np
from dataclasses import dataclass
import typing as t

@dataclass
class Stage:
    name: str
    env_vars: t.Dict[str, str]

cinn_stages = [
    Stage(
        name="dynamic_to_static",
        env_vars=dict(
            PADDLE_DEBUG_ENABLE_CINN=False,
            FLAGS_prim_all=False,
            FLAGS_prim_enable_dynamic=False,
        ),
    ),
    Stage(
        name="prim",
        env_vars=dict(
            PADDLE_DEBUG_ENABLE_CINN=False,
            FLAGS_prim_all=True,
            FLAGS_prim_enable_dynamic=True,
        ),
    ),
    Stage(
        name="infer_symbolic",
        env_vars=dict(
            PADDLE_DEBUG_ENABLE_CINN=False,
            FLAGS_prim_all=True,
            FLAGS_prim_enable_dynamic=True,
            FLAGS_use_cinn=False,
            FLAGS_check_infer_symbolic=True,
        ),
    ),
	Stage(
        name="frontend",
        env_vars=dict(
            PADDLE_DEBUG_ENABLE_CINN=True,
            FLAGS_prim_all=True,
            FLAGS_prim_enable_dynamic=True,
            FLAGS_use_cinn=True,
            FLAGS_check_infer_symbolic=False,
            FLAGS_enable_fusion_fallback=True,
        ), 
    ),
    Stage(
        name="backend",
        env_vars=dict(
            PADDLE_DEBUG_ENABLE_CINN=True,
            FLAGS_prim_all=True,
            FLAGS_prim_enable_dynamic=True,
            FLAGS_use_cinn=True,
            FLAGS_check_infer_symbolic=False,
            FLAGS_enable_fusion_fallback=False,
        ), 
    ),
]

def GetCinnStageByName(name):
    for stage in cinn_stages:
        if stage.name == name:
            return stage
    return None

def GetCurrentCinnStage():
    name = os.getenv('PADDLE_DEBUG_CINN_STAGE_NAME')
    if name is None:
        return None
    stage_names = [stage.name for stage in cinn_stages]
    assert name in stage_names, (
        f"PADDLE_DEBUG_CINN_STAGE_NAME should be in {stage_names}"
    )
    return GetCinnStageByName(name)

def GetPrevCinnStage(stage):
    for i in range(1, len(cinn_stages)):
        if stage is cinn_stages[i]:
            return cinn_stages[i - 1]
    return None

def IsCinnStageEnableDiff():
    value = os.getenv('PADDLE_DEBUG_CINN_STAGE_ENABLE_DIFF')
    enabled = value in {
        '1',
        'true',
        'True',
    }
    if enabled:
        assert GetCurrentCinnStage() is not None
    return enabled

def GetExitCodeAndStdErr(cmd, env):
    env = {
        k:v
        for k, v in env.items()
        if v is not None
    }
    import subprocess
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    return result.returncode, result.stderr

def GetStageExitCodeAndStdErr(stage):
    return GetExitCodeAndStdErr(
        [sys.executable, __file__],
        env=dict(
            PADDLE_DEBUG_CINN_STAGE_NAME=stage.name,
            PADDLE_DEBUG_CINN_STAGE_ENABLE_DIFF='0',
            PYTHONPATH=os.getenv('PYTHONPATH'),
            ATHENA_ENABLE_TRY_RUN="False",
        ),
    )

def AthenaTryRunEnabled():
    return os.getenv('ATHENA_ENABLE_TRY_RUN') not in {
        "0",
        "False",
        "false",
        "OFF"
    }

def GetNeedSkipAndSkipMessage():
    current_stage = GetCurrentCinnStage()
    assert current_stage is not None
    if not IsCinnStageEnableDiff():
        return False, ""
    last_stage = GetPrevCinnStage(current_stage)
    if last_stage is None:
        return False, ""
    exitcode, stderr = GetStageExitCodeAndStdErr(last_stage)
    if exitcode != 0:
        return True, f"last stage failed."
    return False, ""

def GetCurrentStageTryRunExitCodeAndStdErr():
    if not AthenaTryRunEnabled():
        return False, ""
    current_stage = GetCurrentCinnStage()
    assert current_stage is not None
    return GetStageExitCodeAndStdErr(current_stage)

def SetDefaultEnv(**env_var2value):
    for env_var, value in env_var2value.items():
        if os.getenv(env_var) is None:
            os.environ[env_var] = str(value)

SetDefaultEnv(
    PADDLE_DEBUG_CINN_STAGE_NAME="backend",
    PADDLE_DEBUG_CINN_STAGE_ENABLE_DIFF=False,
    PADDLE_DEBUG_ENABLE_CINN=True,
    FLAGS_enable_pir_api=True,
    FLAGS_prim_all=True,
    FLAGS_prim_enable_dynamic=True,
    FLAGS_use_cinn=False,
    FLAGS_check_infer_symbolic=False,
    FLAGS_enable_fusion_fallback=False,
)

need_skip, skip_message = GetNeedSkipAndSkipMessage()
try_run_exit_code, try_run_stderr = GetCurrentStageTryRunExitCodeAndStdErr()
class TestTryRun(unittest.TestCase):
    def test_panic(self):
        if not AthenaTryRunEnabled():
            return
        if try_run_exit_code == 0:
            # All unittest cases passed.
            return
        if try_run_exit_code > 0:
            # program failed but not panic.
            return
        # program panicked.
        kOutputLimit = 65536
        message = try_run_stderr[-kOutputLimit:]
        raise RuntimeError(f"panicked. last {kOutputLimit} characters of stderr: \n{message}")

import paddle

def SetEnvVar(env_var2value):
    for env_var, value in env_var2value.items():
        os.environ[env_var] = str(value)
    paddle.set_flags({
        env_var:value
        for env_var, value in env_var2value.items()
        if env_var.startswith('FLAGS_')
    })

if GetCurrentCinnStage() is not None:
    SetEnvVar(GetCurrentCinnStage().env_vars)

def NumOperationsInBlock(block_idx):
    return [77][block_idx] - 1 # number-of-ops-in-block

def GetPaddleDebugNumAllowedOps():
    try:
        return int(os.getenv('PADDLE_DEBUG_NUM_ALLOWED_OPS'))
    except:
        return None

paddle_debug_num_allowed_ops = GetPaddleDebugNumAllowedOps()


if type(paddle_debug_num_allowed_ops) is not int:
    def EarlyReturn(block_idx, op_idx):
        return False      
else:
    def EarlyReturn(block_idx, op_idx):
        return op_idx >= paddle_debug_num_allowed_ops

class BlockEntries:
    def builtin_module_461_0_0(self, parameter_1, parameter_11, parameter_5, parameter_0, parameter_14, parameter_13, parameter_2, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_3, parameter_9, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7):

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_0 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_1 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_2 = full_int_array_0

        # pd_op.pool2d: (1x20x1x1xf32) <- (1x20x128x256xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(data_0, full_int_array_0, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (1x5x1x1xf32) <- (1x20x1x1xf32, 5x20x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(pool2d_0, parameter_0, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [1, -1, 1, 1]

        # pd_op.reshape: (1x5x1x1xf32, 0x5xi64) <- (5xf32, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_1, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (1x5x1x1xf32) <- (1x5x1x1xf32, 1x5x1x1xf32)
        add_0 = conv2d_0 + reshape_0

        # pd_op.relu: (1x5x1x1xf32) <- (1x5x1x1xf32)
        relu_0 = paddle._C_ops.relu(add_0)

        # pd_op.conv2d: (1x20x1x1xf32) <- (1x5x1x1xf32, 20x5x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(relu_0, parameter_2, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x20x1x1xf32, 0x20xi64) <- (20xf32, 4xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_3, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (1x20x1x1xf32) <- (1x20x1x1xf32, 1x20x1x1xf32)
        add_1 = conv2d_1 + reshape_2

        # pd_op.sigmoid: (1x20x1x1xf32) <- (1x20x1x1xf32)
        sigmoid_0 = paddle.nn.functional.sigmoid(add_1)

        # pd_op.multiply: (1x20x128x256xf32) <- (1x20x128x256xf32, 1x20x1x1xf32)
        multiply_0 = data_0 * sigmoid_0

        # pd_op.pool2d: (1x40x1x1xf32) <- (1x40x64x128xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(data_1, assign_2, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (1x10x1x1xf32) <- (1x40x1x1xf32, 10x40x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(pool2d_1, parameter_4, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x10x1x1xf32, 0x10xi64) <- (10xf32, 4xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_5, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (1x10x1x1xf32) <- (1x10x1x1xf32, 1x10x1x1xf32)
        add_2 = conv2d_2 + reshape_4

        # pd_op.relu: (1x10x1x1xf32) <- (1x10x1x1xf32)
        relu_1 = paddle._C_ops.relu(add_2)

        # pd_op.conv2d: (1x40x1x1xf32) <- (1x10x1x1xf32, 40x10x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(relu_1, parameter_6, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x40x1x1xf32, 0x40xi64) <- (40xf32, 4xi64)
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_7, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (1x40x1x1xf32) <- (1x40x1x1xf32, 1x40x1x1xf32)
        add_3 = conv2d_3 + reshape_6

        # pd_op.sigmoid: (1x40x1x1xf32) <- (1x40x1x1xf32)
        sigmoid_1 = paddle.nn.functional.sigmoid(add_3)

        # pd_op.multiply: (1x40x64x128xf32) <- (1x40x64x128xf32, 1x40x1x1xf32)
        multiply_1 = data_1 * sigmoid_1

        # pd_op.pool2d: (1x80x1x1xf32) <- (1x80x32x64xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(data_2, assign_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (1x20x1x1xf32) <- (1x80x1x1xf32, 20x80x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(pool2d_2, parameter_8, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x20x1x1xf32, 0x20xi64) <- (20xf32, 4xi64)
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_9, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (1x20x1x1xf32) <- (1x20x1x1xf32, 1x20x1x1xf32)
        add_4 = conv2d_4 + reshape_8

        # pd_op.relu: (1x20x1x1xf32) <- (1x20x1x1xf32)
        relu_2 = paddle._C_ops.relu(add_4)

        # pd_op.conv2d: (1x80x1x1xf32) <- (1x20x1x1xf32, 80x20x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(relu_2, parameter_10, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x80x1x1xf32, 0x80xi64) <- (80xf32, 4xi64)
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_11, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (1x80x1x1xf32) <- (1x80x1x1xf32, 1x80x1x1xf32)
        add_5 = conv2d_5 + reshape_10

        # pd_op.sigmoid: (1x80x1x1xf32) <- (1x80x1x1xf32)
        sigmoid_2 = paddle.nn.functional.sigmoid(add_5)

        # pd_op.multiply: (1x80x32x64xf32) <- (1x80x32x64xf32, 1x80x1x1xf32)
        multiply_2 = data_2 * sigmoid_2

        # pd_op.pool2d: (1x160x1x1xf32) <- (1x160x16x32xf32, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(data_3, assign_0, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (1x40x1x1xf32) <- (1x160x1x1xf32, 40x160x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(pool2d_3, parameter_12, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x40x1x1xf32, 0x40xi64) <- (40xf32, 4xi64)
        reshape_12, reshape_13 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_13, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (1x40x1x1xf32) <- (1x40x1x1xf32, 1x40x1x1xf32)
        add_6 = conv2d_6 + reshape_12

        # pd_op.relu: (1x40x1x1xf32) <- (1x40x1x1xf32)
        relu_3 = paddle._C_ops.relu(add_6)

        # pd_op.conv2d: (1x160x1x1xf32) <- (1x40x1x1xf32, 160x40x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(relu_3, parameter_14, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x160x1x1xf32, 0x160xi64) <- (160xf32, 4xi64)
        reshape_14, reshape_15 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_15, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (1x160x1x1xf32) <- (1x160x1x1xf32, 1x160x1x1xf32)
        add_7 = conv2d_7 + reshape_14

        # pd_op.sigmoid: (1x160x1x1xf32) <- (1x160x1x1xf32)
        sigmoid_3 = paddle.nn.functional.sigmoid(add_7)

        # pd_op.multiply: (1x160x16x32xf32) <- (1x160x16x32xf32, 1x160x1x1xf32)
        multiply_3 = data_3 * sigmoid_3

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_3 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_4 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_5 = full_0

        # builtin.combine: ([1x20x128x256xf32, 1x20x128x256xf32]) <- (1x20x128x256xf32, 1x20x128x256xf32)
        combine_0 = [data_4, multiply_0]

        # pd_op.concat: (1x40x128x256xf32) <- ([1x20x128x256xf32, 1x20x128x256xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        # builtin.combine: ([1x40x64x128xf32, 1x40x64x128xf32]) <- (1x40x64x128xf32, 1x40x64x128xf32)
        combine_1 = [data_5, multiply_1]

        # pd_op.concat: (1x80x64x128xf32) <- ([1x40x64x128xf32, 1x40x64x128xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, assign_5)

        # builtin.combine: ([1x80x32x64xf32, 1x80x32x64xf32]) <- (1x80x32x64xf32, 1x80x32x64xf32)
        combine_2 = [data_6, multiply_2]

        # pd_op.concat: (1x160x32x64xf32) <- ([1x80x32x64xf32, 1x80x32x64xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, assign_4)

        # builtin.combine: ([1x160x16x32xf32, 1x160x16x32xf32]) <- (1x160x16x32xf32, 1x160x16x32xf32)
        combine_3 = [data_7, multiply_3]

        # pd_op.concat: (1x320x16x32xf32) <- ([1x160x16x32xf32, 1x160x16x32xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_3, assign_3)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_2 = [1, 2, 20, 128, 256]

        # pd_op.reshape: (1x2x20x128x256xf32, 0x1x40x128x256xi64) <- (1x40x128x256xf32, 5xi64)
        reshape_16, reshape_17 = (lambda x, f: f(x))(paddle._C_ops.reshape(concat_0, full_int_array_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (1x20x2x128x256xf32) <- (1x2x20x128x256xf32)
        transpose_0 = paddle._C_ops.transpose(reshape_16, [0, 2, 1, 3, 4])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [1, 40, 128, 256]

        # pd_op.reshape: (1x40x128x256xf32, 0x1x20x2x128x256xi64) <- (1x20x2x128x256xf32, 4xi64)
        reshape_18, reshape_19 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_0, full_int_array_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_4 = [1, 2, 40, 64, 128]

        # pd_op.reshape: (1x2x40x64x128xf32, 0x1x80x64x128xi64) <- (1x80x64x128xf32, 5xi64)
        reshape_20, reshape_21 = (lambda x, f: f(x))(paddle._C_ops.reshape(concat_1, full_int_array_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (1x40x2x64x128xf32) <- (1x2x40x64x128xf32)
        transpose_1 = paddle._C_ops.transpose(reshape_20, [0, 2, 1, 3, 4])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_5 = [1, 80, 64, 128]

        # pd_op.reshape: (1x80x64x128xf32, 0x1x40x2x64x128xi64) <- (1x40x2x64x128xf32, 4xi64)
        reshape_22, reshape_23 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_1, full_int_array_5), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_6 = [1, 2, 80, 32, 64]

        # pd_op.reshape: (1x2x80x32x64xf32, 0x1x160x32x64xi64) <- (1x160x32x64xf32, 5xi64)
        reshape_24, reshape_25 = (lambda x, f: f(x))(paddle._C_ops.reshape(concat_2, full_int_array_6), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (1x80x2x32x64xf32) <- (1x2x80x32x64xf32)
        transpose_2 = paddle._C_ops.transpose(reshape_24, [0, 2, 1, 3, 4])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_7 = [1, 160, 32, 64]

        # pd_op.reshape: (1x160x32x64xf32, 0x1x80x2x32x64xi64) <- (1x80x2x32x64xf32, 4xi64)
        reshape_26, reshape_27 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_2, full_int_array_7), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_8 = [1, 2, 160, 16, 32]

        # pd_op.reshape: (1x2x160x16x32xf32, 0x1x320x16x32xi64) <- (1x320x16x32xf32, 5xi64)
        reshape_28, reshape_29 = (lambda x, f: f(x))(paddle._C_ops.reshape(concat_3, full_int_array_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (1x160x2x16x32xf32) <- (1x2x160x16x32xf32)
        transpose_3 = paddle._C_ops.transpose(reshape_28, [0, 2, 1, 3, 4])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_9 = [1, 320, 16, 32]

        # pd_op.reshape: (1x320x16x32xf32, 0x1x160x2x16x32xi64) <- (1x160x2x16x32xf32, 4xi64)
        reshape_30, reshape_31 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_3, full_int_array_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))
        return full_int_array_0, pool2d_0, conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, sigmoid_0, multiply_0, assign_2, pool2d_1, conv2d_2, reshape_4, reshape_5, relu_1, conv2d_3, reshape_6, reshape_7, sigmoid_1, multiply_1, assign_1, pool2d_2, conv2d_4, reshape_8, reshape_9, relu_2, conv2d_5, reshape_10, reshape_11, sigmoid_2, multiply_2, assign_0, pool2d_3, conv2d_6, reshape_12, reshape_13, relu_3, conv2d_7, reshape_14, reshape_15, sigmoid_3, multiply_3, full_0, assign_5, assign_4, assign_3, reshape_17, reshape_19, reshape_21, reshape_23, reshape_25, reshape_27, reshape_29, reshape_31, reshape_18, reshape_22, reshape_26, reshape_30



def GetEnvVarEnableJit():
    enable_jit = os.getenv('PADDLE_DEBUG_ENABLE_JIT')
    return enable_jit not in {
        "0",
        "False",
        "false",
        "OFF",
    }

def GetEnvVarEnableCinn():
    enable_cinn = os.getenv('PADDLE_DEBUG_ENABLE_CINN')
    return enable_cinn not in {
        "0",
        "False",
        "false",
        "OFF",
    }


def GetTolerance(dtype):
    if dtype == np.float16:
        return GetFloat16Tolerance()
    if dtype == np.float32:
        return GetFloat32Tolerance()
    return 1e-6

def GetFloat16Tolerance():
    try:
        return float(os.getenv('PADDLE_DEBUG_FLOAT16_TOL'))
    except:
        return 1e-3

def GetFloat32Tolerance():
    try:
        return float(os.getenv('PADDLE_DEBUG_FLOAT32_TOL'))
    except:
        return 1e-6

def IsInteger(dtype):
    return np.dtype(dtype).char in np.typecodes['AllInteger']


class CinnTestBase:
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def _test_entry(self):
        dy_outs = self.entry(use_cinn=False)
        cinn_outs = self.entry(use_cinn=GetEnvVarEnableCinn())

        for cinn_out, dy_out in zip(cinn_outs, dy_outs):
          if type(cinn_out) is list and type(dy_out) is list:
            for x, y in zip(cinn_out, dy_out):
              self.assert_all_close(x, y)
          else:
            self.assert_all_close(cinn_out, dy_out)

    def assert_all_close(self, x, y):
        if (hasattr(x, "numpy") and hasattr(y, "numpy")):
            x_numpy = x.numpy()
            y_numpy = y.numpy()
            assert x_numpy.dtype == y_numpy.dtype
            if IsInteger(x_numpy.dtype):
                np.testing.assert_equal(x_numpy, y_numpy)
            else:
                tol = GetTolerance(x_numpy.dtype)
                np.testing.assert_allclose(x_numpy, y_numpy, atol=tol, rtol=tol)
        else:
            assert x == y

class ModuleOp(paddle.nn.Layer, BlockEntries):
    def __init__(self):
        super().__init__()

    def forward(self, parameter_1, parameter_11, parameter_5, parameter_0, parameter_14, parameter_13, parameter_2, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_3, parameter_9, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7):
        return self.builtin_module_461_0_0(parameter_1, parameter_11, parameter_5, parameter_0, parameter_14, parameter_13, parameter_2, parameter_7, parameter_6, parameter_10, parameter_15, parameter_4, parameter_8, parameter_12, parameter_3, parameter_9, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_461_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_1
            paddle.uniform([5], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([10], dtype='float32', min=0, max=0.5),
            # parameter_0
            paddle.uniform([5, 20, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([160, 40, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([20, 5, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([40, 10, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([80, 20, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([10, 40, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([20, 80, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([40, 160, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # data_0
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            # data_1
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            # data_2
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
            # data_3
            paddle.uniform([1, 160, 16, 32], dtype='float32', min=0, max=0.5),
            # data_4
            paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            # data_5
            paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            # data_6
            paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
            # data_7
            paddle.uniform([1, 160, 16, 32], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_1
            paddle.static.InputSpec(shape=[5], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[10], dtype='float32'),
            # parameter_0
            paddle.static.InputSpec(shape=[5, 20, 1, 1], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[160, 40, 1, 1], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[20, 5, 1, 1], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[40, 10, 1, 1], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[80, 20, 1, 1], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[10, 40, 1, 1], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[20, 80, 1, 1], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[40, 160, 1, 1], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # data_0
            paddle.static.InputSpec(shape=[1, 20, 128, 256], dtype='float32'),
            # data_1
            paddle.static.InputSpec(shape=[1, 40, 64, 128], dtype='float32'),
            # data_2
            paddle.static.InputSpec(shape=[1, 80, 32, 64], dtype='float32'),
            # data_3
            paddle.static.InputSpec(shape=[1, 160, 16, 32], dtype='float32'),
            # data_4
            paddle.static.InputSpec(shape=[1, 20, 128, 256], dtype='float32'),
            # data_5
            paddle.static.InputSpec(shape=[1, 40, 64, 128], dtype='float32'),
            # data_6
            paddle.static.InputSpec(shape=[1, 80, 32, 64], dtype='float32'),
            # data_7
            paddle.static.InputSpec(shape=[1, 160, 16, 32], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def entry(self, use_cinn):
        net = ModuleOp()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        paddle.seed(2024)
        out = net(*self.inputs)
        return out

    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        self._test_entry()

if __name__ == '__main__':
    unittest.main()