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
    return [117][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_742_0_0(self, parameter_37, parameter_9, parameter_8, parameter_15, parameter_23, parameter_38, parameter_1, parameter_31, parameter_19, parameter_0, parameter_28, parameter_22, parameter_21, parameter_39, parameter_32, parameter_18, parameter_33, parameter_12, parameter_7, parameter_30, parameter_5, parameter_17, parameter_16, parameter_29, parameter_35, parameter_6, parameter_27, parameter_2, parameter_36, parameter_3, parameter_11, parameter_10, parameter_25, parameter_26, parameter_13, parameter_24, parameter_20, parameter_14, parameter_34, parameter_4, data_3, data_2, data_1, data_0):

        # pd_op.conv2d: (-1x96x-1x-1xf32) <- (-1x480x-1x-1xf32, 96x480x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(data_0, parameter_0, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [1, 1]

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_0 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_1 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_2 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_3 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_4 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_5 = full_int_array_0

        # pd_op.assign: (2xi64) <- (2xi64)
        assign_6 = full_int_array_0

        # pd_op.pool2d: (-1x96x1x1xf32) <- (-1x96x-1x-1xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(conv2d_0, full_int_array_0, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x24x1x1xf32) <- (-1x96x1x1xf32, 24x96x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(pool2d_0, parameter_1, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [1, -1, 1, 1]

        # pd_op.reshape: (1x24x1x1xf32, 0x24xi64) <- (24xf32, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_2, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x24x1x1xf32) <- (-1x24x1x1xf32, 1x24x1x1xf32)
        add_0 = conv2d_1 + reshape_0

        # pd_op.relu: (-1x24x1x1xf32) <- (-1x24x1x1xf32)
        relu_0 = paddle._C_ops.relu(add_0)

        # pd_op.conv2d: (-1x96x1x1xf32) <- (-1x24x1x1xf32, 96x24x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(relu_0, parameter_3, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x96x1x1xf32, 0x96xi64) <- (96xf32, 4xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_4, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x96x1x1xf32) <- (-1x96x1x1xf32, 1x96x1x1xf32)
        add_1 = conv2d_2 + reshape_2

        # pd_op.hardsigmoid: (-1x96x1x1xf32) <- (-1x96x1x1xf32)
        hardsigmoid_0 = paddle._C_ops.hardsigmoid(add_1, float('0.2'), float('0.5'))

        # pd_op.multiply: (-1x96x-1x-1xf32) <- (-1x96x-1x-1xf32, -1x96x1x1xf32)
        multiply_0 = conv2d_0 * hardsigmoid_0

        # pd_op.add: (-1x96x-1x-1xf32) <- (-1x96x-1x-1xf32, -1x96x-1x-1xf32)
        add_2 = conv2d_0 + multiply_0

        # pd_op.conv2d: (-1x96x-1x-1xf32) <- (-1x56x-1x-1xf32, 96x56x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(data_1, parameter_5, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.pool2d: (-1x96x1x1xf32) <- (-1x96x-1x-1xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(conv2d_3, assign_6, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x24x1x1xf32) <- (-1x96x1x1xf32, 24x96x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(pool2d_1, parameter_6, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x24x1x1xf32, 0x24xi64) <- (24xf32, 4xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_7, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x24x1x1xf32) <- (-1x24x1x1xf32, 1x24x1x1xf32)
        add_3 = conv2d_4 + reshape_4

        # pd_op.relu: (-1x24x1x1xf32) <- (-1x24x1x1xf32)
        relu_1 = paddle._C_ops.relu(add_3)

        # pd_op.conv2d: (-1x96x1x1xf32) <- (-1x24x1x1xf32, 96x24x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(relu_1, parameter_8, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x96x1x1xf32, 0x96xi64) <- (96xf32, 4xi64)
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_9, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x96x1x1xf32) <- (-1x96x1x1xf32, 1x96x1x1xf32)
        add_4 = conv2d_5 + reshape_6

        # pd_op.hardsigmoid: (-1x96x1x1xf32) <- (-1x96x1x1xf32)
        hardsigmoid_1 = paddle._C_ops.hardsigmoid(add_4, float('0.2'), float('0.5'))

        # pd_op.multiply: (-1x96x-1x-1xf32) <- (-1x96x-1x-1xf32, -1x96x1x1xf32)
        multiply_1 = conv2d_3 * hardsigmoid_1

        # pd_op.add: (-1x96x-1x-1xf32) <- (-1x96x-1x-1xf32, -1x96x-1x-1xf32)
        add_5 = conv2d_3 + multiply_1

        # pd_op.conv2d: (-1x96x-1x-1xf32) <- (-1x24x-1x-1xf32, 96x24x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(data_2, parameter_10, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.pool2d: (-1x96x1x1xf32) <- (-1x96x-1x-1xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(conv2d_6, assign_5, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x24x1x1xf32) <- (-1x96x1x1xf32, 24x96x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(pool2d_2, parameter_11, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x24x1x1xf32, 0x24xi64) <- (24xf32, 4xi64)
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_12, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x24x1x1xf32) <- (-1x24x1x1xf32, 1x24x1x1xf32)
        add_6 = conv2d_7 + reshape_8

        # pd_op.relu: (-1x24x1x1xf32) <- (-1x24x1x1xf32)
        relu_2 = paddle._C_ops.relu(add_6)

        # pd_op.conv2d: (-1x96x1x1xf32) <- (-1x24x1x1xf32, 96x24x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(relu_2, parameter_13, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x96x1x1xf32, 0x96xi64) <- (96xf32, 4xi64)
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_14, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x96x1x1xf32) <- (-1x96x1x1xf32, 1x96x1x1xf32)
        add_7 = conv2d_8 + reshape_10

        # pd_op.hardsigmoid: (-1x96x1x1xf32) <- (-1x96x1x1xf32)
        hardsigmoid_2 = paddle._C_ops.hardsigmoid(add_7, float('0.2'), float('0.5'))

        # pd_op.multiply: (-1x96x-1x-1xf32) <- (-1x96x-1x-1xf32, -1x96x1x1xf32)
        multiply_2 = conv2d_6 * hardsigmoid_2

        # pd_op.add: (-1x96x-1x-1xf32) <- (-1x96x-1x-1xf32, -1x96x-1x-1xf32)
        add_8 = conv2d_6 + multiply_2

        # pd_op.conv2d: (-1x96x-1x-1xf32) <- (-1x16x-1x-1xf32, 96x16x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(data_3, parameter_15, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.pool2d: (-1x96x1x1xf32) <- (-1x96x-1x-1xf32, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(conv2d_9, assign_4, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x24x1x1xf32) <- (-1x96x1x1xf32, 24x96x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(pool2d_3, parameter_16, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x24x1x1xf32, 0x24xi64) <- (24xf32, 4xi64)
        reshape_12, reshape_13 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_17, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x24x1x1xf32) <- (-1x24x1x1xf32, 1x24x1x1xf32)
        add_9 = conv2d_10 + reshape_12

        # pd_op.relu: (-1x24x1x1xf32) <- (-1x24x1x1xf32)
        relu_3 = paddle._C_ops.relu(add_9)

        # pd_op.conv2d: (-1x96x1x1xf32) <- (-1x24x1x1xf32, 96x24x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(relu_3, parameter_18, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x96x1x1xf32, 0x96xi64) <- (96xf32, 4xi64)
        reshape_14, reshape_15 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_19, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x96x1x1xf32) <- (-1x96x1x1xf32, 1x96x1x1xf32)
        add_10 = conv2d_11 + reshape_14

        # pd_op.hardsigmoid: (-1x96x1x1xf32) <- (-1x96x1x1xf32)
        hardsigmoid_3 = paddle._C_ops.hardsigmoid(add_10, float('0.2'), float('0.5'))

        # pd_op.multiply: (-1x96x-1x-1xf32) <- (-1x96x-1x-1xf32, -1x96x1x1xf32)
        multiply_3 = conv2d_9 * hardsigmoid_3

        # pd_op.add: (-1x96x-1x-1xf32) <- (-1x96x-1x-1xf32, -1x96x-1x-1xf32)
        add_11 = conv2d_9 + multiply_3

        # pd_op.nearest_interp: (-1x96x-1x-1xf32) <- (-1x96x-1x-1xf32, None, None, None)
        nearest_interp_0 = paddle._C_ops.nearest_interp(add_2, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 1)

        # pd_op.add: (-1x96x-1x-1xf32) <- (-1x96x-1x-1xf32, -1x96x-1x-1xf32)
        add_12 = add_5 + nearest_interp_0

        # pd_op.nearest_interp: (-1x96x-1x-1xf32) <- (-1x96x-1x-1xf32, None, None, None)
        nearest_interp_1 = paddle._C_ops.nearest_interp(add_12, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 1)

        # pd_op.add: (-1x96x-1x-1xf32) <- (-1x96x-1x-1xf32, -1x96x-1x-1xf32)
        add_13 = add_8 + nearest_interp_1

        # pd_op.nearest_interp: (-1x96x-1x-1xf32) <- (-1x96x-1x-1xf32, None, None, None)
        nearest_interp_2 = paddle._C_ops.nearest_interp(add_13, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 1)

        # pd_op.add: (-1x96x-1x-1xf32) <- (-1x96x-1x-1xf32, -1x96x-1x-1xf32)
        add_14 = add_11 + nearest_interp_2

        # pd_op.conv2d: (-1x24x-1x-1xf32) <- (-1x96x-1x-1xf32, 24x96x3x3xf32)
        conv2d_12 = paddle._C_ops.conv2d(add_2, parameter_20, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.pool2d: (-1x24x1x1xf32) <- (-1x24x-1x-1xf32, 2xi64)
        pool2d_4 = paddle._C_ops.pool2d(conv2d_12, assign_3, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x6x1x1xf32) <- (-1x24x1x1xf32, 6x24x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(pool2d_4, parameter_21, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x6x1x1xf32, 0x6xi64) <- (6xf32, 4xi64)
        reshape_16, reshape_17 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_22, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x6x1x1xf32) <- (-1x6x1x1xf32, 1x6x1x1xf32)
        add_15 = conv2d_13 + reshape_16

        # pd_op.relu: (-1x6x1x1xf32) <- (-1x6x1x1xf32)
        relu_4 = paddle._C_ops.relu(add_15)

        # pd_op.conv2d: (-1x24x1x1xf32) <- (-1x6x1x1xf32, 24x6x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(relu_4, parameter_23, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x24x1x1xf32, 0x24xi64) <- (24xf32, 4xi64)
        reshape_18, reshape_19 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_24, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x24x1x1xf32) <- (-1x24x1x1xf32, 1x24x1x1xf32)
        add_16 = conv2d_14 + reshape_18

        # pd_op.hardsigmoid: (-1x24x1x1xf32) <- (-1x24x1x1xf32)
        hardsigmoid_4 = paddle._C_ops.hardsigmoid(add_16, float('0.2'), float('0.5'))

        # pd_op.multiply: (-1x24x-1x-1xf32) <- (-1x24x-1x-1xf32, -1x24x1x1xf32)
        multiply_4 = conv2d_12 * hardsigmoid_4

        # pd_op.add: (-1x24x-1x-1xf32) <- (-1x24x-1x-1xf32, -1x24x-1x-1xf32)
        add_17 = conv2d_12 + multiply_4

        # pd_op.conv2d: (-1x24x-1x-1xf32) <- (-1x96x-1x-1xf32, 24x96x3x3xf32)
        conv2d_15 = paddle._C_ops.conv2d(add_12, parameter_25, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.pool2d: (-1x24x1x1xf32) <- (-1x24x-1x-1xf32, 2xi64)
        pool2d_5 = paddle._C_ops.pool2d(conv2d_15, assign_2, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x6x1x1xf32) <- (-1x24x1x1xf32, 6x24x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(pool2d_5, parameter_26, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x6x1x1xf32, 0x6xi64) <- (6xf32, 4xi64)
        reshape_20, reshape_21 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_27, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x6x1x1xf32) <- (-1x6x1x1xf32, 1x6x1x1xf32)
        add_18 = conv2d_16 + reshape_20

        # pd_op.relu: (-1x6x1x1xf32) <- (-1x6x1x1xf32)
        relu_5 = paddle._C_ops.relu(add_18)

        # pd_op.conv2d: (-1x24x1x1xf32) <- (-1x6x1x1xf32, 24x6x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(relu_5, parameter_28, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x24x1x1xf32, 0x24xi64) <- (24xf32, 4xi64)
        reshape_22, reshape_23 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_29, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x24x1x1xf32) <- (-1x24x1x1xf32, 1x24x1x1xf32)
        add_19 = conv2d_17 + reshape_22

        # pd_op.hardsigmoid: (-1x24x1x1xf32) <- (-1x24x1x1xf32)
        hardsigmoid_5 = paddle._C_ops.hardsigmoid(add_19, float('0.2'), float('0.5'))

        # pd_op.multiply: (-1x24x-1x-1xf32) <- (-1x24x-1x-1xf32, -1x24x1x1xf32)
        multiply_5 = conv2d_15 * hardsigmoid_5

        # pd_op.add: (-1x24x-1x-1xf32) <- (-1x24x-1x-1xf32, -1x24x-1x-1xf32)
        add_20 = conv2d_15 + multiply_5

        # pd_op.conv2d: (-1x24x-1x-1xf32) <- (-1x96x-1x-1xf32, 24x96x3x3xf32)
        conv2d_18 = paddle._C_ops.conv2d(add_13, parameter_30, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.pool2d: (-1x24x1x1xf32) <- (-1x24x-1x-1xf32, 2xi64)
        pool2d_6 = paddle._C_ops.pool2d(conv2d_18, assign_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x6x1x1xf32) <- (-1x24x1x1xf32, 6x24x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(pool2d_6, parameter_31, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x6x1x1xf32, 0x6xi64) <- (6xf32, 4xi64)
        reshape_24, reshape_25 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_32, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x6x1x1xf32) <- (-1x6x1x1xf32, 1x6x1x1xf32)
        add_21 = conv2d_19 + reshape_24

        # pd_op.relu: (-1x6x1x1xf32) <- (-1x6x1x1xf32)
        relu_6 = paddle._C_ops.relu(add_21)

        # pd_op.conv2d: (-1x24x1x1xf32) <- (-1x6x1x1xf32, 24x6x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(relu_6, parameter_33, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x24x1x1xf32, 0x24xi64) <- (24xf32, 4xi64)
        reshape_26, reshape_27 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_34, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x24x1x1xf32) <- (-1x24x1x1xf32, 1x24x1x1xf32)
        add_22 = conv2d_20 + reshape_26

        # pd_op.hardsigmoid: (-1x24x1x1xf32) <- (-1x24x1x1xf32)
        hardsigmoid_6 = paddle._C_ops.hardsigmoid(add_22, float('0.2'), float('0.5'))

        # pd_op.multiply: (-1x24x-1x-1xf32) <- (-1x24x-1x-1xf32, -1x24x1x1xf32)
        multiply_6 = conv2d_18 * hardsigmoid_6

        # pd_op.add: (-1x24x-1x-1xf32) <- (-1x24x-1x-1xf32, -1x24x-1x-1xf32)
        add_23 = conv2d_18 + multiply_6

        # pd_op.conv2d: (-1x24x-1x-1xf32) <- (-1x96x-1x-1xf32, 24x96x3x3xf32)
        conv2d_21 = paddle._C_ops.conv2d(add_14, parameter_35, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.pool2d: (-1x24x1x1xf32) <- (-1x24x-1x-1xf32, 2xi64)
        pool2d_7 = paddle._C_ops.pool2d(conv2d_21, assign_0, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x6x1x1xf32) <- (-1x24x1x1xf32, 6x24x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(pool2d_7, parameter_36, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x6x1x1xf32, 0x6xi64) <- (6xf32, 4xi64)
        reshape_28, reshape_29 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_37, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x6x1x1xf32) <- (-1x6x1x1xf32, 1x6x1x1xf32)
        add_24 = conv2d_22 + reshape_28

        # pd_op.relu: (-1x6x1x1xf32) <- (-1x6x1x1xf32)
        relu_7 = paddle._C_ops.relu(add_24)

        # pd_op.conv2d: (-1x24x1x1xf32) <- (-1x6x1x1xf32, 24x6x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(relu_7, parameter_38, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x24x1x1xf32, 0x24xi64) <- (24xf32, 4xi64)
        reshape_30, reshape_31 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_39, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x24x1x1xf32) <- (-1x24x1x1xf32, 1x24x1x1xf32)
        add_25 = conv2d_23 + reshape_30

        # pd_op.hardsigmoid: (-1x24x1x1xf32) <- (-1x24x1x1xf32)
        hardsigmoid_7 = paddle._C_ops.hardsigmoid(add_25, float('0.2'), float('0.5'))

        # pd_op.multiply: (-1x24x-1x-1xf32) <- (-1x24x-1x-1xf32, -1x24x1x1xf32)
        multiply_7 = conv2d_21 * hardsigmoid_7

        # pd_op.add: (-1x24x-1x-1xf32) <- (-1x24x-1x-1xf32, -1x24x-1x-1xf32)
        add_26 = conv2d_21 + multiply_7

        # pd_op.nearest_interp: (-1x24x-1x-1xf32) <- (-1x24x-1x-1xf32, None, None, None)
        nearest_interp_3 = paddle._C_ops.nearest_interp(add_17, None, None, None, 'NCHW', -1, -1, -1, [float('8'), float('8')], 'nearest', False, 1)

        # pd_op.nearest_interp: (-1x24x-1x-1xf32) <- (-1x24x-1x-1xf32, None, None, None)
        nearest_interp_4 = paddle._C_ops.nearest_interp(add_20, None, None, None, 'NCHW', -1, -1, -1, [float('4'), float('4')], 'nearest', False, 1)

        # pd_op.nearest_interp: (-1x24x-1x-1xf32) <- (-1x24x-1x-1xf32, None, None, None)
        nearest_interp_5 = paddle._C_ops.nearest_interp(add_23, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 1)

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([-1x24x-1x-1xf32, -1x24x-1x-1xf32, -1x24x-1x-1xf32, -1x24x-1x-1xf32]) <- (-1x24x-1x-1xf32, -1x24x-1x-1xf32, -1x24x-1x-1xf32, -1x24x-1x-1xf32)
        combine_0 = [nearest_interp_3, nearest_interp_4, nearest_interp_5, add_26]

        # pd_op.concat: (-1x96x-1x-1xf32) <- ([-1x24x-1x-1xf32, -1x24x-1x-1xf32, -1x24x-1x-1xf32, -1x24x-1x-1xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)
        return conv2d_0, full_int_array_0, pool2d_0, conv2d_1, reshape_0, reshape_1, relu_0, conv2d_2, reshape_2, reshape_3, hardsigmoid_0, multiply_0, add_2, conv2d_3, assign_6, pool2d_1, conv2d_4, reshape_4, reshape_5, relu_1, conv2d_5, reshape_6, reshape_7, hardsigmoid_1, multiply_1, add_5, conv2d_6, assign_5, pool2d_2, conv2d_7, reshape_8, reshape_9, relu_2, conv2d_8, reshape_10, reshape_11, hardsigmoid_2, multiply_2, add_8, conv2d_9, assign_4, pool2d_3, conv2d_10, reshape_12, reshape_13, relu_3, conv2d_11, reshape_14, reshape_15, hardsigmoid_3, multiply_3, add_11, nearest_interp_0, add_12, nearest_interp_1, add_13, nearest_interp_2, add_14, conv2d_12, assign_3, pool2d_4, conv2d_13, reshape_16, reshape_17, relu_4, conv2d_14, reshape_18, reshape_19, hardsigmoid_4, multiply_4, add_17, conv2d_15, assign_2, pool2d_5, conv2d_16, reshape_20, reshape_21, relu_5, conv2d_17, reshape_22, reshape_23, hardsigmoid_5, multiply_5, add_20, conv2d_18, assign_1, pool2d_6, conv2d_19, reshape_24, reshape_25, relu_6, conv2d_20, reshape_26, reshape_27, hardsigmoid_6, multiply_6, add_23, conv2d_21, assign_0, pool2d_7, conv2d_22, reshape_28, reshape_29, relu_7, conv2d_23, reshape_30, reshape_31, hardsigmoid_7, multiply_7, add_26, nearest_interp_3, nearest_interp_4, nearest_interp_5, full_0, concat_0



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

    def forward(self, parameter_37, parameter_9, parameter_8, parameter_15, parameter_23, parameter_38, parameter_1, parameter_31, parameter_19, parameter_0, parameter_28, parameter_22, parameter_21, parameter_39, parameter_32, parameter_18, parameter_33, parameter_12, parameter_7, parameter_30, parameter_5, parameter_17, parameter_16, parameter_29, parameter_35, parameter_6, parameter_27, parameter_2, parameter_36, parameter_3, parameter_11, parameter_10, parameter_25, parameter_26, parameter_13, parameter_24, parameter_20, parameter_14, parameter_34, parameter_4, data_3, data_2, data_1, data_0):
        return self.builtin_module_742_0_0(parameter_37, parameter_9, parameter_8, parameter_15, parameter_23, parameter_38, parameter_1, parameter_31, parameter_19, parameter_0, parameter_28, parameter_22, parameter_21, parameter_39, parameter_32, parameter_18, parameter_33, parameter_12, parameter_7, parameter_30, parameter_5, parameter_17, parameter_16, parameter_29, parameter_35, parameter_6, parameter_27, parameter_2, parameter_36, parameter_3, parameter_11, parameter_10, parameter_25, parameter_26, parameter_13, parameter_24, parameter_20, parameter_14, parameter_34, parameter_4, data_3, data_2, data_1, data_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_742_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_37
            paddle.uniform([6], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([96, 16, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([24, 6, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([24, 6, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_0
            paddle.uniform([96, 480, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([24, 6, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([6], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([6], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([24, 6, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([24, 96, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([96, 56, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([24, 96, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([6], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([24, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([24, 96, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([6, 24, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([96, 24, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([24, 96, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # data_3
            paddle.uniform([2, 16, 240, 240], dtype='float32', min=0, max=0.5),
            # data_2
            paddle.uniform([2, 24, 120, 120], dtype='float32', min=0, max=0.5),
            # data_1
            paddle.uniform([2, 56, 60, 60], dtype='float32', min=0, max=0.5),
            # data_0
            paddle.uniform([2, 480, 30, 30], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_37
            paddle.static.InputSpec(shape=[6], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[96, 24, 1, 1], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[96, 16, 1, 1], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[24, 6, 1, 1], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[24, 6, 1, 1], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[24, 96, 1, 1], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[6, 24, 1, 1], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_0
            paddle.static.InputSpec(shape=[96, 480, 1, 1], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[24, 6, 1, 1], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[6], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[6, 24, 1, 1], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[6], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[96, 24, 1, 1], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[24, 6, 1, 1], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[24, 96, 3, 3], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[96, 56, 1, 1], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[24, 96, 1, 1], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[24, 96, 3, 3], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[24, 96, 1, 1], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[6], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[6, 24, 1, 1], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[96, 24, 1, 1], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[24, 96, 1, 1], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[96, 24, 1, 1], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[24, 96, 3, 3], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[6, 24, 1, 1], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[96, 24, 1, 1], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[24, 96, 3, 3], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # data_3
            paddle.static.InputSpec(shape=[None, 16, None, None], dtype='float32'),
            # data_2
            paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
            # data_1
            paddle.static.InputSpec(shape=[None, 56, None, None], dtype='float32'),
            # data_0
            paddle.static.InputSpec(shape=[None, 480, None, None], dtype='float32'),
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