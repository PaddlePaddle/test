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
    return [231][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_1064_0_0(self, parameter_8, parameter_15, parameter_0, parameter_14, parameter_4, parameter_7, parameter_11, parameter_5, parameter_10, parameter_9, parameter_2, parameter_19, parameter_17, parameter_16, parameter_13, parameter_18, parameter_12, parameter_6, parameter_1, parameter_3, data_0, data_1, data_2, data_3, data_4):

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(data_0, parameter_0, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xi64) <- (256xf32, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_1, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.assign: (0x256xi64) <- (0x256xi64)
        assign_0 = reshape_1

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_1 = reshape_0

        # pd_op.assign: (0x256xi64) <- (0x256xi64)
        assign_2 = reshape_1

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_3 = reshape_0

        # pd_op.assign: (0x256xi64) <- (0x256xi64)
        assign_4 = reshape_1

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_5 = reshape_0

        # pd_op.assign: (0x256xi64) <- (0x256xi64)
        assign_6 = reshape_1

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_7 = reshape_0

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_0 = conv2d_0 + reshape_0

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_0 = paddle._C_ops.relu(add_0)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(data_0, parameter_2, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x256x1x1xf32, 0x256xi64) <- (256xf32, 4xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_3, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.assign: (0x256xi64) <- (0x256xi64)
        assign_8 = reshape_3

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_9 = reshape_2

        # pd_op.assign: (0x256xi64) <- (0x256xi64)
        assign_10 = reshape_3

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_11 = reshape_2

        # pd_op.assign: (0x256xi64) <- (0x256xi64)
        assign_12 = reshape_3

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_13 = reshape_2

        # pd_op.assign: (0x256xi64) <- (0x256xi64)
        assign_14 = reshape_3

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_15 = reshape_2

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_1 = conv2d_1 + reshape_2

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_1 = paddle._C_ops.relu(add_1)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(relu_0, parameter_4, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x256x1x1xf32, 0x256xi64) <- (256xf32, 4xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_5, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.assign: (0x256xi64) <- (0x256xi64)
        assign_16 = reshape_5

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_17 = reshape_4

        # pd_op.assign: (0x256xi64) <- (0x256xi64)
        assign_18 = reshape_5

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_19 = reshape_4

        # pd_op.assign: (0x256xi64) <- (0x256xi64)
        assign_20 = reshape_5

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_21 = reshape_4

        # pd_op.assign: (0x256xi64) <- (0x256xi64)
        assign_22 = reshape_5

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_23 = reshape_4

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_2 = conv2d_2 + reshape_4

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_2 = paddle._C_ops.relu(add_2)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(relu_1, parameter_6, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x256x1x1xf32, 0x256xi64) <- (256xf32, 4xi64)
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_7, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.assign: (0x256xi64) <- (0x256xi64)
        assign_24 = reshape_7

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_25 = reshape_6

        # pd_op.assign: (0x256xi64) <- (0x256xi64)
        assign_26 = reshape_7

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_27 = reshape_6

        # pd_op.assign: (0x256xi64) <- (0x256xi64)
        assign_28 = reshape_7

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_29 = reshape_6

        # pd_op.assign: (0x256xi64) <- (0x256xi64)
        assign_30 = reshape_7

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_31 = reshape_6

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_3 = conv2d_3 + reshape_6

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_3 = paddle._C_ops.relu(add_3)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(relu_2, parameter_8, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x256x1x1xf32, 0x256xi64) <- (256xf32, 4xi64)
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_9, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.assign: (0x256xi64) <- (0x256xi64)
        assign_32 = reshape_9

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_33 = reshape_8

        # pd_op.assign: (0x256xi64) <- (0x256xi64)
        assign_34 = reshape_9

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_35 = reshape_8

        # pd_op.assign: (0x256xi64) <- (0x256xi64)
        assign_36 = reshape_9

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_37 = reshape_8

        # pd_op.assign: (0x256xi64) <- (0x256xi64)
        assign_38 = reshape_9

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_39 = reshape_8

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_4 = conv2d_4 + reshape_8

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_4 = paddle._C_ops.relu(add_4)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(relu_3, parameter_10, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x256x1x1xf32, 0x256xi64) <- (256xf32, 4xi64)
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_11, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.assign: (0x256xi64) <- (0x256xi64)
        assign_40 = reshape_11

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_41 = reshape_10

        # pd_op.assign: (0x256xi64) <- (0x256xi64)
        assign_42 = reshape_11

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_43 = reshape_10

        # pd_op.assign: (0x256xi64) <- (0x256xi64)
        assign_44 = reshape_11

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_45 = reshape_10

        # pd_op.assign: (0x256xi64) <- (0x256xi64)
        assign_46 = reshape_11

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_47 = reshape_10

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_5 = conv2d_5 + reshape_10

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_5 = paddle._C_ops.relu(add_5)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(relu_4, parameter_12, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x256x1x1xf32, 0x256xi64) <- (256xf32, 4xi64)
        reshape_12, reshape_13 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_13, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.assign: (0x256xi64) <- (0x256xi64)
        assign_48 = reshape_13

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_49 = reshape_12

        # pd_op.assign: (0x256xi64) <- (0x256xi64)
        assign_50 = reshape_13

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_51 = reshape_12

        # pd_op.assign: (0x256xi64) <- (0x256xi64)
        assign_52 = reshape_13

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_53 = reshape_12

        # pd_op.assign: (0x256xi64) <- (0x256xi64)
        assign_54 = reshape_13

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_55 = reshape_12

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_6 = conv2d_6 + reshape_12

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_6 = paddle._C_ops.relu(add_6)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_7 = paddle._C_ops.conv2d(relu_5, parameter_14, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x256x1x1xf32, 0x256xi64) <- (256xf32, 4xi64)
        reshape_14, reshape_15 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_15, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.assign: (0x256xi64) <- (0x256xi64)
        assign_56 = reshape_15

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_57 = reshape_14

        # pd_op.assign: (0x256xi64) <- (0x256xi64)
        assign_58 = reshape_15

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_59 = reshape_14

        # pd_op.assign: (0x256xi64) <- (0x256xi64)
        assign_60 = reshape_15

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_61 = reshape_14

        # pd_op.assign: (0x256xi64) <- (0x256xi64)
        assign_62 = reshape_15

        # pd_op.assign: (1x256x1x1xf32) <- (1x256x1x1xf32)
        assign_63 = reshape_14

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_7 = conv2d_7 + reshape_14

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_7 = paddle._C_ops.relu(add_7)

        # pd_op.conv2d: (-1x720x-1x-1xf32) <- (-1x256x-1x-1xf32, 720x256x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(relu_6, parameter_16, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x720x1x1xf32, 0x720xi64) <- (720xf32, 4xi64)
        reshape_16, reshape_17 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_17, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.assign: (0x720xi64) <- (0x720xi64)
        assign_64 = reshape_17

        # pd_op.assign: (1x720x1x1xf32) <- (1x720x1x1xf32)
        assign_65 = reshape_16

        # pd_op.assign: (0x720xi64) <- (0x720xi64)
        assign_66 = reshape_17

        # pd_op.assign: (1x720x1x1xf32) <- (1x720x1x1xf32)
        assign_67 = reshape_16

        # pd_op.assign: (0x720xi64) <- (0x720xi64)
        assign_68 = reshape_17

        # pd_op.assign: (1x720x1x1xf32) <- (1x720x1x1xf32)
        assign_69 = reshape_16

        # pd_op.assign: (0x720xi64) <- (0x720xi64)
        assign_70 = reshape_17

        # pd_op.assign: (1x720x1x1xf32) <- (1x720x1x1xf32)
        assign_71 = reshape_16

        # pd_op.add: (-1x720x-1x-1xf32) <- (-1x720x-1x-1xf32, 1x720x1x1xf32)
        add_8 = conv2d_8 + reshape_16

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x256x-1x-1xf32, 36x256x3x3xf32)
        conv2d_9 = paddle._C_ops.conv2d(relu_7, parameter_18, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x36x1x1xf32, 0x36xi64) <- (36xf32, 4xi64)
        reshape_18, reshape_19 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_19, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.assign: (0x36xi64) <- (0x36xi64)
        assign_72 = reshape_19

        # pd_op.assign: (1x36x1x1xf32) <- (1x36x1x1xf32)
        assign_73 = reshape_18

        # pd_op.assign: (0x36xi64) <- (0x36xi64)
        assign_74 = reshape_19

        # pd_op.assign: (1x36x1x1xf32) <- (1x36x1x1xf32)
        assign_75 = reshape_18

        # pd_op.assign: (0x36xi64) <- (0x36xi64)
        assign_76 = reshape_19

        # pd_op.assign: (1x36x1x1xf32) <- (1x36x1x1xf32)
        assign_77 = reshape_18

        # pd_op.assign: (0x36xi64) <- (0x36xi64)
        assign_78 = reshape_19

        # pd_op.assign: (1x36x1x1xf32) <- (1x36x1x1xf32)
        assign_79 = reshape_18

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 1x36x1x1xf32)
        add_9 = conv2d_9 + reshape_18

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_10 = paddle._C_ops.conv2d(data_1, parameter_0, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_10 = conv2d_10 + assign_7

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_8 = paddle._C_ops.relu(add_10)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(data_1, parameter_2, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_11 = conv2d_11 + assign_15

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_9 = paddle._C_ops.relu(add_11)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_12 = paddle._C_ops.conv2d(relu_8, parameter_4, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_12 = conv2d_12 + assign_23

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_10 = paddle._C_ops.relu(add_12)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_13 = paddle._C_ops.conv2d(relu_9, parameter_6, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_13 = conv2d_13 + assign_31

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_11 = paddle._C_ops.relu(add_13)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_14 = paddle._C_ops.conv2d(relu_10, parameter_8, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_14 = conv2d_14 + assign_39

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_12 = paddle._C_ops.relu(add_14)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_15 = paddle._C_ops.conv2d(relu_11, parameter_10, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_15 = conv2d_15 + assign_47

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_13 = paddle._C_ops.relu(add_15)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_16 = paddle._C_ops.conv2d(relu_12, parameter_12, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_16 = conv2d_16 + assign_55

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_14 = paddle._C_ops.relu(add_16)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_17 = paddle._C_ops.conv2d(relu_13, parameter_14, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_17 = conv2d_17 + assign_63

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_15 = paddle._C_ops.relu(add_17)

        # pd_op.conv2d: (-1x720x-1x-1xf32) <- (-1x256x-1x-1xf32, 720x256x3x3xf32)
        conv2d_18 = paddle._C_ops.conv2d(relu_14, parameter_16, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x720x-1x-1xf32) <- (-1x720x-1x-1xf32, 1x720x1x1xf32)
        add_18 = conv2d_18 + assign_71

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x256x-1x-1xf32, 36x256x3x3xf32)
        conv2d_19 = paddle._C_ops.conv2d(relu_15, parameter_18, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 1x36x1x1xf32)
        add_19 = conv2d_19 + assign_79

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_20 = paddle._C_ops.conv2d(data_2, parameter_0, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_20 = conv2d_20 + assign_5

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_16 = paddle._C_ops.relu(add_20)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_21 = paddle._C_ops.conv2d(data_2, parameter_2, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_21 = conv2d_21 + assign_13

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_17 = paddle._C_ops.relu(add_21)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_22 = paddle._C_ops.conv2d(relu_16, parameter_4, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_22 = conv2d_22 + assign_21

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_18 = paddle._C_ops.relu(add_22)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_23 = paddle._C_ops.conv2d(relu_17, parameter_6, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_23 = conv2d_23 + assign_29

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_19 = paddle._C_ops.relu(add_23)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_24 = paddle._C_ops.conv2d(relu_18, parameter_8, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_24 = conv2d_24 + assign_37

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_20 = paddle._C_ops.relu(add_24)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_25 = paddle._C_ops.conv2d(relu_19, parameter_10, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_25 = conv2d_25 + assign_45

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_21 = paddle._C_ops.relu(add_25)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_26 = paddle._C_ops.conv2d(relu_20, parameter_12, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_26 = conv2d_26 + assign_53

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_22 = paddle._C_ops.relu(add_26)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_27 = paddle._C_ops.conv2d(relu_21, parameter_14, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_27 = conv2d_27 + assign_61

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_23 = paddle._C_ops.relu(add_27)

        # pd_op.conv2d: (-1x720x-1x-1xf32) <- (-1x256x-1x-1xf32, 720x256x3x3xf32)
        conv2d_28 = paddle._C_ops.conv2d(relu_22, parameter_16, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x720x-1x-1xf32) <- (-1x720x-1x-1xf32, 1x720x1x1xf32)
        add_28 = conv2d_28 + assign_69

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x256x-1x-1xf32, 36x256x3x3xf32)
        conv2d_29 = paddle._C_ops.conv2d(relu_23, parameter_18, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 1x36x1x1xf32)
        add_29 = conv2d_29 + assign_77

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_30 = paddle._C_ops.conv2d(data_3, parameter_0, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_30 = conv2d_30 + assign_3

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_24 = paddle._C_ops.relu(add_30)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_31 = paddle._C_ops.conv2d(data_3, parameter_2, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_31 = conv2d_31 + assign_11

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_25 = paddle._C_ops.relu(add_31)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_32 = paddle._C_ops.conv2d(relu_24, parameter_4, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_32 = conv2d_32 + assign_19

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_26 = paddle._C_ops.relu(add_32)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_33 = paddle._C_ops.conv2d(relu_25, parameter_6, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_33 = conv2d_33 + assign_27

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_27 = paddle._C_ops.relu(add_33)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_34 = paddle._C_ops.conv2d(relu_26, parameter_8, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_34 = conv2d_34 + assign_35

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_28 = paddle._C_ops.relu(add_34)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_35 = paddle._C_ops.conv2d(relu_27, parameter_10, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_35 = conv2d_35 + assign_43

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_29 = paddle._C_ops.relu(add_35)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_36 = paddle._C_ops.conv2d(relu_28, parameter_12, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_36 = conv2d_36 + assign_51

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_30 = paddle._C_ops.relu(add_36)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_37 = paddle._C_ops.conv2d(relu_29, parameter_14, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_37 = conv2d_37 + assign_59

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_31 = paddle._C_ops.relu(add_37)

        # pd_op.conv2d: (-1x720x-1x-1xf32) <- (-1x256x-1x-1xf32, 720x256x3x3xf32)
        conv2d_38 = paddle._C_ops.conv2d(relu_30, parameter_16, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x720x-1x-1xf32) <- (-1x720x-1x-1xf32, 1x720x1x1xf32)
        add_38 = conv2d_38 + assign_67

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x256x-1x-1xf32, 36x256x3x3xf32)
        conv2d_39 = paddle._C_ops.conv2d(relu_31, parameter_18, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 1x36x1x1xf32)
        add_39 = conv2d_39 + assign_75

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_40 = paddle._C_ops.conv2d(data_4, parameter_0, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_40 = conv2d_40 + assign_1

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_32 = paddle._C_ops.relu(add_40)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_41 = paddle._C_ops.conv2d(data_4, parameter_2, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_41 = conv2d_41 + assign_9

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_33 = paddle._C_ops.relu(add_41)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_42 = paddle._C_ops.conv2d(relu_32, parameter_4, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_42 = conv2d_42 + assign_17

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_34 = paddle._C_ops.relu(add_42)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_43 = paddle._C_ops.conv2d(relu_33, parameter_6, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_43 = conv2d_43 + assign_25

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_35 = paddle._C_ops.relu(add_43)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_44 = paddle._C_ops.conv2d(relu_34, parameter_8, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_44 = conv2d_44 + assign_33

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_36 = paddle._C_ops.relu(add_44)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_45 = paddle._C_ops.conv2d(relu_35, parameter_10, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_45 = conv2d_45 + assign_41

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_37 = paddle._C_ops.relu(add_45)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_46 = paddle._C_ops.conv2d(relu_36, parameter_12, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_46 = conv2d_46 + assign_49

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_38 = paddle._C_ops.relu(add_46)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_47 = paddle._C_ops.conv2d(relu_37, parameter_14, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_47 = conv2d_47 + assign_57

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_39 = paddle._C_ops.relu(add_47)

        # pd_op.conv2d: (-1x720x-1x-1xf32) <- (-1x256x-1x-1xf32, 720x256x3x3xf32)
        conv2d_48 = paddle._C_ops.conv2d(relu_38, parameter_16, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x720x-1x-1xf32) <- (-1x720x-1x-1xf32, 1x720x1x1xf32)
        add_48 = conv2d_48 + assign_65

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x256x-1x-1xf32, 36x256x3x3xf32)
        conv2d_49 = paddle._C_ops.conv2d(relu_39, parameter_18, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 1x36x1x1xf32)
        add_49 = conv2d_49 + assign_73
        return conv2d_0, reshape_0, reshape_1, relu_0, conv2d_1, reshape_2, reshape_3, relu_1, conv2d_2, reshape_4, reshape_5, relu_2, conv2d_3, reshape_6, reshape_7, relu_3, conv2d_4, reshape_8, reshape_9, relu_4, conv2d_5, reshape_10, reshape_11, relu_5, conv2d_6, reshape_12, reshape_13, relu_6, conv2d_7, reshape_14, reshape_15, relu_7, conv2d_8, reshape_16, reshape_17, conv2d_9, reshape_18, reshape_19, conv2d_10, assign_7, assign_6, relu_8, conv2d_11, assign_15, assign_14, relu_9, conv2d_12, assign_23, assign_22, relu_10, conv2d_13, assign_31, assign_30, relu_11, conv2d_14, assign_39, assign_38, relu_12, conv2d_15, assign_47, assign_46, relu_13, conv2d_16, assign_55, assign_54, relu_14, conv2d_17, assign_63, assign_62, relu_15, conv2d_18, assign_71, assign_70, conv2d_19, assign_79, assign_78, conv2d_20, assign_5, assign_4, relu_16, conv2d_21, assign_13, assign_12, relu_17, conv2d_22, assign_21, assign_20, relu_18, conv2d_23, assign_29, assign_28, relu_19, conv2d_24, assign_37, assign_36, relu_20, conv2d_25, assign_45, assign_44, relu_21, conv2d_26, assign_53, assign_52, relu_22, conv2d_27, assign_61, assign_60, relu_23, conv2d_28, assign_69, assign_68, conv2d_29, assign_77, assign_76, conv2d_30, assign_3, assign_2, relu_24, conv2d_31, assign_11, assign_10, relu_25, conv2d_32, assign_19, assign_18, relu_26, conv2d_33, assign_27, assign_26, relu_27, conv2d_34, assign_35, assign_34, relu_28, conv2d_35, assign_43, assign_42, relu_29, conv2d_36, assign_51, assign_50, relu_30, conv2d_37, assign_59, assign_58, relu_31, conv2d_38, assign_67, assign_66, conv2d_39, assign_75, assign_74, conv2d_40, assign_1, assign_0, relu_32, conv2d_41, assign_9, assign_8, relu_33, conv2d_42, assign_17, assign_16, relu_34, conv2d_43, assign_25, assign_24, relu_35, conv2d_44, assign_33, assign_32, relu_36, conv2d_45, assign_41, assign_40, relu_37, conv2d_46, assign_49, assign_48, relu_38, conv2d_47, assign_57, assign_56, relu_39, conv2d_48, assign_65, assign_64, conv2d_49, assign_73, assign_72, add_8, add_18, add_28, add_38, add_48, add_9, add_19, add_29, add_39, add_49



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

    def forward(self, parameter_8, parameter_15, parameter_0, parameter_14, parameter_4, parameter_7, parameter_11, parameter_5, parameter_10, parameter_9, parameter_2, parameter_19, parameter_17, parameter_16, parameter_13, parameter_18, parameter_12, parameter_6, parameter_1, parameter_3, data_0, data_1, data_2, data_3, data_4):
        return self.builtin_module_1064_0_0(parameter_8, parameter_15, parameter_0, parameter_14, parameter_4, parameter_7, parameter_11, parameter_5, parameter_10, parameter_9, parameter_2, parameter_19, parameter_17, parameter_16, parameter_13, parameter_18, parameter_12, parameter_6, parameter_1, parameter_3, data_0, data_1, data_2, data_3, data_4)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_1064_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_8
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_0
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([720], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([720, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # data_0
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            # data_1
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            # data_2
            paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            # data_3
            paddle.uniform([1, 256, 12, 18], dtype='float32', min=0, max=0.5),
            # data_4
            paddle.uniform([1, 256, 6, 9], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_8
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_0
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[720], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[720, 256, 3, 3], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[36, 256, 3, 3], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # data_0
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            # data_1
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            # data_2
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            # data_3
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            # data_4
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
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