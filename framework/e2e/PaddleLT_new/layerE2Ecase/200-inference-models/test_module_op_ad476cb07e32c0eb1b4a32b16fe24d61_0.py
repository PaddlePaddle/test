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
    return [111][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_147_0_0(self, parameter_0, parameter_1, parameter_2, parameter_3, parameter_4, parameter_5, parameter_6, parameter_7, parameter_8, parameter_9, parameter_10, parameter_11, parameter_12, parameter_13, parameter_14, parameter_15, parameter_16, parameter_17, parameter_18, parameter_19, parameter_20, parameter_21, parameter_22, parameter_23, parameter_24, feed_0):

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-2]

        # pd_op.unsqueeze: (256x100x1x3xf32, None) <- (256x100x3xf32, 1xi64)
        unsqueeze_0, unsqueeze_1 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(parameter_0, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [-2]

        # pd_op.unsqueeze: (-1x400x1x100xf32, None) <- (-1x400x100xf32, 1xi64)
        unsqueeze_2, unsqueeze_3 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(feed_0, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x256x1x100xf32) <- (-1x400x1x100xf32, 256x100x1x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(unsqueeze_2, unsqueeze_0, [1, 1], [0, 1], 'EXPLICIT', [1, 1], 4, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_2 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xi64) <- (256xf32, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_1, full_int_array_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x1x100xf32) <- (-1x256x1x100xf32, 1x256x1x1xf32)
        add__0 = paddle._C_ops.add_(conv2d_0, reshape_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [-2]

        # pd_op.squeeze_: (-1x256x100xf32, None) <- (-1x256x1x100xf32, 1xi64)
        squeeze__0, squeeze__1 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(add__0, full_int_array_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.relu_: (-1x256x100xf32) <- (-1x256x100xf32)
        relu__0 = paddle._C_ops.relu_(squeeze__0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [-2]

        # pd_op.unsqueeze: (256x64x1x3xf32, None) <- (256x64x3xf32, 1xi64)
        unsqueeze_4, unsqueeze_5 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(parameter_2, full_int_array_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [-2]

        # pd_op.unsqueeze_: (-1x256x1x100xf32, None) <- (-1x256x100xf32, 1xi64)
        unsqueeze__0, unsqueeze__1 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(relu__0, full_int_array_5), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x256x1x100xf32) <- (-1x256x1x100xf32, 256x64x1x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(unsqueeze__0, unsqueeze_4, [1, 1], [0, 1], 'EXPLICIT', [1, 1], 4, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_6 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xi64) <- (256xf32, 4xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_3, full_int_array_6), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x1x100xf32) <- (-1x256x1x100xf32, 1x256x1x1xf32)
        add__1 = paddle._C_ops.add_(conv2d_1, reshape_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [-2]

        # pd_op.squeeze_: (-1x256x100xf32, None) <- (-1x256x1x100xf32, 1xi64)
        squeeze__2, squeeze__3 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(add__1, full_int_array_7), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.relu_: (-1x256x100xf32) <- (-1x256x100xf32)
        relu__1 = paddle._C_ops.relu_(squeeze__2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [-2]

        # pd_op.unsqueeze: (256x64x1x3xf32, None) <- (256x64x3xf32, 1xi64)
        unsqueeze_6, unsqueeze_7 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(parameter_4, full_int_array_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [-2]

        # pd_op.unsqueeze: (-1x256x1x100xf32, None) <- (-1x256x100xf32, 1xi64)
        unsqueeze_8, unsqueeze_9 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(relu__1, full_int_array_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x256x1x100xf32) <- (-1x256x1x100xf32, 256x64x1x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(unsqueeze_8, unsqueeze_6, [1, 1], [0, 1], 'EXPLICIT', [1, 1], 4, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_10 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xi64) <- (256xf32, 4xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_5, full_int_array_10), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x1x100xf32) <- (-1x256x1x100xf32, 1x256x1x1xf32)
        add__2 = paddle._C_ops.add_(conv2d_2, reshape_4)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_11 = [-2]

        # pd_op.squeeze_: (-1x256x100xf32, None) <- (-1x256x1x100xf32, 1xi64)
        squeeze__4, squeeze__5 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(add__2, full_int_array_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.relu_: (-1x256x100xf32) <- (-1x256x100xf32)
        relu__2 = paddle._C_ops.relu_(squeeze__4)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_12 = [-2]

        # pd_op.unsqueeze: (1x256x1x1xf32, None) <- (1x256x1xf32, 1xi64)
        unsqueeze_10, unsqueeze_11 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(parameter_6, full_int_array_12), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_13 = [-2]

        # pd_op.unsqueeze_: (-1x256x1x100xf32, None) <- (-1x256x100xf32, 1xi64)
        unsqueeze__2, unsqueeze__3 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(relu__2, full_int_array_13), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x1x1x100xf32) <- (-1x256x1x100xf32, 1x256x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(unsqueeze__2, unsqueeze_10, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_14 = [1, 1, 1, 1]

        # pd_op.reshape: (1x1x1x1xf32, 0x1xi64) <- (1xf32, 4xi64)
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_7, full_int_array_14), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1x1x100xf32) <- (-1x1x1x100xf32, 1x1x1x1xf32)
        add__3 = paddle._C_ops.add_(conv2d_3, reshape_6)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_15 = [-2]

        # pd_op.squeeze_: (-1x1x100xf32, None) <- (-1x1x1x100xf32, 1xi64)
        squeeze__6, squeeze__7 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(add__3, full_int_array_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.sigmoid_: (-1x1x100xf32) <- (-1x1x100xf32)
        sigmoid__0 = paddle._C_ops.sigmoid_(squeeze__6)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_16 = [1]

        # pd_op.squeeze_: (-1x100xf32, None) <- (-1x1x100xf32, 1xi64)
        squeeze__8, squeeze__9 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(sigmoid__0, full_int_array_16), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_17 = [-2]

        # pd_op.unsqueeze: (256x64x1x3xf32, None) <- (256x64x3xf32, 1xi64)
        unsqueeze_12, unsqueeze_13 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(parameter_8, full_int_array_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_18 = [-2]

        # pd_op.unsqueeze: (-1x256x1x100xf32, None) <- (-1x256x100xf32, 1xi64)
        unsqueeze_14, unsqueeze_15 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(relu__1, full_int_array_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x256x1x100xf32) <- (-1x256x1x100xf32, 256x64x1x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(unsqueeze_14, unsqueeze_12, [1, 1], [0, 1], 'EXPLICIT', [1, 1], 4, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_19 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xi64) <- (256xf32, 4xi64)
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_9, full_int_array_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x1x100xf32) <- (-1x256x1x100xf32, 1x256x1x1xf32)
        add__4 = paddle._C_ops.add_(conv2d_4, reshape_8)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_20 = [-2]

        # pd_op.squeeze_: (-1x256x100xf32, None) <- (-1x256x1x100xf32, 1xi64)
        squeeze__10, squeeze__11 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(add__4, full_int_array_20), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.relu_: (-1x256x100xf32) <- (-1x256x100xf32)
        relu__3 = paddle._C_ops.relu_(squeeze__10)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_21 = [-2]

        # pd_op.unsqueeze: (1x256x1x1xf32, None) <- (1x256x1xf32, 1xi64)
        unsqueeze_16, unsqueeze_17 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(parameter_10, full_int_array_21), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_22 = [-2]

        # pd_op.unsqueeze_: (-1x256x1x100xf32, None) <- (-1x256x100xf32, 1xi64)
        unsqueeze__4, unsqueeze__5 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(relu__3, full_int_array_22), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x1x1x100xf32) <- (-1x256x1x100xf32, 1x256x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(unsqueeze__4, unsqueeze_16, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_23 = [1, 1, 1, 1]

        # pd_op.reshape: (1x1x1x1xf32, 0x1xi64) <- (1xf32, 4xi64)
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_11, full_int_array_23), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1x1x100xf32) <- (-1x1x1x100xf32, 1x1x1x1xf32)
        add__5 = paddle._C_ops.add_(conv2d_5, reshape_10)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_24 = [-2]

        # pd_op.squeeze_: (-1x1x100xf32, None) <- (-1x1x1x100xf32, 1xi64)
        squeeze__12, squeeze__13 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(add__5, full_int_array_24), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.sigmoid_: (-1x1x100xf32) <- (-1x1x100xf32)
        sigmoid__1 = paddle._C_ops.sigmoid_(squeeze__12)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_25 = [1]

        # pd_op.squeeze_: (-1x100xf32, None) <- (-1x1x100xf32, 1xi64)
        squeeze__14, squeeze__15 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(sigmoid__1, full_int_array_25), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_26 = [-2]

        # pd_op.unsqueeze: (128x256x1x3xf32, None) <- (128x256x3xf32, 1xi64)
        unsqueeze_18, unsqueeze_19 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(parameter_12, full_int_array_26), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_27 = [-2]

        # pd_op.unsqueeze_: (-1x256x1x100xf32, None) <- (-1x256x100xf32, 1xi64)
        unsqueeze__6, unsqueeze__7 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(relu__1, full_int_array_27), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x128x1x100xf32) <- (-1x256x1x100xf32, 128x256x1x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(unsqueeze__6, unsqueeze_18, [1, 1], [0, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_28 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xi64) <- (128xf32, 4xi64)
        reshape_12, reshape_13 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_13, full_int_array_28), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x1x100xf32) <- (-1x128x1x100xf32, 1x128x1x1xf32)
        add__6 = paddle._C_ops.add_(conv2d_6, reshape_12)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_29 = [-2]

        # pd_op.squeeze_: (-1x128x100xf32, None) <- (-1x128x1x100xf32, 1xi64)
        squeeze__16, squeeze__17 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(add__6, full_int_array_29), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.relu_: (-1x128x100xf32) <- (-1x128x100xf32)
        relu__4 = paddle._C_ops.relu_(squeeze__16)

        # pd_op.matmul: (-1x128x320000xf32) <- (-1x128x100xf32, 100x320000xf32)
        matmul_0 = paddle._C_ops.matmul(relu__4, parameter_14, False, False)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_30 = [0, 0, -1, 100, 100]

        # pd_op.reshape_: (-1x128x-1x100x100xf32, 0x-1x128x320000xf32) <- (-1x128x320000xf32, 5xi64)
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_0, full_int_array_30), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv3d: (-1x512x-1x100x100xf32) <- (-1x128x-1x100x100xf32, 512x128x32x1x1xf32)
        conv3d_0 = paddle._C_ops.conv3d(reshape__0, parameter_15, [32, 1, 1], [0, 0, 0], 'EXPLICIT', 1, [1, 1, 1], 'NCDHW')

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_31 = [1, 512, 1, 1, 1]

        # pd_op.reshape: (1x512x1x1x1xf32, 0x512xf32) <- (512xf32, 5xi64)
        reshape_14, reshape_15 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_16, full_int_array_31), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x512x-1x100x100xf32) <- (-1x512x-1x100x100xf32, 1x512x1x1x1xf32)
        add_0 = paddle._C_ops.add(conv3d_0, reshape_14)

        # pd_op.relu: (-1x512x-1x100x100xf32) <- (-1x512x-1x100x100xf32)
        relu_0 = paddle._C_ops.relu(add_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_32 = [2]

        # pd_op.squeeze_: (-1x512x100x100xf32, None) <- (-1x512x-1x100x100xf32, 1xi64)
        squeeze__18, squeeze__19 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(relu_0, full_int_array_32), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x128x100x100xf32) <- (-1x512x100x100xf32, 128x512x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(squeeze__18, parameter_17, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_33 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_16, reshape_17 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_18, full_int_array_33), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x100x100xf32) <- (-1x128x100x100xf32, 1x128x1x1xf32)
        add__7 = paddle._C_ops.add_(conv2d_7, reshape_16)

        # pd_op.relu_: (-1x128x100x100xf32) <- (-1x128x100x100xf32)
        relu__5 = paddle._C_ops.relu_(add__7)

        # pd_op.conv2d: (-1x128x100x100xf32) <- (-1x128x100x100xf32, 128x128x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(relu__5, parameter_19, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_34 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_18, reshape_19 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_20, full_int_array_34), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x100x100xf32) <- (-1x128x100x100xf32, 1x128x1x1xf32)
        add__8 = paddle._C_ops.add_(conv2d_8, reshape_18)

        # pd_op.relu_: (-1x128x100x100xf32) <- (-1x128x100x100xf32)
        relu__6 = paddle._C_ops.relu_(add__8)

        # pd_op.conv2d: (-1x128x100x100xf32) <- (-1x128x100x100xf32, 128x128x3x3xf32)
        conv2d_9 = paddle._C_ops.conv2d(relu__6, parameter_21, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_35 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_20, reshape_21 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_22, full_int_array_35), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x100x100xf32) <- (-1x128x100x100xf32, 1x128x1x1xf32)
        add__9 = paddle._C_ops.add_(conv2d_9, reshape_20)

        # pd_op.relu_: (-1x128x100x100xf32) <- (-1x128x100x100xf32)
        relu__7 = paddle._C_ops.relu_(add__9)

        # pd_op.conv2d: (-1x2x100x100xf32) <- (-1x128x100x100xf32, 2x128x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(relu__7, parameter_23, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_36 = [1, 2, 1, 1]

        # pd_op.reshape: (1x2x1x1xf32, 0x2xf32) <- (2xf32, 4xi64)
        reshape_22, reshape_23 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_24, full_int_array_36), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x2x100x100xf32) <- (-1x2x100x100xf32, 1x2x1x1xf32)
        add__10 = paddle._C_ops.add_(conv2d_10, reshape_22)

        # pd_op.sigmoid_: (-1x2x100x100xf32) <- (-1x2x100x100xf32)
        sigmoid__2 = paddle._C_ops.sigmoid_(add__10)
        return sigmoid__2, squeeze__8, squeeze__14



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

    def forward(self, parameter_0, parameter_1, parameter_2, parameter_3, parameter_4, parameter_5, parameter_6, parameter_7, parameter_8, parameter_9, parameter_10, parameter_11, parameter_12, parameter_13, parameter_14, parameter_15, parameter_16, parameter_17, parameter_18, parameter_19, parameter_20, parameter_21, parameter_22, parameter_23, parameter_24, feed_0):
        return self.builtin_module_147_0_0(parameter_0, parameter_1, parameter_2, parameter_3, parameter_4, parameter_5, parameter_6, parameter_7, parameter_8, parameter_9, parameter_10, parameter_11, parameter_12, parameter_13, parameter_14, parameter_15, parameter_16, parameter_17, parameter_18, parameter_19, parameter_20, parameter_21, parameter_22, parameter_23, parameter_24, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_147_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([256, 100, 3], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([256, 64, 3], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([256, 64, 3], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([1, 256, 1], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([256, 64, 3], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([1, 256, 1], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([128, 256, 3], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([100, 320000], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([512, 128, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([128, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([2, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([2], dtype='float32', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 400, 100], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_0
            paddle.static.InputSpec(shape=[256, 100, 3], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[256, 64, 3], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[256, 64, 3], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[1, 256, 1], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[256, 64, 3], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[1, 256, 1], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[128, 256, 3], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[100, 320000], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[512, 128, 32, 1, 1], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[2, 128, 1, 1], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[2], dtype='float32'),
            # feed_0
            paddle.static.InputSpec(shape=[None, 400, 100], dtype='float32'),
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