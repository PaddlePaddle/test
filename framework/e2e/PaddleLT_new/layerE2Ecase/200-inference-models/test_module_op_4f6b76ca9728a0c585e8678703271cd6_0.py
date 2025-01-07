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
    return [2, 2, 346, 2, 1597][block_idx] - 1 # number-of-ops-in-block

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
    def pd_op_if_2867_0_0(self, reshape__0):

        # pd_op.flip: (1x1x2x180x320xf32) <- (1x1x2x180x320xf32)
        flip_0 = paddle._C_ops.flip(reshape__0, [1])

        # pd_op.assign_: (1x1x2x180x320xf32) <- (1x1x2x180x320xf32)
        assign__0 = paddle._C_ops.assign_(flip_0)
        return assign__0
    def pd_op_if_2867_1_0(self, ):

        # pd_op.full: (1x1x2x180x320xf32) <- ()
        full_0 = paddle._C_ops.full([1, 1, 2, 180, 320], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x1x2x180x320xf32) <- (1x1x2x180x320xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def pd_op_if_2875_0_0(self, reshape__0, reshape__1, parameter_0, parameter_1, parameter_2, parameter_3, parameter_4, parameter_5, parameter_6, parameter_7, parameter_8, parameter_9, parameter_10, parameter_11, parameter_12, parameter_13, parameter_14, parameter_15, parameter_16, parameter_17, parameter_18, parameter_19, parameter_20, parameter_21, parameter_22, parameter_23, parameter_24, parameter_25, parameter_26, parameter_27, parameter_28, parameter_29, parameter_30, parameter_31, parameter_32, parameter_33, parameter_34, parameter_35, parameter_36, parameter_37, parameter_38, parameter_39, parameter_40, parameter_41, parameter_42, parameter_43, parameter_44, parameter_45, parameter_46, parameter_47, parameter_48, parameter_49, parameter_50, parameter_51, parameter_52, parameter_53, parameter_54, parameter_55, parameter_56, parameter_57, parameter_58, parameter_59, parameter_60, parameter_61, parameter_62, parameter_63, parameter_64, parameter_65, parameter_66, parameter_67, parameter_68, parameter_69, parameter_70, parameter_71, parameter_72, parameter_73):

        # pd_op.bilinear_interp: (1x3x192x320xf32) <- (1x3x180x320xf32, None, None, None)
        bilinear_interp_0 = paddle._C_ops.bilinear_interp(reshape__0, None, None, None, 'NCHW', -1, 192, 320, [], 'bilinear', False, 0)

        # pd_op.bilinear_interp: (1x3x192x320xf32) <- (1x3x180x320xf32, None, None, None)
        bilinear_interp_1 = paddle._C_ops.bilinear_interp(reshape__1, None, None, None, 'NCHW', -1, 192, 320, [], 'bilinear', False, 0)

        # pd_op.subtract_: (1x3x192x320xf32) <- (1x3x192x320xf32, 1x3x1x1xf32)
        subtract__0 = paddle._C_ops.subtract_(bilinear_interp_0, parameter_0)

        # pd_op.divide_: (1x3x192x320xf32) <- (1x3x192x320xf32, 1x3x1x1xf32)
        divide__0 = paddle._C_ops.divide_(subtract__0, parameter_1)

        # pd_op.subtract_: (1x3x192x320xf32) <- (1x3x192x320xf32, 1x3x1x1xf32)
        subtract__1 = paddle._C_ops.subtract_(bilinear_interp_1, parameter_0)

        # pd_op.divide_: (1x3x192x320xf32) <- (1x3x192x320xf32, 1x3x1x1xf32)
        divide__1 = paddle._C_ops.divide_(subtract__1, parameter_1)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [2, 2]

        # pd_op.pool2d: (1x3x96x160xf32) <- (1x3x192x320xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(divide__0, full_int_array_0, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [2, 2]

        # pd_op.pool2d: (1x3x96x160xf32) <- (1x3x192x320xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(divide__1, full_int_array_1, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [2, 2]

        # pd_op.pool2d: (1x3x48x80xf32) <- (1x3x96x160xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(pool2d_0, full_int_array_2, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [2, 2]

        # pd_op.pool2d: (1x3x48x80xf32) <- (1x3x96x160xf32, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(pool2d_1, full_int_array_3, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.assign_value: (1x2x48x80xf32) <- ()
        assign_value_0 = paddle.to_tensor([float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0')], dtype=paddle.float32).reshape([1, 2, 48, 80])

        # pd_op.transpose: (1x48x80x2xf32) <- (1x2x48x80xf32)
        transpose_0 = paddle._C_ops.transpose(assign_value_0, [0, 2, 3, 1])

        # pd_op.full: (1xi64) <- ()
        full_0 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_1 = paddle._C_ops.full([1], float('48'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (48xi64) <- (1xi64, 1xi64, 1xi64)
        arange_0 = paddle.arange(full_0, full_1, full_2, dtype='int64')

        # pd_op.full: (1xi64) <- ()
        full_3 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_4 = paddle._C_ops.full([1], float('80'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_5 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (80xi64) <- (1xi64, 1xi64, 1xi64)
        arange_1 = paddle.arange(full_3, full_4, full_5, dtype='int64')

        # builtin.combine: ([48xi64, 80xi64]) <- (48xi64, 80xi64)
        combine_0 = [arange_0, arange_1]

        # pd_op.meshgrid: ([48x80xi64, 48x80xi64]) <- ([48xi64, 80xi64])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_0)

        # builtin.slice: (48x80xi64) <- ([48x80xi64, 48x80xi64])
        slice_0 = meshgrid_0[1]

        # builtin.slice: (48x80xi64) <- ([48x80xi64, 48x80xi64])
        slice_1 = meshgrid_0[0]

        # builtin.combine: ([48x80xi64, 48x80xi64]) <- (48x80xi64, 48x80xi64)
        combine_1 = [slice_0, slice_1]

        # pd_op.stack: (48x80x2xi64) <- ([48x80xi64, 48x80xi64])
        stack_0 = paddle._C_ops.stack(combine_1, 2)

        # pd_op.cast: (48x80x2xf32) <- (48x80x2xi64)
        cast_0 = paddle._C_ops.cast(stack_0, paddle.float32)

        # pd_op.add_: (1x48x80x2xf32) <- (48x80x2xf32, 1x48x80x2xf32)
        add__0 = paddle._C_ops.add_(cast_0, transpose_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [1]

        # pd_op.slice: (1x48x80xf32) <- (1x48x80x2xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(add__0, [3], full_int_array_4, full_int_array_5, [1], [3])

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full([1], float('2'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x48x80xf32) <- (1x48x80xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(slice_2, full_6, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('0.0126582'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x48x80xf32) <- (1x48x80xf32, 1xf32)
        scale__1 = paddle._C_ops.scale_(scale__0, full_7, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x48x80xf32) <- (1x48x80xf32, 1xf32)
        scale__2 = paddle._C_ops.scale_(scale__1, full_8, float('-1'), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [2]

        # pd_op.slice: (1x48x80xf32) <- (1x48x80x2xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(add__0, [3], full_int_array_6, full_int_array_7, [1], [3])

        # pd_op.full: (1xf32) <- ()
        full_9 = paddle._C_ops.full([1], float('2'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x48x80xf32) <- (1x48x80xf32, 1xf32)
        scale__3 = paddle._C_ops.scale_(slice_3, full_9, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_10 = paddle._C_ops.full([1], float('0.0212766'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x48x80xf32) <- (1x48x80xf32, 1xf32)
        scale__4 = paddle._C_ops.scale_(scale__3, full_10, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_11 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x48x80xf32) <- (1x48x80xf32, 1xf32)
        scale__5 = paddle._C_ops.scale_(scale__4, full_11, float('-1'), True)

        # builtin.combine: ([1x48x80xf32, 1x48x80xf32]) <- (1x48x80xf32, 1x48x80xf32)
        combine_2 = [scale__2, scale__5]

        # pd_op.stack: (1x48x80x2xf32) <- ([1x48x80xf32, 1x48x80xf32])
        stack_1 = paddle._C_ops.stack(combine_2, 3)

        # pd_op.grid_sample: (1x3x48x80xf32) <- (1x3x48x80xf32, 1x48x80x2xf32)
        grid_sample_0 = paddle._C_ops.grid_sample(pool2d_3, stack_1, 'bilinear', 'border', True)

        # builtin.combine: ([1x3x48x80xf32, 1x3x48x80xf32, 1x2x48x80xf32]) <- (1x3x48x80xf32, 1x3x48x80xf32, 1x2x48x80xf32)
        combine_3 = [pool2d_2, grid_sample_0, assign_value_0]

        # pd_op.full: (1xi32) <- ()
        full_12 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x8x48x80xf32) <- ([1x3x48x80xf32, 1x3x48x80xf32, 1x2x48x80xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_3, full_12)

        # pd_op.conv2d: (1x16x48x80xf32) <- (1x8x48x80xf32, 16x8x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(concat_0, parameter_2, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_8 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_3, full_int_array_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x48x80xf32) <- (1x16x48x80xf32, 1x16x1x1xf32)
        add__1 = paddle._C_ops.add_(conv2d_0, reshape_0)

        # pd_op.leaky_relu_: (1x16x48x80xf32) <- (1x16x48x80xf32)
        leaky_relu__0 = paddle._C_ops.leaky_relu_(add__1, float('0.1'))

        # pd_op.conv2d: (1x16x48x80xf32) <- (1x16x48x80xf32, 16x16x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(leaky_relu__0, parameter_4, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_9 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_5, full_int_array_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x48x80xf32) <- (1x16x48x80xf32, 1x16x1x1xf32)
        add__2 = paddle._C_ops.add_(conv2d_1, reshape_2)

        # pd_op.leaky_relu_: (1x16x48x80xf32) <- (1x16x48x80xf32)
        leaky_relu__1 = paddle._C_ops.leaky_relu_(add__2, float('0.1'))

        # pd_op.conv2d: (1x32x48x80xf32) <- (1x16x48x80xf32, 32x16x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(leaky_relu__1, parameter_6, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_10 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_7, full_int_array_10), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x48x80xf32) <- (1x32x48x80xf32, 1x32x1x1xf32)
        add__3 = paddle._C_ops.add_(conv2d_2, reshape_4)

        # pd_op.leaky_relu_: (1x32x48x80xf32) <- (1x32x48x80xf32)
        leaky_relu__2 = paddle._C_ops.leaky_relu_(add__3, float('0.1'))

        # pd_op.conv2d: (1x32x48x80xf32) <- (1x32x48x80xf32, 32x32x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(leaky_relu__2, parameter_8, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_11 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_9, full_int_array_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x48x80xf32) <- (1x32x48x80xf32, 1x32x1x1xf32)
        add__4 = paddle._C_ops.add_(conv2d_3, reshape_6)

        # pd_op.leaky_relu_: (1x32x48x80xf32) <- (1x32x48x80xf32)
        leaky_relu__3 = paddle._C_ops.leaky_relu_(add__4, float('0.1'))

        # pd_op.conv2d: (1x32x48x80xf32) <- (1x32x48x80xf32, 32x32x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(leaky_relu__3, parameter_10, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_12 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_11, full_int_array_12), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x48x80xf32) <- (1x32x48x80xf32, 1x32x1x1xf32)
        add__5 = paddle._C_ops.add_(conv2d_4, reshape_8)

        # pd_op.leaky_relu_: (1x32x48x80xf32) <- (1x32x48x80xf32)
        leaky_relu__4 = paddle._C_ops.leaky_relu_(add__5, float('0.1'))

        # pd_op.conv2d: (1x32x48x80xf32) <- (1x32x48x80xf32, 32x32x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(leaky_relu__4, parameter_12, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_13 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_13, full_int_array_13), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x48x80xf32) <- (1x32x48x80xf32, 1x32x1x1xf32)
        add__6 = paddle._C_ops.add_(conv2d_5, reshape_10)

        # pd_op.leaky_relu_: (1x32x48x80xf32) <- (1x32x48x80xf32)
        leaky_relu__5 = paddle._C_ops.leaky_relu_(add__6, float('0.1'))

        # pd_op.conv2d: (1x16x48x80xf32) <- (1x32x48x80xf32, 16x32x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(leaky_relu__5, parameter_14, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_14 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_12, reshape_13 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_15, full_int_array_14), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x48x80xf32) <- (1x16x48x80xf32, 1x16x1x1xf32)
        add__7 = paddle._C_ops.add_(conv2d_6, reshape_12)

        # pd_op.leaky_relu_: (1x16x48x80xf32) <- (1x16x48x80xf32)
        leaky_relu__6 = paddle._C_ops.leaky_relu_(add__7, float('0.1'))

        # pd_op.conv2d: (1x16x48x80xf32) <- (1x16x48x80xf32, 16x16x3x3xf32)
        conv2d_7 = paddle._C_ops.conv2d(leaky_relu__6, parameter_16, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_15 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_14, reshape_15 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_17, full_int_array_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x48x80xf32) <- (1x16x48x80xf32, 1x16x1x1xf32)
        add__8 = paddle._C_ops.add_(conv2d_7, reshape_14)

        # pd_op.leaky_relu_: (1x16x48x80xf32) <- (1x16x48x80xf32)
        leaky_relu__7 = paddle._C_ops.leaky_relu_(add__8, float('0.1'))

        # pd_op.conv2d: (1x16x48x80xf32) <- (1x16x48x80xf32, 16x16x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(leaky_relu__7, parameter_18, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_16 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_16, reshape_17 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_19, full_int_array_16), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x48x80xf32) <- (1x16x48x80xf32, 1x16x1x1xf32)
        add__9 = paddle._C_ops.add_(conv2d_8, reshape_16)

        # pd_op.leaky_relu_: (1x16x48x80xf32) <- (1x16x48x80xf32)
        leaky_relu__8 = paddle._C_ops.leaky_relu_(add__9, float('0.1'))

        # pd_op.conv2d: (1x8x48x80xf32) <- (1x16x48x80xf32, 8x16x3x3xf32)
        conv2d_9 = paddle._C_ops.conv2d(leaky_relu__8, parameter_20, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_17 = [1, 8, 1, 1]

        # pd_op.reshape: (1x8x1x1xf32, 0x8xf32) <- (8xf32, 4xi64)
        reshape_18, reshape_19 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_21, full_int_array_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x8x48x80xf32) <- (1x8x48x80xf32, 1x8x1x1xf32)
        add__10 = paddle._C_ops.add_(conv2d_9, reshape_18)

        # pd_op.leaky_relu_: (1x8x48x80xf32) <- (1x8x48x80xf32)
        leaky_relu__9 = paddle._C_ops.leaky_relu_(add__10, float('0.1'))

        # pd_op.conv2d: (1x8x48x80xf32) <- (1x8x48x80xf32, 8x8x3x3xf32)
        conv2d_10 = paddle._C_ops.conv2d(leaky_relu__9, parameter_22, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_18 = [1, 8, 1, 1]

        # pd_op.reshape: (1x8x1x1xf32, 0x8xf32) <- (8xf32, 4xi64)
        reshape_20, reshape_21 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_23, full_int_array_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x8x48x80xf32) <- (1x8x48x80xf32, 1x8x1x1xf32)
        add__11 = paddle._C_ops.add_(conv2d_10, reshape_20)

        # pd_op.leaky_relu_: (1x8x48x80xf32) <- (1x8x48x80xf32)
        leaky_relu__10 = paddle._C_ops.leaky_relu_(add__11, float('0.1'))

        # pd_op.conv2d: (1x2x48x80xf32) <- (1x8x48x80xf32, 2x8x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(leaky_relu__10, parameter_24, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_19 = [1, 2, 1, 1]

        # pd_op.reshape: (1x2x1x1xf32, 0x2xf32) <- (2xf32, 4xi64)
        reshape_22, reshape_23 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_25, full_int_array_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x2x48x80xf32) <- (1x2x48x80xf32, 1x2x1x1xf32)
        add__12 = paddle._C_ops.add_(conv2d_11, reshape_22)

        # pd_op.add_: (1x2x48x80xf32) <- (1x2x48x80xf32, 1x2x48x80xf32)
        add__13 = paddle._C_ops.add_(assign_value_0, add__12)

        # pd_op.bilinear_interp: (1x2x96x160xf32) <- (1x2x48x80xf32, None, None, None)
        bilinear_interp_2 = paddle._C_ops.bilinear_interp(add__13, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'bilinear', True, 0)

        # pd_op.full: (1xf32) <- ()
        full_13 = paddle._C_ops.full([1], float('2'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x2x96x160xf32) <- (1x2x96x160xf32, 1xf32)
        scale__6 = paddle._C_ops.scale_(bilinear_interp_2, full_13, float('0'), True)

        # pd_op.transpose: (1x96x160x2xf32) <- (1x2x96x160xf32)
        transpose_1 = paddle._C_ops.transpose(scale__6, [0, 2, 3, 1])

        # pd_op.full: (1xi64) <- ()
        full_14 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_15 = paddle._C_ops.full([1], float('96'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_16 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (96xi64) <- (1xi64, 1xi64, 1xi64)
        arange_2 = paddle.arange(full_14, full_15, full_16, dtype='int64')

        # pd_op.full: (1xi64) <- ()
        full_17 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_18 = paddle._C_ops.full([1], float('160'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_19 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (160xi64) <- (1xi64, 1xi64, 1xi64)
        arange_3 = paddle.arange(full_17, full_18, full_19, dtype='int64')

        # builtin.combine: ([96xi64, 160xi64]) <- (96xi64, 160xi64)
        combine_4 = [arange_2, arange_3]

        # pd_op.meshgrid: ([96x160xi64, 96x160xi64]) <- ([96xi64, 160xi64])
        meshgrid_1 = paddle._C_ops.meshgrid(combine_4)

        # builtin.slice: (96x160xi64) <- ([96x160xi64, 96x160xi64])
        slice_4 = meshgrid_1[1]

        # builtin.slice: (96x160xi64) <- ([96x160xi64, 96x160xi64])
        slice_5 = meshgrid_1[0]

        # builtin.combine: ([96x160xi64, 96x160xi64]) <- (96x160xi64, 96x160xi64)
        combine_5 = [slice_4, slice_5]

        # pd_op.stack: (96x160x2xi64) <- ([96x160xi64, 96x160xi64])
        stack_2 = paddle._C_ops.stack(combine_5, 2)

        # pd_op.cast: (96x160x2xf32) <- (96x160x2xi64)
        cast_1 = paddle._C_ops.cast(stack_2, paddle.float32)

        # pd_op.add_: (1x96x160x2xf32) <- (96x160x2xf32, 1x96x160x2xf32)
        add__14 = paddle._C_ops.add_(cast_1, transpose_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_20 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_21 = [1]

        # pd_op.slice: (1x96x160xf32) <- (1x96x160x2xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(add__14, [3], full_int_array_20, full_int_array_21, [1], [3])

        # pd_op.full: (1xf32) <- ()
        full_20 = paddle._C_ops.full([1], float('2'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x96x160xf32) <- (1x96x160xf32, 1xf32)
        scale__7 = paddle._C_ops.scale_(slice_6, full_20, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_21 = paddle._C_ops.full([1], float('0.00628931'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x96x160xf32) <- (1x96x160xf32, 1xf32)
        scale__8 = paddle._C_ops.scale_(scale__7, full_21, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_22 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x96x160xf32) <- (1x96x160xf32, 1xf32)
        scale__9 = paddle._C_ops.scale_(scale__8, full_22, float('-1'), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_22 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_23 = [2]

        # pd_op.slice: (1x96x160xf32) <- (1x96x160x2xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(add__14, [3], full_int_array_22, full_int_array_23, [1], [3])

        # pd_op.full: (1xf32) <- ()
        full_23 = paddle._C_ops.full([1], float('2'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x96x160xf32) <- (1x96x160xf32, 1xf32)
        scale__10 = paddle._C_ops.scale_(slice_7, full_23, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_24 = paddle._C_ops.full([1], float('0.0105263'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x96x160xf32) <- (1x96x160xf32, 1xf32)
        scale__11 = paddle._C_ops.scale_(scale__10, full_24, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_25 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x96x160xf32) <- (1x96x160xf32, 1xf32)
        scale__12 = paddle._C_ops.scale_(scale__11, full_25, float('-1'), True)

        # builtin.combine: ([1x96x160xf32, 1x96x160xf32]) <- (1x96x160xf32, 1x96x160xf32)
        combine_6 = [scale__9, scale__12]

        # pd_op.stack: (1x96x160x2xf32) <- ([1x96x160xf32, 1x96x160xf32])
        stack_3 = paddle._C_ops.stack(combine_6, 3)

        # pd_op.grid_sample: (1x3x96x160xf32) <- (1x3x96x160xf32, 1x96x160x2xf32)
        grid_sample_1 = paddle._C_ops.grid_sample(pool2d_1, stack_3, 'bilinear', 'border', True)

        # builtin.combine: ([1x3x96x160xf32, 1x3x96x160xf32, 1x2x96x160xf32]) <- (1x3x96x160xf32, 1x3x96x160xf32, 1x2x96x160xf32)
        combine_7 = [pool2d_0, grid_sample_1, scale__6]

        # pd_op.full: (1xi32) <- ()
        full_26 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x8x96x160xf32) <- ([1x3x96x160xf32, 1x3x96x160xf32, 1x2x96x160xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_7, full_26)

        # pd_op.conv2d: (1x16x96x160xf32) <- (1x8x96x160xf32, 16x8x3x3xf32)
        conv2d_12 = paddle._C_ops.conv2d(concat_1, parameter_26, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_24 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_24, reshape_25 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_27, full_int_array_24), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x96x160xf32) <- (1x16x96x160xf32, 1x16x1x1xf32)
        add__15 = paddle._C_ops.add_(conv2d_12, reshape_24)

        # pd_op.leaky_relu_: (1x16x96x160xf32) <- (1x16x96x160xf32)
        leaky_relu__11 = paddle._C_ops.leaky_relu_(add__15, float('0.1'))

        # pd_op.conv2d: (1x16x96x160xf32) <- (1x16x96x160xf32, 16x16x3x3xf32)
        conv2d_13 = paddle._C_ops.conv2d(leaky_relu__11, parameter_28, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_25 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_26, reshape_27 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_29, full_int_array_25), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x96x160xf32) <- (1x16x96x160xf32, 1x16x1x1xf32)
        add__16 = paddle._C_ops.add_(conv2d_13, reshape_26)

        # pd_op.leaky_relu_: (1x16x96x160xf32) <- (1x16x96x160xf32)
        leaky_relu__12 = paddle._C_ops.leaky_relu_(add__16, float('0.1'))

        # pd_op.conv2d: (1x32x96x160xf32) <- (1x16x96x160xf32, 32x16x3x3xf32)
        conv2d_14 = paddle._C_ops.conv2d(leaky_relu__12, parameter_30, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_26 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_28, reshape_29 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_31, full_int_array_26), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x96x160xf32) <- (1x32x96x160xf32, 1x32x1x1xf32)
        add__17 = paddle._C_ops.add_(conv2d_14, reshape_28)

        # pd_op.leaky_relu_: (1x32x96x160xf32) <- (1x32x96x160xf32)
        leaky_relu__13 = paddle._C_ops.leaky_relu_(add__17, float('0.1'))

        # pd_op.conv2d: (1x32x96x160xf32) <- (1x32x96x160xf32, 32x32x3x3xf32)
        conv2d_15 = paddle._C_ops.conv2d(leaky_relu__13, parameter_32, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_27 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_30, reshape_31 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_33, full_int_array_27), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x96x160xf32) <- (1x32x96x160xf32, 1x32x1x1xf32)
        add__18 = paddle._C_ops.add_(conv2d_15, reshape_30)

        # pd_op.leaky_relu_: (1x32x96x160xf32) <- (1x32x96x160xf32)
        leaky_relu__14 = paddle._C_ops.leaky_relu_(add__18, float('0.1'))

        # pd_op.conv2d: (1x32x96x160xf32) <- (1x32x96x160xf32, 32x32x3x3xf32)
        conv2d_16 = paddle._C_ops.conv2d(leaky_relu__14, parameter_34, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_28 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_32, reshape_33 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_35, full_int_array_28), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x96x160xf32) <- (1x32x96x160xf32, 1x32x1x1xf32)
        add__19 = paddle._C_ops.add_(conv2d_16, reshape_32)

        # pd_op.leaky_relu_: (1x32x96x160xf32) <- (1x32x96x160xf32)
        leaky_relu__15 = paddle._C_ops.leaky_relu_(add__19, float('0.1'))

        # pd_op.conv2d: (1x32x96x160xf32) <- (1x32x96x160xf32, 32x32x3x3xf32)
        conv2d_17 = paddle._C_ops.conv2d(leaky_relu__15, parameter_36, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_29 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_34, reshape_35 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_37, full_int_array_29), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x96x160xf32) <- (1x32x96x160xf32, 1x32x1x1xf32)
        add__20 = paddle._C_ops.add_(conv2d_17, reshape_34)

        # pd_op.leaky_relu_: (1x32x96x160xf32) <- (1x32x96x160xf32)
        leaky_relu__16 = paddle._C_ops.leaky_relu_(add__20, float('0.1'))

        # pd_op.conv2d: (1x16x96x160xf32) <- (1x32x96x160xf32, 16x32x3x3xf32)
        conv2d_18 = paddle._C_ops.conv2d(leaky_relu__16, parameter_38, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_30 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_36, reshape_37 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_39, full_int_array_30), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x96x160xf32) <- (1x16x96x160xf32, 1x16x1x1xf32)
        add__21 = paddle._C_ops.add_(conv2d_18, reshape_36)

        # pd_op.leaky_relu_: (1x16x96x160xf32) <- (1x16x96x160xf32)
        leaky_relu__17 = paddle._C_ops.leaky_relu_(add__21, float('0.1'))

        # pd_op.conv2d: (1x16x96x160xf32) <- (1x16x96x160xf32, 16x16x3x3xf32)
        conv2d_19 = paddle._C_ops.conv2d(leaky_relu__17, parameter_40, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_31 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_38, reshape_39 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_41, full_int_array_31), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x96x160xf32) <- (1x16x96x160xf32, 1x16x1x1xf32)
        add__22 = paddle._C_ops.add_(conv2d_19, reshape_38)

        # pd_op.leaky_relu_: (1x16x96x160xf32) <- (1x16x96x160xf32)
        leaky_relu__18 = paddle._C_ops.leaky_relu_(add__22, float('0.1'))

        # pd_op.conv2d: (1x16x96x160xf32) <- (1x16x96x160xf32, 16x16x3x3xf32)
        conv2d_20 = paddle._C_ops.conv2d(leaky_relu__18, parameter_42, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_32 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_40, reshape_41 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_43, full_int_array_32), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x96x160xf32) <- (1x16x96x160xf32, 1x16x1x1xf32)
        add__23 = paddle._C_ops.add_(conv2d_20, reshape_40)

        # pd_op.leaky_relu_: (1x16x96x160xf32) <- (1x16x96x160xf32)
        leaky_relu__19 = paddle._C_ops.leaky_relu_(add__23, float('0.1'))

        # pd_op.conv2d: (1x8x96x160xf32) <- (1x16x96x160xf32, 8x16x3x3xf32)
        conv2d_21 = paddle._C_ops.conv2d(leaky_relu__19, parameter_44, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_33 = [1, 8, 1, 1]

        # pd_op.reshape: (1x8x1x1xf32, 0x8xf32) <- (8xf32, 4xi64)
        reshape_42, reshape_43 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_45, full_int_array_33), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x8x96x160xf32) <- (1x8x96x160xf32, 1x8x1x1xf32)
        add__24 = paddle._C_ops.add_(conv2d_21, reshape_42)

        # pd_op.leaky_relu_: (1x8x96x160xf32) <- (1x8x96x160xf32)
        leaky_relu__20 = paddle._C_ops.leaky_relu_(add__24, float('0.1'))

        # pd_op.conv2d: (1x8x96x160xf32) <- (1x8x96x160xf32, 8x8x3x3xf32)
        conv2d_22 = paddle._C_ops.conv2d(leaky_relu__20, parameter_46, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_34 = [1, 8, 1, 1]

        # pd_op.reshape: (1x8x1x1xf32, 0x8xf32) <- (8xf32, 4xi64)
        reshape_44, reshape_45 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_47, full_int_array_34), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x8x96x160xf32) <- (1x8x96x160xf32, 1x8x1x1xf32)
        add__25 = paddle._C_ops.add_(conv2d_22, reshape_44)

        # pd_op.leaky_relu_: (1x8x96x160xf32) <- (1x8x96x160xf32)
        leaky_relu__21 = paddle._C_ops.leaky_relu_(add__25, float('0.1'))

        # pd_op.conv2d: (1x2x96x160xf32) <- (1x8x96x160xf32, 2x8x3x3xf32)
        conv2d_23 = paddle._C_ops.conv2d(leaky_relu__21, parameter_48, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_35 = [1, 2, 1, 1]

        # pd_op.reshape: (1x2x1x1xf32, 0x2xf32) <- (2xf32, 4xi64)
        reshape_46, reshape_47 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_49, full_int_array_35), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x2x96x160xf32) <- (1x2x96x160xf32, 1x2x1x1xf32)
        add__26 = paddle._C_ops.add_(conv2d_23, reshape_46)

        # pd_op.add_: (1x2x96x160xf32) <- (1x2x96x160xf32, 1x2x96x160xf32)
        add__27 = paddle._C_ops.add_(scale__6, add__26)

        # pd_op.bilinear_interp: (1x2x192x320xf32) <- (1x2x96x160xf32, None, None, None)
        bilinear_interp_3 = paddle._C_ops.bilinear_interp(add__27, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'bilinear', True, 0)

        # pd_op.full: (1xf32) <- ()
        full_27 = paddle._C_ops.full([1], float('2'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x2x192x320xf32) <- (1x2x192x320xf32, 1xf32)
        scale__13 = paddle._C_ops.scale_(bilinear_interp_3, full_27, float('0'), True)

        # pd_op.transpose: (1x192x320x2xf32) <- (1x2x192x320xf32)
        transpose_2 = paddle._C_ops.transpose(scale__13, [0, 2, 3, 1])

        # pd_op.full: (1xi64) <- ()
        full_28 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_29 = paddle._C_ops.full([1], float('192'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_30 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (192xi64) <- (1xi64, 1xi64, 1xi64)
        arange_4 = paddle.arange(full_28, full_29, full_30, dtype='int64')

        # pd_op.full: (1xi64) <- ()
        full_31 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_32 = paddle._C_ops.full([1], float('320'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_33 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (320xi64) <- (1xi64, 1xi64, 1xi64)
        arange_5 = paddle.arange(full_31, full_32, full_33, dtype='int64')

        # builtin.combine: ([192xi64, 320xi64]) <- (192xi64, 320xi64)
        combine_8 = [arange_4, arange_5]

        # pd_op.meshgrid: ([192x320xi64, 192x320xi64]) <- ([192xi64, 320xi64])
        meshgrid_2 = paddle._C_ops.meshgrid(combine_8)

        # builtin.slice: (192x320xi64) <- ([192x320xi64, 192x320xi64])
        slice_8 = meshgrid_2[1]

        # builtin.slice: (192x320xi64) <- ([192x320xi64, 192x320xi64])
        slice_9 = meshgrid_2[0]

        # builtin.combine: ([192x320xi64, 192x320xi64]) <- (192x320xi64, 192x320xi64)
        combine_9 = [slice_8, slice_9]

        # pd_op.stack: (192x320x2xi64) <- ([192x320xi64, 192x320xi64])
        stack_4 = paddle._C_ops.stack(combine_9, 2)

        # pd_op.cast: (192x320x2xf32) <- (192x320x2xi64)
        cast_2 = paddle._C_ops.cast(stack_4, paddle.float32)

        # pd_op.add_: (1x192x320x2xf32) <- (192x320x2xf32, 1x192x320x2xf32)
        add__28 = paddle._C_ops.add_(cast_2, transpose_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_36 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_37 = [1]

        # pd_op.slice: (1x192x320xf32) <- (1x192x320x2xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(add__28, [3], full_int_array_36, full_int_array_37, [1], [3])

        # pd_op.full: (1xf32) <- ()
        full_34 = paddle._C_ops.full([1], float('2'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x192x320xf32) <- (1x192x320xf32, 1xf32)
        scale__14 = paddle._C_ops.scale_(slice_10, full_34, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_35 = paddle._C_ops.full([1], float('0.0031348'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x192x320xf32) <- (1x192x320xf32, 1xf32)
        scale__15 = paddle._C_ops.scale_(scale__14, full_35, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_36 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x192x320xf32) <- (1x192x320xf32, 1xf32)
        scale__16 = paddle._C_ops.scale_(scale__15, full_36, float('-1'), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_38 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_39 = [2]

        # pd_op.slice: (1x192x320xf32) <- (1x192x320x2xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(add__28, [3], full_int_array_38, full_int_array_39, [1], [3])

        # pd_op.full: (1xf32) <- ()
        full_37 = paddle._C_ops.full([1], float('2'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x192x320xf32) <- (1x192x320xf32, 1xf32)
        scale__17 = paddle._C_ops.scale_(slice_11, full_37, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_38 = paddle._C_ops.full([1], float('0.0052356'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x192x320xf32) <- (1x192x320xf32, 1xf32)
        scale__18 = paddle._C_ops.scale_(scale__17, full_38, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_39 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x192x320xf32) <- (1x192x320xf32, 1xf32)
        scale__19 = paddle._C_ops.scale_(scale__18, full_39, float('-1'), True)

        # builtin.combine: ([1x192x320xf32, 1x192x320xf32]) <- (1x192x320xf32, 1x192x320xf32)
        combine_10 = [scale__16, scale__19]

        # pd_op.stack: (1x192x320x2xf32) <- ([1x192x320xf32, 1x192x320xf32])
        stack_5 = paddle._C_ops.stack(combine_10, 3)

        # pd_op.grid_sample: (1x3x192x320xf32) <- (1x3x192x320xf32, 1x192x320x2xf32)
        grid_sample_2 = paddle._C_ops.grid_sample(divide__1, stack_5, 'bilinear', 'border', True)

        # builtin.combine: ([1x3x192x320xf32, 1x3x192x320xf32, 1x2x192x320xf32]) <- (1x3x192x320xf32, 1x3x192x320xf32, 1x2x192x320xf32)
        combine_11 = [divide__0, grid_sample_2, scale__13]

        # pd_op.full: (1xi32) <- ()
        full_40 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x8x192x320xf32) <- ([1x3x192x320xf32, 1x3x192x320xf32, 1x2x192x320xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_11, full_40)

        # pd_op.conv2d: (1x16x192x320xf32) <- (1x8x192x320xf32, 16x8x3x3xf32)
        conv2d_24 = paddle._C_ops.conv2d(concat_2, parameter_50, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_40 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_48, reshape_49 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_51, full_int_array_40), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x192x320xf32) <- (1x16x192x320xf32, 1x16x1x1xf32)
        add__29 = paddle._C_ops.add_(conv2d_24, reshape_48)

        # pd_op.leaky_relu_: (1x16x192x320xf32) <- (1x16x192x320xf32)
        leaky_relu__22 = paddle._C_ops.leaky_relu_(add__29, float('0.1'))

        # pd_op.conv2d: (1x16x192x320xf32) <- (1x16x192x320xf32, 16x16x3x3xf32)
        conv2d_25 = paddle._C_ops.conv2d(leaky_relu__22, parameter_52, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_41 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_50, reshape_51 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_53, full_int_array_41), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x192x320xf32) <- (1x16x192x320xf32, 1x16x1x1xf32)
        add__30 = paddle._C_ops.add_(conv2d_25, reshape_50)

        # pd_op.leaky_relu_: (1x16x192x320xf32) <- (1x16x192x320xf32)
        leaky_relu__23 = paddle._C_ops.leaky_relu_(add__30, float('0.1'))

        # pd_op.conv2d: (1x32x192x320xf32) <- (1x16x192x320xf32, 32x16x3x3xf32)
        conv2d_26 = paddle._C_ops.conv2d(leaky_relu__23, parameter_54, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_42 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_52, reshape_53 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_55, full_int_array_42), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x192x320xf32) <- (1x32x192x320xf32, 1x32x1x1xf32)
        add__31 = paddle._C_ops.add_(conv2d_26, reshape_52)

        # pd_op.leaky_relu_: (1x32x192x320xf32) <- (1x32x192x320xf32)
        leaky_relu__24 = paddle._C_ops.leaky_relu_(add__31, float('0.1'))

        # pd_op.conv2d: (1x32x192x320xf32) <- (1x32x192x320xf32, 32x32x3x3xf32)
        conv2d_27 = paddle._C_ops.conv2d(leaky_relu__24, parameter_56, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_43 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_54, reshape_55 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_57, full_int_array_43), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x192x320xf32) <- (1x32x192x320xf32, 1x32x1x1xf32)
        add__32 = paddle._C_ops.add_(conv2d_27, reshape_54)

        # pd_op.leaky_relu_: (1x32x192x320xf32) <- (1x32x192x320xf32)
        leaky_relu__25 = paddle._C_ops.leaky_relu_(add__32, float('0.1'))

        # pd_op.conv2d: (1x32x192x320xf32) <- (1x32x192x320xf32, 32x32x3x3xf32)
        conv2d_28 = paddle._C_ops.conv2d(leaky_relu__25, parameter_58, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_44 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_56, reshape_57 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_59, full_int_array_44), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x192x320xf32) <- (1x32x192x320xf32, 1x32x1x1xf32)
        add__33 = paddle._C_ops.add_(conv2d_28, reshape_56)

        # pd_op.leaky_relu_: (1x32x192x320xf32) <- (1x32x192x320xf32)
        leaky_relu__26 = paddle._C_ops.leaky_relu_(add__33, float('0.1'))

        # pd_op.conv2d: (1x32x192x320xf32) <- (1x32x192x320xf32, 32x32x3x3xf32)
        conv2d_29 = paddle._C_ops.conv2d(leaky_relu__26, parameter_60, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_45 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_58, reshape_59 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_61, full_int_array_45), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x192x320xf32) <- (1x32x192x320xf32, 1x32x1x1xf32)
        add__34 = paddle._C_ops.add_(conv2d_29, reshape_58)

        # pd_op.leaky_relu_: (1x32x192x320xf32) <- (1x32x192x320xf32)
        leaky_relu__27 = paddle._C_ops.leaky_relu_(add__34, float('0.1'))

        # pd_op.conv2d: (1x16x192x320xf32) <- (1x32x192x320xf32, 16x32x3x3xf32)
        conv2d_30 = paddle._C_ops.conv2d(leaky_relu__27, parameter_62, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_46 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_60, reshape_61 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_63, full_int_array_46), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x192x320xf32) <- (1x16x192x320xf32, 1x16x1x1xf32)
        add__35 = paddle._C_ops.add_(conv2d_30, reshape_60)

        # pd_op.leaky_relu_: (1x16x192x320xf32) <- (1x16x192x320xf32)
        leaky_relu__28 = paddle._C_ops.leaky_relu_(add__35, float('0.1'))

        # pd_op.conv2d: (1x16x192x320xf32) <- (1x16x192x320xf32, 16x16x3x3xf32)
        conv2d_31 = paddle._C_ops.conv2d(leaky_relu__28, parameter_64, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_47 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_62, reshape_63 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_65, full_int_array_47), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x192x320xf32) <- (1x16x192x320xf32, 1x16x1x1xf32)
        add__36 = paddle._C_ops.add_(conv2d_31, reshape_62)

        # pd_op.leaky_relu_: (1x16x192x320xf32) <- (1x16x192x320xf32)
        leaky_relu__29 = paddle._C_ops.leaky_relu_(add__36, float('0.1'))

        # pd_op.conv2d: (1x16x192x320xf32) <- (1x16x192x320xf32, 16x16x3x3xf32)
        conv2d_32 = paddle._C_ops.conv2d(leaky_relu__29, parameter_66, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_48 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_64, reshape_65 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_67, full_int_array_48), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x192x320xf32) <- (1x16x192x320xf32, 1x16x1x1xf32)
        add__37 = paddle._C_ops.add_(conv2d_32, reshape_64)

        # pd_op.leaky_relu_: (1x16x192x320xf32) <- (1x16x192x320xf32)
        leaky_relu__30 = paddle._C_ops.leaky_relu_(add__37, float('0.1'))

        # pd_op.conv2d: (1x8x192x320xf32) <- (1x16x192x320xf32, 8x16x3x3xf32)
        conv2d_33 = paddle._C_ops.conv2d(leaky_relu__30, parameter_68, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_49 = [1, 8, 1, 1]

        # pd_op.reshape: (1x8x1x1xf32, 0x8xf32) <- (8xf32, 4xi64)
        reshape_66, reshape_67 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_69, full_int_array_49), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x8x192x320xf32) <- (1x8x192x320xf32, 1x8x1x1xf32)
        add__38 = paddle._C_ops.add_(conv2d_33, reshape_66)

        # pd_op.leaky_relu_: (1x8x192x320xf32) <- (1x8x192x320xf32)
        leaky_relu__31 = paddle._C_ops.leaky_relu_(add__38, float('0.1'))

        # pd_op.conv2d: (1x8x192x320xf32) <- (1x8x192x320xf32, 8x8x3x3xf32)
        conv2d_34 = paddle._C_ops.conv2d(leaky_relu__31, parameter_70, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_50 = [1, 8, 1, 1]

        # pd_op.reshape: (1x8x1x1xf32, 0x8xf32) <- (8xf32, 4xi64)
        reshape_68, reshape_69 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_71, full_int_array_50), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x8x192x320xf32) <- (1x8x192x320xf32, 1x8x1x1xf32)
        add__39 = paddle._C_ops.add_(conv2d_34, reshape_68)

        # pd_op.leaky_relu_: (1x8x192x320xf32) <- (1x8x192x320xf32)
        leaky_relu__32 = paddle._C_ops.leaky_relu_(add__39, float('0.1'))

        # pd_op.conv2d: (1x2x192x320xf32) <- (1x8x192x320xf32, 2x8x3x3xf32)
        conv2d_35 = paddle._C_ops.conv2d(leaky_relu__32, parameter_72, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_51 = [1, 2, 1, 1]

        # pd_op.reshape: (1x2x1x1xf32, 0x2xf32) <- (2xf32, 4xi64)
        reshape_70, reshape_71 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_73, full_int_array_51), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x2x192x320xf32) <- (1x2x192x320xf32, 1x2x1x1xf32)
        add__40 = paddle._C_ops.add_(conv2d_35, reshape_70)

        # pd_op.add_: (1x2x192x320xf32) <- (1x2x192x320xf32, 1x2x192x320xf32)
        add__41 = paddle._C_ops.add_(scale__13, add__40)

        # pd_op.bilinear_interp: (1x2x180x320xf32) <- (1x2x192x320xf32, None, None, None)
        bilinear_interp_4 = paddle._C_ops.bilinear_interp(add__41, None, None, None, 'NCHW', -1, 180, 320, [], 'bilinear', False, 0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_52 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_53 = [1]

        # pd_op.slice: (1x180x320xf32) <- (1x2x180x320xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(bilinear_interp_4, [1], full_int_array_52, full_int_array_53, [1], [1])

        # pd_op.full: (1xf32) <- ()
        full_41 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x180x320xf32) <- (1x180x320xf32, 1xf32)
        scale__20 = paddle._C_ops.scale_(slice_12, full_41, float('0'), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_54 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_55 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_56 = [1]

        # pd_op.set_value_with_tensor_: (1x2x180x320xf32) <- (1x2x180x320xf32, 1x180x320xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__0 = paddle._C_ops.set_value_with_tensor_(bilinear_interp_4, scale__20, full_int_array_54, full_int_array_55, full_int_array_56, [1], [1], [])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_57 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_58 = [2]

        # pd_op.slice: (1x180x320xf32) <- (1x2x180x320xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(set_value_with_tensor__0, [1], full_int_array_57, full_int_array_58, [1], [1])

        # pd_op.full: (1xf32) <- ()
        full_42 = paddle._C_ops.full([1], float('0.9375'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x180x320xf32) <- (1x180x320xf32, 1xf32)
        scale__21 = paddle._C_ops.scale_(slice_13, full_42, float('0'), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_59 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_60 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_61 = [1]

        # pd_op.set_value_with_tensor_: (1x2x180x320xf32) <- (1x2x180x320xf32, 1x180x320xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__1 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__0, scale__21, full_int_array_59, full_int_array_60, full_int_array_61, [1], [1], [])

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_62 = [1, 1, 2, 180, 320]

        # pd_op.reshape_: (1x1x2x180x320xf32, 0x1x2x180x320xf32) <- (1x2x180x320xf32, 5xi64)
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape_(set_value_with_tensor__1, full_int_array_62), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.assign_: (1x1x2x180x320xf32) <- (1x1x2x180x320xf32)
        assign__0 = paddle._C_ops.assign_(reshape__2)
        return assign__0
    def pd_op_if_2875_1_0(self, ):

        # pd_op.full: (1x1x2x180x320xf32) <- ()
        full_0 = paddle._C_ops.full([1, 1, 2, 180, 320], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.memcpy_h2d: (1x1x2x180x320xf32) <- (1x1x2x180x320xf32)
        memcpy_h2d_0 = paddle._C_ops.memcpy_h2d(full_0, 1)
        return memcpy_h2d_0
    def builtin_module_2208_0_0(self, parameter_0, parameter_1, parameter_2, parameter_3, parameter_4, parameter_5, parameter_6, parameter_7, parameter_8, parameter_9, parameter_10, parameter_11, parameter_12, parameter_13, parameter_14, parameter_15, parameter_16, parameter_17, parameter_18, parameter_19, parameter_20, parameter_21, parameter_22, parameter_23, parameter_24, parameter_25, parameter_26, parameter_27, parameter_28, parameter_29, parameter_30, parameter_31, parameter_32, parameter_33, parameter_34, parameter_35, parameter_36, parameter_37, parameter_38, parameter_39, parameter_40, parameter_41, parameter_42, parameter_43, parameter_44, parameter_45, parameter_46, parameter_47, parameter_48, parameter_49, parameter_50, parameter_51, parameter_52, parameter_53, parameter_54, parameter_55, parameter_56, parameter_57, parameter_58, parameter_59, parameter_60, parameter_61, parameter_62, parameter_63, parameter_64, parameter_65, parameter_66, parameter_67, parameter_68, parameter_69, parameter_70, parameter_71, parameter_72, parameter_73, parameter_74, parameter_75, parameter_76, parameter_77, parameter_78, parameter_79, parameter_80, parameter_81, parameter_82, parameter_83, parameter_84, parameter_85, parameter_86, parameter_87, parameter_88, parameter_89, parameter_90, parameter_91, parameter_92, parameter_93, parameter_94, parameter_95, parameter_96, parameter_97, parameter_98, parameter_99, parameter_100, parameter_101, parameter_102, parameter_103, parameter_104, parameter_105, parameter_106, parameter_107, parameter_108, parameter_109, parameter_110, parameter_111, parameter_112, parameter_113, parameter_114, parameter_115, parameter_116, parameter_117, parameter_118, parameter_119, parameter_120, parameter_121, parameter_122, parameter_123, parameter_124, parameter_125, parameter_126, parameter_127, parameter_128, parameter_129, parameter_130, parameter_131, parameter_132, parameter_133, parameter_134, parameter_135, parameter_136, parameter_137, parameter_138, parameter_139, parameter_140, parameter_141, parameter_142, parameter_143, parameter_144, parameter_145, parameter_146, parameter_147, parameter_148, parameter_149, parameter_150, parameter_151, parameter_152, parameter_153, parameter_154, parameter_155, parameter_156, parameter_157, parameter_158, parameter_159, parameter_160, parameter_161, parameter_162, parameter_163, parameter_164, parameter_165, parameter_166, parameter_167, parameter_168, parameter_169, parameter_170, parameter_171, parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179, parameter_180, parameter_181, parameter_182, parameter_183, parameter_184, parameter_185, parameter_186, parameter_187, parameter_188, parameter_189, parameter_190, parameter_191, parameter_192, parameter_193, parameter_194, parameter_195, parameter_196, parameter_197, parameter_198, parameter_199, parameter_200, parameter_201, parameter_202, parameter_203, parameter_204, parameter_205, parameter_206, parameter_207, parameter_208, parameter_209, parameter_210, parameter_211, parameter_212, parameter_213, parameter_214, parameter_215, parameter_216, parameter_217, parameter_218, parameter_219, parameter_220, parameter_221, parameter_222, parameter_223, parameter_224, parameter_225, parameter_226, parameter_227, parameter_228, parameter_229, parameter_230, parameter_231, parameter_232, parameter_233, parameter_234, parameter_235, parameter_236, parameter_237, parameter_238, parameter_239, parameter_240, parameter_241, parameter_242, parameter_243, parameter_244, parameter_245, parameter_246, parameter_247, feed_0):

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([1x1x3x180x320xf32, 1x1x3x180x320xf32]) <- (1x2x3x180x320xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(feed_0, 2, full_0)

        # builtin.slice: (1x1x3x180x320xf32) <- ([1x1x3x180x320xf32, 1x1x3x180x320xf32])
        slice_0 = split_with_num_0[1]

        # pd_op.flip: (1x1x3x180x320xf32) <- (1x1x3x180x320xf32)
        flip_0 = paddle._C_ops.flip(slice_0, [1])

        # builtin.slice: (1x1x3x180x320xf32) <- ([1x1x3x180x320xf32, 1x1x3x180x320xf32])
        slice_1 = split_with_num_0[0]

        # pd_op.subtract_: (1x1x3x180x320xf32) <- (1x1x3x180x320xf32, 1x1x3x180x320xf32)
        subtract__0 = paddle._C_ops.subtract_(slice_1, flip_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.frobenius_norm: (xf32) <- (1x1x3x180x320xf32, 1xi64)
        frobenius_norm_0 = paddle._C_ops.frobenius_norm(subtract__0, full_int_array_0, False, True)

        # pd_op.full: (xf32) <- ()
        full_1 = paddle._C_ops.full([], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.equal: (xb) <- (xf32, xf32)
        equal_0 = paddle._C_ops.equal(frobenius_norm_0, full_1)

        # pd_op.cast: (xi32) <- (xb)
        cast_0 = paddle._C_ops.cast(equal_0, paddle.int32)

        # pd_op.full: (xb) <- ()
        full_2 = paddle._C_ops.full([], float('0'), paddle.bool, paddle.framework._current_expected_place())

        # pd_op.full: (xb) <- ()
        full_3 = paddle._C_ops.full([], float('1'), paddle.bool, paddle.framework._current_expected_place())

        # pd_op.select_input: (xb) <- (xi32, xb, xb)
        select_input_0 = (full_2 if cast_0 == 0 else full_3)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [-1, 3, 180, 320]

        # pd_op.reshape: (2x3x180x320xf32, 0x1x2x3x180x320xf32) <- (1x2x3x180x320xf32, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(feed_0, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (2x32x180x320xf32) <- (2x3x180x320xf32, 32x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(reshape_0, parameter_0, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_2 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_1, full_int_array_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (2x32x180x320xf32) <- (2x32x180x320xf32, 1x32x1x1xf32)
        add__0 = paddle._C_ops.add_(conv2d_0, reshape_2)

        # pd_op.leaky_relu_: (2x32x180x320xf32) <- (2x32x180x320xf32)
        leaky_relu__0 = paddle._C_ops.leaky_relu_(add__0, float('0.1'))

        # pd_op.conv2d: (2x32x180x320xf32) <- (2x32x180x320xf32, 32x32x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(leaky_relu__0, parameter_2, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_3, full_int_array_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (2x32x180x320xf32) <- (2x32x180x320xf32, 1x32x1x1xf32)
        add__1 = paddle._C_ops.add_(conv2d_1, reshape_4)

        # pd_op.relu_: (2x32x180x320xf32) <- (2x32x180x320xf32)
        relu__0 = paddle._C_ops.relu_(add__1)

        # pd_op.conv2d: (2x32x180x320xf32) <- (2x32x180x320xf32, 32x32x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(relu__0, parameter_4, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_4 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_5, full_int_array_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (2x32x180x320xf32) <- (2x32x180x320xf32, 1x32x1x1xf32)
        add__2 = paddle._C_ops.add_(conv2d_2, reshape_6)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (2x32x180x320xf32) <- (2x32x180x320xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(add__2, full_4, float('0'), True)

        # pd_op.add_: (2x32x180x320xf32) <- (2x32x180x320xf32, 2x32x180x320xf32)
        add__3 = paddle._C_ops.add_(leaky_relu__0, scale__0)

        # pd_op.conv2d: (2x32x180x320xf32) <- (2x32x180x320xf32, 32x32x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(add__3, parameter_6, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_5 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_7, full_int_array_5), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (2x32x180x320xf32) <- (2x32x180x320xf32, 1x32x1x1xf32)
        add__4 = paddle._C_ops.add_(conv2d_3, reshape_8)

        # pd_op.relu_: (2x32x180x320xf32) <- (2x32x180x320xf32)
        relu__1 = paddle._C_ops.relu_(add__4)

        # pd_op.conv2d: (2x32x180x320xf32) <- (2x32x180x320xf32, 32x32x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(relu__1, parameter_8, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_6 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_9, full_int_array_6), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (2x32x180x320xf32) <- (2x32x180x320xf32, 1x32x1x1xf32)
        add__5 = paddle._C_ops.add_(conv2d_4, reshape_10)

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (2x32x180x320xf32) <- (2x32x180x320xf32, 1xf32)
        scale__1 = paddle._C_ops.scale_(add__5, full_5, float('0'), True)

        # pd_op.add_: (2x32x180x320xf32) <- (2x32x180x320xf32, 2x32x180x320xf32)
        add__6 = paddle._C_ops.add_(add__3, scale__1)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_7 = [1, 2, -1, 180, 320]

        # pd_op.reshape_: (1x2x32x180x320xf32, 0x2x32x180x320xf32) <- (2x32x180x320xf32, 5xi64)
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__6, full_int_array_7), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [1]

        # pd_op.slice: (1x32x180x320xf32) <- (1x2x32x180x320xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(reshape__0, [1], full_int_array_8, full_int_array_9, [1], [1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_10 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_11 = [2]

        # pd_op.slice: (1x32x180x320xf32) <- (1x2x32x180x320xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(reshape__0, [1], full_int_array_10, full_int_array_11, [1], [1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_12 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_13 = [-1]

        # pd_op.slice: (1x1x3x180x320xf32) <- (1x2x3x180x320xf32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(feed_0, [1], full_int_array_12, full_int_array_13, [1], [])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_14 = [-1, 3, 180, 320]

        # pd_op.reshape_: (1x3x180x320xf32, 0x1x1x3x180x320xf32) <- (1x1x3x180x320xf32, 4xi64)
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape_(slice_4, full_int_array_14), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_15 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_16 = [2]

        # pd_op.slice: (1x1x3x180x320xf32) <- (1x2x3x180x320xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(feed_0, [1], full_int_array_15, full_int_array_16, [1], [])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_17 = [-1, 3, 180, 320]

        # pd_op.reshape_: (1x3x180x320xf32, 0x1x1x3x180x320xf32) <- (1x1x3x180x320xf32, 4xi64)
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape_(slice_5, full_int_array_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.bilinear_interp: (1x3x192x320xf32) <- (1x3x180x320xf32, None, None, None)
        bilinear_interp_0 = paddle._C_ops.bilinear_interp(reshape__2, None, None, None, 'NCHW', -1, 192, 320, [], 'bilinear', False, 0)

        # pd_op.bilinear_interp: (1x3x192x320xf32) <- (1x3x180x320xf32, None, None, None)
        bilinear_interp_1 = paddle._C_ops.bilinear_interp(reshape__4, None, None, None, 'NCHW', -1, 192, 320, [], 'bilinear', False, 0)

        # pd_op.subtract_: (1x3x192x320xf32) <- (1x3x192x320xf32, 1x3x1x1xf32)
        subtract__1 = paddle._C_ops.subtract_(bilinear_interp_0, parameter_10)

        # pd_op.divide_: (1x3x192x320xf32) <- (1x3x192x320xf32, 1x3x1x1xf32)
        divide__0 = paddle._C_ops.divide_(subtract__1, parameter_11)

        # pd_op.subtract_: (1x3x192x320xf32) <- (1x3x192x320xf32, 1x3x1x1xf32)
        subtract__2 = paddle._C_ops.subtract_(bilinear_interp_1, parameter_10)

        # pd_op.divide_: (1x3x192x320xf32) <- (1x3x192x320xf32, 1x3x1x1xf32)
        divide__1 = paddle._C_ops.divide_(subtract__2, parameter_11)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_18 = [2, 2]

        # pd_op.pool2d: (1x3x96x160xf32) <- (1x3x192x320xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(divide__0, full_int_array_18, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_19 = [2, 2]

        # pd_op.pool2d: (1x3x96x160xf32) <- (1x3x192x320xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(divide__1, full_int_array_19, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_20 = [2, 2]

        # pd_op.pool2d: (1x3x48x80xf32) <- (1x3x96x160xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(pool2d_0, full_int_array_20, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_21 = [2, 2]

        # pd_op.pool2d: (1x3x48x80xf32) <- (1x3x96x160xf32, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(pool2d_1, full_int_array_21, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.assign_value: (1x2x48x80xf32) <- ()
        assign_value_0 = paddle.to_tensor([float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0'), float('0')], dtype=paddle.float32).reshape([1, 2, 48, 80])

        # pd_op.transpose: (1x48x80x2xf32) <- (1x2x48x80xf32)
        transpose_0 = paddle._C_ops.transpose(assign_value_0, [0, 2, 3, 1])

        # pd_op.full: (1xi64) <- ()
        full_6 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_7 = paddle._C_ops.full([1], float('48'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_8 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (48xi64) <- (1xi64, 1xi64, 1xi64)
        arange_0 = paddle.arange(full_6, full_7, full_8, dtype='int64')

        # pd_op.full: (1xi64) <- ()
        full_9 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_10 = paddle._C_ops.full([1], float('80'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_11 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (80xi64) <- (1xi64, 1xi64, 1xi64)
        arange_1 = paddle.arange(full_9, full_10, full_11, dtype='int64')

        # builtin.combine: ([48xi64, 80xi64]) <- (48xi64, 80xi64)
        combine_0 = [arange_0, arange_1]

        # pd_op.meshgrid: ([48x80xi64, 48x80xi64]) <- ([48xi64, 80xi64])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_0)

        # builtin.slice: (48x80xi64) <- ([48x80xi64, 48x80xi64])
        slice_6 = meshgrid_0[1]

        # builtin.slice: (48x80xi64) <- ([48x80xi64, 48x80xi64])
        slice_7 = meshgrid_0[0]

        # builtin.combine: ([48x80xi64, 48x80xi64]) <- (48x80xi64, 48x80xi64)
        combine_1 = [slice_6, slice_7]

        # pd_op.stack: (48x80x2xi64) <- ([48x80xi64, 48x80xi64])
        stack_0 = paddle._C_ops.stack(combine_1, 2)

        # pd_op.cast: (48x80x2xf32) <- (48x80x2xi64)
        cast_1 = paddle._C_ops.cast(stack_0, paddle.float32)

        # pd_op.add_: (1x48x80x2xf32) <- (48x80x2xf32, 1x48x80x2xf32)
        add__7 = paddle._C_ops.add_(cast_1, transpose_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_22 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_23 = [1]

        # pd_op.slice: (1x48x80xf32) <- (1x48x80x2xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(add__7, [3], full_int_array_22, full_int_array_23, [1], [3])

        # pd_op.full: (1xf32) <- ()
        full_12 = paddle._C_ops.full([1], float('2'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x48x80xf32) <- (1x48x80xf32, 1xf32)
        scale__2 = paddle._C_ops.scale_(slice_8, full_12, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_13 = paddle._C_ops.full([1], float('0.0126582'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x48x80xf32) <- (1x48x80xf32, 1xf32)
        scale__3 = paddle._C_ops.scale_(scale__2, full_13, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_14 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x48x80xf32) <- (1x48x80xf32, 1xf32)
        scale__4 = paddle._C_ops.scale_(scale__3, full_14, float('-1'), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_24 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_25 = [2]

        # pd_op.slice: (1x48x80xf32) <- (1x48x80x2xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(add__7, [3], full_int_array_24, full_int_array_25, [1], [3])

        # pd_op.full: (1xf32) <- ()
        full_15 = paddle._C_ops.full([1], float('2'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x48x80xf32) <- (1x48x80xf32, 1xf32)
        scale__5 = paddle._C_ops.scale_(slice_9, full_15, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_16 = paddle._C_ops.full([1], float('0.0212766'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x48x80xf32) <- (1x48x80xf32, 1xf32)
        scale__6 = paddle._C_ops.scale_(scale__5, full_16, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_17 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x48x80xf32) <- (1x48x80xf32, 1xf32)
        scale__7 = paddle._C_ops.scale_(scale__6, full_17, float('-1'), True)

        # builtin.combine: ([1x48x80xf32, 1x48x80xf32]) <- (1x48x80xf32, 1x48x80xf32)
        combine_2 = [scale__4, scale__7]

        # pd_op.stack: (1x48x80x2xf32) <- ([1x48x80xf32, 1x48x80xf32])
        stack_1 = paddle._C_ops.stack(combine_2, 3)

        # pd_op.grid_sample: (1x3x48x80xf32) <- (1x3x48x80xf32, 1x48x80x2xf32)
        grid_sample_0 = paddle._C_ops.grid_sample(pool2d_3, stack_1, 'bilinear', 'border', True)

        # builtin.combine: ([1x3x48x80xf32, 1x3x48x80xf32, 1x2x48x80xf32]) <- (1x3x48x80xf32, 1x3x48x80xf32, 1x2x48x80xf32)
        combine_3 = [pool2d_2, grid_sample_0, assign_value_0]

        # pd_op.full: (1xi32) <- ()
        full_18 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x8x48x80xf32) <- ([1x3x48x80xf32, 1x3x48x80xf32, 1x2x48x80xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_3, full_18)

        # pd_op.conv2d: (1x16x48x80xf32) <- (1x8x48x80xf32, 16x8x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(concat_0, parameter_12, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_26 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_12, reshape_13 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_13, full_int_array_26), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x48x80xf32) <- (1x16x48x80xf32, 1x16x1x1xf32)
        add__8 = paddle._C_ops.add_(conv2d_5, reshape_12)

        # pd_op.leaky_relu_: (1x16x48x80xf32) <- (1x16x48x80xf32)
        leaky_relu__1 = paddle._C_ops.leaky_relu_(add__8, float('0.1'))

        # pd_op.conv2d: (1x16x48x80xf32) <- (1x16x48x80xf32, 16x16x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(leaky_relu__1, parameter_14, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_27 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_14, reshape_15 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_15, full_int_array_27), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x48x80xf32) <- (1x16x48x80xf32, 1x16x1x1xf32)
        add__9 = paddle._C_ops.add_(conv2d_6, reshape_14)

        # pd_op.leaky_relu_: (1x16x48x80xf32) <- (1x16x48x80xf32)
        leaky_relu__2 = paddle._C_ops.leaky_relu_(add__9, float('0.1'))

        # pd_op.conv2d: (1x32x48x80xf32) <- (1x16x48x80xf32, 32x16x3x3xf32)
        conv2d_7 = paddle._C_ops.conv2d(leaky_relu__2, parameter_16, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_28 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_16, reshape_17 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_17, full_int_array_28), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x48x80xf32) <- (1x32x48x80xf32, 1x32x1x1xf32)
        add__10 = paddle._C_ops.add_(conv2d_7, reshape_16)

        # pd_op.leaky_relu_: (1x32x48x80xf32) <- (1x32x48x80xf32)
        leaky_relu__3 = paddle._C_ops.leaky_relu_(add__10, float('0.1'))

        # pd_op.conv2d: (1x32x48x80xf32) <- (1x32x48x80xf32, 32x32x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(leaky_relu__3, parameter_18, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_29 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_18, reshape_19 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_19, full_int_array_29), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x48x80xf32) <- (1x32x48x80xf32, 1x32x1x1xf32)
        add__11 = paddle._C_ops.add_(conv2d_8, reshape_18)

        # pd_op.leaky_relu_: (1x32x48x80xf32) <- (1x32x48x80xf32)
        leaky_relu__4 = paddle._C_ops.leaky_relu_(add__11, float('0.1'))

        # pd_op.conv2d: (1x32x48x80xf32) <- (1x32x48x80xf32, 32x32x3x3xf32)
        conv2d_9 = paddle._C_ops.conv2d(leaky_relu__4, parameter_20, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_30 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_20, reshape_21 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_21, full_int_array_30), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x48x80xf32) <- (1x32x48x80xf32, 1x32x1x1xf32)
        add__12 = paddle._C_ops.add_(conv2d_9, reshape_20)

        # pd_op.leaky_relu_: (1x32x48x80xf32) <- (1x32x48x80xf32)
        leaky_relu__5 = paddle._C_ops.leaky_relu_(add__12, float('0.1'))

        # pd_op.conv2d: (1x32x48x80xf32) <- (1x32x48x80xf32, 32x32x3x3xf32)
        conv2d_10 = paddle._C_ops.conv2d(leaky_relu__5, parameter_22, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_31 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_22, reshape_23 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_23, full_int_array_31), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x48x80xf32) <- (1x32x48x80xf32, 1x32x1x1xf32)
        add__13 = paddle._C_ops.add_(conv2d_10, reshape_22)

        # pd_op.leaky_relu_: (1x32x48x80xf32) <- (1x32x48x80xf32)
        leaky_relu__6 = paddle._C_ops.leaky_relu_(add__13, float('0.1'))

        # pd_op.conv2d: (1x16x48x80xf32) <- (1x32x48x80xf32, 16x32x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(leaky_relu__6, parameter_24, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_32 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_24, reshape_25 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_25, full_int_array_32), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x48x80xf32) <- (1x16x48x80xf32, 1x16x1x1xf32)
        add__14 = paddle._C_ops.add_(conv2d_11, reshape_24)

        # pd_op.leaky_relu_: (1x16x48x80xf32) <- (1x16x48x80xf32)
        leaky_relu__7 = paddle._C_ops.leaky_relu_(add__14, float('0.1'))

        # pd_op.conv2d: (1x16x48x80xf32) <- (1x16x48x80xf32, 16x16x3x3xf32)
        conv2d_12 = paddle._C_ops.conv2d(leaky_relu__7, parameter_26, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_33 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_26, reshape_27 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_27, full_int_array_33), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x48x80xf32) <- (1x16x48x80xf32, 1x16x1x1xf32)
        add__15 = paddle._C_ops.add_(conv2d_12, reshape_26)

        # pd_op.leaky_relu_: (1x16x48x80xf32) <- (1x16x48x80xf32)
        leaky_relu__8 = paddle._C_ops.leaky_relu_(add__15, float('0.1'))

        # pd_op.conv2d: (1x16x48x80xf32) <- (1x16x48x80xf32, 16x16x3x3xf32)
        conv2d_13 = paddle._C_ops.conv2d(leaky_relu__8, parameter_28, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_34 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_28, reshape_29 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_29, full_int_array_34), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x48x80xf32) <- (1x16x48x80xf32, 1x16x1x1xf32)
        add__16 = paddle._C_ops.add_(conv2d_13, reshape_28)

        # pd_op.leaky_relu_: (1x16x48x80xf32) <- (1x16x48x80xf32)
        leaky_relu__9 = paddle._C_ops.leaky_relu_(add__16, float('0.1'))

        # pd_op.conv2d: (1x8x48x80xf32) <- (1x16x48x80xf32, 8x16x3x3xf32)
        conv2d_14 = paddle._C_ops.conv2d(leaky_relu__9, parameter_30, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_35 = [1, 8, 1, 1]

        # pd_op.reshape: (1x8x1x1xf32, 0x8xf32) <- (8xf32, 4xi64)
        reshape_30, reshape_31 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_31, full_int_array_35), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x8x48x80xf32) <- (1x8x48x80xf32, 1x8x1x1xf32)
        add__17 = paddle._C_ops.add_(conv2d_14, reshape_30)

        # pd_op.leaky_relu_: (1x8x48x80xf32) <- (1x8x48x80xf32)
        leaky_relu__10 = paddle._C_ops.leaky_relu_(add__17, float('0.1'))

        # pd_op.conv2d: (1x8x48x80xf32) <- (1x8x48x80xf32, 8x8x3x3xf32)
        conv2d_15 = paddle._C_ops.conv2d(leaky_relu__10, parameter_32, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_36 = [1, 8, 1, 1]

        # pd_op.reshape: (1x8x1x1xf32, 0x8xf32) <- (8xf32, 4xi64)
        reshape_32, reshape_33 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_33, full_int_array_36), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x8x48x80xf32) <- (1x8x48x80xf32, 1x8x1x1xf32)
        add__18 = paddle._C_ops.add_(conv2d_15, reshape_32)

        # pd_op.leaky_relu_: (1x8x48x80xf32) <- (1x8x48x80xf32)
        leaky_relu__11 = paddle._C_ops.leaky_relu_(add__18, float('0.1'))

        # pd_op.conv2d: (1x2x48x80xf32) <- (1x8x48x80xf32, 2x8x3x3xf32)
        conv2d_16 = paddle._C_ops.conv2d(leaky_relu__11, parameter_34, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_37 = [1, 2, 1, 1]

        # pd_op.reshape: (1x2x1x1xf32, 0x2xf32) <- (2xf32, 4xi64)
        reshape_34, reshape_35 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_35, full_int_array_37), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x2x48x80xf32) <- (1x2x48x80xf32, 1x2x1x1xf32)
        add__19 = paddle._C_ops.add_(conv2d_16, reshape_34)

        # pd_op.add_: (1x2x48x80xf32) <- (1x2x48x80xf32, 1x2x48x80xf32)
        add__20 = paddle._C_ops.add_(assign_value_0, add__19)

        # pd_op.bilinear_interp: (1x2x96x160xf32) <- (1x2x48x80xf32, None, None, None)
        bilinear_interp_2 = paddle._C_ops.bilinear_interp(add__20, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'bilinear', True, 0)

        # pd_op.full: (1xf32) <- ()
        full_19 = paddle._C_ops.full([1], float('2'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x2x96x160xf32) <- (1x2x96x160xf32, 1xf32)
        scale__8 = paddle._C_ops.scale_(bilinear_interp_2, full_19, float('0'), True)

        # pd_op.transpose: (1x96x160x2xf32) <- (1x2x96x160xf32)
        transpose_1 = paddle._C_ops.transpose(scale__8, [0, 2, 3, 1])

        # pd_op.full: (1xi64) <- ()
        full_20 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_21 = paddle._C_ops.full([1], float('96'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_22 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (96xi64) <- (1xi64, 1xi64, 1xi64)
        arange_2 = paddle.arange(full_20, full_21, full_22, dtype='int64')

        # pd_op.full: (1xi64) <- ()
        full_23 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_24 = paddle._C_ops.full([1], float('160'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_25 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (160xi64) <- (1xi64, 1xi64, 1xi64)
        arange_3 = paddle.arange(full_23, full_24, full_25, dtype='int64')

        # builtin.combine: ([96xi64, 160xi64]) <- (96xi64, 160xi64)
        combine_4 = [arange_2, arange_3]

        # pd_op.meshgrid: ([96x160xi64, 96x160xi64]) <- ([96xi64, 160xi64])
        meshgrid_1 = paddle._C_ops.meshgrid(combine_4)

        # builtin.slice: (96x160xi64) <- ([96x160xi64, 96x160xi64])
        slice_10 = meshgrid_1[1]

        # builtin.slice: (96x160xi64) <- ([96x160xi64, 96x160xi64])
        slice_11 = meshgrid_1[0]

        # builtin.combine: ([96x160xi64, 96x160xi64]) <- (96x160xi64, 96x160xi64)
        combine_5 = [slice_10, slice_11]

        # pd_op.stack: (96x160x2xi64) <- ([96x160xi64, 96x160xi64])
        stack_2 = paddle._C_ops.stack(combine_5, 2)

        # pd_op.cast: (96x160x2xf32) <- (96x160x2xi64)
        cast_2 = paddle._C_ops.cast(stack_2, paddle.float32)

        # pd_op.add_: (1x96x160x2xf32) <- (96x160x2xf32, 1x96x160x2xf32)
        add__21 = paddle._C_ops.add_(cast_2, transpose_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_38 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_39 = [1]

        # pd_op.slice: (1x96x160xf32) <- (1x96x160x2xf32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(add__21, [3], full_int_array_38, full_int_array_39, [1], [3])

        # pd_op.full: (1xf32) <- ()
        full_26 = paddle._C_ops.full([1], float('2'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x96x160xf32) <- (1x96x160xf32, 1xf32)
        scale__9 = paddle._C_ops.scale_(slice_12, full_26, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_27 = paddle._C_ops.full([1], float('0.00628931'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x96x160xf32) <- (1x96x160xf32, 1xf32)
        scale__10 = paddle._C_ops.scale_(scale__9, full_27, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_28 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x96x160xf32) <- (1x96x160xf32, 1xf32)
        scale__11 = paddle._C_ops.scale_(scale__10, full_28, float('-1'), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_40 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_41 = [2]

        # pd_op.slice: (1x96x160xf32) <- (1x96x160x2xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(add__21, [3], full_int_array_40, full_int_array_41, [1], [3])

        # pd_op.full: (1xf32) <- ()
        full_29 = paddle._C_ops.full([1], float('2'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x96x160xf32) <- (1x96x160xf32, 1xf32)
        scale__12 = paddle._C_ops.scale_(slice_13, full_29, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_30 = paddle._C_ops.full([1], float('0.0105263'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x96x160xf32) <- (1x96x160xf32, 1xf32)
        scale__13 = paddle._C_ops.scale_(scale__12, full_30, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_31 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x96x160xf32) <- (1x96x160xf32, 1xf32)
        scale__14 = paddle._C_ops.scale_(scale__13, full_31, float('-1'), True)

        # builtin.combine: ([1x96x160xf32, 1x96x160xf32]) <- (1x96x160xf32, 1x96x160xf32)
        combine_6 = [scale__11, scale__14]

        # pd_op.stack: (1x96x160x2xf32) <- ([1x96x160xf32, 1x96x160xf32])
        stack_3 = paddle._C_ops.stack(combine_6, 3)

        # pd_op.grid_sample: (1x3x96x160xf32) <- (1x3x96x160xf32, 1x96x160x2xf32)
        grid_sample_1 = paddle._C_ops.grid_sample(pool2d_1, stack_3, 'bilinear', 'border', True)

        # builtin.combine: ([1x3x96x160xf32, 1x3x96x160xf32, 1x2x96x160xf32]) <- (1x3x96x160xf32, 1x3x96x160xf32, 1x2x96x160xf32)
        combine_7 = [pool2d_0, grid_sample_1, scale__8]

        # pd_op.full: (1xi32) <- ()
        full_32 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x8x96x160xf32) <- ([1x3x96x160xf32, 1x3x96x160xf32, 1x2x96x160xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_7, full_32)

        # pd_op.conv2d: (1x16x96x160xf32) <- (1x8x96x160xf32, 16x8x3x3xf32)
        conv2d_17 = paddle._C_ops.conv2d(concat_1, parameter_36, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_42 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_36, reshape_37 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_37, full_int_array_42), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x96x160xf32) <- (1x16x96x160xf32, 1x16x1x1xf32)
        add__22 = paddle._C_ops.add_(conv2d_17, reshape_36)

        # pd_op.leaky_relu_: (1x16x96x160xf32) <- (1x16x96x160xf32)
        leaky_relu__12 = paddle._C_ops.leaky_relu_(add__22, float('0.1'))

        # pd_op.conv2d: (1x16x96x160xf32) <- (1x16x96x160xf32, 16x16x3x3xf32)
        conv2d_18 = paddle._C_ops.conv2d(leaky_relu__12, parameter_38, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_43 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_38, reshape_39 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_39, full_int_array_43), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x96x160xf32) <- (1x16x96x160xf32, 1x16x1x1xf32)
        add__23 = paddle._C_ops.add_(conv2d_18, reshape_38)

        # pd_op.leaky_relu_: (1x16x96x160xf32) <- (1x16x96x160xf32)
        leaky_relu__13 = paddle._C_ops.leaky_relu_(add__23, float('0.1'))

        # pd_op.conv2d: (1x32x96x160xf32) <- (1x16x96x160xf32, 32x16x3x3xf32)
        conv2d_19 = paddle._C_ops.conv2d(leaky_relu__13, parameter_40, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_44 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_40, reshape_41 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_41, full_int_array_44), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x96x160xf32) <- (1x32x96x160xf32, 1x32x1x1xf32)
        add__24 = paddle._C_ops.add_(conv2d_19, reshape_40)

        # pd_op.leaky_relu_: (1x32x96x160xf32) <- (1x32x96x160xf32)
        leaky_relu__14 = paddle._C_ops.leaky_relu_(add__24, float('0.1'))

        # pd_op.conv2d: (1x32x96x160xf32) <- (1x32x96x160xf32, 32x32x3x3xf32)
        conv2d_20 = paddle._C_ops.conv2d(leaky_relu__14, parameter_42, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_45 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_42, reshape_43 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_43, full_int_array_45), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x96x160xf32) <- (1x32x96x160xf32, 1x32x1x1xf32)
        add__25 = paddle._C_ops.add_(conv2d_20, reshape_42)

        # pd_op.leaky_relu_: (1x32x96x160xf32) <- (1x32x96x160xf32)
        leaky_relu__15 = paddle._C_ops.leaky_relu_(add__25, float('0.1'))

        # pd_op.conv2d: (1x32x96x160xf32) <- (1x32x96x160xf32, 32x32x3x3xf32)
        conv2d_21 = paddle._C_ops.conv2d(leaky_relu__15, parameter_44, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_46 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_44, reshape_45 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_45, full_int_array_46), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x96x160xf32) <- (1x32x96x160xf32, 1x32x1x1xf32)
        add__26 = paddle._C_ops.add_(conv2d_21, reshape_44)

        # pd_op.leaky_relu_: (1x32x96x160xf32) <- (1x32x96x160xf32)
        leaky_relu__16 = paddle._C_ops.leaky_relu_(add__26, float('0.1'))

        # pd_op.conv2d: (1x32x96x160xf32) <- (1x32x96x160xf32, 32x32x3x3xf32)
        conv2d_22 = paddle._C_ops.conv2d(leaky_relu__16, parameter_46, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_47 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_46, reshape_47 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_47, full_int_array_47), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x96x160xf32) <- (1x32x96x160xf32, 1x32x1x1xf32)
        add__27 = paddle._C_ops.add_(conv2d_22, reshape_46)

        # pd_op.leaky_relu_: (1x32x96x160xf32) <- (1x32x96x160xf32)
        leaky_relu__17 = paddle._C_ops.leaky_relu_(add__27, float('0.1'))

        # pd_op.conv2d: (1x16x96x160xf32) <- (1x32x96x160xf32, 16x32x3x3xf32)
        conv2d_23 = paddle._C_ops.conv2d(leaky_relu__17, parameter_48, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_48 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_48, reshape_49 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_49, full_int_array_48), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x96x160xf32) <- (1x16x96x160xf32, 1x16x1x1xf32)
        add__28 = paddle._C_ops.add_(conv2d_23, reshape_48)

        # pd_op.leaky_relu_: (1x16x96x160xf32) <- (1x16x96x160xf32)
        leaky_relu__18 = paddle._C_ops.leaky_relu_(add__28, float('0.1'))

        # pd_op.conv2d: (1x16x96x160xf32) <- (1x16x96x160xf32, 16x16x3x3xf32)
        conv2d_24 = paddle._C_ops.conv2d(leaky_relu__18, parameter_50, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_49 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_50, reshape_51 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_51, full_int_array_49), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x96x160xf32) <- (1x16x96x160xf32, 1x16x1x1xf32)
        add__29 = paddle._C_ops.add_(conv2d_24, reshape_50)

        # pd_op.leaky_relu_: (1x16x96x160xf32) <- (1x16x96x160xf32)
        leaky_relu__19 = paddle._C_ops.leaky_relu_(add__29, float('0.1'))

        # pd_op.conv2d: (1x16x96x160xf32) <- (1x16x96x160xf32, 16x16x3x3xf32)
        conv2d_25 = paddle._C_ops.conv2d(leaky_relu__19, parameter_52, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_50 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_52, reshape_53 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_53, full_int_array_50), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x96x160xf32) <- (1x16x96x160xf32, 1x16x1x1xf32)
        add__30 = paddle._C_ops.add_(conv2d_25, reshape_52)

        # pd_op.leaky_relu_: (1x16x96x160xf32) <- (1x16x96x160xf32)
        leaky_relu__20 = paddle._C_ops.leaky_relu_(add__30, float('0.1'))

        # pd_op.conv2d: (1x8x96x160xf32) <- (1x16x96x160xf32, 8x16x3x3xf32)
        conv2d_26 = paddle._C_ops.conv2d(leaky_relu__20, parameter_54, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_51 = [1, 8, 1, 1]

        # pd_op.reshape: (1x8x1x1xf32, 0x8xf32) <- (8xf32, 4xi64)
        reshape_54, reshape_55 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_55, full_int_array_51), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x8x96x160xf32) <- (1x8x96x160xf32, 1x8x1x1xf32)
        add__31 = paddle._C_ops.add_(conv2d_26, reshape_54)

        # pd_op.leaky_relu_: (1x8x96x160xf32) <- (1x8x96x160xf32)
        leaky_relu__21 = paddle._C_ops.leaky_relu_(add__31, float('0.1'))

        # pd_op.conv2d: (1x8x96x160xf32) <- (1x8x96x160xf32, 8x8x3x3xf32)
        conv2d_27 = paddle._C_ops.conv2d(leaky_relu__21, parameter_56, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_52 = [1, 8, 1, 1]

        # pd_op.reshape: (1x8x1x1xf32, 0x8xf32) <- (8xf32, 4xi64)
        reshape_56, reshape_57 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_57, full_int_array_52), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x8x96x160xf32) <- (1x8x96x160xf32, 1x8x1x1xf32)
        add__32 = paddle._C_ops.add_(conv2d_27, reshape_56)

        # pd_op.leaky_relu_: (1x8x96x160xf32) <- (1x8x96x160xf32)
        leaky_relu__22 = paddle._C_ops.leaky_relu_(add__32, float('0.1'))

        # pd_op.conv2d: (1x2x96x160xf32) <- (1x8x96x160xf32, 2x8x3x3xf32)
        conv2d_28 = paddle._C_ops.conv2d(leaky_relu__22, parameter_58, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_53 = [1, 2, 1, 1]

        # pd_op.reshape: (1x2x1x1xf32, 0x2xf32) <- (2xf32, 4xi64)
        reshape_58, reshape_59 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_59, full_int_array_53), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x2x96x160xf32) <- (1x2x96x160xf32, 1x2x1x1xf32)
        add__33 = paddle._C_ops.add_(conv2d_28, reshape_58)

        # pd_op.add_: (1x2x96x160xf32) <- (1x2x96x160xf32, 1x2x96x160xf32)
        add__34 = paddle._C_ops.add_(scale__8, add__33)

        # pd_op.bilinear_interp: (1x2x192x320xf32) <- (1x2x96x160xf32, None, None, None)
        bilinear_interp_3 = paddle._C_ops.bilinear_interp(add__34, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'bilinear', True, 0)

        # pd_op.full: (1xf32) <- ()
        full_33 = paddle._C_ops.full([1], float('2'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x2x192x320xf32) <- (1x2x192x320xf32, 1xf32)
        scale__15 = paddle._C_ops.scale_(bilinear_interp_3, full_33, float('0'), True)

        # pd_op.transpose: (1x192x320x2xf32) <- (1x2x192x320xf32)
        transpose_2 = paddle._C_ops.transpose(scale__15, [0, 2, 3, 1])

        # pd_op.full: (1xi64) <- ()
        full_34 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_35 = paddle._C_ops.full([1], float('192'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_36 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (192xi64) <- (1xi64, 1xi64, 1xi64)
        arange_4 = paddle.arange(full_34, full_35, full_36, dtype='int64')

        # pd_op.full: (1xi64) <- ()
        full_37 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_38 = paddle._C_ops.full([1], float('320'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_39 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (320xi64) <- (1xi64, 1xi64, 1xi64)
        arange_5 = paddle.arange(full_37, full_38, full_39, dtype='int64')

        # builtin.combine: ([192xi64, 320xi64]) <- (192xi64, 320xi64)
        combine_8 = [arange_4, arange_5]

        # pd_op.meshgrid: ([192x320xi64, 192x320xi64]) <- ([192xi64, 320xi64])
        meshgrid_2 = paddle._C_ops.meshgrid(combine_8)

        # builtin.slice: (192x320xi64) <- ([192x320xi64, 192x320xi64])
        slice_14 = meshgrid_2[1]

        # builtin.slice: (192x320xi64) <- ([192x320xi64, 192x320xi64])
        slice_15 = meshgrid_2[0]

        # builtin.combine: ([192x320xi64, 192x320xi64]) <- (192x320xi64, 192x320xi64)
        combine_9 = [slice_14, slice_15]

        # pd_op.stack: (192x320x2xi64) <- ([192x320xi64, 192x320xi64])
        stack_4 = paddle._C_ops.stack(combine_9, 2)

        # pd_op.cast: (192x320x2xf32) <- (192x320x2xi64)
        cast_3 = paddle._C_ops.cast(stack_4, paddle.float32)

        # pd_op.add_: (1x192x320x2xf32) <- (192x320x2xf32, 1x192x320x2xf32)
        add__35 = paddle._C_ops.add_(cast_3, transpose_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_54 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_55 = [1]

        # pd_op.slice: (1x192x320xf32) <- (1x192x320x2xf32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(add__35, [3], full_int_array_54, full_int_array_55, [1], [3])

        # pd_op.full: (1xf32) <- ()
        full_40 = paddle._C_ops.full([1], float('2'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x192x320xf32) <- (1x192x320xf32, 1xf32)
        scale__16 = paddle._C_ops.scale_(slice_16, full_40, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_41 = paddle._C_ops.full([1], float('0.0031348'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x192x320xf32) <- (1x192x320xf32, 1xf32)
        scale__17 = paddle._C_ops.scale_(scale__16, full_41, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_42 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x192x320xf32) <- (1x192x320xf32, 1xf32)
        scale__18 = paddle._C_ops.scale_(scale__17, full_42, float('-1'), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_56 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_57 = [2]

        # pd_op.slice: (1x192x320xf32) <- (1x192x320x2xf32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(add__35, [3], full_int_array_56, full_int_array_57, [1], [3])

        # pd_op.full: (1xf32) <- ()
        full_43 = paddle._C_ops.full([1], float('2'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x192x320xf32) <- (1x192x320xf32, 1xf32)
        scale__19 = paddle._C_ops.scale_(slice_17, full_43, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_44 = paddle._C_ops.full([1], float('0.0052356'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x192x320xf32) <- (1x192x320xf32, 1xf32)
        scale__20 = paddle._C_ops.scale_(scale__19, full_44, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_45 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x192x320xf32) <- (1x192x320xf32, 1xf32)
        scale__21 = paddle._C_ops.scale_(scale__20, full_45, float('-1'), True)

        # builtin.combine: ([1x192x320xf32, 1x192x320xf32]) <- (1x192x320xf32, 1x192x320xf32)
        combine_10 = [scale__18, scale__21]

        # pd_op.stack: (1x192x320x2xf32) <- ([1x192x320xf32, 1x192x320xf32])
        stack_5 = paddle._C_ops.stack(combine_10, 3)

        # pd_op.grid_sample: (1x3x192x320xf32) <- (1x3x192x320xf32, 1x192x320x2xf32)
        grid_sample_2 = paddle._C_ops.grid_sample(divide__1, stack_5, 'bilinear', 'border', True)

        # builtin.combine: ([1x3x192x320xf32, 1x3x192x320xf32, 1x2x192x320xf32]) <- (1x3x192x320xf32, 1x3x192x320xf32, 1x2x192x320xf32)
        combine_11 = [divide__0, grid_sample_2, scale__15]

        # pd_op.full: (1xi32) <- ()
        full_46 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x8x192x320xf32) <- ([1x3x192x320xf32, 1x3x192x320xf32, 1x2x192x320xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_11, full_46)

        # pd_op.conv2d: (1x16x192x320xf32) <- (1x8x192x320xf32, 16x8x3x3xf32)
        conv2d_29 = paddle._C_ops.conv2d(concat_2, parameter_60, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_58 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_60, reshape_61 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_61, full_int_array_58), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x192x320xf32) <- (1x16x192x320xf32, 1x16x1x1xf32)
        add__36 = paddle._C_ops.add_(conv2d_29, reshape_60)

        # pd_op.leaky_relu_: (1x16x192x320xf32) <- (1x16x192x320xf32)
        leaky_relu__23 = paddle._C_ops.leaky_relu_(add__36, float('0.1'))

        # pd_op.conv2d: (1x16x192x320xf32) <- (1x16x192x320xf32, 16x16x3x3xf32)
        conv2d_30 = paddle._C_ops.conv2d(leaky_relu__23, parameter_62, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_59 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_62, reshape_63 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_63, full_int_array_59), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x192x320xf32) <- (1x16x192x320xf32, 1x16x1x1xf32)
        add__37 = paddle._C_ops.add_(conv2d_30, reshape_62)

        # pd_op.leaky_relu_: (1x16x192x320xf32) <- (1x16x192x320xf32)
        leaky_relu__24 = paddle._C_ops.leaky_relu_(add__37, float('0.1'))

        # pd_op.conv2d: (1x32x192x320xf32) <- (1x16x192x320xf32, 32x16x3x3xf32)
        conv2d_31 = paddle._C_ops.conv2d(leaky_relu__24, parameter_64, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_60 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_64, reshape_65 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_65, full_int_array_60), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x192x320xf32) <- (1x32x192x320xf32, 1x32x1x1xf32)
        add__38 = paddle._C_ops.add_(conv2d_31, reshape_64)

        # pd_op.leaky_relu_: (1x32x192x320xf32) <- (1x32x192x320xf32)
        leaky_relu__25 = paddle._C_ops.leaky_relu_(add__38, float('0.1'))

        # pd_op.conv2d: (1x32x192x320xf32) <- (1x32x192x320xf32, 32x32x3x3xf32)
        conv2d_32 = paddle._C_ops.conv2d(leaky_relu__25, parameter_66, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_61 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_66, reshape_67 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_67, full_int_array_61), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x192x320xf32) <- (1x32x192x320xf32, 1x32x1x1xf32)
        add__39 = paddle._C_ops.add_(conv2d_32, reshape_66)

        # pd_op.leaky_relu_: (1x32x192x320xf32) <- (1x32x192x320xf32)
        leaky_relu__26 = paddle._C_ops.leaky_relu_(add__39, float('0.1'))

        # pd_op.conv2d: (1x32x192x320xf32) <- (1x32x192x320xf32, 32x32x3x3xf32)
        conv2d_33 = paddle._C_ops.conv2d(leaky_relu__26, parameter_68, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_62 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_68, reshape_69 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_69, full_int_array_62), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x192x320xf32) <- (1x32x192x320xf32, 1x32x1x1xf32)
        add__40 = paddle._C_ops.add_(conv2d_33, reshape_68)

        # pd_op.leaky_relu_: (1x32x192x320xf32) <- (1x32x192x320xf32)
        leaky_relu__27 = paddle._C_ops.leaky_relu_(add__40, float('0.1'))

        # pd_op.conv2d: (1x32x192x320xf32) <- (1x32x192x320xf32, 32x32x3x3xf32)
        conv2d_34 = paddle._C_ops.conv2d(leaky_relu__27, parameter_70, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_63 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_70, reshape_71 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_71, full_int_array_63), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x192x320xf32) <- (1x32x192x320xf32, 1x32x1x1xf32)
        add__41 = paddle._C_ops.add_(conv2d_34, reshape_70)

        # pd_op.leaky_relu_: (1x32x192x320xf32) <- (1x32x192x320xf32)
        leaky_relu__28 = paddle._C_ops.leaky_relu_(add__41, float('0.1'))

        # pd_op.conv2d: (1x16x192x320xf32) <- (1x32x192x320xf32, 16x32x3x3xf32)
        conv2d_35 = paddle._C_ops.conv2d(leaky_relu__28, parameter_72, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_64 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_72, reshape_73 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_73, full_int_array_64), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x192x320xf32) <- (1x16x192x320xf32, 1x16x1x1xf32)
        add__42 = paddle._C_ops.add_(conv2d_35, reshape_72)

        # pd_op.leaky_relu_: (1x16x192x320xf32) <- (1x16x192x320xf32)
        leaky_relu__29 = paddle._C_ops.leaky_relu_(add__42, float('0.1'))

        # pd_op.conv2d: (1x16x192x320xf32) <- (1x16x192x320xf32, 16x16x3x3xf32)
        conv2d_36 = paddle._C_ops.conv2d(leaky_relu__29, parameter_74, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_65 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_74, reshape_75 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_75, full_int_array_65), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x192x320xf32) <- (1x16x192x320xf32, 1x16x1x1xf32)
        add__43 = paddle._C_ops.add_(conv2d_36, reshape_74)

        # pd_op.leaky_relu_: (1x16x192x320xf32) <- (1x16x192x320xf32)
        leaky_relu__30 = paddle._C_ops.leaky_relu_(add__43, float('0.1'))

        # pd_op.conv2d: (1x16x192x320xf32) <- (1x16x192x320xf32, 16x16x3x3xf32)
        conv2d_37 = paddle._C_ops.conv2d(leaky_relu__30, parameter_76, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_66 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf32, 0x16xf32) <- (16xf32, 4xi64)
        reshape_76, reshape_77 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_77, full_int_array_66), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x16x192x320xf32) <- (1x16x192x320xf32, 1x16x1x1xf32)
        add__44 = paddle._C_ops.add_(conv2d_37, reshape_76)

        # pd_op.leaky_relu_: (1x16x192x320xf32) <- (1x16x192x320xf32)
        leaky_relu__31 = paddle._C_ops.leaky_relu_(add__44, float('0.1'))

        # pd_op.conv2d: (1x8x192x320xf32) <- (1x16x192x320xf32, 8x16x3x3xf32)
        conv2d_38 = paddle._C_ops.conv2d(leaky_relu__31, parameter_78, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_67 = [1, 8, 1, 1]

        # pd_op.reshape: (1x8x1x1xf32, 0x8xf32) <- (8xf32, 4xi64)
        reshape_78, reshape_79 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_79, full_int_array_67), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x8x192x320xf32) <- (1x8x192x320xf32, 1x8x1x1xf32)
        add__45 = paddle._C_ops.add_(conv2d_38, reshape_78)

        # pd_op.leaky_relu_: (1x8x192x320xf32) <- (1x8x192x320xf32)
        leaky_relu__32 = paddle._C_ops.leaky_relu_(add__45, float('0.1'))

        # pd_op.conv2d: (1x8x192x320xf32) <- (1x8x192x320xf32, 8x8x3x3xf32)
        conv2d_39 = paddle._C_ops.conv2d(leaky_relu__32, parameter_80, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_68 = [1, 8, 1, 1]

        # pd_op.reshape: (1x8x1x1xf32, 0x8xf32) <- (8xf32, 4xi64)
        reshape_80, reshape_81 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_81, full_int_array_68), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x8x192x320xf32) <- (1x8x192x320xf32, 1x8x1x1xf32)
        add__46 = paddle._C_ops.add_(conv2d_39, reshape_80)

        # pd_op.leaky_relu_: (1x8x192x320xf32) <- (1x8x192x320xf32)
        leaky_relu__33 = paddle._C_ops.leaky_relu_(add__46, float('0.1'))

        # pd_op.conv2d: (1x2x192x320xf32) <- (1x8x192x320xf32, 2x8x3x3xf32)
        conv2d_40 = paddle._C_ops.conv2d(leaky_relu__33, parameter_82, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_69 = [1, 2, 1, 1]

        # pd_op.reshape: (1x2x1x1xf32, 0x2xf32) <- (2xf32, 4xi64)
        reshape_82, reshape_83 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_83, full_int_array_69), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x2x192x320xf32) <- (1x2x192x320xf32, 1x2x1x1xf32)
        add__47 = paddle._C_ops.add_(conv2d_40, reshape_82)

        # pd_op.add_: (1x2x192x320xf32) <- (1x2x192x320xf32, 1x2x192x320xf32)
        add__48 = paddle._C_ops.add_(scale__15, add__47)

        # pd_op.bilinear_interp: (1x2x180x320xf32) <- (1x2x192x320xf32, None, None, None)
        bilinear_interp_4 = paddle._C_ops.bilinear_interp(add__48, None, None, None, 'NCHW', -1, 180, 320, [], 'bilinear', False, 0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_70 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_71 = [1]

        # pd_op.slice: (1x180x320xf32) <- (1x2x180x320xf32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(bilinear_interp_4, [1], full_int_array_70, full_int_array_71, [1], [1])

        # pd_op.full: (1xf32) <- ()
        full_47 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x180x320xf32) <- (1x180x320xf32, 1xf32)
        scale__22 = paddle._C_ops.scale_(slice_18, full_47, float('0'), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_72 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_73 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_74 = [1]

        # pd_op.set_value_with_tensor_: (1x2x180x320xf32) <- (1x2x180x320xf32, 1x180x320xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__0 = paddle._C_ops.set_value_with_tensor_(bilinear_interp_4, scale__22, full_int_array_72, full_int_array_73, full_int_array_74, [1], [1], [])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_75 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_76 = [2]

        # pd_op.slice: (1x180x320xf32) <- (1x2x180x320xf32, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(set_value_with_tensor__0, [1], full_int_array_75, full_int_array_76, [1], [1])

        # pd_op.full: (1xf32) <- ()
        full_48 = paddle._C_ops.full([1], float('0.9375'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x180x320xf32) <- (1x180x320xf32, 1xf32)
        scale__23 = paddle._C_ops.scale_(slice_19, full_48, float('0'), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_77 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_78 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_79 = [1]

        # pd_op.set_value_with_tensor_: (1x2x180x320xf32) <- (1x2x180x320xf32, 1x180x320xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__1 = paddle._C_ops.set_value_with_tensor_(set_value_with_tensor__0, scale__23, full_int_array_77, full_int_array_78, full_int_array_79, [1], [1], [])

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_80 = [1, 1, 2, 180, 320]

        # pd_op.reshape_: (1x1x2x180x320xf32, 0x1x2x180x320xf32) <- (1x2x180x320xf32, 5xi64)
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape_(set_value_with_tensor__1, full_int_array_80), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.if: (1x1x2x180x320xf32) <- (xb)
        if select_input_0:
            if_0, = self.pd_op_if_2867_0_0(reshape__6)
        else:
            if_0, = self.pd_op_if_2867_1_0()

        # pd_op.logical_not: (xb) <- (xb)
        logical_not_0 = paddle._C_ops.logical_not(select_input_0)

        # pd_op.if: (1x1x2x180x320xf32) <- (xb)
        if logical_not_0:
            if_1, = self.pd_op_if_2875_0_0(reshape__4, reshape__2, parameter_10, parameter_11, parameter_12, parameter_13, parameter_14, parameter_15, parameter_16, parameter_17, parameter_18, parameter_19, parameter_20, parameter_21, parameter_22, parameter_23, parameter_24, parameter_25, parameter_26, parameter_27, parameter_28, parameter_29, parameter_30, parameter_31, parameter_32, parameter_33, parameter_34, parameter_35, parameter_36, parameter_37, parameter_38, parameter_39, parameter_40, parameter_41, parameter_42, parameter_43, parameter_44, parameter_45, parameter_46, parameter_47, parameter_48, parameter_49, parameter_50, parameter_51, parameter_52, parameter_53, parameter_54, parameter_55, parameter_56, parameter_57, parameter_58, parameter_59, parameter_60, parameter_61, parameter_62, parameter_63, parameter_64, parameter_65, parameter_66, parameter_67, parameter_68, parameter_69, parameter_70, parameter_71, parameter_72, parameter_73, parameter_74, parameter_75, parameter_76, parameter_77, parameter_78, parameter_79, parameter_80, parameter_81, parameter_82, parameter_83)
        else:
            if_1, = self.pd_op_if_2875_1_0()

        # pd_op.cast: (xi32) <- (xb)
        cast_4 = paddle._C_ops.cast(select_input_0, paddle.int32)

        # pd_op.select_input: (1x1x2x180x320xf32) <- (xi32, 1x1x2x180x320xf32, 1x1x2x180x320xf32)
        select_input_1 = (if_1 if cast_4 == 0 else if_0)

        # pd_op.full: (1x32x180x320xf32) <- ()
        full_49 = paddle._C_ops.full([1, 32, 180, 320], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_81 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_82 = [1]

        # pd_op.slice: (1x2x180x320xf32) <- (1x1x2x180x320xf32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(select_input_1, [1], full_int_array_81, full_int_array_82, [1], [1])

        # pd_op.transpose: (1x180x320x2xf32) <- (1x2x180x320xf32)
        transpose_3 = paddle._C_ops.transpose(slice_20, [0, 2, 3, 1])

        # pd_op.full: (1xi64) <- ()
        full_50 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_51 = paddle._C_ops.full([1], float('180'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_52 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (180xi64) <- (1xi64, 1xi64, 1xi64)
        arange_6 = paddle.arange(full_50, full_51, full_52, dtype='int64')

        # pd_op.full: (1xi64) <- ()
        full_53 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_54 = paddle._C_ops.full([1], float('320'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_55 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (320xi64) <- (1xi64, 1xi64, 1xi64)
        arange_7 = paddle.arange(full_53, full_54, full_55, dtype='int64')

        # builtin.combine: ([180xi64, 320xi64]) <- (180xi64, 320xi64)
        combine_12 = [arange_6, arange_7]

        # pd_op.meshgrid: ([180x320xi64, 180x320xi64]) <- ([180xi64, 320xi64])
        meshgrid_3 = paddle._C_ops.meshgrid(combine_12)

        # builtin.slice: (180x320xi64) <- ([180x320xi64, 180x320xi64])
        slice_21 = meshgrid_3[1]

        # builtin.slice: (180x320xi64) <- ([180x320xi64, 180x320xi64])
        slice_22 = meshgrid_3[0]

        # builtin.combine: ([180x320xi64, 180x320xi64]) <- (180x320xi64, 180x320xi64)
        combine_13 = [slice_21, slice_22]

        # pd_op.stack: (180x320x2xi64) <- ([180x320xi64, 180x320xi64])
        stack_6 = paddle._C_ops.stack(combine_13, 2)

        # pd_op.cast: (180x320x2xf32) <- (180x320x2xi64)
        cast_5 = paddle._C_ops.cast(stack_6, paddle.float32)

        # pd_op.add_: (1x180x320x2xf32) <- (180x320x2xf32, 1x180x320x2xf32)
        add__49 = paddle._C_ops.add_(cast_5, transpose_3)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_83 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_84 = [1]

        # pd_op.slice: (1x180x320xf32) <- (1x180x320x2xf32, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(add__49, [3], full_int_array_83, full_int_array_84, [1], [3])

        # pd_op.full: (1xf32) <- ()
        full_56 = paddle._C_ops.full([1], float('2'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x180x320xf32) <- (1x180x320xf32, 1xf32)
        scale__24 = paddle._C_ops.scale_(slice_23, full_56, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_57 = paddle._C_ops.full([1], float('0.0031348'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x180x320xf32) <- (1x180x320xf32, 1xf32)
        scale__25 = paddle._C_ops.scale_(scale__24, full_57, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_58 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x180x320xf32) <- (1x180x320xf32, 1xf32)
        scale__26 = paddle._C_ops.scale_(scale__25, full_58, float('-1'), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_85 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_86 = [2]

        # pd_op.slice: (1x180x320xf32) <- (1x180x320x2xf32, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(add__49, [3], full_int_array_85, full_int_array_86, [1], [3])

        # pd_op.full: (1xf32) <- ()
        full_59 = paddle._C_ops.full([1], float('2'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x180x320xf32) <- (1x180x320xf32, 1xf32)
        scale__27 = paddle._C_ops.scale_(slice_24, full_59, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_60 = paddle._C_ops.full([1], float('0.00558659'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x180x320xf32) <- (1x180x320xf32, 1xf32)
        scale__28 = paddle._C_ops.scale_(scale__27, full_60, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_61 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x180x320xf32) <- (1x180x320xf32, 1xf32)
        scale__29 = paddle._C_ops.scale_(scale__28, full_61, float('-1'), True)

        # builtin.combine: ([1x180x320xf32, 1x180x320xf32]) <- (1x180x320xf32, 1x180x320xf32)
        combine_14 = [scale__26, scale__29]

        # pd_op.stack: (1x180x320x2xf32) <- ([1x180x320xf32, 1x180x320xf32])
        stack_7 = paddle._C_ops.stack(combine_14, 3)

        # pd_op.grid_sample: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x180x320x2xf32)
        grid_sample_3 = paddle._C_ops.grid_sample(slice_2, stack_7, 'bilinear', 'zeros', True)

        # builtin.combine: ([1x32x180x320xf32, 1x32x180x320xf32]) <- (1x32x180x320xf32, 1x32x180x320xf32)
        combine_15 = [grid_sample_3, slice_3]

        # pd_op.full: (1xi32) <- ()
        full_62 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x64x180x320xf32) <- ([1x32x180x320xf32, 1x32x180x320xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_15, full_62)

        # builtin.combine: ([1x64x180x320xf32, 1x2x180x320xf32]) <- (1x64x180x320xf32, 1x2x180x320xf32)
        combine_16 = [concat_3, slice_20]

        # pd_op.full: (1xi32) <- ()
        full_63 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x66x180x320xf32) <- ([1x64x180x320xf32, 1x2x180x320xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_16, full_63)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x66x180x320xf32, 32x66x3x3xf32)
        conv2d_41 = paddle._C_ops.conv2d(concat_4, parameter_84, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_87 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_84, reshape_85 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_85, full_int_array_87), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__50 = paddle._C_ops.add_(conv2d_41, reshape_84)

        # pd_op.leaky_relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        leaky_relu__34 = paddle._C_ops.leaky_relu_(add__50, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_42 = paddle._C_ops.conv2d(leaky_relu__34, parameter_86, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_88 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_86, reshape_87 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_87, full_int_array_88), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__51 = paddle._C_ops.add_(conv2d_42, reshape_86)

        # pd_op.leaky_relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        leaky_relu__35 = paddle._C_ops.leaky_relu_(add__51, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_43 = paddle._C_ops.conv2d(leaky_relu__35, parameter_88, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_89 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_88, reshape_89 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_89, full_int_array_89), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__52 = paddle._C_ops.add_(conv2d_43, reshape_88)

        # pd_op.leaky_relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        leaky_relu__36 = paddle._C_ops.leaky_relu_(add__52, float('0.1'))

        # pd_op.conv2d: (1x216x180x320xf32) <- (1x32x180x320xf32, 216x32x3x3xf32)
        conv2d_44 = paddle._C_ops.conv2d(leaky_relu__36, parameter_90, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_90 = [1, 216, 1, 1]

        # pd_op.reshape: (1x216x1x1xf32, 0x216xf32) <- (216xf32, 4xi64)
        reshape_90, reshape_91 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_91, full_int_array_90), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x216x180x320xf32) <- (1x216x180x320xf32, 1x216x1x1xf32)
        add__53 = paddle._C_ops.add_(conv2d_44, reshape_90)

        # pd_op.full: (1xi32) <- ()
        full_64 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([1x72x180x320xf32, 1x72x180x320xf32, 1x72x180x320xf32]) <- (1x216x180x320xf32, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(add__53, 3, full_64)

        # builtin.slice: (1x72x180x320xf32) <- ([1x72x180x320xf32, 1x72x180x320xf32, 1x72x180x320xf32])
        slice_25 = split_with_num_1[0]

        # builtin.slice: (1x72x180x320xf32) <- ([1x72x180x320xf32, 1x72x180x320xf32, 1x72x180x320xf32])
        slice_26 = split_with_num_1[1]

        # builtin.combine: ([1x72x180x320xf32, 1x72x180x320xf32]) <- (1x72x180x320xf32, 1x72x180x320xf32)
        combine_17 = [slice_25, slice_26]

        # pd_op.full: (1xi32) <- ()
        full_65 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x144x180x320xf32) <- ([1x72x180x320xf32, 1x72x180x320xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_17, full_65)

        # pd_op.tanh_: (1x144x180x320xf32) <- (1x144x180x320xf32)
        tanh__0 = paddle._C_ops.tanh_(concat_5)

        # pd_op.full: (1xf32) <- ()
        full_66 = paddle._C_ops.full([1], float('10'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x144x180x320xf32) <- (1x144x180x320xf32, 1xf32)
        scale__30 = paddle._C_ops.scale_(tanh__0, full_66, float('0'), True)

        # pd_op.flip: (1x2x180x320xf32) <- (1x2x180x320xf32)
        flip_1 = paddle._C_ops.flip(slice_20, [1])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_91 = [1, 72, 1, 1]

        # pd_op.tile: (1x144x180x320xf32) <- (1x2x180x320xf32, 4xi64)
        tile_0 = paddle._C_ops.tile(flip_1, full_int_array_91)

        # pd_op.add_: (1x144x180x320xf32) <- (1x144x180x320xf32, 1x144x180x320xf32)
        add__54 = paddle._C_ops.add_(scale__30, tile_0)

        # builtin.slice: (1x72x180x320xf32) <- ([1x72x180x320xf32, 1x72x180x320xf32, 1x72x180x320xf32])
        slice_27 = split_with_num_1[2]

        # pd_op.sigmoid_: (1x72x180x320xf32) <- (1x72x180x320xf32)
        sigmoid__0 = paddle._C_ops.sigmoid_(slice_27)

        # pd_op.deformable_conv: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x144x180x320xf32, 32x32x3x3xf32, 1x72x180x320xf32)
        deformable_conv_0 = paddle._C_ops.deformable_conv(slice_2, add__54, parameter_92, sigmoid__0, [1, 1], [1, 1], [1, 1], 8, 1, 1)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_92 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xi64) <- (32xf32, 4xi64)
        reshape_92, reshape_93 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_93, full_int_array_92), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__55 = paddle._C_ops.add_(deformable_conv_0, reshape_92)

        # builtin.combine: ([1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32]) <- (1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32)
        combine_18 = [slice_3, full_49, add__55]

        # pd_op.full: (1xi32) <- ()
        full_67 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x96x180x320xf32) <- ([1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_18, full_67)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x96x180x320xf32, 32x96x3x3xf32)
        conv2d_45 = paddle._C_ops.conv2d(concat_6, parameter_94, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_93 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_94, reshape_95 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_95, full_int_array_93), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__56 = paddle._C_ops.add_(conv2d_45, reshape_94)

        # pd_op.leaky_relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        leaky_relu__37 = paddle._C_ops.leaky_relu_(add__56, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_46 = paddle._C_ops.conv2d(leaky_relu__37, parameter_96, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_94 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_96, reshape_97 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_97, full_int_array_94), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__57 = paddle._C_ops.add_(conv2d_46, reshape_96)

        # pd_op.relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        relu__2 = paddle._C_ops.relu_(add__57)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_47 = paddle._C_ops.conv2d(relu__2, parameter_98, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_95 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_98, reshape_99 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_99, full_int_array_95), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__58 = paddle._C_ops.add_(conv2d_47, reshape_98)

        # pd_op.full: (1xf32) <- ()
        full_68 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1xf32)
        scale__31 = paddle._C_ops.scale_(add__58, full_68, float('0'), True)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__59 = paddle._C_ops.add_(leaky_relu__37, scale__31)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_48 = paddle._C_ops.conv2d(add__59, parameter_100, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_96 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_100, reshape_101 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_101, full_int_array_96), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__60 = paddle._C_ops.add_(conv2d_48, reshape_100)

        # pd_op.relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        relu__3 = paddle._C_ops.relu_(add__60)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_49 = paddle._C_ops.conv2d(relu__3, parameter_102, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_97 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_102, reshape_103 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_103, full_int_array_97), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__61 = paddle._C_ops.add_(conv2d_49, reshape_102)

        # pd_op.full: (1xf32) <- ()
        full_69 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1xf32)
        scale__32 = paddle._C_ops.scale_(add__61, full_69, float('0'), True)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__62 = paddle._C_ops.add_(add__59, scale__32)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_50 = paddle._C_ops.conv2d(add__62, parameter_104, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_98 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_104, reshape_105 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_105, full_int_array_98), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__63 = paddle._C_ops.add_(conv2d_50, reshape_104)

        # pd_op.relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        relu__4 = paddle._C_ops.relu_(add__63)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_51 = paddle._C_ops.conv2d(relu__4, parameter_106, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_99 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_106, reshape_107 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_107, full_int_array_99), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__64 = paddle._C_ops.add_(conv2d_51, reshape_106)

        # pd_op.full: (1xf32) <- ()
        full_70 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1xf32)
        scale__33 = paddle._C_ops.scale_(add__64, full_70, float('0'), True)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__65 = paddle._C_ops.add_(add__62, scale__33)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_100 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_101 = [1]

        # pd_op.slice: (1x2x180x320xf32) <- (1x1x2x180x320xf32, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(reshape__6, [1], full_int_array_100, full_int_array_101, [1], [1])

        # pd_op.transpose: (1x180x320x2xf32) <- (1x2x180x320xf32)
        transpose_4 = paddle._C_ops.transpose(slice_28, [0, 2, 3, 1])

        # pd_op.full: (1xi64) <- ()
        full_71 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_72 = paddle._C_ops.full([1], float('180'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_73 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (180xi64) <- (1xi64, 1xi64, 1xi64)
        arange_8 = paddle.arange(full_71, full_72, full_73, dtype='int64')

        # pd_op.full: (1xi64) <- ()
        full_74 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_75 = paddle._C_ops.full([1], float('320'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_76 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (320xi64) <- (1xi64, 1xi64, 1xi64)
        arange_9 = paddle.arange(full_74, full_75, full_76, dtype='int64')

        # builtin.combine: ([180xi64, 320xi64]) <- (180xi64, 320xi64)
        combine_19 = [arange_8, arange_9]

        # pd_op.meshgrid: ([180x320xi64, 180x320xi64]) <- ([180xi64, 320xi64])
        meshgrid_4 = paddle._C_ops.meshgrid(combine_19)

        # builtin.slice: (180x320xi64) <- ([180x320xi64, 180x320xi64])
        slice_29 = meshgrid_4[1]

        # builtin.slice: (180x320xi64) <- ([180x320xi64, 180x320xi64])
        slice_30 = meshgrid_4[0]

        # builtin.combine: ([180x320xi64, 180x320xi64]) <- (180x320xi64, 180x320xi64)
        combine_20 = [slice_29, slice_30]

        # pd_op.stack: (180x320x2xi64) <- ([180x320xi64, 180x320xi64])
        stack_8 = paddle._C_ops.stack(combine_20, 2)

        # pd_op.cast: (180x320x2xf32) <- (180x320x2xi64)
        cast_6 = paddle._C_ops.cast(stack_8, paddle.float32)

        # pd_op.add_: (1x180x320x2xf32) <- (180x320x2xf32, 1x180x320x2xf32)
        add__66 = paddle._C_ops.add_(cast_6, transpose_4)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_102 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_103 = [1]

        # pd_op.slice: (1x180x320xf32) <- (1x180x320x2xf32, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(add__66, [3], full_int_array_102, full_int_array_103, [1], [3])

        # pd_op.full: (1xf32) <- ()
        full_77 = paddle._C_ops.full([1], float('2'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x180x320xf32) <- (1x180x320xf32, 1xf32)
        scale__34 = paddle._C_ops.scale_(slice_31, full_77, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_78 = paddle._C_ops.full([1], float('0.0031348'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x180x320xf32) <- (1x180x320xf32, 1xf32)
        scale__35 = paddle._C_ops.scale_(scale__34, full_78, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_79 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x180x320xf32) <- (1x180x320xf32, 1xf32)
        scale__36 = paddle._C_ops.scale_(scale__35, full_79, float('-1'), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_104 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_105 = [2]

        # pd_op.slice: (1x180x320xf32) <- (1x180x320x2xf32, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(add__66, [3], full_int_array_104, full_int_array_105, [1], [3])

        # pd_op.full: (1xf32) <- ()
        full_80 = paddle._C_ops.full([1], float('2'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x180x320xf32) <- (1x180x320xf32, 1xf32)
        scale__37 = paddle._C_ops.scale_(slice_32, full_80, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_81 = paddle._C_ops.full([1], float('0.00558659'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x180x320xf32) <- (1x180x320xf32, 1xf32)
        scale__38 = paddle._C_ops.scale_(scale__37, full_81, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_82 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x180x320xf32) <- (1x180x320xf32, 1xf32)
        scale__39 = paddle._C_ops.scale_(scale__38, full_82, float('-1'), True)

        # builtin.combine: ([1x180x320xf32, 1x180x320xf32]) <- (1x180x320xf32, 1x180x320xf32)
        combine_21 = [scale__36, scale__39]

        # pd_op.stack: (1x180x320x2xf32) <- ([1x180x320xf32, 1x180x320xf32])
        stack_9 = paddle._C_ops.stack(combine_21, 3)

        # pd_op.grid_sample: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x180x320x2xf32)
        grid_sample_4 = paddle._C_ops.grid_sample(slice_3, stack_9, 'bilinear', 'zeros', True)

        # builtin.combine: ([1x32x180x320xf32, 1x32x180x320xf32]) <- (1x32x180x320xf32, 1x32x180x320xf32)
        combine_22 = [grid_sample_4, slice_2]

        # pd_op.full: (1xi32) <- ()
        full_83 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x64x180x320xf32) <- ([1x32x180x320xf32, 1x32x180x320xf32], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_22, full_83)

        # builtin.combine: ([1x64x180x320xf32, 1x2x180x320xf32]) <- (1x64x180x320xf32, 1x2x180x320xf32)
        combine_23 = [concat_7, slice_28]

        # pd_op.full: (1xi32) <- ()
        full_84 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x66x180x320xf32) <- ([1x64x180x320xf32, 1x2x180x320xf32], 1xi32)
        concat_8 = paddle._C_ops.concat(combine_23, full_84)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x66x180x320xf32, 32x66x3x3xf32)
        conv2d_52 = paddle._C_ops.conv2d(concat_8, parameter_84, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_106 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_108, reshape_109 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_85, full_int_array_106), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__67 = paddle._C_ops.add_(conv2d_52, reshape_108)

        # pd_op.leaky_relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        leaky_relu__38 = paddle._C_ops.leaky_relu_(add__67, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_53 = paddle._C_ops.conv2d(leaky_relu__38, parameter_86, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_107 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_110, reshape_111 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_87, full_int_array_107), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__68 = paddle._C_ops.add_(conv2d_53, reshape_110)

        # pd_op.leaky_relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        leaky_relu__39 = paddle._C_ops.leaky_relu_(add__68, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_54 = paddle._C_ops.conv2d(leaky_relu__39, parameter_88, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_108 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_112, reshape_113 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_89, full_int_array_108), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__69 = paddle._C_ops.add_(conv2d_54, reshape_112)

        # pd_op.leaky_relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        leaky_relu__40 = paddle._C_ops.leaky_relu_(add__69, float('0.1'))

        # pd_op.conv2d: (1x216x180x320xf32) <- (1x32x180x320xf32, 216x32x3x3xf32)
        conv2d_55 = paddle._C_ops.conv2d(leaky_relu__40, parameter_90, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_109 = [1, 216, 1, 1]

        # pd_op.reshape: (1x216x1x1xf32, 0x216xf32) <- (216xf32, 4xi64)
        reshape_114, reshape_115 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_91, full_int_array_109), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x216x180x320xf32) <- (1x216x180x320xf32, 1x216x1x1xf32)
        add__70 = paddle._C_ops.add_(conv2d_55, reshape_114)

        # pd_op.full: (1xi32) <- ()
        full_85 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([1x72x180x320xf32, 1x72x180x320xf32, 1x72x180x320xf32]) <- (1x216x180x320xf32, 1xi32)
        split_with_num_2 = paddle._C_ops.split_with_num(add__70, 3, full_85)

        # builtin.slice: (1x72x180x320xf32) <- ([1x72x180x320xf32, 1x72x180x320xf32, 1x72x180x320xf32])
        slice_33 = split_with_num_2[0]

        # builtin.slice: (1x72x180x320xf32) <- ([1x72x180x320xf32, 1x72x180x320xf32, 1x72x180x320xf32])
        slice_34 = split_with_num_2[1]

        # builtin.combine: ([1x72x180x320xf32, 1x72x180x320xf32]) <- (1x72x180x320xf32, 1x72x180x320xf32)
        combine_24 = [slice_33, slice_34]

        # pd_op.full: (1xi32) <- ()
        full_86 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x144x180x320xf32) <- ([1x72x180x320xf32, 1x72x180x320xf32], 1xi32)
        concat_9 = paddle._C_ops.concat(combine_24, full_86)

        # pd_op.tanh_: (1x144x180x320xf32) <- (1x144x180x320xf32)
        tanh__1 = paddle._C_ops.tanh_(concat_9)

        # pd_op.full: (1xf32) <- ()
        full_87 = paddle._C_ops.full([1], float('10'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x144x180x320xf32) <- (1x144x180x320xf32, 1xf32)
        scale__40 = paddle._C_ops.scale_(tanh__1, full_87, float('0'), True)

        # pd_op.flip: (1x2x180x320xf32) <- (1x2x180x320xf32)
        flip_2 = paddle._C_ops.flip(slice_28, [1])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_110 = [1, 72, 1, 1]

        # pd_op.tile: (1x144x180x320xf32) <- (1x2x180x320xf32, 4xi64)
        tile_1 = paddle._C_ops.tile(flip_2, full_int_array_110)

        # pd_op.add_: (1x144x180x320xf32) <- (1x144x180x320xf32, 1x144x180x320xf32)
        add__71 = paddle._C_ops.add_(scale__40, tile_1)

        # builtin.slice: (1x72x180x320xf32) <- ([1x72x180x320xf32, 1x72x180x320xf32, 1x72x180x320xf32])
        slice_35 = split_with_num_2[2]

        # pd_op.sigmoid_: (1x72x180x320xf32) <- (1x72x180x320xf32)
        sigmoid__1 = paddle._C_ops.sigmoid_(slice_35)

        # pd_op.deformable_conv: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x144x180x320xf32, 32x32x3x3xf32, 1x72x180x320xf32)
        deformable_conv_1 = paddle._C_ops.deformable_conv(slice_3, add__71, parameter_92, sigmoid__1, [1, 1], [1, 1], [1, 1], 8, 1, 1)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_111 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xi64) <- (32xf32, 4xi64)
        reshape_116, reshape_117 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_93, full_int_array_111), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__72 = paddle._C_ops.add_(deformable_conv_1, reshape_116)

        # pd_op.full: (1x32x180x320xf32) <- ()
        full_88 = paddle._C_ops.full([1, 32, 180, 320], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # builtin.combine: ([1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32]) <- (1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32)
        combine_25 = [slice_2, add__72, full_88]

        # pd_op.full: (1xi32) <- ()
        full_89 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x96x180x320xf32) <- ([1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32], 1xi32)
        concat_10 = paddle._C_ops.concat(combine_25, full_89)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x96x180x320xf32, 32x96x3x3xf32)
        conv2d_56 = paddle._C_ops.conv2d(concat_10, parameter_94, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_112 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_118, reshape_119 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_95, full_int_array_112), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__73 = paddle._C_ops.add_(conv2d_56, reshape_118)

        # pd_op.leaky_relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        leaky_relu__41 = paddle._C_ops.leaky_relu_(add__73, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_57 = paddle._C_ops.conv2d(leaky_relu__41, parameter_96, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_113 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_120, reshape_121 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_97, full_int_array_113), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__74 = paddle._C_ops.add_(conv2d_57, reshape_120)

        # pd_op.relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        relu__5 = paddle._C_ops.relu_(add__74)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_58 = paddle._C_ops.conv2d(relu__5, parameter_98, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_114 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_122, reshape_123 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_99, full_int_array_114), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__75 = paddle._C_ops.add_(conv2d_58, reshape_122)

        # pd_op.full: (1xf32) <- ()
        full_90 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1xf32)
        scale__41 = paddle._C_ops.scale_(add__75, full_90, float('0'), True)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__76 = paddle._C_ops.add_(leaky_relu__41, scale__41)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_59 = paddle._C_ops.conv2d(add__76, parameter_100, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_115 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_124, reshape_125 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_101, full_int_array_115), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__77 = paddle._C_ops.add_(conv2d_59, reshape_124)

        # pd_op.relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        relu__6 = paddle._C_ops.relu_(add__77)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_60 = paddle._C_ops.conv2d(relu__6, parameter_102, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_116 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_126, reshape_127 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_103, full_int_array_116), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__78 = paddle._C_ops.add_(conv2d_60, reshape_126)

        # pd_op.full: (1xf32) <- ()
        full_91 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1xf32)
        scale__42 = paddle._C_ops.scale_(add__78, full_91, float('0'), True)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__79 = paddle._C_ops.add_(add__76, scale__42)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_61 = paddle._C_ops.conv2d(add__79, parameter_104, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_117 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_128, reshape_129 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_105, full_int_array_117), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__80 = paddle._C_ops.add_(conv2d_61, reshape_128)

        # pd_op.relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        relu__7 = paddle._C_ops.relu_(add__80)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_62 = paddle._C_ops.conv2d(relu__7, parameter_106, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_118 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_130, reshape_131 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_107, full_int_array_118), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__81 = paddle._C_ops.add_(conv2d_62, reshape_130)

        # pd_op.full: (1xf32) <- ()
        full_92 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1xf32)
        scale__43 = paddle._C_ops.scale_(add__81, full_92, float('0'), True)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__82 = paddle._C_ops.add_(add__79, scale__43)

        # pd_op.full: (1x32x180x320xf32) <- ()
        full_93 = paddle._C_ops.full([1, 32, 180, 320], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # builtin.combine: ([1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32]) <- (1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32)
        combine_26 = [slice_3, add__65, full_93]

        # pd_op.full: (1xi32) <- ()
        full_94 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x96x180x320xf32) <- ([1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32], 1xi32)
        concat_11 = paddle._C_ops.concat(combine_26, full_94)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x96x180x320xf32, 32x96x3x3xf32)
        conv2d_63 = paddle._C_ops.conv2d(concat_11, parameter_108, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_119 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_132, reshape_133 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_109, full_int_array_119), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__83 = paddle._C_ops.add_(conv2d_63, reshape_132)

        # pd_op.leaky_relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        leaky_relu__42 = paddle._C_ops.leaky_relu_(add__83, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_64 = paddle._C_ops.conv2d(leaky_relu__42, parameter_110, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_120 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_134, reshape_135 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_111, full_int_array_120), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__84 = paddle._C_ops.add_(conv2d_64, reshape_134)

        # pd_op.relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        relu__8 = paddle._C_ops.relu_(add__84)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_65 = paddle._C_ops.conv2d(relu__8, parameter_112, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_121 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_136, reshape_137 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_113, full_int_array_121), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__85 = paddle._C_ops.add_(conv2d_65, reshape_136)

        # pd_op.full: (1xf32) <- ()
        full_95 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1xf32)
        scale__44 = paddle._C_ops.scale_(add__85, full_95, float('0'), True)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__86 = paddle._C_ops.add_(leaky_relu__42, scale__44)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_66 = paddle._C_ops.conv2d(add__86, parameter_114, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_122 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_138, reshape_139 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_115, full_int_array_122), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__87 = paddle._C_ops.add_(conv2d_66, reshape_138)

        # pd_op.relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        relu__9 = paddle._C_ops.relu_(add__87)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_67 = paddle._C_ops.conv2d(relu__9, parameter_116, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_123 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_140, reshape_141 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_117, full_int_array_123), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__88 = paddle._C_ops.add_(conv2d_67, reshape_140)

        # pd_op.full: (1xf32) <- ()
        full_96 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1xf32)
        scale__45 = paddle._C_ops.scale_(add__88, full_96, float('0'), True)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__89 = paddle._C_ops.add_(add__86, scale__45)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_68 = paddle._C_ops.conv2d(add__89, parameter_118, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_124 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_142, reshape_143 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_119, full_int_array_124), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__90 = paddle._C_ops.add_(conv2d_68, reshape_142)

        # pd_op.relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        relu__10 = paddle._C_ops.relu_(add__90)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_69 = paddle._C_ops.conv2d(relu__10, parameter_120, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_125 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_144, reshape_145 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_121, full_int_array_125), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__91 = paddle._C_ops.add_(conv2d_69, reshape_144)

        # pd_op.full: (1xf32) <- ()
        full_97 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1xf32)
        scale__46 = paddle._C_ops.scale_(add__91, full_97, float('0'), True)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__92 = paddle._C_ops.add_(add__89, scale__46)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__93 = paddle._C_ops.add_(full_93, add__92)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_126 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_127 = [1]

        # pd_op.slice: (1x2x180x320xf32) <- (1x1x2x180x320xf32, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(reshape__6, [1], full_int_array_126, full_int_array_127, [1], [1])

        # pd_op.transpose: (1x180x320x2xf32) <- (1x2x180x320xf32)
        transpose_5 = paddle._C_ops.transpose(slice_36, [0, 2, 3, 1])

        # pd_op.full: (1xi64) <- ()
        full_98 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_99 = paddle._C_ops.full([1], float('180'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_100 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (180xi64) <- (1xi64, 1xi64, 1xi64)
        arange_10 = paddle.arange(full_98, full_99, full_100, dtype='int64')

        # pd_op.full: (1xi64) <- ()
        full_101 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_102 = paddle._C_ops.full([1], float('320'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_103 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (320xi64) <- (1xi64, 1xi64, 1xi64)
        arange_11 = paddle.arange(full_101, full_102, full_103, dtype='int64')

        # builtin.combine: ([180xi64, 320xi64]) <- (180xi64, 320xi64)
        combine_27 = [arange_10, arange_11]

        # pd_op.meshgrid: ([180x320xi64, 180x320xi64]) <- ([180xi64, 320xi64])
        meshgrid_5 = paddle._C_ops.meshgrid(combine_27)

        # builtin.slice: (180x320xi64) <- ([180x320xi64, 180x320xi64])
        slice_37 = meshgrid_5[1]

        # builtin.slice: (180x320xi64) <- ([180x320xi64, 180x320xi64])
        slice_38 = meshgrid_5[0]

        # builtin.combine: ([180x320xi64, 180x320xi64]) <- (180x320xi64, 180x320xi64)
        combine_28 = [slice_37, slice_38]

        # pd_op.stack: (180x320x2xi64) <- ([180x320xi64, 180x320xi64])
        stack_10 = paddle._C_ops.stack(combine_28, 2)

        # pd_op.cast: (180x320x2xf32) <- (180x320x2xi64)
        cast_7 = paddle._C_ops.cast(stack_10, paddle.float32)

        # pd_op.add_: (1x180x320x2xf32) <- (180x320x2xf32, 1x180x320x2xf32)
        add__94 = paddle._C_ops.add_(cast_7, transpose_5)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_128 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_129 = [1]

        # pd_op.slice: (1x180x320xf32) <- (1x180x320x2xf32, 1xi64, 1xi64)
        slice_39 = paddle._C_ops.slice(add__94, [3], full_int_array_128, full_int_array_129, [1], [3])

        # pd_op.full: (1xf32) <- ()
        full_104 = paddle._C_ops.full([1], float('2'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x180x320xf32) <- (1x180x320xf32, 1xf32)
        scale__47 = paddle._C_ops.scale_(slice_39, full_104, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_105 = paddle._C_ops.full([1], float('0.0031348'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x180x320xf32) <- (1x180x320xf32, 1xf32)
        scale__48 = paddle._C_ops.scale_(scale__47, full_105, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_106 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x180x320xf32) <- (1x180x320xf32, 1xf32)
        scale__49 = paddle._C_ops.scale_(scale__48, full_106, float('-1'), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_130 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_131 = [2]

        # pd_op.slice: (1x180x320xf32) <- (1x180x320x2xf32, 1xi64, 1xi64)
        slice_40 = paddle._C_ops.slice(add__94, [3], full_int_array_130, full_int_array_131, [1], [3])

        # pd_op.full: (1xf32) <- ()
        full_107 = paddle._C_ops.full([1], float('2'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x180x320xf32) <- (1x180x320xf32, 1xf32)
        scale__50 = paddle._C_ops.scale_(slice_40, full_107, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_108 = paddle._C_ops.full([1], float('0.00558659'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x180x320xf32) <- (1x180x320xf32, 1xf32)
        scale__51 = paddle._C_ops.scale_(scale__50, full_108, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_109 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x180x320xf32) <- (1x180x320xf32, 1xf32)
        scale__52 = paddle._C_ops.scale_(scale__51, full_109, float('-1'), True)

        # builtin.combine: ([1x180x320xf32, 1x180x320xf32]) <- (1x180x320xf32, 1x180x320xf32)
        combine_29 = [scale__49, scale__52]

        # pd_op.stack: (1x180x320x2xf32) <- ([1x180x320xf32, 1x180x320xf32])
        stack_11 = paddle._C_ops.stack(combine_29, 3)

        # pd_op.grid_sample: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x180x320x2xf32)
        grid_sample_5 = paddle._C_ops.grid_sample(add__93, stack_11, 'bilinear', 'zeros', True)

        # builtin.combine: ([1x32x180x320xf32, 1x32x180x320xf32]) <- (1x32x180x320xf32, 1x32x180x320xf32)
        combine_30 = [grid_sample_5, slice_2]

        # pd_op.full: (1xi32) <- ()
        full_110 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x64x180x320xf32) <- ([1x32x180x320xf32, 1x32x180x320xf32], 1xi32)
        concat_12 = paddle._C_ops.concat(combine_30, full_110)

        # builtin.combine: ([1x64x180x320xf32, 1x2x180x320xf32]) <- (1x64x180x320xf32, 1x2x180x320xf32)
        combine_31 = [concat_12, slice_36]

        # pd_op.full: (1xi32) <- ()
        full_111 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x66x180x320xf32) <- ([1x64x180x320xf32, 1x2x180x320xf32], 1xi32)
        concat_13 = paddle._C_ops.concat(combine_31, full_111)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x66x180x320xf32, 32x66x3x3xf32)
        conv2d_70 = paddle._C_ops.conv2d(concat_13, parameter_122, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_132 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_146, reshape_147 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_123, full_int_array_132), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__95 = paddle._C_ops.add_(conv2d_70, reshape_146)

        # pd_op.leaky_relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        leaky_relu__43 = paddle._C_ops.leaky_relu_(add__95, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_71 = paddle._C_ops.conv2d(leaky_relu__43, parameter_124, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_133 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_148, reshape_149 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_125, full_int_array_133), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__96 = paddle._C_ops.add_(conv2d_71, reshape_148)

        # pd_op.leaky_relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        leaky_relu__44 = paddle._C_ops.leaky_relu_(add__96, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_72 = paddle._C_ops.conv2d(leaky_relu__44, parameter_126, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_134 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_150, reshape_151 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_127, full_int_array_134), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__97 = paddle._C_ops.add_(conv2d_72, reshape_150)

        # pd_op.leaky_relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        leaky_relu__45 = paddle._C_ops.leaky_relu_(add__97, float('0.1'))

        # pd_op.conv2d: (1x108x180x320xf32) <- (1x32x180x320xf32, 108x32x3x3xf32)
        conv2d_73 = paddle._C_ops.conv2d(leaky_relu__45, parameter_128, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_135 = [1, 108, 1, 1]

        # pd_op.reshape: (1x108x1x1xf32, 0x108xf32) <- (108xf32, 4xi64)
        reshape_152, reshape_153 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_129, full_int_array_135), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x108x180x320xf32) <- (1x108x180x320xf32, 1x108x1x1xf32)
        add__98 = paddle._C_ops.add_(conv2d_73, reshape_152)

        # pd_op.full: (1xi32) <- ()
        full_112 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([1x36x180x320xf32, 1x36x180x320xf32, 1x36x180x320xf32]) <- (1x108x180x320xf32, 1xi32)
        split_with_num_3 = paddle._C_ops.split_with_num(add__98, 3, full_112)

        # builtin.slice: (1x36x180x320xf32) <- ([1x36x180x320xf32, 1x36x180x320xf32, 1x36x180x320xf32])
        slice_41 = split_with_num_3[0]

        # builtin.slice: (1x36x180x320xf32) <- ([1x36x180x320xf32, 1x36x180x320xf32, 1x36x180x320xf32])
        slice_42 = split_with_num_3[1]

        # builtin.combine: ([1x36x180x320xf32, 1x36x180x320xf32]) <- (1x36x180x320xf32, 1x36x180x320xf32)
        combine_32 = [slice_41, slice_42]

        # pd_op.full: (1xi32) <- ()
        full_113 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x72x180x320xf32) <- ([1x36x180x320xf32, 1x36x180x320xf32], 1xi32)
        concat_14 = paddle._C_ops.concat(combine_32, full_113)

        # pd_op.tanh_: (1x72x180x320xf32) <- (1x72x180x320xf32)
        tanh__2 = paddle._C_ops.tanh_(concat_14)

        # pd_op.full: (1xf32) <- ()
        full_114 = paddle._C_ops.full([1], float('10'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x72x180x320xf32) <- (1x72x180x320xf32, 1xf32)
        scale__53 = paddle._C_ops.scale_(tanh__2, full_114, float('0'), True)

        # pd_op.flip: (1x2x180x320xf32) <- (1x2x180x320xf32)
        flip_3 = paddle._C_ops.flip(slice_36, [1])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_136 = [1, 36, 1, 1]

        # pd_op.tile: (1x72x180x320xf32) <- (1x2x180x320xf32, 4xi64)
        tile_2 = paddle._C_ops.tile(flip_3, full_int_array_136)

        # pd_op.add_: (1x72x180x320xf32) <- (1x72x180x320xf32, 1x72x180x320xf32)
        add__99 = paddle._C_ops.add_(scale__53, tile_2)

        # builtin.slice: (1x36x180x320xf32) <- ([1x36x180x320xf32, 1x36x180x320xf32, 1x36x180x320xf32])
        slice_43 = split_with_num_3[2]

        # pd_op.sigmoid_: (1x36x180x320xf32) <- (1x36x180x320xf32)
        sigmoid__2 = paddle._C_ops.sigmoid_(slice_43)

        # pd_op.deformable_conv: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x72x180x320xf32, 32x32x3x3xf32, 1x36x180x320xf32)
        deformable_conv_2 = paddle._C_ops.deformable_conv(add__93, add__99, parameter_130, sigmoid__2, [1, 1], [1, 1], [1, 1], 4, 1, 1)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_137 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xi64) <- (32xf32, 4xi64)
        reshape_154, reshape_155 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_131, full_int_array_137), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__100 = paddle._C_ops.add_(deformable_conv_2, reshape_154)

        # builtin.combine: ([1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32]) <- (1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32)
        combine_33 = [slice_2, add__82, add__100]

        # pd_op.full: (1xi32) <- ()
        full_115 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x96x180x320xf32) <- ([1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32], 1xi32)
        concat_15 = paddle._C_ops.concat(combine_33, full_115)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x96x180x320xf32, 32x96x3x3xf32)
        conv2d_74 = paddle._C_ops.conv2d(concat_15, parameter_108, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_138 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_156, reshape_157 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_109, full_int_array_138), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__101 = paddle._C_ops.add_(conv2d_74, reshape_156)

        # pd_op.leaky_relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        leaky_relu__46 = paddle._C_ops.leaky_relu_(add__101, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_75 = paddle._C_ops.conv2d(leaky_relu__46, parameter_110, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_139 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_158, reshape_159 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_111, full_int_array_139), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__102 = paddle._C_ops.add_(conv2d_75, reshape_158)

        # pd_op.relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        relu__11 = paddle._C_ops.relu_(add__102)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_76 = paddle._C_ops.conv2d(relu__11, parameter_112, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_140 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_160, reshape_161 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_113, full_int_array_140), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__103 = paddle._C_ops.add_(conv2d_76, reshape_160)

        # pd_op.full: (1xf32) <- ()
        full_116 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1xf32)
        scale__54 = paddle._C_ops.scale_(add__103, full_116, float('0'), True)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__104 = paddle._C_ops.add_(leaky_relu__46, scale__54)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_77 = paddle._C_ops.conv2d(add__104, parameter_114, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_141 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_162, reshape_163 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_115, full_int_array_141), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__105 = paddle._C_ops.add_(conv2d_77, reshape_162)

        # pd_op.relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        relu__12 = paddle._C_ops.relu_(add__105)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_78 = paddle._C_ops.conv2d(relu__12, parameter_116, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_142 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_164, reshape_165 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_117, full_int_array_142), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__106 = paddle._C_ops.add_(conv2d_78, reshape_164)

        # pd_op.full: (1xf32) <- ()
        full_117 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1xf32)
        scale__55 = paddle._C_ops.scale_(add__106, full_117, float('0'), True)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__107 = paddle._C_ops.add_(add__104, scale__55)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_79 = paddle._C_ops.conv2d(add__107, parameter_118, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_143 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_166, reshape_167 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_119, full_int_array_143), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__108 = paddle._C_ops.add_(conv2d_79, reshape_166)

        # pd_op.relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        relu__13 = paddle._C_ops.relu_(add__108)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_80 = paddle._C_ops.conv2d(relu__13, parameter_120, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_144 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_168, reshape_169 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_121, full_int_array_144), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__109 = paddle._C_ops.add_(conv2d_80, reshape_168)

        # pd_op.full: (1xf32) <- ()
        full_118 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1xf32)
        scale__56 = paddle._C_ops.scale_(add__109, full_118, float('0'), True)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__110 = paddle._C_ops.add_(add__107, scale__56)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__111 = paddle._C_ops.add_(add__100, add__110)

        # pd_op.full: (1x32x180x320xf32) <- ()
        full_119 = paddle._C_ops.full([1, 32, 180, 320], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # builtin.combine: ([1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32]) <- (1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32)
        combine_34 = [slice_2, add__82, add__111, full_119]

        # pd_op.full: (1xi32) <- ()
        full_120 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x128x180x320xf32) <- ([1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32], 1xi32)
        concat_16 = paddle._C_ops.concat(combine_34, full_120)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x128x180x320xf32, 32x128x3x3xf32)
        conv2d_81 = paddle._C_ops.conv2d(concat_16, parameter_132, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_145 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_170, reshape_171 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_133, full_int_array_145), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__112 = paddle._C_ops.add_(conv2d_81, reshape_170)

        # pd_op.leaky_relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        leaky_relu__47 = paddle._C_ops.leaky_relu_(add__112, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_82 = paddle._C_ops.conv2d(leaky_relu__47, parameter_134, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_146 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_172, reshape_173 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_135, full_int_array_146), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__113 = paddle._C_ops.add_(conv2d_82, reshape_172)

        # pd_op.relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        relu__14 = paddle._C_ops.relu_(add__113)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_83 = paddle._C_ops.conv2d(relu__14, parameter_136, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_147 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_174, reshape_175 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_137, full_int_array_147), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__114 = paddle._C_ops.add_(conv2d_83, reshape_174)

        # pd_op.full: (1xf32) <- ()
        full_121 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1xf32)
        scale__57 = paddle._C_ops.scale_(add__114, full_121, float('0'), True)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__115 = paddle._C_ops.add_(leaky_relu__47, scale__57)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_84 = paddle._C_ops.conv2d(add__115, parameter_138, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_148 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_176, reshape_177 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_139, full_int_array_148), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__116 = paddle._C_ops.add_(conv2d_84, reshape_176)

        # pd_op.relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        relu__15 = paddle._C_ops.relu_(add__116)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_85 = paddle._C_ops.conv2d(relu__15, parameter_140, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_149 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_178, reshape_179 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_141, full_int_array_149), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__117 = paddle._C_ops.add_(conv2d_85, reshape_178)

        # pd_op.full: (1xf32) <- ()
        full_122 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1xf32)
        scale__58 = paddle._C_ops.scale_(add__117, full_122, float('0'), True)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__118 = paddle._C_ops.add_(add__115, scale__58)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_86 = paddle._C_ops.conv2d(add__118, parameter_142, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_150 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_180, reshape_181 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_143, full_int_array_150), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__119 = paddle._C_ops.add_(conv2d_86, reshape_180)

        # pd_op.relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        relu__16 = paddle._C_ops.relu_(add__119)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_87 = paddle._C_ops.conv2d(relu__16, parameter_144, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_151 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_182, reshape_183 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_145, full_int_array_151), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__120 = paddle._C_ops.add_(conv2d_87, reshape_182)

        # pd_op.full: (1xf32) <- ()
        full_123 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1xf32)
        scale__59 = paddle._C_ops.scale_(add__120, full_123, float('0'), True)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__121 = paddle._C_ops.add_(add__118, scale__59)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__122 = paddle._C_ops.add_(full_119, add__121)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_152 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_153 = [1]

        # pd_op.slice: (1x2x180x320xf32) <- (1x1x2x180x320xf32, 1xi64, 1xi64)
        slice_44 = paddle._C_ops.slice(select_input_1, [1], full_int_array_152, full_int_array_153, [1], [1])

        # pd_op.transpose: (1x180x320x2xf32) <- (1x2x180x320xf32)
        transpose_6 = paddle._C_ops.transpose(slice_44, [0, 2, 3, 1])

        # pd_op.full: (1xi64) <- ()
        full_124 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_125 = paddle._C_ops.full([1], float('180'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_126 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (180xi64) <- (1xi64, 1xi64, 1xi64)
        arange_12 = paddle.arange(full_124, full_125, full_126, dtype='int64')

        # pd_op.full: (1xi64) <- ()
        full_127 = paddle._C_ops.full([1], float('0'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_128 = paddle._C_ops.full([1], float('320'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.full: (1xi64) <- ()
        full_129 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.arange: (320xi64) <- (1xi64, 1xi64, 1xi64)
        arange_13 = paddle.arange(full_127, full_128, full_129, dtype='int64')

        # builtin.combine: ([180xi64, 320xi64]) <- (180xi64, 320xi64)
        combine_35 = [arange_12, arange_13]

        # pd_op.meshgrid: ([180x320xi64, 180x320xi64]) <- ([180xi64, 320xi64])
        meshgrid_6 = paddle._C_ops.meshgrid(combine_35)

        # builtin.slice: (180x320xi64) <- ([180x320xi64, 180x320xi64])
        slice_45 = meshgrid_6[1]

        # builtin.slice: (180x320xi64) <- ([180x320xi64, 180x320xi64])
        slice_46 = meshgrid_6[0]

        # builtin.combine: ([180x320xi64, 180x320xi64]) <- (180x320xi64, 180x320xi64)
        combine_36 = [slice_45, slice_46]

        # pd_op.stack: (180x320x2xi64) <- ([180x320xi64, 180x320xi64])
        stack_12 = paddle._C_ops.stack(combine_36, 2)

        # pd_op.cast: (180x320x2xf32) <- (180x320x2xi64)
        cast_8 = paddle._C_ops.cast(stack_12, paddle.float32)

        # pd_op.add_: (1x180x320x2xf32) <- (180x320x2xf32, 1x180x320x2xf32)
        add__123 = paddle._C_ops.add_(cast_8, transpose_6)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_154 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_155 = [1]

        # pd_op.slice: (1x180x320xf32) <- (1x180x320x2xf32, 1xi64, 1xi64)
        slice_47 = paddle._C_ops.slice(add__123, [3], full_int_array_154, full_int_array_155, [1], [3])

        # pd_op.full: (1xf32) <- ()
        full_130 = paddle._C_ops.full([1], float('2'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x180x320xf32) <- (1x180x320xf32, 1xf32)
        scale__60 = paddle._C_ops.scale_(slice_47, full_130, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_131 = paddle._C_ops.full([1], float('0.0031348'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x180x320xf32) <- (1x180x320xf32, 1xf32)
        scale__61 = paddle._C_ops.scale_(scale__60, full_131, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_132 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x180x320xf32) <- (1x180x320xf32, 1xf32)
        scale__62 = paddle._C_ops.scale_(scale__61, full_132, float('-1'), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_156 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_157 = [2]

        # pd_op.slice: (1x180x320xf32) <- (1x180x320x2xf32, 1xi64, 1xi64)
        slice_48 = paddle._C_ops.slice(add__123, [3], full_int_array_156, full_int_array_157, [1], [3])

        # pd_op.full: (1xf32) <- ()
        full_133 = paddle._C_ops.full([1], float('2'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x180x320xf32) <- (1x180x320xf32, 1xf32)
        scale__63 = paddle._C_ops.scale_(slice_48, full_133, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_134 = paddle._C_ops.full([1], float('0.00558659'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x180x320xf32) <- (1x180x320xf32, 1xf32)
        scale__64 = paddle._C_ops.scale_(scale__63, full_134, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_135 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x180x320xf32) <- (1x180x320xf32, 1xf32)
        scale__65 = paddle._C_ops.scale_(scale__64, full_135, float('-1'), True)

        # builtin.combine: ([1x180x320xf32, 1x180x320xf32]) <- (1x180x320xf32, 1x180x320xf32)
        combine_37 = [scale__62, scale__65]

        # pd_op.stack: (1x180x320x2xf32) <- ([1x180x320xf32, 1x180x320xf32])
        stack_13 = paddle._C_ops.stack(combine_37, 3)

        # pd_op.grid_sample: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x180x320x2xf32)
        grid_sample_6 = paddle._C_ops.grid_sample(add__122, stack_13, 'bilinear', 'zeros', True)

        # builtin.combine: ([1x32x180x320xf32, 1x32x180x320xf32]) <- (1x32x180x320xf32, 1x32x180x320xf32)
        combine_38 = [grid_sample_6, slice_3]

        # pd_op.full: (1xi32) <- ()
        full_136 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x64x180x320xf32) <- ([1x32x180x320xf32, 1x32x180x320xf32], 1xi32)
        concat_17 = paddle._C_ops.concat(combine_38, full_136)

        # builtin.combine: ([1x64x180x320xf32, 1x2x180x320xf32]) <- (1x64x180x320xf32, 1x2x180x320xf32)
        combine_39 = [concat_17, slice_44]

        # pd_op.full: (1xi32) <- ()
        full_137 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x66x180x320xf32) <- ([1x64x180x320xf32, 1x2x180x320xf32], 1xi32)
        concat_18 = paddle._C_ops.concat(combine_39, full_137)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x66x180x320xf32, 32x66x3x3xf32)
        conv2d_88 = paddle._C_ops.conv2d(concat_18, parameter_146, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_158 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_184, reshape_185 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_147, full_int_array_158), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__124 = paddle._C_ops.add_(conv2d_88, reshape_184)

        # pd_op.leaky_relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        leaky_relu__48 = paddle._C_ops.leaky_relu_(add__124, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_89 = paddle._C_ops.conv2d(leaky_relu__48, parameter_148, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_159 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_186, reshape_187 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_149, full_int_array_159), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__125 = paddle._C_ops.add_(conv2d_89, reshape_186)

        # pd_op.leaky_relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        leaky_relu__49 = paddle._C_ops.leaky_relu_(add__125, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_90 = paddle._C_ops.conv2d(leaky_relu__49, parameter_150, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_160 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_188, reshape_189 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_151, full_int_array_160), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__126 = paddle._C_ops.add_(conv2d_90, reshape_188)

        # pd_op.leaky_relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        leaky_relu__50 = paddle._C_ops.leaky_relu_(add__126, float('0.1'))

        # pd_op.conv2d: (1x108x180x320xf32) <- (1x32x180x320xf32, 108x32x3x3xf32)
        conv2d_91 = paddle._C_ops.conv2d(leaky_relu__50, parameter_152, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_161 = [1, 108, 1, 1]

        # pd_op.reshape: (1x108x1x1xf32, 0x108xf32) <- (108xf32, 4xi64)
        reshape_190, reshape_191 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_153, full_int_array_161), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x108x180x320xf32) <- (1x108x180x320xf32, 1x108x1x1xf32)
        add__127 = paddle._C_ops.add_(conv2d_91, reshape_190)

        # pd_op.full: (1xi32) <- ()
        full_138 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([1x36x180x320xf32, 1x36x180x320xf32, 1x36x180x320xf32]) <- (1x108x180x320xf32, 1xi32)
        split_with_num_4 = paddle._C_ops.split_with_num(add__127, 3, full_138)

        # builtin.slice: (1x36x180x320xf32) <- ([1x36x180x320xf32, 1x36x180x320xf32, 1x36x180x320xf32])
        slice_49 = split_with_num_4[0]

        # builtin.slice: (1x36x180x320xf32) <- ([1x36x180x320xf32, 1x36x180x320xf32, 1x36x180x320xf32])
        slice_50 = split_with_num_4[1]

        # builtin.combine: ([1x36x180x320xf32, 1x36x180x320xf32]) <- (1x36x180x320xf32, 1x36x180x320xf32)
        combine_40 = [slice_49, slice_50]

        # pd_op.full: (1xi32) <- ()
        full_139 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x72x180x320xf32) <- ([1x36x180x320xf32, 1x36x180x320xf32], 1xi32)
        concat_19 = paddle._C_ops.concat(combine_40, full_139)

        # pd_op.tanh_: (1x72x180x320xf32) <- (1x72x180x320xf32)
        tanh__3 = paddle._C_ops.tanh_(concat_19)

        # pd_op.full: (1xf32) <- ()
        full_140 = paddle._C_ops.full([1], float('10'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x72x180x320xf32) <- (1x72x180x320xf32, 1xf32)
        scale__66 = paddle._C_ops.scale_(tanh__3, full_140, float('0'), True)

        # pd_op.flip: (1x2x180x320xf32) <- (1x2x180x320xf32)
        flip_4 = paddle._C_ops.flip(slice_44, [1])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_162 = [1, 36, 1, 1]

        # pd_op.tile: (1x72x180x320xf32) <- (1x2x180x320xf32, 4xi64)
        tile_3 = paddle._C_ops.tile(flip_4, full_int_array_162)

        # pd_op.add_: (1x72x180x320xf32) <- (1x72x180x320xf32, 1x72x180x320xf32)
        add__128 = paddle._C_ops.add_(scale__66, tile_3)

        # builtin.slice: (1x36x180x320xf32) <- ([1x36x180x320xf32, 1x36x180x320xf32, 1x36x180x320xf32])
        slice_51 = split_with_num_4[2]

        # pd_op.sigmoid_: (1x36x180x320xf32) <- (1x36x180x320xf32)
        sigmoid__3 = paddle._C_ops.sigmoid_(slice_51)

        # pd_op.deformable_conv: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x72x180x320xf32, 32x32x3x3xf32, 1x36x180x320xf32)
        deformable_conv_3 = paddle._C_ops.deformable_conv(add__122, add__128, parameter_154, sigmoid__3, [1, 1], [1, 1], [1, 1], 4, 1, 1)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_163 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xi64) <- (32xf32, 4xi64)
        reshape_192, reshape_193 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_155, full_int_array_163), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__129 = paddle._C_ops.add_(deformable_conv_3, reshape_192)

        # builtin.combine: ([1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32]) <- (1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32)
        combine_41 = [slice_3, add__65, add__93, add__129]

        # pd_op.full: (1xi32) <- ()
        full_141 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x128x180x320xf32) <- ([1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32], 1xi32)
        concat_20 = paddle._C_ops.concat(combine_41, full_141)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x128x180x320xf32, 32x128x3x3xf32)
        conv2d_92 = paddle._C_ops.conv2d(concat_20, parameter_132, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_164 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_194, reshape_195 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_133, full_int_array_164), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__130 = paddle._C_ops.add_(conv2d_92, reshape_194)

        # pd_op.leaky_relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        leaky_relu__51 = paddle._C_ops.leaky_relu_(add__130, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_93 = paddle._C_ops.conv2d(leaky_relu__51, parameter_134, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_165 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_196, reshape_197 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_135, full_int_array_165), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__131 = paddle._C_ops.add_(conv2d_93, reshape_196)

        # pd_op.relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        relu__17 = paddle._C_ops.relu_(add__131)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_94 = paddle._C_ops.conv2d(relu__17, parameter_136, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_166 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_198, reshape_199 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_137, full_int_array_166), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__132 = paddle._C_ops.add_(conv2d_94, reshape_198)

        # pd_op.full: (1xf32) <- ()
        full_142 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1xf32)
        scale__67 = paddle._C_ops.scale_(add__132, full_142, float('0'), True)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__133 = paddle._C_ops.add_(leaky_relu__51, scale__67)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_95 = paddle._C_ops.conv2d(add__133, parameter_138, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_167 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_200, reshape_201 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_139, full_int_array_167), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__134 = paddle._C_ops.add_(conv2d_95, reshape_200)

        # pd_op.relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        relu__18 = paddle._C_ops.relu_(add__134)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_96 = paddle._C_ops.conv2d(relu__18, parameter_140, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_168 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_202, reshape_203 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_141, full_int_array_168), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__135 = paddle._C_ops.add_(conv2d_96, reshape_202)

        # pd_op.full: (1xf32) <- ()
        full_143 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1xf32)
        scale__68 = paddle._C_ops.scale_(add__135, full_143, float('0'), True)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__136 = paddle._C_ops.add_(add__133, scale__68)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_97 = paddle._C_ops.conv2d(add__136, parameter_142, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_169 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_204, reshape_205 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_143, full_int_array_169), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__137 = paddle._C_ops.add_(conv2d_97, reshape_204)

        # pd_op.relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        relu__19 = paddle._C_ops.relu_(add__137)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_98 = paddle._C_ops.conv2d(relu__19, parameter_144, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_170 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_206, reshape_207 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_145, full_int_array_170), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__138 = paddle._C_ops.add_(conv2d_98, reshape_206)

        # pd_op.full: (1xf32) <- ()
        full_144 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1xf32)
        scale__69 = paddle._C_ops.scale_(add__138, full_144, float('0'), True)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__139 = paddle._C_ops.add_(add__136, scale__69)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__140 = paddle._C_ops.add_(add__129, add__139)

        # builtin.combine: ([1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32]) <- (1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32)
        combine_42 = [slice_2, add__82, add__111, add__122]

        # pd_op.full: (1xi32) <- ()
        full_145 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x128x180x320xf32) <- ([1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32], 1xi32)
        concat_21 = paddle._C_ops.concat(combine_42, full_145)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x128x180x320xf32, 32x128x3x3xf32)
        conv2d_99 = paddle._C_ops.conv2d(concat_21, parameter_156, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_171 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_208, reshape_209 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_157, full_int_array_171), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__141 = paddle._C_ops.add_(conv2d_99, reshape_208)

        # pd_op.leaky_relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        leaky_relu__52 = paddle._C_ops.leaky_relu_(add__141, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_100 = paddle._C_ops.conv2d(leaky_relu__52, parameter_158, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_172 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_210, reshape_211 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_159, full_int_array_172), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__142 = paddle._C_ops.add_(conv2d_100, reshape_210)

        # pd_op.relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        relu__20 = paddle._C_ops.relu_(add__142)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_101 = paddle._C_ops.conv2d(relu__20, parameter_160, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_173 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_212, reshape_213 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_161, full_int_array_173), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__143 = paddle._C_ops.add_(conv2d_101, reshape_212)

        # pd_op.full: (1xf32) <- ()
        full_146 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1xf32)
        scale__70 = paddle._C_ops.scale_(add__143, full_146, float('0'), True)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__144 = paddle._C_ops.add_(leaky_relu__52, scale__70)

        # pd_op.conv2d: (1x128x180x320xf32) <- (1x32x180x320xf32, 128x32x3x3xf32)
        conv2d_102 = paddle._C_ops.conv2d(add__144, parameter_162, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_174 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_214, reshape_215 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_163, full_int_array_174), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x128x180x320xf32) <- (1x128x180x320xf32, 1x128x1x1xf32)
        add__145 = paddle._C_ops.add_(conv2d_102, reshape_214)

        # pd_op.pixel_shuffle: (1x32x360x640xf32) <- (1x128x180x320xf32)
        pixel_shuffle_0 = paddle._C_ops.pixel_shuffle(add__145, 2, 'NCHW')

        # pd_op.leaky_relu_: (1x32x360x640xf32) <- (1x32x360x640xf32)
        leaky_relu__53 = paddle._C_ops.leaky_relu_(pixel_shuffle_0, float('0.1'))

        # pd_op.conv2d: (1x128x360x640xf32) <- (1x32x360x640xf32, 128x32x3x3xf32)
        conv2d_103 = paddle._C_ops.conv2d(leaky_relu__53, parameter_164, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_175 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_216, reshape_217 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_165, full_int_array_175), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x128x360x640xf32) <- (1x128x360x640xf32, 1x128x1x1xf32)
        add__146 = paddle._C_ops.add_(conv2d_103, reshape_216)

        # pd_op.pixel_shuffle: (1x32x720x1280xf32) <- (1x128x360x640xf32)
        pixel_shuffle_1 = paddle._C_ops.pixel_shuffle(add__146, 2, 'NCHW')

        # pd_op.leaky_relu_: (1x32x720x1280xf32) <- (1x32x720x1280xf32)
        leaky_relu__54 = paddle._C_ops.leaky_relu_(pixel_shuffle_1, float('0.1'))

        # pd_op.conv2d: (1x3x720x1280xf32) <- (1x32x720x1280xf32, 3x32x3x3xf32)
        conv2d_104 = paddle._C_ops.conv2d(leaky_relu__54, parameter_166, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_176 = [1, 3, 1, 1]

        # pd_op.reshape: (1x3x1x1xf32, 0x3xf32) <- (3xf32, 4xi64)
        reshape_218, reshape_219 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_167, full_int_array_176), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x3x720x1280xf32) <- (1x3x720x1280xf32, 1x3x1x1xf32)
        add__147 = paddle._C_ops.add_(conv2d_104, reshape_218)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_177 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_178 = [1]

        # pd_op.slice: (1x3x180x320xf32) <- (1x2x3x180x320xf32, 1xi64, 1xi64)
        slice_52 = paddle._C_ops.slice(feed_0, [1], full_int_array_177, full_int_array_178, [1], [1])

        # pd_op.bilinear_interp: (1x3x720x1280xf32) <- (1x3x180x320xf32, None, None, None)
        bilinear_interp_5 = paddle._C_ops.bilinear_interp(slice_52, None, None, None, 'NCHW', -1, -1, -1, [float('4'), float('4')], 'bilinear', False, 0)

        # pd_op.add_: (1x3x720x1280xf32) <- (1x3x720x1280xf32, 1x3x720x1280xf32)
        add__148 = paddle._C_ops.add_(add__147, bilinear_interp_5)

        # builtin.combine: ([1x3x720x1280xf32, 1x32x720x1280xf32]) <- (1x3x720x1280xf32, 1x32x720x1280xf32)
        combine_43 = [add__148, leaky_relu__54]

        # pd_op.full: (1xi32) <- ()
        full_147 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x35x720x1280xf32) <- ([1x3x720x1280xf32, 1x32x720x1280xf32], 1xi32)
        concat_22 = paddle._C_ops.concat(combine_43, full_147)

        # pd_op.conv2d: (1x32x360x640xf32) <- (1x35x720x1280xf32, 32x35x3x3xf32)
        conv2d_105 = paddle._C_ops.conv2d(concat_22, parameter_168, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_179 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_220, reshape_221 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_169, full_int_array_179), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x360x640xf32) <- (1x32x360x640xf32, 1x32x1x1xf32)
        add__149 = paddle._C_ops.add_(conv2d_105, reshape_220)

        # pd_op.leaky_relu_: (1x32x360x640xf32) <- (1x32x360x640xf32)
        leaky_relu__55 = paddle._C_ops.leaky_relu_(add__149, float('0.1'))

        # pd_op.conv2d: (1x32x360x640xf32) <- (1x32x360x640xf32, 32x32x3x3xf32)
        conv2d_106 = paddle._C_ops.conv2d(leaky_relu__55, parameter_170, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_180 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_222, reshape_223 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_171, full_int_array_180), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x360x640xf32) <- (1x32x360x640xf32, 1x32x1x1xf32)
        add__150 = paddle._C_ops.add_(conv2d_106, reshape_222)

        # builtin.combine: ([1x32x360x640xf32, 1x32x360x640xf32]) <- (1x32x360x640xf32, 1x32x360x640xf32)
        combine_44 = [add__150, leaky_relu__53]

        # pd_op.full: (1xi32) <- ()
        full_148 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x64x360x640xf32) <- ([1x32x360x640xf32, 1x32x360x640xf32], 1xi32)
        concat_23 = paddle._C_ops.concat(combine_44, full_148)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x64x360x640xf32, 32x64x3x3xf32)
        conv2d_107 = paddle._C_ops.conv2d(concat_23, parameter_172, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_181 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_224, reshape_225 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_173, full_int_array_181), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__151 = paddle._C_ops.add_(conv2d_107, reshape_224)

        # pd_op.leaky_relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        leaky_relu__56 = paddle._C_ops.leaky_relu_(add__151, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_108 = paddle._C_ops.conv2d(leaky_relu__56, parameter_174, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_182 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_226, reshape_227 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_175, full_int_array_182), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__152 = paddle._C_ops.add_(conv2d_108, reshape_226)

        # builtin.combine: ([1x32x180x320xf32, 1x32x180x320xf32]) <- (1x32x180x320xf32, 1x32x180x320xf32)
        combine_45 = [add__152, add__144]

        # pd_op.full: (1xi32) <- ()
        full_149 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x64x180x320xf32) <- ([1x32x180x320xf32, 1x32x180x320xf32], 1xi32)
        concat_24 = paddle._C_ops.concat(combine_45, full_149)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x64x180x320xf32, 32x64x3x3xf32)
        conv2d_109 = paddle._C_ops.conv2d(concat_24, parameter_176, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_183 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_228, reshape_229 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_177, full_int_array_183), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__153 = paddle._C_ops.add_(conv2d_109, reshape_228)

        # builtin.combine: ([1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32]) <- (1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32)
        combine_46 = [slice_3, add__65, add__93, add__140]

        # pd_op.full: (1xi32) <- ()
        full_150 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x128x180x320xf32) <- ([1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32], 1xi32)
        concat_25 = paddle._C_ops.concat(combine_46, full_150)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x128x180x320xf32, 32x128x3x3xf32)
        conv2d_110 = paddle._C_ops.conv2d(concat_25, parameter_156, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_184 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_230, reshape_231 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_157, full_int_array_184), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__154 = paddle._C_ops.add_(conv2d_110, reshape_230)

        # pd_op.leaky_relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        leaky_relu__57 = paddle._C_ops.leaky_relu_(add__154, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_111 = paddle._C_ops.conv2d(leaky_relu__57, parameter_158, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_185 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_232, reshape_233 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_159, full_int_array_185), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__155 = paddle._C_ops.add_(conv2d_111, reshape_232)

        # pd_op.relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        relu__21 = paddle._C_ops.relu_(add__155)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_112 = paddle._C_ops.conv2d(relu__21, parameter_160, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_186 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_234, reshape_235 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_161, full_int_array_186), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__156 = paddle._C_ops.add_(conv2d_112, reshape_234)

        # pd_op.full: (1xf32) <- ()
        full_151 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1xf32)
        scale__71 = paddle._C_ops.scale_(add__156, full_151, float('0'), True)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__157 = paddle._C_ops.add_(leaky_relu__57, scale__71)

        # pd_op.conv2d: (1x128x180x320xf32) <- (1x32x180x320xf32, 128x32x3x3xf32)
        conv2d_113 = paddle._C_ops.conv2d(add__157, parameter_162, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_187 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_236, reshape_237 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_163, full_int_array_187), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x128x180x320xf32) <- (1x128x180x320xf32, 1x128x1x1xf32)
        add__158 = paddle._C_ops.add_(conv2d_113, reshape_236)

        # pd_op.pixel_shuffle: (1x32x360x640xf32) <- (1x128x180x320xf32)
        pixel_shuffle_2 = paddle._C_ops.pixel_shuffle(add__158, 2, 'NCHW')

        # pd_op.leaky_relu_: (1x32x360x640xf32) <- (1x32x360x640xf32)
        leaky_relu__58 = paddle._C_ops.leaky_relu_(pixel_shuffle_2, float('0.1'))

        # pd_op.conv2d: (1x128x360x640xf32) <- (1x32x360x640xf32, 128x32x3x3xf32)
        conv2d_114 = paddle._C_ops.conv2d(leaky_relu__58, parameter_164, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_188 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_238, reshape_239 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_165, full_int_array_188), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x128x360x640xf32) <- (1x128x360x640xf32, 1x128x1x1xf32)
        add__159 = paddle._C_ops.add_(conv2d_114, reshape_238)

        # pd_op.pixel_shuffle: (1x32x720x1280xf32) <- (1x128x360x640xf32)
        pixel_shuffle_3 = paddle._C_ops.pixel_shuffle(add__159, 2, 'NCHW')

        # pd_op.leaky_relu_: (1x32x720x1280xf32) <- (1x32x720x1280xf32)
        leaky_relu__59 = paddle._C_ops.leaky_relu_(pixel_shuffle_3, float('0.1'))

        # pd_op.conv2d: (1x3x720x1280xf32) <- (1x32x720x1280xf32, 3x32x3x3xf32)
        conv2d_115 = paddle._C_ops.conv2d(leaky_relu__59, parameter_166, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_189 = [1, 3, 1, 1]

        # pd_op.reshape: (1x3x1x1xf32, 0x3xf32) <- (3xf32, 4xi64)
        reshape_240, reshape_241 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_167, full_int_array_189), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x3x720x1280xf32) <- (1x3x720x1280xf32, 1x3x1x1xf32)
        add__160 = paddle._C_ops.add_(conv2d_115, reshape_240)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_190 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_191 = [2]

        # pd_op.slice: (1x3x180x320xf32) <- (1x2x3x180x320xf32, 1xi64, 1xi64)
        slice_53 = paddle._C_ops.slice(feed_0, [1], full_int_array_190, full_int_array_191, [1], [1])

        # pd_op.bilinear_interp: (1x3x720x1280xf32) <- (1x3x180x320xf32, None, None, None)
        bilinear_interp_6 = paddle._C_ops.bilinear_interp(slice_53, None, None, None, 'NCHW', -1, -1, -1, [float('4'), float('4')], 'bilinear', False, 0)

        # pd_op.add_: (1x3x720x1280xf32) <- (1x3x720x1280xf32, 1x3x720x1280xf32)
        add__161 = paddle._C_ops.add_(add__160, bilinear_interp_6)

        # builtin.combine: ([1x3x720x1280xf32, 1x32x720x1280xf32]) <- (1x3x720x1280xf32, 1x32x720x1280xf32)
        combine_47 = [add__161, leaky_relu__59]

        # pd_op.full: (1xi32) <- ()
        full_152 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x35x720x1280xf32) <- ([1x3x720x1280xf32, 1x32x720x1280xf32], 1xi32)
        concat_26 = paddle._C_ops.concat(combine_47, full_152)

        # pd_op.conv2d: (1x32x360x640xf32) <- (1x35x720x1280xf32, 32x35x3x3xf32)
        conv2d_116 = paddle._C_ops.conv2d(concat_26, parameter_168, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_192 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_242, reshape_243 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_169, full_int_array_192), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x360x640xf32) <- (1x32x360x640xf32, 1x32x1x1xf32)
        add__162 = paddle._C_ops.add_(conv2d_116, reshape_242)

        # pd_op.leaky_relu_: (1x32x360x640xf32) <- (1x32x360x640xf32)
        leaky_relu__60 = paddle._C_ops.leaky_relu_(add__162, float('0.1'))

        # pd_op.conv2d: (1x32x360x640xf32) <- (1x32x360x640xf32, 32x32x3x3xf32)
        conv2d_117 = paddle._C_ops.conv2d(leaky_relu__60, parameter_170, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_193 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_244, reshape_245 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_171, full_int_array_193), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x360x640xf32) <- (1x32x360x640xf32, 1x32x1x1xf32)
        add__163 = paddle._C_ops.add_(conv2d_117, reshape_244)

        # builtin.combine: ([1x32x360x640xf32, 1x32x360x640xf32]) <- (1x32x360x640xf32, 1x32x360x640xf32)
        combine_48 = [add__163, leaky_relu__58]

        # pd_op.full: (1xi32) <- ()
        full_153 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x64x360x640xf32) <- ([1x32x360x640xf32, 1x32x360x640xf32], 1xi32)
        concat_27 = paddle._C_ops.concat(combine_48, full_153)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x64x360x640xf32, 32x64x3x3xf32)
        conv2d_118 = paddle._C_ops.conv2d(concat_27, parameter_172, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_194 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_246, reshape_247 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_173, full_int_array_194), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__164 = paddle._C_ops.add_(conv2d_118, reshape_246)

        # pd_op.leaky_relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        leaky_relu__61 = paddle._C_ops.leaky_relu_(add__164, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_119 = paddle._C_ops.conv2d(leaky_relu__61, parameter_174, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_195 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_248, reshape_249 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_175, full_int_array_195), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__165 = paddle._C_ops.add_(conv2d_119, reshape_248)

        # builtin.combine: ([1x32x180x320xf32, 1x32x180x320xf32]) <- (1x32x180x320xf32, 1x32x180x320xf32)
        combine_49 = [add__165, add__157]

        # pd_op.full: (1xi32) <- ()
        full_154 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x64x180x320xf32) <- ([1x32x180x320xf32, 1x32x180x320xf32], 1xi32)
        concat_28 = paddle._C_ops.concat(combine_49, full_154)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x64x180x320xf32, 32x64x3x3xf32)
        conv2d_120 = paddle._C_ops.conv2d(concat_28, parameter_176, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_196 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_250, reshape_251 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_177, full_int_array_196), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__166 = paddle._C_ops.add_(conv2d_120, reshape_250)

        # pd_op.full: (1x32x180x320xf32) <- ()
        full_155 = paddle._C_ops.full([1, 32, 180, 320], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # builtin.combine: ([1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32]) <- (1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32)
        combine_50 = [add__166, add__65, add__93, add__140, full_155]

        # pd_op.full: (1xi32) <- ()
        full_156 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x160x180x320xf32) <- ([1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32], 1xi32)
        concat_29 = paddle._C_ops.concat(combine_50, full_156)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x160x180x320xf32, 32x160x3x3xf32)
        conv2d_121 = paddle._C_ops.conv2d(concat_29, parameter_178, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_197 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_252, reshape_253 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_179, full_int_array_197), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__167 = paddle._C_ops.add_(conv2d_121, reshape_252)

        # pd_op.leaky_relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        leaky_relu__62 = paddle._C_ops.leaky_relu_(add__167, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_122 = paddle._C_ops.conv2d(leaky_relu__62, parameter_180, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_198 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_254, reshape_255 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_181, full_int_array_198), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__168 = paddle._C_ops.add_(conv2d_122, reshape_254)

        # pd_op.relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        relu__22 = paddle._C_ops.relu_(add__168)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_123 = paddle._C_ops.conv2d(relu__22, parameter_182, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_199 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_256, reshape_257 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_183, full_int_array_199), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__169 = paddle._C_ops.add_(conv2d_123, reshape_256)

        # pd_op.full: (1xf32) <- ()
        full_157 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1xf32)
        scale__72 = paddle._C_ops.scale_(add__169, full_157, float('0'), True)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__170 = paddle._C_ops.add_(leaky_relu__62, scale__72)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_124 = paddle._C_ops.conv2d(add__170, parameter_184, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_200 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_258, reshape_259 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_185, full_int_array_200), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__171 = paddle._C_ops.add_(conv2d_124, reshape_258)

        # pd_op.relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        relu__23 = paddle._C_ops.relu_(add__171)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_125 = paddle._C_ops.conv2d(relu__23, parameter_186, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_201 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_260, reshape_261 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_187, full_int_array_201), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__172 = paddle._C_ops.add_(conv2d_125, reshape_260)

        # pd_op.full: (1xf32) <- ()
        full_158 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1xf32)
        scale__73 = paddle._C_ops.scale_(add__172, full_158, float('0'), True)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__173 = paddle._C_ops.add_(add__170, scale__73)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_126 = paddle._C_ops.conv2d(add__173, parameter_188, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_202 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_262, reshape_263 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_189, full_int_array_202), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__174 = paddle._C_ops.add_(conv2d_126, reshape_262)

        # pd_op.relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        relu__24 = paddle._C_ops.relu_(add__174)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_127 = paddle._C_ops.conv2d(relu__24, parameter_190, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_203 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_264, reshape_265 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_191, full_int_array_203), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__175 = paddle._C_ops.add_(conv2d_127, reshape_264)

        # pd_op.full: (1xf32) <- ()
        full_159 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1xf32)
        scale__74 = paddle._C_ops.scale_(add__175, full_159, float('0'), True)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__176 = paddle._C_ops.add_(add__173, scale__74)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__177 = paddle._C_ops.add_(full_155, add__176)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_204 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_205 = [1]

        # pd_op.slice: (1x2x180x320xf32) <- (1x1x2x180x320xf32, 1xi64, 1xi64)
        slice_54 = paddle._C_ops.slice(reshape__6, [1], full_int_array_204, full_int_array_205, [1], [1])

        # pd_op.deformable_conv: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x72x180x320xf32, 32x32x3x3xf32, 1x36x180x320xf32)
        deformable_conv_4 = paddle._C_ops.deformable_conv(add__177, add__99, parameter_192, sigmoid__2, [1, 1], [1, 1], [1, 1], 4, 1, 1)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_206 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xi64) <- (32xf32, 4xi64)
        reshape_266, reshape_267 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_193, full_int_array_206), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__178 = paddle._C_ops.add_(deformable_conv_4, reshape_266)

        # builtin.combine: ([1x32x180x320xf32, 1x32x180x320xf32, 1x2x180x320xf32]) <- (1x32x180x320xf32, 1x32x180x320xf32, 1x2x180x320xf32)
        combine_51 = [add__178, add__153, slice_54]

        # pd_op.full: (1xi32) <- ()
        full_160 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x66x180x320xf32) <- ([1x32x180x320xf32, 1x32x180x320xf32, 1x2x180x320xf32], 1xi32)
        concat_30 = paddle._C_ops.concat(combine_51, full_160)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x66x180x320xf32, 32x66x3x3xf32)
        conv2d_128 = paddle._C_ops.conv2d(concat_30, parameter_194, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_207 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_268, reshape_269 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_195, full_int_array_207), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__179 = paddle._C_ops.add_(conv2d_128, reshape_268)

        # pd_op.leaky_relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        leaky_relu__63 = paddle._C_ops.leaky_relu_(add__179, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_129 = paddle._C_ops.conv2d(leaky_relu__63, parameter_196, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_208 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_270, reshape_271 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_197, full_int_array_208), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__180 = paddle._C_ops.add_(conv2d_129, reshape_270)

        # pd_op.leaky_relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        leaky_relu__64 = paddle._C_ops.leaky_relu_(add__180, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_130 = paddle._C_ops.conv2d(leaky_relu__64, parameter_198, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_209 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_272, reshape_273 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_199, full_int_array_209), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__181 = paddle._C_ops.add_(conv2d_130, reshape_272)

        # pd_op.leaky_relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        leaky_relu__65 = paddle._C_ops.leaky_relu_(add__181, float('0.1'))

        # pd_op.conv2d: (1x108x180x320xf32) <- (1x32x180x320xf32, 108x32x3x3xf32)
        conv2d_131 = paddle._C_ops.conv2d(leaky_relu__65, parameter_200, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_210 = [1, 108, 1, 1]

        # pd_op.reshape: (1x108x1x1xf32, 0x108xf32) <- (108xf32, 4xi64)
        reshape_274, reshape_275 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_201, full_int_array_210), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x108x180x320xf32) <- (1x108x180x320xf32, 1x108x1x1xf32)
        add__182 = paddle._C_ops.add_(conv2d_131, reshape_274)

        # pd_op.full: (1xi32) <- ()
        full_161 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([1x36x180x320xf32, 1x36x180x320xf32, 1x36x180x320xf32]) <- (1x108x180x320xf32, 1xi32)
        split_with_num_5 = paddle._C_ops.split_with_num(add__182, 3, full_161)

        # builtin.slice: (1x36x180x320xf32) <- ([1x36x180x320xf32, 1x36x180x320xf32, 1x36x180x320xf32])
        slice_55 = split_with_num_5[0]

        # builtin.slice: (1x36x180x320xf32) <- ([1x36x180x320xf32, 1x36x180x320xf32, 1x36x180x320xf32])
        slice_56 = split_with_num_5[1]

        # builtin.combine: ([1x36x180x320xf32, 1x36x180x320xf32]) <- (1x36x180x320xf32, 1x36x180x320xf32)
        combine_52 = [slice_55, slice_56]

        # pd_op.full: (1xi32) <- ()
        full_162 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x72x180x320xf32) <- ([1x36x180x320xf32, 1x36x180x320xf32], 1xi32)
        concat_31 = paddle._C_ops.concat(combine_52, full_162)

        # pd_op.tanh_: (1x72x180x320xf32) <- (1x72x180x320xf32)
        tanh__4 = paddle._C_ops.tanh_(concat_31)

        # pd_op.full: (1xf32) <- ()
        full_163 = paddle._C_ops.full([1], float('10'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x72x180x320xf32) <- (1x72x180x320xf32, 1xf32)
        scale__75 = paddle._C_ops.scale_(tanh__4, full_163, float('0'), True)

        # pd_op.add_: (1x72x180x320xf32) <- (1x72x180x320xf32, 1x72x180x320xf32)
        add__183 = paddle._C_ops.add_(scale__75, add__99)

        # builtin.slice: (1x36x180x320xf32) <- ([1x36x180x320xf32, 1x36x180x320xf32, 1x36x180x320xf32])
        slice_57 = split_with_num_5[2]

        # pd_op.sigmoid_: (1x36x180x320xf32) <- (1x36x180x320xf32)
        sigmoid__4 = paddle._C_ops.sigmoid_(slice_57)

        # pd_op.add_: (1x36x180x320xf32) <- (1x36x180x320xf32, 1x36x180x320xf32)
        add__184 = paddle._C_ops.add_(sigmoid__4, sigmoid__2)

        # pd_op.full: (1xf32) <- ()
        full_164 = paddle._C_ops.full([1], float('0.5'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x36x180x320xf32) <- (1x36x180x320xf32, 1xf32)
        scale__76 = paddle._C_ops.scale_(add__184, full_164, float('0'), True)

        # pd_op.deformable_conv: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x72x180x320xf32, 32x32x3x3xf32, 1x36x180x320xf32)
        deformable_conv_5 = paddle._C_ops.deformable_conv(add__177, add__183, parameter_202, scale__76, [1, 1], [1, 1], [1, 1], 4, 1, 1)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_211 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xi64) <- (32xf32, 4xi64)
        reshape_276, reshape_277 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_203, full_int_array_211), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__185 = paddle._C_ops.add_(deformable_conv_5, reshape_276)

        # builtin.combine: ([1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32]) <- (1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32)
        combine_53 = [add__153, add__82, add__111, add__122, add__185]

        # pd_op.full: (1xi32) <- ()
        full_165 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x160x180x320xf32) <- ([1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32], 1xi32)
        concat_32 = paddle._C_ops.concat(combine_53, full_165)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x160x180x320xf32, 32x160x3x3xf32)
        conv2d_132 = paddle._C_ops.conv2d(concat_32, parameter_178, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_212 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_278, reshape_279 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_179, full_int_array_212), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__186 = paddle._C_ops.add_(conv2d_132, reshape_278)

        # pd_op.leaky_relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        leaky_relu__66 = paddle._C_ops.leaky_relu_(add__186, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_133 = paddle._C_ops.conv2d(leaky_relu__66, parameter_180, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_213 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_280, reshape_281 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_181, full_int_array_213), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__187 = paddle._C_ops.add_(conv2d_133, reshape_280)

        # pd_op.relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        relu__25 = paddle._C_ops.relu_(add__187)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_134 = paddle._C_ops.conv2d(relu__25, parameter_182, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_214 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_282, reshape_283 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_183, full_int_array_214), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__188 = paddle._C_ops.add_(conv2d_134, reshape_282)

        # pd_op.full: (1xf32) <- ()
        full_166 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1xf32)
        scale__77 = paddle._C_ops.scale_(add__188, full_166, float('0'), True)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__189 = paddle._C_ops.add_(leaky_relu__66, scale__77)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_135 = paddle._C_ops.conv2d(add__189, parameter_184, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_215 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_284, reshape_285 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_185, full_int_array_215), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__190 = paddle._C_ops.add_(conv2d_135, reshape_284)

        # pd_op.relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        relu__26 = paddle._C_ops.relu_(add__190)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_136 = paddle._C_ops.conv2d(relu__26, parameter_186, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_216 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_286, reshape_287 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_187, full_int_array_216), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__191 = paddle._C_ops.add_(conv2d_136, reshape_286)

        # pd_op.full: (1xf32) <- ()
        full_167 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1xf32)
        scale__78 = paddle._C_ops.scale_(add__191, full_167, float('0'), True)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__192 = paddle._C_ops.add_(add__189, scale__78)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_137 = paddle._C_ops.conv2d(add__192, parameter_188, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_217 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_288, reshape_289 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_189, full_int_array_217), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__193 = paddle._C_ops.add_(conv2d_137, reshape_288)

        # pd_op.relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        relu__27 = paddle._C_ops.relu_(add__193)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_138 = paddle._C_ops.conv2d(relu__27, parameter_190, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_218 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_290, reshape_291 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_191, full_int_array_218), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__194 = paddle._C_ops.add_(conv2d_138, reshape_290)

        # pd_op.full: (1xf32) <- ()
        full_168 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1xf32)
        scale__79 = paddle._C_ops.scale_(add__194, full_168, float('0'), True)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__195 = paddle._C_ops.add_(add__192, scale__79)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__196 = paddle._C_ops.add_(add__185, add__195)

        # pd_op.full: (1x32x180x320xf32) <- ()
        full_169 = paddle._C_ops.full([1, 32, 180, 320], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # builtin.combine: ([1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32]) <- (1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32)
        combine_54 = [add__153, add__82, add__111, add__122, add__196, full_169]

        # pd_op.full: (1xi32) <- ()
        full_170 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x192x180x320xf32) <- ([1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32], 1xi32)
        concat_33 = paddle._C_ops.concat(combine_54, full_170)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x192x180x320xf32, 32x192x3x3xf32)
        conv2d_139 = paddle._C_ops.conv2d(concat_33, parameter_204, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_219 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_292, reshape_293 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_205, full_int_array_219), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__197 = paddle._C_ops.add_(conv2d_139, reshape_292)

        # pd_op.leaky_relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        leaky_relu__67 = paddle._C_ops.leaky_relu_(add__197, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_140 = paddle._C_ops.conv2d(leaky_relu__67, parameter_206, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_220 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_294, reshape_295 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_207, full_int_array_220), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__198 = paddle._C_ops.add_(conv2d_140, reshape_294)

        # pd_op.relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        relu__28 = paddle._C_ops.relu_(add__198)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_141 = paddle._C_ops.conv2d(relu__28, parameter_208, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_221 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_296, reshape_297 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_209, full_int_array_221), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__199 = paddle._C_ops.add_(conv2d_141, reshape_296)

        # pd_op.full: (1xf32) <- ()
        full_171 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1xf32)
        scale__80 = paddle._C_ops.scale_(add__199, full_171, float('0'), True)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__200 = paddle._C_ops.add_(leaky_relu__67, scale__80)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_142 = paddle._C_ops.conv2d(add__200, parameter_210, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_222 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_298, reshape_299 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_211, full_int_array_222), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__201 = paddle._C_ops.add_(conv2d_142, reshape_298)

        # pd_op.relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        relu__29 = paddle._C_ops.relu_(add__201)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_143 = paddle._C_ops.conv2d(relu__29, parameter_212, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_223 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_300, reshape_301 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_213, full_int_array_223), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__202 = paddle._C_ops.add_(conv2d_143, reshape_300)

        # pd_op.full: (1xf32) <- ()
        full_172 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1xf32)
        scale__81 = paddle._C_ops.scale_(add__202, full_172, float('0'), True)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__203 = paddle._C_ops.add_(add__200, scale__81)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_144 = paddle._C_ops.conv2d(add__203, parameter_214, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_224 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_302, reshape_303 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_215, full_int_array_224), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__204 = paddle._C_ops.add_(conv2d_144, reshape_302)

        # pd_op.relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        relu__30 = paddle._C_ops.relu_(add__204)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_145 = paddle._C_ops.conv2d(relu__30, parameter_216, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_225 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_304, reshape_305 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_217, full_int_array_225), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__205 = paddle._C_ops.add_(conv2d_145, reshape_304)

        # pd_op.full: (1xf32) <- ()
        full_173 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1xf32)
        scale__82 = paddle._C_ops.scale_(add__205, full_173, float('0'), True)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__206 = paddle._C_ops.add_(add__203, scale__82)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__207 = paddle._C_ops.add_(full_169, add__206)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_226 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_227 = [1]

        # pd_op.slice: (1x2x180x320xf32) <- (1x1x2x180x320xf32, 1xi64, 1xi64)
        slice_58 = paddle._C_ops.slice(select_input_1, [1], full_int_array_226, full_int_array_227, [1], [1])

        # pd_op.deformable_conv: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x72x180x320xf32, 32x32x3x3xf32, 1x36x180x320xf32)
        deformable_conv_6 = paddle._C_ops.deformable_conv(add__207, add__128, parameter_218, sigmoid__3, [1, 1], [1, 1], [1, 1], 4, 1, 1)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_228 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xi64) <- (32xf32, 4xi64)
        reshape_306, reshape_307 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_219, full_int_array_228), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__208 = paddle._C_ops.add_(deformable_conv_6, reshape_306)

        # builtin.combine: ([1x32x180x320xf32, 1x32x180x320xf32, 1x2x180x320xf32]) <- (1x32x180x320xf32, 1x32x180x320xf32, 1x2x180x320xf32)
        combine_55 = [add__208, add__166, slice_58]

        # pd_op.full: (1xi32) <- ()
        full_174 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x66x180x320xf32) <- ([1x32x180x320xf32, 1x32x180x320xf32, 1x2x180x320xf32], 1xi32)
        concat_34 = paddle._C_ops.concat(combine_55, full_174)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x66x180x320xf32, 32x66x3x3xf32)
        conv2d_146 = paddle._C_ops.conv2d(concat_34, parameter_220, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_229 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_308, reshape_309 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_221, full_int_array_229), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__209 = paddle._C_ops.add_(conv2d_146, reshape_308)

        # pd_op.leaky_relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        leaky_relu__68 = paddle._C_ops.leaky_relu_(add__209, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_147 = paddle._C_ops.conv2d(leaky_relu__68, parameter_222, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_230 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_310, reshape_311 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_223, full_int_array_230), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__210 = paddle._C_ops.add_(conv2d_147, reshape_310)

        # pd_op.leaky_relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        leaky_relu__69 = paddle._C_ops.leaky_relu_(add__210, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_148 = paddle._C_ops.conv2d(leaky_relu__69, parameter_224, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_231 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_312, reshape_313 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_225, full_int_array_231), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__211 = paddle._C_ops.add_(conv2d_148, reshape_312)

        # pd_op.leaky_relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        leaky_relu__70 = paddle._C_ops.leaky_relu_(add__211, float('0.1'))

        # pd_op.conv2d: (1x108x180x320xf32) <- (1x32x180x320xf32, 108x32x3x3xf32)
        conv2d_149 = paddle._C_ops.conv2d(leaky_relu__70, parameter_226, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_232 = [1, 108, 1, 1]

        # pd_op.reshape: (1x108x1x1xf32, 0x108xf32) <- (108xf32, 4xi64)
        reshape_314, reshape_315 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_227, full_int_array_232), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x108x180x320xf32) <- (1x108x180x320xf32, 1x108x1x1xf32)
        add__212 = paddle._C_ops.add_(conv2d_149, reshape_314)

        # pd_op.full: (1xi32) <- ()
        full_175 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([1x36x180x320xf32, 1x36x180x320xf32, 1x36x180x320xf32]) <- (1x108x180x320xf32, 1xi32)
        split_with_num_6 = paddle._C_ops.split_with_num(add__212, 3, full_175)

        # builtin.slice: (1x36x180x320xf32) <- ([1x36x180x320xf32, 1x36x180x320xf32, 1x36x180x320xf32])
        slice_59 = split_with_num_6[0]

        # builtin.slice: (1x36x180x320xf32) <- ([1x36x180x320xf32, 1x36x180x320xf32, 1x36x180x320xf32])
        slice_60 = split_with_num_6[1]

        # builtin.combine: ([1x36x180x320xf32, 1x36x180x320xf32]) <- (1x36x180x320xf32, 1x36x180x320xf32)
        combine_56 = [slice_59, slice_60]

        # pd_op.full: (1xi32) <- ()
        full_176 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x72x180x320xf32) <- ([1x36x180x320xf32, 1x36x180x320xf32], 1xi32)
        concat_35 = paddle._C_ops.concat(combine_56, full_176)

        # pd_op.tanh_: (1x72x180x320xf32) <- (1x72x180x320xf32)
        tanh__5 = paddle._C_ops.tanh_(concat_35)

        # pd_op.full: (1xf32) <- ()
        full_177 = paddle._C_ops.full([1], float('10'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x72x180x320xf32) <- (1x72x180x320xf32, 1xf32)
        scale__83 = paddle._C_ops.scale_(tanh__5, full_177, float('0'), True)

        # pd_op.add_: (1x72x180x320xf32) <- (1x72x180x320xf32, 1x72x180x320xf32)
        add__213 = paddle._C_ops.add_(scale__83, add__128)

        # builtin.slice: (1x36x180x320xf32) <- ([1x36x180x320xf32, 1x36x180x320xf32, 1x36x180x320xf32])
        slice_61 = split_with_num_6[2]

        # pd_op.sigmoid_: (1x36x180x320xf32) <- (1x36x180x320xf32)
        sigmoid__5 = paddle._C_ops.sigmoid_(slice_61)

        # pd_op.add_: (1x36x180x320xf32) <- (1x36x180x320xf32, 1x36x180x320xf32)
        add__214 = paddle._C_ops.add_(sigmoid__5, sigmoid__3)

        # pd_op.full: (1xf32) <- ()
        full_178 = paddle._C_ops.full([1], float('0.5'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x36x180x320xf32) <- (1x36x180x320xf32, 1xf32)
        scale__84 = paddle._C_ops.scale_(add__214, full_178, float('0'), True)

        # pd_op.deformable_conv: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x72x180x320xf32, 32x32x3x3xf32, 1x36x180x320xf32)
        deformable_conv_7 = paddle._C_ops.deformable_conv(add__207, add__213, parameter_228, scale__84, [1, 1], [1, 1], [1, 1], 4, 1, 1)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_233 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xi64) <- (32xf32, 4xi64)
        reshape_316, reshape_317 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_229, full_int_array_233), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__215 = paddle._C_ops.add_(deformable_conv_7, reshape_316)

        # builtin.combine: ([1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32]) <- (1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32)
        combine_57 = [add__166, add__65, add__93, add__140, add__177, add__215]

        # pd_op.full: (1xi32) <- ()
        full_179 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x192x180x320xf32) <- ([1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32], 1xi32)
        concat_36 = paddle._C_ops.concat(combine_57, full_179)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x192x180x320xf32, 32x192x3x3xf32)
        conv2d_150 = paddle._C_ops.conv2d(concat_36, parameter_204, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_234 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_318, reshape_319 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_205, full_int_array_234), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__216 = paddle._C_ops.add_(conv2d_150, reshape_318)

        # pd_op.leaky_relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        leaky_relu__71 = paddle._C_ops.leaky_relu_(add__216, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_151 = paddle._C_ops.conv2d(leaky_relu__71, parameter_206, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_235 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_320, reshape_321 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_207, full_int_array_235), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__217 = paddle._C_ops.add_(conv2d_151, reshape_320)

        # pd_op.relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        relu__31 = paddle._C_ops.relu_(add__217)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_152 = paddle._C_ops.conv2d(relu__31, parameter_208, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_236 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_322, reshape_323 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_209, full_int_array_236), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__218 = paddle._C_ops.add_(conv2d_152, reshape_322)

        # pd_op.full: (1xf32) <- ()
        full_180 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1xf32)
        scale__85 = paddle._C_ops.scale_(add__218, full_180, float('0'), True)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__219 = paddle._C_ops.add_(leaky_relu__71, scale__85)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_153 = paddle._C_ops.conv2d(add__219, parameter_210, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_237 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_324, reshape_325 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_211, full_int_array_237), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__220 = paddle._C_ops.add_(conv2d_153, reshape_324)

        # pd_op.relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        relu__32 = paddle._C_ops.relu_(add__220)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_154 = paddle._C_ops.conv2d(relu__32, parameter_212, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_238 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_326, reshape_327 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_213, full_int_array_238), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__221 = paddle._C_ops.add_(conv2d_154, reshape_326)

        # pd_op.full: (1xf32) <- ()
        full_181 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1xf32)
        scale__86 = paddle._C_ops.scale_(add__221, full_181, float('0'), True)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__222 = paddle._C_ops.add_(add__219, scale__86)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_155 = paddle._C_ops.conv2d(add__222, parameter_214, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_239 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_328, reshape_329 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_215, full_int_array_239), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__223 = paddle._C_ops.add_(conv2d_155, reshape_328)

        # pd_op.relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        relu__33 = paddle._C_ops.relu_(add__223)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_156 = paddle._C_ops.conv2d(relu__33, parameter_216, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_240 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_330, reshape_331 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_217, full_int_array_240), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__224 = paddle._C_ops.add_(conv2d_156, reshape_330)

        # pd_op.full: (1xf32) <- ()
        full_182 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1xf32)
        scale__87 = paddle._C_ops.scale_(add__224, full_182, float('0'), True)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__225 = paddle._C_ops.add_(add__222, scale__87)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__226 = paddle._C_ops.add_(add__215, add__225)

        # builtin.combine: ([1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32]) <- (1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32)
        combine_58 = [slice_2, add__82, add__111, add__122, add__196, add__207]

        # pd_op.full: (1xi32) <- ()
        full_183 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x192x180x320xf32) <- ([1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32], 1xi32)
        concat_37 = paddle._C_ops.concat(combine_58, full_183)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x192x180x320xf32, 32x192x3x3xf32)
        conv2d_157 = paddle._C_ops.conv2d(concat_37, parameter_230, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_241 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_332, reshape_333 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_231, full_int_array_241), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__227 = paddle._C_ops.add_(conv2d_157, reshape_332)

        # pd_op.leaky_relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        leaky_relu__72 = paddle._C_ops.leaky_relu_(add__227, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_158 = paddle._C_ops.conv2d(leaky_relu__72, parameter_232, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_242 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_334, reshape_335 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_233, full_int_array_242), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__228 = paddle._C_ops.add_(conv2d_158, reshape_334)

        # pd_op.relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        relu__34 = paddle._C_ops.relu_(add__228)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_159 = paddle._C_ops.conv2d(relu__34, parameter_234, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_243 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_336, reshape_337 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_235, full_int_array_243), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__229 = paddle._C_ops.add_(conv2d_159, reshape_336)

        # pd_op.full: (1xf32) <- ()
        full_184 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1xf32)
        scale__88 = paddle._C_ops.scale_(add__229, full_184, float('0'), True)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__230 = paddle._C_ops.add_(leaky_relu__72, scale__88)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_160 = paddle._C_ops.conv2d(add__230, parameter_236, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_244 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_338, reshape_339 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_237, full_int_array_244), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__231 = paddle._C_ops.add_(conv2d_160, reshape_338)

        # pd_op.relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        relu__35 = paddle._C_ops.relu_(add__231)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_161 = paddle._C_ops.conv2d(relu__35, parameter_238, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_245 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_340, reshape_341 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_239, full_int_array_245), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__232 = paddle._C_ops.add_(conv2d_161, reshape_340)

        # pd_op.full: (1xf32) <- ()
        full_185 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1xf32)
        scale__89 = paddle._C_ops.scale_(add__232, full_185, float('0'), True)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__233 = paddle._C_ops.add_(add__230, scale__89)

        # pd_op.conv2d: (1x128x180x320xf32) <- (1x32x180x320xf32, 128x32x3x3xf32)
        conv2d_162 = paddle._C_ops.conv2d(add__233, parameter_240, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_246 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_342, reshape_343 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_241, full_int_array_246), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x128x180x320xf32) <- (1x128x180x320xf32, 1x128x1x1xf32)
        add__234 = paddle._C_ops.add_(conv2d_162, reshape_342)

        # pd_op.pixel_shuffle: (1x32x360x640xf32) <- (1x128x180x320xf32)
        pixel_shuffle_4 = paddle._C_ops.pixel_shuffle(add__234, 2, 'NCHW')

        # pd_op.leaky_relu_: (1x32x360x640xf32) <- (1x32x360x640xf32)
        leaky_relu__73 = paddle._C_ops.leaky_relu_(pixel_shuffle_4, float('0.1'))

        # pd_op.conv2d: (1x128x360x640xf32) <- (1x32x360x640xf32, 128x32x3x3xf32)
        conv2d_163 = paddle._C_ops.conv2d(leaky_relu__73, parameter_242, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_247 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_344, reshape_345 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_243, full_int_array_247), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x128x360x640xf32) <- (1x128x360x640xf32, 1x128x1x1xf32)
        add__235 = paddle._C_ops.add_(conv2d_163, reshape_344)

        # pd_op.pixel_shuffle: (1x32x720x1280xf32) <- (1x128x360x640xf32)
        pixel_shuffle_5 = paddle._C_ops.pixel_shuffle(add__235, 2, 'NCHW')

        # pd_op.leaky_relu_: (1x32x720x1280xf32) <- (1x32x720x1280xf32)
        leaky_relu__74 = paddle._C_ops.leaky_relu_(pixel_shuffle_5, float('0.1'))

        # pd_op.conv2d: (1x3x720x1280xf32) <- (1x32x720x1280xf32, 3x32x3x3xf32)
        conv2d_164 = paddle._C_ops.conv2d(leaky_relu__74, parameter_244, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_248 = [1, 3, 1, 1]

        # pd_op.reshape: (1x3x1x1xf32, 0x3xf32) <- (3xf32, 4xi64)
        reshape_346, reshape_347 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_245, full_int_array_248), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x3x720x1280xf32) <- (1x3x720x1280xf32, 1x3x1x1xf32)
        add__236 = paddle._C_ops.add_(conv2d_164, reshape_346)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_249 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_250 = [1]

        # pd_op.slice: (1x3x180x320xf32) <- (1x2x3x180x320xf32, 1xi64, 1xi64)
        slice_62 = paddle._C_ops.slice(feed_0, [1], full_int_array_249, full_int_array_250, [1], [1])

        # pd_op.bilinear_interp: (1x3x720x1280xf32) <- (1x3x180x320xf32, None, None, None)
        bilinear_interp_7 = paddle._C_ops.bilinear_interp(slice_62, None, None, None, 'NCHW', -1, -1, -1, [float('4'), float('4')], 'bilinear', False, 0)

        # pd_op.add_: (1x3x720x1280xf32) <- (1x3x720x1280xf32, 1x3x720x1280xf32)
        add__237 = paddle._C_ops.add_(add__236, bilinear_interp_7)

        # pd_op.conv2d: (1x128x180x320xf32) <- (1x32x180x320xf32, 128x32x3x3xf32)
        conv2d_165 = paddle._C_ops.conv2d(add__82, parameter_162, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_251 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_348, reshape_349 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_163, full_int_array_251), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x128x180x320xf32) <- (1x128x180x320xf32, 1x128x1x1xf32)
        add__238 = paddle._C_ops.add_(conv2d_165, reshape_348)

        # pd_op.pixel_shuffle: (1x32x360x640xf32) <- (1x128x180x320xf32)
        pixel_shuffle_6 = paddle._C_ops.pixel_shuffle(add__238, 2, 'NCHW')

        # pd_op.leaky_relu_: (1x32x360x640xf32) <- (1x32x360x640xf32)
        leaky_relu__75 = paddle._C_ops.leaky_relu_(pixel_shuffle_6, float('0.1'))

        # pd_op.conv2d: (1x128x360x640xf32) <- (1x32x360x640xf32, 128x32x3x3xf32)
        conv2d_166 = paddle._C_ops.conv2d(leaky_relu__75, parameter_164, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_252 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_350, reshape_351 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_165, full_int_array_252), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x128x360x640xf32) <- (1x128x360x640xf32, 1x128x1x1xf32)
        add__239 = paddle._C_ops.add_(conv2d_166, reshape_350)

        # pd_op.pixel_shuffle: (1x32x720x1280xf32) <- (1x128x360x640xf32)
        pixel_shuffle_7 = paddle._C_ops.pixel_shuffle(add__239, 2, 'NCHW')

        # pd_op.leaky_relu_: (1x32x720x1280xf32) <- (1x32x720x1280xf32)
        leaky_relu__76 = paddle._C_ops.leaky_relu_(pixel_shuffle_7, float('0.1'))

        # pd_op.conv2d: (1x3x720x1280xf32) <- (1x32x720x1280xf32, 3x32x3x3xf32)
        conv2d_167 = paddle._C_ops.conv2d(leaky_relu__76, parameter_246, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_253 = [1, 3, 1, 1]

        # pd_op.reshape: (1x3x1x1xf32, 0x3xf32) <- (3xf32, 4xi64)
        reshape_352, reshape_353 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_247, full_int_array_253), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x3x720x1280xf32) <- (1x3x720x1280xf32, 1x3x1x1xf32)
        add__240 = paddle._C_ops.add_(conv2d_167, reshape_352)

        # pd_op.add_: (1x3x720x1280xf32) <- (1x3x720x1280xf32, 1x3x720x1280xf32)
        add__241 = paddle._C_ops.add_(add__240, add__237)

        # builtin.combine: ([1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32]) <- (1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32)
        combine_59 = [slice_3, add__65, add__93, add__140, add__177, add__226]

        # pd_op.full: (1xi32) <- ()
        full_186 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x192x180x320xf32) <- ([1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32, 1x32x180x320xf32], 1xi32)
        concat_38 = paddle._C_ops.concat(combine_59, full_186)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x192x180x320xf32, 32x192x3x3xf32)
        conv2d_168 = paddle._C_ops.conv2d(concat_38, parameter_230, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_254 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_354, reshape_355 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_231, full_int_array_254), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__242 = paddle._C_ops.add_(conv2d_168, reshape_354)

        # pd_op.leaky_relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        leaky_relu__77 = paddle._C_ops.leaky_relu_(add__242, float('0.1'))

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_169 = paddle._C_ops.conv2d(leaky_relu__77, parameter_232, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_255 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_356, reshape_357 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_233, full_int_array_255), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__243 = paddle._C_ops.add_(conv2d_169, reshape_356)

        # pd_op.relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        relu__36 = paddle._C_ops.relu_(add__243)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_170 = paddle._C_ops.conv2d(relu__36, parameter_234, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_256 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_358, reshape_359 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_235, full_int_array_256), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__244 = paddle._C_ops.add_(conv2d_170, reshape_358)

        # pd_op.full: (1xf32) <- ()
        full_187 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1xf32)
        scale__90 = paddle._C_ops.scale_(add__244, full_187, float('0'), True)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__245 = paddle._C_ops.add_(leaky_relu__77, scale__90)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_171 = paddle._C_ops.conv2d(add__245, parameter_236, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_257 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_360, reshape_361 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_237, full_int_array_257), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__246 = paddle._C_ops.add_(conv2d_171, reshape_360)

        # pd_op.relu_: (1x32x180x320xf32) <- (1x32x180x320xf32)
        relu__37 = paddle._C_ops.relu_(add__246)

        # pd_op.conv2d: (1x32x180x320xf32) <- (1x32x180x320xf32, 32x32x3x3xf32)
        conv2d_172 = paddle._C_ops.conv2d(relu__37, parameter_238, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_258 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf32, 0x32xf32) <- (32xf32, 4xi64)
        reshape_362, reshape_363 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_239, full_int_array_258), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x1x1xf32)
        add__247 = paddle._C_ops.add_(conv2d_172, reshape_362)

        # pd_op.full: (1xf32) <- ()
        full_188 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1xf32)
        scale__91 = paddle._C_ops.scale_(add__247, full_188, float('0'), True)

        # pd_op.add_: (1x32x180x320xf32) <- (1x32x180x320xf32, 1x32x180x320xf32)
        add__248 = paddle._C_ops.add_(add__245, scale__91)

        # pd_op.conv2d: (1x128x180x320xf32) <- (1x32x180x320xf32, 128x32x3x3xf32)
        conv2d_173 = paddle._C_ops.conv2d(add__248, parameter_240, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_259 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_364, reshape_365 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_241, full_int_array_259), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x128x180x320xf32) <- (1x128x180x320xf32, 1x128x1x1xf32)
        add__249 = paddle._C_ops.add_(conv2d_173, reshape_364)

        # pd_op.pixel_shuffle: (1x32x360x640xf32) <- (1x128x180x320xf32)
        pixel_shuffle_8 = paddle._C_ops.pixel_shuffle(add__249, 2, 'NCHW')

        # pd_op.leaky_relu_: (1x32x360x640xf32) <- (1x32x360x640xf32)
        leaky_relu__78 = paddle._C_ops.leaky_relu_(pixel_shuffle_8, float('0.1'))

        # pd_op.conv2d: (1x128x360x640xf32) <- (1x32x360x640xf32, 128x32x3x3xf32)
        conv2d_174 = paddle._C_ops.conv2d(leaky_relu__78, parameter_242, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_260 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_366, reshape_367 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_243, full_int_array_260), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x128x360x640xf32) <- (1x128x360x640xf32, 1x128x1x1xf32)
        add__250 = paddle._C_ops.add_(conv2d_174, reshape_366)

        # pd_op.pixel_shuffle: (1x32x720x1280xf32) <- (1x128x360x640xf32)
        pixel_shuffle_9 = paddle._C_ops.pixel_shuffle(add__250, 2, 'NCHW')

        # pd_op.leaky_relu_: (1x32x720x1280xf32) <- (1x32x720x1280xf32)
        leaky_relu__79 = paddle._C_ops.leaky_relu_(pixel_shuffle_9, float('0.1'))

        # pd_op.conv2d: (1x3x720x1280xf32) <- (1x32x720x1280xf32, 3x32x3x3xf32)
        conv2d_175 = paddle._C_ops.conv2d(leaky_relu__79, parameter_244, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_261 = [1, 3, 1, 1]

        # pd_op.reshape: (1x3x1x1xf32, 0x3xf32) <- (3xf32, 4xi64)
        reshape_368, reshape_369 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_245, full_int_array_261), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x3x720x1280xf32) <- (1x3x720x1280xf32, 1x3x1x1xf32)
        add__251 = paddle._C_ops.add_(conv2d_175, reshape_368)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_262 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_263 = [2]

        # pd_op.slice: (1x3x180x320xf32) <- (1x2x3x180x320xf32, 1xi64, 1xi64)
        slice_63 = paddle._C_ops.slice(feed_0, [1], full_int_array_262, full_int_array_263, [1], [1])

        # pd_op.bilinear_interp: (1x3x720x1280xf32) <- (1x3x180x320xf32, None, None, None)
        bilinear_interp_8 = paddle._C_ops.bilinear_interp(slice_63, None, None, None, 'NCHW', -1, -1, -1, [float('4'), float('4')], 'bilinear', False, 0)

        # pd_op.add_: (1x3x720x1280xf32) <- (1x3x720x1280xf32, 1x3x720x1280xf32)
        add__252 = paddle._C_ops.add_(add__251, bilinear_interp_8)

        # pd_op.conv2d: (1x128x180x320xf32) <- (1x32x180x320xf32, 128x32x3x3xf32)
        conv2d_176 = paddle._C_ops.conv2d(add__65, parameter_162, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_264 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_370, reshape_371 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_163, full_int_array_264), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x128x180x320xf32) <- (1x128x180x320xf32, 1x128x1x1xf32)
        add__253 = paddle._C_ops.add_(conv2d_176, reshape_370)

        # pd_op.pixel_shuffle: (1x32x360x640xf32) <- (1x128x180x320xf32)
        pixel_shuffle_10 = paddle._C_ops.pixel_shuffle(add__253, 2, 'NCHW')

        # pd_op.leaky_relu_: (1x32x360x640xf32) <- (1x32x360x640xf32)
        leaky_relu__80 = paddle._C_ops.leaky_relu_(pixel_shuffle_10, float('0.1'))

        # pd_op.conv2d: (1x128x360x640xf32) <- (1x32x360x640xf32, 128x32x3x3xf32)
        conv2d_177 = paddle._C_ops.conv2d(leaky_relu__80, parameter_164, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_265 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_372, reshape_373 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_165, full_int_array_265), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x128x360x640xf32) <- (1x128x360x640xf32, 1x128x1x1xf32)
        add__254 = paddle._C_ops.add_(conv2d_177, reshape_372)

        # pd_op.pixel_shuffle: (1x32x720x1280xf32) <- (1x128x360x640xf32)
        pixel_shuffle_11 = paddle._C_ops.pixel_shuffle(add__254, 2, 'NCHW')

        # pd_op.leaky_relu_: (1x32x720x1280xf32) <- (1x32x720x1280xf32)
        leaky_relu__81 = paddle._C_ops.leaky_relu_(pixel_shuffle_11, float('0.1'))

        # pd_op.conv2d: (1x3x720x1280xf32) <- (1x32x720x1280xf32, 3x32x3x3xf32)
        conv2d_178 = paddle._C_ops.conv2d(leaky_relu__81, parameter_246, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_266 = [1, 3, 1, 1]

        # pd_op.reshape: (1x3x1x1xf32, 0x3xf32) <- (3xf32, 4xi64)
        reshape_374, reshape_375 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_247, full_int_array_266), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (1x3x720x1280xf32) <- (1x3x720x1280xf32, 1x3x1x1xf32)
        add__255 = paddle._C_ops.add_(conv2d_178, reshape_374)

        # pd_op.add_: (1x3x720x1280xf32) <- (1x3x720x1280xf32, 1x3x720x1280xf32)
        add__256 = paddle._C_ops.add_(add__255, add__252)

        # builtin.combine: ([1x3x720x1280xf32, 1x3x720x1280xf32]) <- (1x3x720x1280xf32, 1x3x720x1280xf32)
        combine_60 = [add__148, add__161]

        # pd_op.stack: (1x2x3x720x1280xf32) <- ([1x3x720x1280xf32, 1x3x720x1280xf32])
        stack_14 = paddle._C_ops.stack(combine_60, 1)

        # builtin.combine: ([1x3x720x1280xf32, 1x3x720x1280xf32]) <- (1x3x720x1280xf32, 1x3x720x1280xf32)
        combine_61 = [add__241, add__256]

        # pd_op.stack: (1x2x3x720x1280xf32) <- ([1x3x720x1280xf32, 1x3x720x1280xf32])
        stack_15 = paddle._C_ops.stack(combine_61, 1)
        return stack_14, stack_15



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

    def forward(self, parameter_0, parameter_1, parameter_2, parameter_3, parameter_4, parameter_5, parameter_6, parameter_7, parameter_8, parameter_9, parameter_10, parameter_11, parameter_12, parameter_13, parameter_14, parameter_15, parameter_16, parameter_17, parameter_18, parameter_19, parameter_20, parameter_21, parameter_22, parameter_23, parameter_24, parameter_25, parameter_26, parameter_27, parameter_28, parameter_29, parameter_30, parameter_31, parameter_32, parameter_33, parameter_34, parameter_35, parameter_36, parameter_37, parameter_38, parameter_39, parameter_40, parameter_41, parameter_42, parameter_43, parameter_44, parameter_45, parameter_46, parameter_47, parameter_48, parameter_49, parameter_50, parameter_51, parameter_52, parameter_53, parameter_54, parameter_55, parameter_56, parameter_57, parameter_58, parameter_59, parameter_60, parameter_61, parameter_62, parameter_63, parameter_64, parameter_65, parameter_66, parameter_67, parameter_68, parameter_69, parameter_70, parameter_71, parameter_72, parameter_73, parameter_74, parameter_75, parameter_76, parameter_77, parameter_78, parameter_79, parameter_80, parameter_81, parameter_82, parameter_83, parameter_84, parameter_85, parameter_86, parameter_87, parameter_88, parameter_89, parameter_90, parameter_91, parameter_92, parameter_93, parameter_94, parameter_95, parameter_96, parameter_97, parameter_98, parameter_99, parameter_100, parameter_101, parameter_102, parameter_103, parameter_104, parameter_105, parameter_106, parameter_107, parameter_108, parameter_109, parameter_110, parameter_111, parameter_112, parameter_113, parameter_114, parameter_115, parameter_116, parameter_117, parameter_118, parameter_119, parameter_120, parameter_121, parameter_122, parameter_123, parameter_124, parameter_125, parameter_126, parameter_127, parameter_128, parameter_129, parameter_130, parameter_131, parameter_132, parameter_133, parameter_134, parameter_135, parameter_136, parameter_137, parameter_138, parameter_139, parameter_140, parameter_141, parameter_142, parameter_143, parameter_144, parameter_145, parameter_146, parameter_147, parameter_148, parameter_149, parameter_150, parameter_151, parameter_152, parameter_153, parameter_154, parameter_155, parameter_156, parameter_157, parameter_158, parameter_159, parameter_160, parameter_161, parameter_162, parameter_163, parameter_164, parameter_165, parameter_166, parameter_167, parameter_168, parameter_169, parameter_170, parameter_171, parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179, parameter_180, parameter_181, parameter_182, parameter_183, parameter_184, parameter_185, parameter_186, parameter_187, parameter_188, parameter_189, parameter_190, parameter_191, parameter_192, parameter_193, parameter_194, parameter_195, parameter_196, parameter_197, parameter_198, parameter_199, parameter_200, parameter_201, parameter_202, parameter_203, parameter_204, parameter_205, parameter_206, parameter_207, parameter_208, parameter_209, parameter_210, parameter_211, parameter_212, parameter_213, parameter_214, parameter_215, parameter_216, parameter_217, parameter_218, parameter_219, parameter_220, parameter_221, parameter_222, parameter_223, parameter_224, parameter_225, parameter_226, parameter_227, parameter_228, parameter_229, parameter_230, parameter_231, parameter_232, parameter_233, parameter_234, parameter_235, parameter_236, parameter_237, parameter_238, parameter_239, parameter_240, parameter_241, parameter_242, parameter_243, parameter_244, parameter_245, parameter_246, parameter_247, feed_0):
        return self.builtin_module_2208_0_0(parameter_0, parameter_1, parameter_2, parameter_3, parameter_4, parameter_5, parameter_6, parameter_7, parameter_8, parameter_9, parameter_10, parameter_11, parameter_12, parameter_13, parameter_14, parameter_15, parameter_16, parameter_17, parameter_18, parameter_19, parameter_20, parameter_21, parameter_22, parameter_23, parameter_24, parameter_25, parameter_26, parameter_27, parameter_28, parameter_29, parameter_30, parameter_31, parameter_32, parameter_33, parameter_34, parameter_35, parameter_36, parameter_37, parameter_38, parameter_39, parameter_40, parameter_41, parameter_42, parameter_43, parameter_44, parameter_45, parameter_46, parameter_47, parameter_48, parameter_49, parameter_50, parameter_51, parameter_52, parameter_53, parameter_54, parameter_55, parameter_56, parameter_57, parameter_58, parameter_59, parameter_60, parameter_61, parameter_62, parameter_63, parameter_64, parameter_65, parameter_66, parameter_67, parameter_68, parameter_69, parameter_70, parameter_71, parameter_72, parameter_73, parameter_74, parameter_75, parameter_76, parameter_77, parameter_78, parameter_79, parameter_80, parameter_81, parameter_82, parameter_83, parameter_84, parameter_85, parameter_86, parameter_87, parameter_88, parameter_89, parameter_90, parameter_91, parameter_92, parameter_93, parameter_94, parameter_95, parameter_96, parameter_97, parameter_98, parameter_99, parameter_100, parameter_101, parameter_102, parameter_103, parameter_104, parameter_105, parameter_106, parameter_107, parameter_108, parameter_109, parameter_110, parameter_111, parameter_112, parameter_113, parameter_114, parameter_115, parameter_116, parameter_117, parameter_118, parameter_119, parameter_120, parameter_121, parameter_122, parameter_123, parameter_124, parameter_125, parameter_126, parameter_127, parameter_128, parameter_129, parameter_130, parameter_131, parameter_132, parameter_133, parameter_134, parameter_135, parameter_136, parameter_137, parameter_138, parameter_139, parameter_140, parameter_141, parameter_142, parameter_143, parameter_144, parameter_145, parameter_146, parameter_147, parameter_148, parameter_149, parameter_150, parameter_151, parameter_152, parameter_153, parameter_154, parameter_155, parameter_156, parameter_157, parameter_158, parameter_159, parameter_160, parameter_161, parameter_162, parameter_163, parameter_164, parameter_165, parameter_166, parameter_167, parameter_168, parameter_169, parameter_170, parameter_171, parameter_172, parameter_173, parameter_174, parameter_175, parameter_176, parameter_177, parameter_178, parameter_179, parameter_180, parameter_181, parameter_182, parameter_183, parameter_184, parameter_185, parameter_186, parameter_187, parameter_188, parameter_189, parameter_190, parameter_191, parameter_192, parameter_193, parameter_194, parameter_195, parameter_196, parameter_197, parameter_198, parameter_199, parameter_200, parameter_201, parameter_202, parameter_203, parameter_204, parameter_205, parameter_206, parameter_207, parameter_208, parameter_209, parameter_210, parameter_211, parameter_212, parameter_213, parameter_214, parameter_215, parameter_216, parameter_217, parameter_218, parameter_219, parameter_220, parameter_221, parameter_222, parameter_223, parameter_224, parameter_225, parameter_226, parameter_227, parameter_228, parameter_229, parameter_230, parameter_231, parameter_232, parameter_233, parameter_234, parameter_235, parameter_236, parameter_237, parameter_238, parameter_239, parameter_240, parameter_241, parameter_242, parameter_243, parameter_244, parameter_245, parameter_246, parameter_247, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_2208_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([32, 3, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([1, 3, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([1, 3, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([16, 8, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([16, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([32, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([16, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([16, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([16, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([8, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([8], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([8, 8, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([8], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([2, 8, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([2], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([16, 8, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([16, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([32, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([16, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([16, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([16, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([8, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([8], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([8, 8, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([8], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([2, 8, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([2], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([16, 8, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([16, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([32, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([16, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([16, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([16, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([8, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([8], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([8, 8, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([8], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([2, 8, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([2], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([32, 66, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([216, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([216], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([32, 96, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([32, 96, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([32, 66, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([108, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([108], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([32, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_139
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([32, 66, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_149
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([108, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([108], dtype='float32', min=0, max=0.5),
            # parameter_154
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([32, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([128, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([128, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([3, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([3], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([32, 35, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([32, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([32, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([32, 160, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_189
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.uniform([32, 66, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([108, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([108], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([32, 192, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_209
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_213
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_214
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_215
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_217
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_218
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_219
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([32, 66, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_221
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_223
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_224
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_225
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([108, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_227
            paddle.uniform([108], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_229
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([32, 192, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_233
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_234
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_238
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_239
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([128, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_241
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([128, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_243
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_244
            paddle.uniform([3, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_245
            paddle.uniform([3], dtype='float32', min=0, max=0.5),
            # parameter_246
            paddle.uniform([3, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([3], dtype='float32', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 2, 3, 180, 320], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_0
            paddle.static.InputSpec(shape=[32, 3, 3, 3], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[1, 3, 1, 1], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[1, 3, 1, 1], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[16, 8, 3, 3], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[16, 16, 3, 3], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[32, 16, 3, 3], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[16, 32, 3, 3], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[16, 16, 3, 3], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[16, 16, 3, 3], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[8, 16, 3, 3], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[8], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[8, 8, 3, 3], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[8], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[2, 8, 3, 3], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[2], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[16, 8, 3, 3], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[16, 16, 3, 3], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[32, 16, 3, 3], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[16, 32, 3, 3], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[16, 16, 3, 3], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[16, 16, 3, 3], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[8, 16, 3, 3], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[8], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[8, 8, 3, 3], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[8], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[2, 8, 3, 3], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[2], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[16, 8, 3, 3], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[16, 16, 3, 3], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[32, 16, 3, 3], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[16, 32, 3, 3], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[16, 16, 3, 3], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[16, 16, 3, 3], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[8, 16, 3, 3], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[8], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[8, 8, 3, 3], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[8], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[2, 8, 3, 3], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[2], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[32, 66, 3, 3], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[216, 32, 3, 3], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[216], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[32, 96, 3, 3], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[32, 96, 3, 3], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[32, 66, 3, 3], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[108, 32, 3, 3], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[108], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[32, 128, 3, 3], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_139
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[32, 66, 3, 3], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_149
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[108, 32, 3, 3], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[108], dtype='float32'),
            # parameter_154
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[32, 128, 3, 3], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[128, 32, 3, 3], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[128, 32, 3, 3], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[3, 32, 3, 3], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[3], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[32, 35, 3, 3], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[32, 64, 3, 3], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[32, 64, 3, 3], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[32, 160, 3, 3], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_189
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[32, 66, 3, 3], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[108, 32, 3, 3], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[108], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[32, 192, 3, 3], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_209
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_213
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_214
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_215
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_217
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_218
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_219
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[32, 66, 3, 3], dtype='float32'),
            # parameter_221
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_223
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_224
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_225
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[108, 32, 3, 3], dtype='float32'),
            # parameter_227
            paddle.static.InputSpec(shape=[108], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_229
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[32, 192, 3, 3], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_233
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_234
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_238
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_239
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[128, 32, 3, 3], dtype='float32'),
            # parameter_241
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[128, 32, 3, 3], dtype='float32'),
            # parameter_243
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_244
            paddle.static.InputSpec(shape=[3, 32, 3, 3], dtype='float32'),
            # parameter_245
            paddle.static.InputSpec(shape=[3], dtype='float32'),
            # parameter_246
            paddle.static.InputSpec(shape=[3, 32, 3, 3], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[3], dtype='float32'),
            # feed_0
            paddle.static.InputSpec(shape=[1, 2, 3, 180, 320], dtype='float32'),
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