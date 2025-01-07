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
    return [150][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_469_0_0(self, ):

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full([1], float('152'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.arange: (152xi64) <- (1xf32, 1xf32, 1xf32)
        arange_0 = paddle.arange(full_0, full_1, full_2, dtype='int64')

        # pd_op.cast: (152xf32) <- (152xi64)
        cast_0 = paddle._C_ops.cast(arange_0, paddle.float32)

        # pd_op.scale: (152xf32) <- (152xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(cast_0, full_2, float('0.5'), True)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('8'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (152xf32) <- (152xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(scale_0, full_3, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('100'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.arange: (100xi64) <- (1xf32, 1xf32, 1xf32)
        arange_1 = paddle.arange(full_0, full_4, full_2, dtype='int64')

        # pd_op.cast: (100xf32) <- (100xi64)
        cast_1 = paddle._C_ops.cast(arange_1, paddle.float32)

        # pd_op.scale: (100xf32) <- (100xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(cast_1, full_2, float('0.5'), True)

        # pd_op.scale: (100xf32) <- (100xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(scale_2, full_3, float('0'), True)

        # builtin.combine: ([100xf32, 152xf32]) <- (100xf32, 152xf32)
        combine_0 = [scale_3, scale_1]

        # pd_op.meshgrid: ([100x152xf32, 100x152xf32]) <- ([100xf32, 152xf32])
        meshgrid_0 = paddle._C_ops.meshgrid(combine_0)

        # builtin.split: (100x152xf32, 100x152xf32) <- ([100x152xf32, 100x152xf32])
        split_0, split_1, = meshgrid_0

        # pd_op.scale: (100x152xf32) <- (100x152xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(split_1, full_2, float('-32'), True)

        # pd_op.scale: (100x152xf32) <- (100x152xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(split_0, full_2, float('-32'), True)

        # pd_op.scale: (100x152xf32) <- (100x152xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(split_1, full_2, float('32'), True)

        # pd_op.scale: (100x152xf32) <- (100x152xf32, 1xf32)
        scale_7 = paddle._C_ops.scale(split_0, full_2, float('32'), True)

        # builtin.combine: ([100x152xf32, 100x152xf32, 100x152xf32, 100x152xf32]) <- (100x152xf32, 100x152xf32, 100x152xf32, 100x152xf32)
        combine_1 = [scale_4, scale_5, scale_6, scale_7]

        # pd_op.stack: (100x152x4xf32) <- ([100x152xf32, 100x152xf32, 100x152xf32, 100x152xf32])
        stack_0 = paddle._C_ops.stack(combine_1, -1)

        # pd_op.cast: (100x152x4xf32) <- (100x152x4xf32)
        cast_2 = paddle._C_ops.cast(stack_0, paddle.float32)

        # builtin.combine: ([100x152xf32, 100x152xf32]) <- (100x152xf32, 100x152xf32)
        combine_2 = [split_1, split_0]

        # pd_op.stack: (100x152x2xf32) <- ([100x152xf32, 100x152xf32])
        stack_1 = paddle._C_ops.stack(combine_2, -1)

        # pd_op.cast: (100x152x2xf32) <- (100x152x2xf32)
        cast_3 = paddle._C_ops.cast(stack_1, paddle.float32)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [-1, 4]

        # pd_op.reshape: (15200x4xf32, 0x100x152x4xi64) <- (100x152x4xf32, 2xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(cast_2, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [-1, 2]

        # pd_op.reshape: (15200x2xf32, 0x100x152x2xi64) <- (100x152x2xf32, 2xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(cast_3, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (15200x1xf32) <- ()
        full_5 = paddle._C_ops.full([15200, 1], float('8'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full([1], float('76'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.arange: (76xi64) <- (1xf32, 1xf32, 1xf32)
        arange_2 = paddle.arange(full_0, full_6, full_2, dtype='int64')

        # pd_op.cast: (76xf32) <- (76xi64)
        cast_4 = paddle._C_ops.cast(arange_2, paddle.float32)

        # pd_op.scale: (76xf32) <- (76xf32, 1xf32)
        scale_8 = paddle._C_ops.scale(cast_4, full_2, float('0.5'), True)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('16'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (76xf32) <- (76xf32, 1xf32)
        scale_9 = paddle._C_ops.scale(scale_8, full_7, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full([1], float('50'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.arange: (50xi64) <- (1xf32, 1xf32, 1xf32)
        arange_3 = paddle.arange(full_0, full_8, full_2, dtype='int64')

        # pd_op.cast: (50xf32) <- (50xi64)
        cast_5 = paddle._C_ops.cast(arange_3, paddle.float32)

        # pd_op.scale: (50xf32) <- (50xf32, 1xf32)
        scale_10 = paddle._C_ops.scale(cast_5, full_2, float('0.5'), True)

        # pd_op.scale: (50xf32) <- (50xf32, 1xf32)
        scale_11 = paddle._C_ops.scale(scale_10, full_7, float('0'), True)

        # builtin.combine: ([50xf32, 76xf32]) <- (50xf32, 76xf32)
        combine_3 = [scale_11, scale_9]

        # pd_op.meshgrid: ([50x76xf32, 50x76xf32]) <- ([50xf32, 76xf32])
        meshgrid_1 = paddle._C_ops.meshgrid(combine_3)

        # builtin.split: (50x76xf32, 50x76xf32) <- ([50x76xf32, 50x76xf32])
        split_2, split_3, = meshgrid_1

        # pd_op.scale: (50x76xf32) <- (50x76xf32, 1xf32)
        scale_12 = paddle._C_ops.scale(split_3, full_2, float('-64'), True)

        # pd_op.scale: (50x76xf32) <- (50x76xf32, 1xf32)
        scale_13 = paddle._C_ops.scale(split_2, full_2, float('-64'), True)

        # pd_op.scale: (50x76xf32) <- (50x76xf32, 1xf32)
        scale_14 = paddle._C_ops.scale(split_3, full_2, float('64'), True)

        # pd_op.scale: (50x76xf32) <- (50x76xf32, 1xf32)
        scale_15 = paddle._C_ops.scale(split_2, full_2, float('64'), True)

        # builtin.combine: ([50x76xf32, 50x76xf32, 50x76xf32, 50x76xf32]) <- (50x76xf32, 50x76xf32, 50x76xf32, 50x76xf32)
        combine_4 = [scale_12, scale_13, scale_14, scale_15]

        # pd_op.stack: (50x76x4xf32) <- ([50x76xf32, 50x76xf32, 50x76xf32, 50x76xf32])
        stack_2 = paddle._C_ops.stack(combine_4, -1)

        # pd_op.cast: (50x76x4xf32) <- (50x76x4xf32)
        cast_6 = paddle._C_ops.cast(stack_2, paddle.float32)

        # builtin.combine: ([50x76xf32, 50x76xf32]) <- (50x76xf32, 50x76xf32)
        combine_5 = [split_3, split_2]

        # pd_op.stack: (50x76x2xf32) <- ([50x76xf32, 50x76xf32])
        stack_3 = paddle._C_ops.stack(combine_5, -1)

        # pd_op.cast: (50x76x2xf32) <- (50x76x2xf32)
        cast_7 = paddle._C_ops.cast(stack_3, paddle.float32)

        # pd_op.reshape: (3800x4xf32, 0x50x76x4xi64) <- (50x76x4xf32, 2xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(cast_6, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.reshape: (3800x2xf32, 0x50x76x2xi64) <- (50x76x2xf32, 2xi64)
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(cast_7, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (3800x1xf32) <- ()
        full_9 = paddle._C_ops.full([3800, 1], float('16'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.full: (1xf32) <- ()
        full_10 = paddle._C_ops.full([1], float('38'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.arange: (38xi64) <- (1xf32, 1xf32, 1xf32)
        arange_4 = paddle.arange(full_0, full_10, full_2, dtype='int64')

        # pd_op.cast: (38xf32) <- (38xi64)
        cast_8 = paddle._C_ops.cast(arange_4, paddle.float32)

        # pd_op.scale: (38xf32) <- (38xf32, 1xf32)
        scale_16 = paddle._C_ops.scale(cast_8, full_2, float('0.5'), True)

        # pd_op.full: (1xf32) <- ()
        full_11 = paddle._C_ops.full([1], float('32'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (38xf32) <- (38xf32, 1xf32)
        scale_17 = paddle._C_ops.scale(scale_16, full_11, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_12 = paddle._C_ops.full([1], float('25'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.arange: (25xi64) <- (1xf32, 1xf32, 1xf32)
        arange_5 = paddle.arange(full_0, full_12, full_2, dtype='int64')

        # pd_op.cast: (25xf32) <- (25xi64)
        cast_9 = paddle._C_ops.cast(arange_5, paddle.float32)

        # pd_op.scale: (25xf32) <- (25xf32, 1xf32)
        scale_18 = paddle._C_ops.scale(cast_9, full_2, float('0.5'), True)

        # pd_op.scale: (25xf32) <- (25xf32, 1xf32)
        scale_19 = paddle._C_ops.scale(scale_18, full_11, float('0'), True)

        # builtin.combine: ([25xf32, 38xf32]) <- (25xf32, 38xf32)
        combine_6 = [scale_19, scale_17]

        # pd_op.meshgrid: ([25x38xf32, 25x38xf32]) <- ([25xf32, 38xf32])
        meshgrid_2 = paddle._C_ops.meshgrid(combine_6)

        # builtin.split: (25x38xf32, 25x38xf32) <- ([25x38xf32, 25x38xf32])
        split_4, split_5, = meshgrid_2

        # pd_op.scale: (25x38xf32) <- (25x38xf32, 1xf32)
        scale_20 = paddle._C_ops.scale(split_5, full_2, float('-128'), True)

        # pd_op.scale: (25x38xf32) <- (25x38xf32, 1xf32)
        scale_21 = paddle._C_ops.scale(split_4, full_2, float('-128'), True)

        # pd_op.scale: (25x38xf32) <- (25x38xf32, 1xf32)
        scale_22 = paddle._C_ops.scale(split_5, full_2, float('128'), True)

        # pd_op.scale: (25x38xf32) <- (25x38xf32, 1xf32)
        scale_23 = paddle._C_ops.scale(split_4, full_2, float('128'), True)

        # builtin.combine: ([25x38xf32, 25x38xf32, 25x38xf32, 25x38xf32]) <- (25x38xf32, 25x38xf32, 25x38xf32, 25x38xf32)
        combine_7 = [scale_20, scale_21, scale_22, scale_23]

        # pd_op.stack: (25x38x4xf32) <- ([25x38xf32, 25x38xf32, 25x38xf32, 25x38xf32])
        stack_4 = paddle._C_ops.stack(combine_7, -1)

        # pd_op.cast: (25x38x4xf32) <- (25x38x4xf32)
        cast_10 = paddle._C_ops.cast(stack_4, paddle.float32)

        # builtin.combine: ([25x38xf32, 25x38xf32]) <- (25x38xf32, 25x38xf32)
        combine_8 = [split_5, split_4]

        # pd_op.stack: (25x38x2xf32) <- ([25x38xf32, 25x38xf32])
        stack_5 = paddle._C_ops.stack(combine_8, -1)

        # pd_op.cast: (25x38x2xf32) <- (25x38x2xf32)
        cast_11 = paddle._C_ops.cast(stack_5, paddle.float32)

        # pd_op.reshape: (950x4xf32, 0x25x38x4xi64) <- (25x38x4xf32, 2xi64)
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(cast_10, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.reshape: (950x2xf32, 0x25x38x2xi64) <- (25x38x2xf32, 2xi64)
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(cast_11, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (950x1xf32) <- ()
        full_13 = paddle._C_ops.full([950, 1], float('32'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.full: (1xf32) <- ()
        full_14 = paddle._C_ops.full([1], float('19'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.arange: (19xi64) <- (1xf32, 1xf32, 1xf32)
        arange_6 = paddle.arange(full_0, full_14, full_2, dtype='int64')

        # pd_op.cast: (19xf32) <- (19xi64)
        cast_12 = paddle._C_ops.cast(arange_6, paddle.float32)

        # pd_op.scale: (19xf32) <- (19xf32, 1xf32)
        scale_24 = paddle._C_ops.scale(cast_12, full_2, float('0.5'), True)

        # pd_op.full: (1xf32) <- ()
        full_15 = paddle._C_ops.full([1], float('64'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (19xf32) <- (19xf32, 1xf32)
        scale_25 = paddle._C_ops.scale(scale_24, full_15, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_16 = paddle._C_ops.full([1], float('13'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.arange: (13xi64) <- (1xf32, 1xf32, 1xf32)
        arange_7 = paddle.arange(full_0, full_16, full_2, dtype='int64')

        # pd_op.cast: (13xf32) <- (13xi64)
        cast_13 = paddle._C_ops.cast(arange_7, paddle.float32)

        # pd_op.scale: (13xf32) <- (13xf32, 1xf32)
        scale_26 = paddle._C_ops.scale(cast_13, full_2, float('0.5'), True)

        # pd_op.scale: (13xf32) <- (13xf32, 1xf32)
        scale_27 = paddle._C_ops.scale(scale_26, full_15, float('0'), True)

        # builtin.combine: ([13xf32, 19xf32]) <- (13xf32, 19xf32)
        combine_9 = [scale_27, scale_25]

        # pd_op.meshgrid: ([13x19xf32, 13x19xf32]) <- ([13xf32, 19xf32])
        meshgrid_3 = paddle._C_ops.meshgrid(combine_9)

        # builtin.split: (13x19xf32, 13x19xf32) <- ([13x19xf32, 13x19xf32])
        split_6, split_7, = meshgrid_3

        # pd_op.scale: (13x19xf32) <- (13x19xf32, 1xf32)
        scale_28 = paddle._C_ops.scale(split_7, full_2, float('-256'), True)

        # pd_op.scale: (13x19xf32) <- (13x19xf32, 1xf32)
        scale_29 = paddle._C_ops.scale(split_6, full_2, float('-256'), True)

        # pd_op.scale: (13x19xf32) <- (13x19xf32, 1xf32)
        scale_30 = paddle._C_ops.scale(split_7, full_2, float('256'), True)

        # pd_op.scale: (13x19xf32) <- (13x19xf32, 1xf32)
        scale_31 = paddle._C_ops.scale(split_6, full_2, float('256'), True)

        # builtin.combine: ([13x19xf32, 13x19xf32, 13x19xf32, 13x19xf32]) <- (13x19xf32, 13x19xf32, 13x19xf32, 13x19xf32)
        combine_10 = [scale_28, scale_29, scale_30, scale_31]

        # pd_op.stack: (13x19x4xf32) <- ([13x19xf32, 13x19xf32, 13x19xf32, 13x19xf32])
        stack_6 = paddle._C_ops.stack(combine_10, -1)

        # pd_op.cast: (13x19x4xf32) <- (13x19x4xf32)
        cast_14 = paddle._C_ops.cast(stack_6, paddle.float32)

        # builtin.combine: ([13x19xf32, 13x19xf32]) <- (13x19xf32, 13x19xf32)
        combine_11 = [split_7, split_6]

        # pd_op.stack: (13x19x2xf32) <- ([13x19xf32, 13x19xf32])
        stack_7 = paddle._C_ops.stack(combine_11, -1)

        # pd_op.cast: (13x19x2xf32) <- (13x19x2xf32)
        cast_15 = paddle._C_ops.cast(stack_7, paddle.float32)

        # pd_op.reshape: (247x4xf32, 0x13x19x4xi64) <- (13x19x4xf32, 2xi64)
        reshape_12, reshape_13 = (lambda x, f: f(x))(paddle._C_ops.reshape(cast_14, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.reshape: (247x2xf32, 0x13x19x2xi64) <- (13x19x2xf32, 2xi64)
        reshape_14, reshape_15 = (lambda x, f: f(x))(paddle._C_ops.reshape(cast_15, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (247x1xf32) <- ()
        full_17 = paddle._C_ops.full([247, 1], float('64'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.full: (1xf32) <- ()
        full_18 = paddle._C_ops.full([1], float('10'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.arange: (10xi64) <- (1xf32, 1xf32, 1xf32)
        arange_8 = paddle.arange(full_0, full_18, full_2, dtype='int64')

        # pd_op.cast: (10xf32) <- (10xi64)
        cast_16 = paddle._C_ops.cast(arange_8, paddle.float32)

        # pd_op.scale: (10xf32) <- (10xf32, 1xf32)
        scale_32 = paddle._C_ops.scale(cast_16, full_2, float('0.5'), True)

        # pd_op.full: (1xf32) <- ()
        full_19 = paddle._C_ops.full([1], float('128'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (10xf32) <- (10xf32, 1xf32)
        scale_33 = paddle._C_ops.scale(scale_32, full_19, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_20 = paddle._C_ops.full([1], float('7'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.arange: (7xi64) <- (1xf32, 1xf32, 1xf32)
        arange_9 = paddle.arange(full_0, full_20, full_2, dtype='int64')

        # pd_op.cast: (7xf32) <- (7xi64)
        cast_17 = paddle._C_ops.cast(arange_9, paddle.float32)

        # pd_op.scale: (7xf32) <- (7xf32, 1xf32)
        scale_34 = paddle._C_ops.scale(cast_17, full_2, float('0.5'), True)

        # pd_op.scale: (7xf32) <- (7xf32, 1xf32)
        scale_35 = paddle._C_ops.scale(scale_34, full_19, float('0'), True)

        # builtin.combine: ([7xf32, 10xf32]) <- (7xf32, 10xf32)
        combine_12 = [scale_35, scale_33]

        # pd_op.meshgrid: ([7x10xf32, 7x10xf32]) <- ([7xf32, 10xf32])
        meshgrid_4 = paddle._C_ops.meshgrid(combine_12)

        # builtin.split: (7x10xf32, 7x10xf32) <- ([7x10xf32, 7x10xf32])
        split_8, split_9, = meshgrid_4

        # pd_op.scale: (7x10xf32) <- (7x10xf32, 1xf32)
        scale_36 = paddle._C_ops.scale(split_9, full_2, float('-512'), True)

        # pd_op.scale: (7x10xf32) <- (7x10xf32, 1xf32)
        scale_37 = paddle._C_ops.scale(split_8, full_2, float('-512'), True)

        # pd_op.scale: (7x10xf32) <- (7x10xf32, 1xf32)
        scale_38 = paddle._C_ops.scale(split_9, full_2, float('512'), True)

        # pd_op.scale: (7x10xf32) <- (7x10xf32, 1xf32)
        scale_39 = paddle._C_ops.scale(split_8, full_2, float('512'), True)

        # builtin.combine: ([7x10xf32, 7x10xf32, 7x10xf32, 7x10xf32]) <- (7x10xf32, 7x10xf32, 7x10xf32, 7x10xf32)
        combine_13 = [scale_36, scale_37, scale_38, scale_39]

        # pd_op.stack: (7x10x4xf32) <- ([7x10xf32, 7x10xf32, 7x10xf32, 7x10xf32])
        stack_8 = paddle._C_ops.stack(combine_13, -1)

        # pd_op.cast: (7x10x4xf32) <- (7x10x4xf32)
        cast_18 = paddle._C_ops.cast(stack_8, paddle.float32)

        # builtin.combine: ([7x10xf32, 7x10xf32]) <- (7x10xf32, 7x10xf32)
        combine_14 = [split_9, split_8]

        # pd_op.stack: (7x10x2xf32) <- ([7x10xf32, 7x10xf32])
        stack_9 = paddle._C_ops.stack(combine_14, -1)

        # pd_op.cast: (7x10x2xf32) <- (7x10x2xf32)
        cast_19 = paddle._C_ops.cast(stack_9, paddle.float32)

        # pd_op.reshape: (70x4xf32, 0x7x10x4xi64) <- (7x10x4xf32, 2xi64)
        reshape_16, reshape_17 = (lambda x, f: f(x))(paddle._C_ops.reshape(cast_18, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.reshape: (70x2xf32, 0x7x10x2xi64) <- (7x10x2xf32, 2xi64)
        reshape_18, reshape_19 = (lambda x, f: f(x))(paddle._C_ops.reshape(cast_19, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (70x1xf32) <- ()
        full_21 = paddle._C_ops.full([70, 1], float('128'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.full: (1xi32) <- ()
        full_22 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([15200x4xf32, 3800x4xf32, 950x4xf32, 247x4xf32, 70x4xf32]) <- (15200x4xf32, 3800x4xf32, 950x4xf32, 247x4xf32, 70x4xf32)
        combine_15 = [reshape_0, reshape_4, reshape_8, reshape_12, reshape_16]

        # pd_op.concat: (20267x4xf32) <- ([15200x4xf32, 3800x4xf32, 950x4xf32, 247x4xf32, 70x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_15, full_22)

        # builtin.combine: ([15200x2xf32, 3800x2xf32, 950x2xf32, 247x2xf32, 70x2xf32]) <- (15200x2xf32, 3800x2xf32, 950x2xf32, 247x2xf32, 70x2xf32)
        combine_16 = [reshape_2, reshape_6, reshape_10, reshape_14, reshape_18]

        # pd_op.concat: (20267x2xf32) <- ([15200x2xf32, 3800x2xf32, 950x2xf32, 247x2xf32, 70x2xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_16, full_22)

        # builtin.combine: ([15200x1xf32, 3800x1xf32, 950x1xf32, 247x1xf32, 70x1xf32]) <- (15200x1xf32, 3800x1xf32, 950x1xf32, 247x1xf32, 70x1xf32)
        combine_17 = [full_5, full_9, full_13, full_17, full_21]

        # pd_op.concat: (20267x1xf32) <- ([15200x1xf32, 3800x1xf32, 950x1xf32, 247x1xf32, 70x1xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_17, full_22)

        # pd_op.divide: (20267x2xf32) <- (20267x2xf32, 20267x1xf32)
        divide_0 = concat_1 / concat_2

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_2 = [15200, 3800, 950, 247, 70]

        # pd_op.split: ([15200x2xf32, 3800x2xf32, 950x2xf32, 247x2xf32, 70x2xf32]) <- (20267x2xf32, 5xi64, 1xi32)
        split_10 = paddle._C_ops.split(divide_0, full_int_array_2, full_22)

        # builtin.split: (15200x2xf32, 3800x2xf32, 950x2xf32, 247x2xf32, 70x2xf32) <- ([15200x2xf32, 3800x2xf32, 950x2xf32, 247x2xf32, 70x2xf32])
        split_11, split_12, split_13, split_14, split_15, = split_10
        return divide_0, concat_0, reshape_0, reshape_4, reshape_8, reshape_12, reshape_16, concat_2



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

    def forward(self, ):
        return self.builtin_module_469_0_0()

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_469_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
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