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
    return [116][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_477_0_0(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6):

        # pd_op.cast: (-1x-1xi32) <- (-1x-1xb)
        cast_0 = paddle._C_ops.cast(data_0, paddle.int32)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-1]

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_0 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_1 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_2 = full_int_array_0

        # pd_op.assign: (1xi64) <- (1xi64)
        assign_3 = full_int_array_0

        # pd_op.unsqueeze: (-1x-1x1xi32, 0x-1x-1xi32) <- (-1x-1xi32, 1xi64)
        unsqueeze_0, unsqueeze_1 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(cast_0, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_1 = [1, 1, 4]

        # pd_op.tile: (-1x-1x4xi32) <- (-1x-1x1xi32, 3xi64)
        tile_0 = paddle._C_ops.tile(unsqueeze_0, full_int_array_1)

        # pd_op.cast: (-1x-1x4xb) <- (-1x-1x4xi32)
        cast_1 = paddle._C_ops.cast(tile_0, paddle.bool)

        # pd_op.masked_select: (-1xf32) <- (-1x-1x-1xf32, -1x-1x4xb)
        masked_select_0 = paddle._C_ops.masked_select(data_1, cast_1)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [-1, 4]

        # pd_op.reshape: (-1x4xf32, 0x-1xi64) <- (-1xf32, 2xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(masked_select_0, full_int_array_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.masked_select: (-1xf32) <- (-1x-1x-1xf32, -1x-1x4xb)
        masked_select_1 = paddle._C_ops.masked_select(data_2, cast_1)

        # pd_op.reshape: (-1x4xf32, 0x-1xi64) <- (-1xf32, 2xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(masked_select_1, full_int_array_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.sum: (-1x-1xf32) <- (-1x-1x-1xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(data_3, assign_3, None, False)

        # pd_op.masked_select: (-1xf32) <- (-1x-1xf32, -1x-1xb)
        masked_select_2 = paddle._C_ops.masked_select(sum_0, data_0)

        # pd_op.unsqueeze: (-1x1xf32, 0x-1xf32) <- (-1xf32, 1xi64)
        unsqueeze_2, unsqueeze_3 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(masked_select_2, assign_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.subtract: (-1x4xf32) <- (-1x4xf32, -1x4xf32)
        subtract_0 = reshape_0 - reshape_2

        # pd_op.abs: (-1x4xf32) <- (-1x4xf32)
        abs_0 = paddle._C_ops.abs(subtract_0)

        # pd_op.mean_all: (xf32) <- (-1x4xf32)
        mean_all_0 = paddle._C_ops.mean_all(abs_0)

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_4 = full_0

        # pd_op.split_with_num: ([-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32]) <- (-1x4xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(reshape_0, 4, full_0)

        # builtin.split: (-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32) <- ([-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32])
        split_0, split_1, split_2, split_3, = split_with_num_0

        # pd_op.split_with_num: ([-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32]) <- (-1x4xf32, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(reshape_2, 4, assign_4)

        # builtin.split: (-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32) <- ([-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32])
        split_4, split_5, split_6, split_7, = split_with_num_1

        # pd_op.maximum: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        maximum_0 = paddle.maximum(split_0, split_4)

        # pd_op.maximum: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        maximum_1 = paddle.maximum(split_1, split_5)

        # pd_op.minimum: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        minimum_0 = paddle._C_ops.minimum(split_2, split_6)

        # pd_op.minimum: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        minimum_1 = paddle._C_ops.minimum(split_3, split_7)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_1 = minimum_0 - maximum_0

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full([1], float('0'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_5 = full_1

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_6 = full_1

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('3.40282e+38'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_7 = full_2

        # pd_op.clip: (-1x1xf32) <- (-1x1xf32, 1xf32, 1xf32)
        clip_0 = paddle._C_ops.clip(subtract_1, full_1, full_2)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_2 = minimum_1 - maximum_1

        # pd_op.clip: (-1x1xf32) <- (-1x1xf32, 1xf32, 1xf32)
        clip_1 = paddle._C_ops.clip(subtract_2, assign_6, assign_7)

        # pd_op.multiply: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        multiply_0 = clip_0 * clip_1

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_3 = split_2 - split_0

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_4 = split_3 - split_1

        # pd_op.multiply: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        multiply_1 = subtract_3 * subtract_4

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_5 = split_6 - split_4

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_6 = split_7 - split_5

        # pd_op.multiply: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        multiply_2 = subtract_5 * subtract_6

        # pd_op.add: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        add_0 = multiply_1 + multiply_2

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_7 = add_0 - multiply_0

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_8 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_9 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_10 = full_3

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(subtract_7, full_3, float('1e-10'), True)

        # pd_op.divide: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        divide_0 = multiply_0 / scale_0

        # pd_op.minimum: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        minimum_2 = paddle._C_ops.minimum(split_0, split_4)

        # pd_op.minimum: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        minimum_3 = paddle._C_ops.minimum(split_1, split_5)

        # pd_op.maximum: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        maximum_2 = paddle.maximum(split_2, split_6)

        # pd_op.maximum: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        maximum_3 = paddle.maximum(split_3, split_7)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_8 = maximum_2 - minimum_2

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_9 = maximum_3 - minimum_3

        # pd_op.multiply: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        multiply_3 = subtract_8 * subtract_9

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(multiply_3, assign_10, float('1e-10'), True)

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_10 = scale_1 - scale_0

        # pd_op.divide: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        divide_1 = subtract_10 / scale_1

        # pd_op.subtract: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        subtract_11 = divide_0 - divide_1

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('-1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_11 = full_4

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(subtract_11, full_4, float('1'), True)

        # pd_op.scale: (-1x1xf32) <- (-1x1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(scale_2, assign_9, float('0'), True)

        # pd_op.multiply: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        multiply_4 = scale_3 * unsqueeze_2

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_3 = []

        # pd_op.assign: (0xi64) <- (0xi64)
        assign_12 = full_int_array_3

        # pd_op.sum: (xf32) <- (-1x1xf32, 0xi64)
        sum_1 = paddle._C_ops.sum(multiply_4, full_int_array_3, None, False)

        # pd_op.divide: (-1xf32) <- (xf32, -1xf32)
        divide_2 = sum_1 / data_4

        # pd_op.unsqueeze: (-1x-1x1xb, 0x-1x-1xb) <- (-1x-1xb, 1xi64)
        unsqueeze_4, unsqueeze_5 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(data_0, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.cast: (-1x-1x1xi32) <- (-1x-1x1xb)
        cast_2 = paddle._C_ops.cast(unsqueeze_4, paddle.int32)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_4 = [1, 1, 68]

        # pd_op.tile: (-1x-1x68xi32) <- (-1x-1x1xi32, 3xi64)
        tile_1 = paddle._C_ops.tile(cast_2, full_int_array_4)

        # pd_op.cast: (-1x-1x68xb) <- (-1x-1x68xi32)
        cast_3 = paddle._C_ops.cast(tile_1, paddle.bool)

        # pd_op.masked_select: (-1xf32) <- (-1x-1x-1xf32, -1x-1x68xb)
        masked_select_3 = paddle._C_ops.masked_select(data_5, cast_3)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_5 = [-1, 4, 17]

        # pd_op.reshape: (-1x4x17xf32, 0x-1xi64) <- (-1xf32, 3xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(masked_select_3, full_int_array_5), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x-1x-1xf32, -1x-1x-1xf32]) <- (-1x-1x-1xf32, 1xi32)
        split_with_num_2 = paddle._C_ops.split_with_num(data_2, 2, full_5)

        # builtin.split: (-1x-1x-1xf32, -1x-1x-1xf32) <- ([-1x-1x-1xf32, -1x-1x-1xf32])
        split_8, split_9, = split_with_num_2

        # pd_op.subtract: (-1x-1x-1xf32) <- (-1x-1xf32, -1x-1x-1xf32)
        subtract_12 = data_6 - split_8

        # pd_op.subtract: (-1x-1x-1xf32) <- (-1x-1x-1xf32, -1x-1xf32)
        subtract_13 = split_9 - data_6

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([-1x-1x-1xf32, -1x-1x-1xf32]) <- (-1x-1x-1xf32, -1x-1x-1xf32)
        combine_0 = [subtract_12, subtract_13]

        # pd_op.concat: (-1x-1x-1xf32) <- ([-1x-1x-1xf32, -1x-1x-1xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_6)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('15.99'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.clip: (-1x-1x-1xf32) <- (-1x-1x-1xf32, 1xf32, 1xf32)
        clip_2 = paddle._C_ops.clip(concat_0, assign_5, full_7)

        # pd_op.masked_select: (-1xf32) <- (-1x-1x-1xf32, -1x-1x4xb)
        masked_select_4 = paddle._C_ops.masked_select(clip_2, cast_1)

        # pd_op.reshape: (-1x4xf32, 0x-1xi64) <- (-1xf32, 2xi64)
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(masked_select_4, full_int_array_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.floor: (-1x4xf32) <- (-1x4xf32)
        floor_0 = paddle._C_ops.floor(reshape_6)

        # pd_op.cast: (-1x4xi64) <- (-1x4xf32)
        cast_4 = paddle._C_ops.cast(floor_0, paddle.int64)

        # pd_op.scale: (-1x4xi64) <- (-1x4xi64, 1xf32)
        scale_4 = paddle._C_ops.scale(cast_4, assign_8, float('1'), True)

        # pd_op.cast: (-1x4xf32) <- (-1x4xi64)
        cast_5 = paddle._C_ops.cast(scale_4, paddle.float32)

        # pd_op.subtract: (-1x4xf32) <- (-1x4xf32, -1x4xf32)
        subtract_14 = cast_5 - reshape_6

        # pd_op.scale: (-1x4xf32) <- (-1x4xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(subtract_14, assign_11, float('1'), True)

        # pd_op.scale: (-1x4xi64) <- (-1x4xi64, 1xf32)
        scale_6 = paddle._C_ops.scale(cast_4, full_3, float('0'), True)

        # pd_op.unsqueeze: (-1x4x1xi64, 0x-1x4xi64) <- (-1x4xi64, 1xi64)
        unsqueeze_6, unsqueeze_7 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(scale_6, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.cross_entropy_with_softmax: (-1x4x17xf32, -1x4x1xf32) <- (-1x4x17xf32, -1x4x1xi64)
        cross_entropy_with_softmax_0, cross_entropy_with_softmax_1 = (lambda x, f: f(x))(paddle._C_ops.cross_entropy_with_softmax(reshape_4, unsqueeze_6, False, True, True, -100, -1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.squeeze: (-1x4xf32, 0x-1x4x1xf32) <- (-1x4x1xf32, 1xi64)
        squeeze_0, squeeze_1 = (lambda x, f: f(x))(paddle._C_ops.squeeze(cross_entropy_with_softmax_1, assign_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x4xf32) <- (-1x4xf32, -1x4xf32)
        multiply_5 = squeeze_0 * subtract_14

        # pd_op.scale: (-1x4xi64) <- (-1x4xi64, 1xf32)
        scale_7 = paddle._C_ops.scale(scale_4, full_3, float('0'), True)

        # pd_op.unsqueeze: (-1x4x1xi64, 0x-1x4xi64) <- (-1x4xi64, 1xi64)
        unsqueeze_8, unsqueeze_9 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(scale_7, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.cross_entropy_with_softmax: (-1x4x17xf32, -1x4x1xf32) <- (-1x4x17xf32, -1x4x1xi64)
        cross_entropy_with_softmax_2, cross_entropy_with_softmax_3 = (lambda x, f: f(x))(paddle._C_ops.cross_entropy_with_softmax(reshape_4, unsqueeze_8, False, True, True, -100, -1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.squeeze: (-1x4xf32, 0x-1x4x1xf32) <- (-1x4x1xf32, 1xi64)
        squeeze_2, squeeze_3 = (lambda x, f: f(x))(paddle._C_ops.squeeze(cross_entropy_with_softmax_3, assign_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x4xf32) <- (-1x4xf32, -1x4xf32)
        multiply_6 = squeeze_2 * scale_5

        # pd_op.add: (-1x4xf32) <- (-1x4xf32, -1x4xf32)
        add_1 = multiply_5 + multiply_6

        # pd_op.mean: (-1x1xf32) <- (-1x4xf32)
        mean_0 = paddle._C_ops.mean(add_1, [-1], True)

        # pd_op.multiply: (-1x1xf32) <- (-1x1xf32, -1x1xf32)
        multiply_7 = mean_0 * unsqueeze_2

        # pd_op.sum: (xf32) <- (-1x1xf32, 0xi64)
        sum_2 = paddle._C_ops.sum(multiply_7, assign_12, None, False)

        # pd_op.divide: (-1xf32) <- (xf32, -1xf32)
        divide_3 = sum_2 / data_4
        return cast_1, reshape_0, reshape_1, reshape_2, reshape_3, assign_3, sum_0, assign_2, unsqueeze_2, unsqueeze_3, subtract_0, abs_0, full_0, split_0, split_1, split_2, split_3, assign_4, split_4, split_5, split_6, split_7, maximum_0, maximum_1, minimum_0, minimum_1, subtract_1, full_1, full_2, clip_0, subtract_2, assign_6, assign_7, clip_1, multiply_0, subtract_3, subtract_4, multiply_1, subtract_5, subtract_6, multiply_2, add_0, full_3, scale_0, divide_0, minimum_2, minimum_3, maximum_2, maximum_3, subtract_8, subtract_9, assign_10, scale_1, subtract_10, divide_1, full_4, assign_9, scale_3, multiply_4, full_int_array_3, sum_1, cast_3, reshape_5, full_5, split_8, split_9, subtract_12, subtract_13, full_6, concat_0, assign_5, full_7, clip_2, reshape_6, reshape_7, assign_8, cast_5, subtract_14, assign_11, scale_5, unsqueeze_6, cross_entropy_with_softmax_0, assign_1, squeeze_0, squeeze_1, multiply_5, unsqueeze_8, cross_entropy_with_softmax_2, assign_0, squeeze_2, squeeze_3, multiply_6, add_1, mean_0, multiply_7, assign_12, sum_2, mean_all_0, divide_2, divide_3



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

    def forward(self, data_0, data_1, data_2, data_3, data_4, data_5, data_6):
        return self.builtin_module_477_0_0(data_0, data_1, data_2, data_3, data_4, data_5, data_6)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_477_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # data_0
            paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
            # data_1
            paddle.uniform([1, 2100, 4], dtype='float32', min=0, max=0.5),
            # data_2
            paddle.uniform([1, 2100, 4], dtype='float32', min=0, max=0.5),
            # data_3
            paddle.uniform([1, 2100, 20], dtype='float32', min=0, max=0.5),
            # data_4
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # data_5
            paddle.uniform([1, 2100, 68], dtype='float32', min=0, max=0.5),
            # data_6
            paddle.uniform([2100, 2], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # data_0
            paddle.static.InputSpec(shape=[None, None], dtype='bool'),
            # data_1
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            # data_2
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            # data_3
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            # data_4
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            # data_5
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            # data_6
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
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