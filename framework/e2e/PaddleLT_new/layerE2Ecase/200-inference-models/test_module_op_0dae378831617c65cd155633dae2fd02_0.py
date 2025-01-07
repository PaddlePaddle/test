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
    return [95][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_171_0_0(self, parameter_0, parameter_1, parameter_5, parameter_2, parameter_4, parameter_3, parameter_6, parameter_10, parameter_7, parameter_9, parameter_8, parameter_11, parameter_15, parameter_12, parameter_14, parameter_13, parameter_16, parameter_20, parameter_17, parameter_19, parameter_18, parameter_21, parameter_25, parameter_22, parameter_24, parameter_23, parameter_26, parameter_30, parameter_27, parameter_29, parameter_28, parameter_31, parameter_32, parameter_36, parameter_33, parameter_35, parameter_34, parameter_37, parameter_41, parameter_38, parameter_40, parameter_39, parameter_42, parameter_46, parameter_43, parameter_45, parameter_44, parameter_47, parameter_51, parameter_48, parameter_50, parameter_49, parameter_52, parameter_56, parameter_53, parameter_55, parameter_54, parameter_57, parameter_61, parameter_58, parameter_60, parameter_59, parameter_62, parameter_66, parameter_63, parameter_65, parameter_64, parameter_67, parameter_68, feed_0):

        # pd_op.cast: (-1x3x-1x-1xf16) <- (-1x3x-1x-1xf32)
        cast_0 = paddle._C_ops.cast(feed_0, paddle.float16)

        # pd_op.conv2d: (-1x64x-1x-1xf16) <- (-1x3x-1x-1xf16, 64x3x4x4xf16)
        conv2d_0 = paddle._C_ops.conv2d(cast_0, parameter_0, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.leaky_relu: (-1x64x-1x-1xf16) <- (-1x64x-1x-1xf16)
        leaky_relu_0 = paddle._C_ops.leaky_relu(conv2d_0, float('0.2'))

        # pd_op.conv2d: (-1x128x-1x-1xf16) <- (-1x64x-1x-1xf16, 128x64x4x4xf16)
        conv2d_1 = paddle._C_ops.conv2d(leaky_relu_0, parameter_1, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x-1x-1xf16, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x-1x-1xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_1, parameter_2, parameter_3, parameter_4, parameter_5, True, float('0.9'), float('1e-05'), 'NCHW', False, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.leaky_relu: (-1x128x-1x-1xf16) <- (-1x128x-1x-1xf16)
        leaky_relu_1 = paddle._C_ops.leaky_relu(batch_norm__0, float('0.2'))

        # pd_op.conv2d: (-1x256x-1x-1xf16) <- (-1x128x-1x-1xf16, 256x128x4x4xf16)
        conv2d_2 = paddle._C_ops.conv2d(leaky_relu_1, parameter_6, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x-1x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x-1x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_2, parameter_7, parameter_8, parameter_9, parameter_10, True, float('0.9'), float('1e-05'), 'NCHW', False, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.leaky_relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        leaky_relu_2 = paddle._C_ops.leaky_relu(batch_norm__6, float('0.2'))

        # pd_op.conv2d: (-1x512x-1x-1xf16) <- (-1x256x-1x-1xf16, 512x256x4x4xf16)
        conv2d_3 = paddle._C_ops.conv2d(leaky_relu_2, parameter_11, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x-1x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x-1x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_3, parameter_12, parameter_13, parameter_14, parameter_15, True, float('0.9'), float('1e-05'), 'NCHW', False, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.leaky_relu: (-1x512x-1x-1xf16) <- (-1x512x-1x-1xf16)
        leaky_relu_3 = paddle._C_ops.leaky_relu(batch_norm__12, float('0.2'))

        # pd_op.conv2d: (-1x512x-1x-1xf16) <- (-1x512x-1x-1xf16, 512x512x4x4xf16)
        conv2d_4 = paddle._C_ops.conv2d(leaky_relu_3, parameter_16, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x-1x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x-1x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_4, parameter_17, parameter_18, parameter_19, parameter_20, True, float('0.9'), float('1e-05'), 'NCHW', False, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.leaky_relu: (-1x512x-1x-1xf16) <- (-1x512x-1x-1xf16)
        leaky_relu_4 = paddle._C_ops.leaky_relu(batch_norm__18, float('0.2'))

        # pd_op.conv2d: (-1x512x-1x-1xf16) <- (-1x512x-1x-1xf16, 512x512x4x4xf16)
        conv2d_5 = paddle._C_ops.conv2d(leaky_relu_4, parameter_21, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x-1x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x-1x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_5, parameter_22, parameter_23, parameter_24, parameter_25, True, float('0.9'), float('1e-05'), 'NCHW', False, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.leaky_relu: (-1x512x-1x-1xf16) <- (-1x512x-1x-1xf16)
        leaky_relu_5 = paddle._C_ops.leaky_relu(batch_norm__24, float('0.2'))

        # pd_op.conv2d: (-1x512x-1x-1xf16) <- (-1x512x-1x-1xf16, 512x512x4x4xf16)
        conv2d_6 = paddle._C_ops.conv2d(leaky_relu_5, parameter_26, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x-1x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x-1x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_6, parameter_27, parameter_28, parameter_29, parameter_30, True, float('0.9'), float('1e-05'), 'NCHW', False, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.leaky_relu: (-1x512x-1x-1xf16) <- (-1x512x-1x-1xf16)
        leaky_relu_6 = paddle._C_ops.leaky_relu(batch_norm__30, float('0.2'))

        # pd_op.conv2d: (-1x512x-1x-1xf16) <- (-1x512x-1x-1xf16, 512x512x4x4xf16)
        conv2d_7 = paddle._C_ops.conv2d(leaky_relu_6, parameter_31, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.relu: (-1x512x-1x-1xf16) <- (-1x512x-1x-1xf16)
        relu_0 = paddle._C_ops.relu(conv2d_7)

        # pd_op.cast: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf16)
        cast_1 = paddle._C_ops.cast(relu_0, paddle.float32)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_0 = []

        # pd_op.conv2d_transpose: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x512x4x4xf32, 0xi64)
        conv2d_transpose_0 = paddle._C_ops.conv2d_transpose(cast_1, parameter_32, [2, 2], [1, 1], [], full_int_array_0, 'EXPLICIT', 1, [1, 1], 'NCHW')

        # pd_op.cast: (-1x512x-1x-1xf16) <- (-1x512x-1x-1xf32)
        cast_2 = paddle._C_ops.cast(conv2d_transpose_0, paddle.float16)

        # pd_op.batch_norm_: (-1x512x-1x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x-1x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(cast_2, parameter_33, parameter_34, parameter_35, parameter_36, True, float('0.9'), float('1e-05'), 'NCHW', False, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # builtin.combine: ([-1x512x-1x-1xf16, -1x512x-1x-1xf16]) <- (-1x512x-1x-1xf16, -1x512x-1x-1xf16)
        combine_0 = [batch_norm__30, batch_norm__36]

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024x-1x-1xf16) <- ([-1x512x-1x-1xf16, -1x512x-1x-1xf16], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        # pd_op.relu: (-1x1024x-1x-1xf16) <- (-1x1024x-1x-1xf16)
        relu_1 = paddle._C_ops.relu(concat_0)

        # pd_op.cast: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf16)
        cast_3 = paddle._C_ops.cast(relu_1, paddle.float32)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_1 = []

        # pd_op.conv2d_transpose: (-1x512x-1x-1xf32) <- (-1x1024x-1x-1xf32, 1024x512x4x4xf32, 0xi64)
        conv2d_transpose_1 = paddle._C_ops.conv2d_transpose(cast_3, parameter_37, [2, 2], [1, 1], [], full_int_array_1, 'EXPLICIT', 1, [1, 1], 'NCHW')

        # pd_op.cast: (-1x512x-1x-1xf16) <- (-1x512x-1x-1xf32)
        cast_4 = paddle._C_ops.cast(conv2d_transpose_1, paddle.float16)

        # pd_op.batch_norm_: (-1x512x-1x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x-1x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(cast_4, parameter_38, parameter_39, parameter_40, parameter_41, True, float('0.9'), float('1e-05'), 'NCHW', False, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # builtin.combine: ([-1x512x-1x-1xf16, -1x512x-1x-1xf16]) <- (-1x512x-1x-1xf16, -1x512x-1x-1xf16)
        combine_1 = [batch_norm__24, batch_norm__42]

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024x-1x-1xf16) <- ([-1x512x-1x-1xf16, -1x512x-1x-1xf16], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_1)

        # pd_op.relu: (-1x1024x-1x-1xf16) <- (-1x1024x-1x-1xf16)
        relu_2 = paddle._C_ops.relu(concat_1)

        # pd_op.cast: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf16)
        cast_5 = paddle._C_ops.cast(relu_2, paddle.float32)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_2 = []

        # pd_op.conv2d_transpose: (-1x512x-1x-1xf32) <- (-1x1024x-1x-1xf32, 1024x512x4x4xf32, 0xi64)
        conv2d_transpose_2 = paddle._C_ops.conv2d_transpose(cast_5, parameter_42, [2, 2], [1, 1], [], full_int_array_2, 'EXPLICIT', 1, [1, 1], 'NCHW')

        # pd_op.cast: (-1x512x-1x-1xf16) <- (-1x512x-1x-1xf32)
        cast_6 = paddle._C_ops.cast(conv2d_transpose_2, paddle.float16)

        # pd_op.batch_norm_: (-1x512x-1x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x-1x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(cast_6, parameter_43, parameter_44, parameter_45, parameter_46, True, float('0.9'), float('1e-05'), 'NCHW', False, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # builtin.combine: ([-1x512x-1x-1xf16, -1x512x-1x-1xf16]) <- (-1x512x-1x-1xf16, -1x512x-1x-1xf16)
        combine_2 = [batch_norm__18, batch_norm__48]

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024x-1x-1xf16) <- ([-1x512x-1x-1xf16, -1x512x-1x-1xf16], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_2)

        # pd_op.relu: (-1x1024x-1x-1xf16) <- (-1x1024x-1x-1xf16)
        relu_3 = paddle._C_ops.relu(concat_2)

        # pd_op.cast: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf16)
        cast_7 = paddle._C_ops.cast(relu_3, paddle.float32)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_3 = []

        # pd_op.conv2d_transpose: (-1x512x-1x-1xf32) <- (-1x1024x-1x-1xf32, 1024x512x4x4xf32, 0xi64)
        conv2d_transpose_3 = paddle._C_ops.conv2d_transpose(cast_7, parameter_47, [2, 2], [1, 1], [], full_int_array_3, 'EXPLICIT', 1, [1, 1], 'NCHW')

        # pd_op.cast: (-1x512x-1x-1xf16) <- (-1x512x-1x-1xf32)
        cast_8 = paddle._C_ops.cast(conv2d_transpose_3, paddle.float16)

        # pd_op.batch_norm_: (-1x512x-1x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x-1x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(cast_8, parameter_48, parameter_49, parameter_50, parameter_51, True, float('0.9'), float('1e-05'), 'NCHW', False, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # builtin.combine: ([-1x512x-1x-1xf16, -1x512x-1x-1xf16]) <- (-1x512x-1x-1xf16, -1x512x-1x-1xf16)
        combine_3 = [batch_norm__12, batch_norm__54]

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024x-1x-1xf16) <- ([-1x512x-1x-1xf16, -1x512x-1x-1xf16], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_3, full_3)

        # pd_op.relu: (-1x1024x-1x-1xf16) <- (-1x1024x-1x-1xf16)
        relu_4 = paddle._C_ops.relu(concat_3)

        # pd_op.cast: (-1x1024x-1x-1xf32) <- (-1x1024x-1x-1xf16)
        cast_9 = paddle._C_ops.cast(relu_4, paddle.float32)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_4 = []

        # pd_op.conv2d_transpose: (-1x256x-1x-1xf32) <- (-1x1024x-1x-1xf32, 1024x256x4x4xf32, 0xi64)
        conv2d_transpose_4 = paddle._C_ops.conv2d_transpose(cast_9, parameter_52, [2, 2], [1, 1], [], full_int_array_4, 'EXPLICIT', 1, [1, 1], 'NCHW')

        # pd_op.cast: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf32)
        cast_10 = paddle._C_ops.cast(conv2d_transpose_4, paddle.float16)

        # pd_op.batch_norm_: (-1x256x-1x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x-1x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(cast_10, parameter_53, parameter_54, parameter_55, parameter_56, True, float('0.9'), float('1e-05'), 'NCHW', False, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # builtin.combine: ([-1x256x-1x-1xf16, -1x256x-1x-1xf16]) <- (-1x256x-1x-1xf16, -1x256x-1x-1xf16)
        combine_4 = [batch_norm__6, batch_norm__60]

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x512x-1x-1xf16) <- ([-1x256x-1x-1xf16, -1x256x-1x-1xf16], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_4, full_4)

        # pd_op.relu: (-1x512x-1x-1xf16) <- (-1x512x-1x-1xf16)
        relu_5 = paddle._C_ops.relu(concat_4)

        # pd_op.cast: (-1x512x-1x-1xf32) <- (-1x512x-1x-1xf16)
        cast_11 = paddle._C_ops.cast(relu_5, paddle.float32)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_5 = []

        # pd_op.conv2d_transpose: (-1x128x-1x-1xf32) <- (-1x512x-1x-1xf32, 512x128x4x4xf32, 0xi64)
        conv2d_transpose_5 = paddle._C_ops.conv2d_transpose(cast_11, parameter_57, [2, 2], [1, 1], [], full_int_array_5, 'EXPLICIT', 1, [1, 1], 'NCHW')

        # pd_op.cast: (-1x128x-1x-1xf16) <- (-1x128x-1x-1xf32)
        cast_12 = paddle._C_ops.cast(conv2d_transpose_5, paddle.float16)

        # pd_op.batch_norm_: (-1x128x-1x-1xf16, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x-1x-1xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(cast_12, parameter_58, parameter_59, parameter_60, parameter_61, True, float('0.9'), float('1e-05'), 'NCHW', False, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # builtin.combine: ([-1x128x-1x-1xf16, -1x128x-1x-1xf16]) <- (-1x128x-1x-1xf16, -1x128x-1x-1xf16)
        combine_5 = [batch_norm__0, batch_norm__66]

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x256x-1x-1xf16) <- ([-1x128x-1x-1xf16, -1x128x-1x-1xf16], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_5, full_5)

        # pd_op.relu: (-1x256x-1x-1xf16) <- (-1x256x-1x-1xf16)
        relu_6 = paddle._C_ops.relu(concat_5)

        # pd_op.cast: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf16)
        cast_13 = paddle._C_ops.cast(relu_6, paddle.float32)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_6 = []

        # pd_op.conv2d_transpose: (-1x64x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x64x4x4xf32, 0xi64)
        conv2d_transpose_6 = paddle._C_ops.conv2d_transpose(cast_13, parameter_62, [2, 2], [1, 1], [], full_int_array_6, 'EXPLICIT', 1, [1, 1], 'NCHW')

        # pd_op.cast: (-1x64x-1x-1xf16) <- (-1x64x-1x-1xf32)
        cast_14 = paddle._C_ops.cast(conv2d_transpose_6, paddle.float16)

        # pd_op.batch_norm_: (-1x64x-1x-1xf16, 64xf32, 64xf32, 64xf32, 64xf32, None) <- (-1x64x-1x-1xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(cast_14, parameter_63, parameter_64, parameter_65, parameter_66, True, float('0.9'), float('1e-05'), 'NCHW', False, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # builtin.combine: ([-1x64x-1x-1xf16, -1x64x-1x-1xf16]) <- (-1x64x-1x-1xf16, -1x64x-1x-1xf16)
        combine_6 = [conv2d_0, batch_norm__72]

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x128x-1x-1xf16) <- ([-1x64x-1x-1xf16, -1x64x-1x-1xf16], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_6, full_6)

        # pd_op.relu: (-1x128x-1x-1xf16) <- (-1x128x-1x-1xf16)
        relu_7 = paddle._C_ops.relu(concat_6)

        # pd_op.cast: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf16)
        cast_15 = paddle._C_ops.cast(relu_7, paddle.float32)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_7 = []

        # pd_op.conv2d_transpose: (-1x3x-1x-1xf32) <- (-1x128x-1x-1xf32, 128x3x4x4xf32, 0xi64)
        conv2d_transpose_7 = paddle._C_ops.conv2d_transpose(cast_15, parameter_67, [2, 2], [1, 1], [], full_int_array_7, 'EXPLICIT', 1, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_8 = [1, 3, 1, 1]

        # pd_op.reshape: (1x3x1x1xf16, 0x3xf16) <- (3xf16, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_68, full_int_array_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.cast: (-1x3x-1x-1xf16) <- (-1x3x-1x-1xf32)
        cast_16 = paddle._C_ops.cast(conv2d_transpose_7, paddle.float16)

        # pd_op.add: (-1x3x-1x-1xf16) <- (-1x3x-1x-1xf16, 1x3x1x1xf16)
        add_0 = paddle._C_ops.add(cast_16, reshape_0)

        # pd_op.tanh: (-1x3x-1x-1xf16) <- (-1x3x-1x-1xf16)
        tanh_0 = paddle._C_ops.tanh(add_0)

        # pd_op.cast: (-1x3x-1x-1xf32) <- (-1x3x-1x-1xf16)
        cast_17 = paddle._C_ops.cast(tanh_0, paddle.float32)
        return cast_17



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

    def forward(self, parameter_0, parameter_1, parameter_5, parameter_2, parameter_4, parameter_3, parameter_6, parameter_10, parameter_7, parameter_9, parameter_8, parameter_11, parameter_15, parameter_12, parameter_14, parameter_13, parameter_16, parameter_20, parameter_17, parameter_19, parameter_18, parameter_21, parameter_25, parameter_22, parameter_24, parameter_23, parameter_26, parameter_30, parameter_27, parameter_29, parameter_28, parameter_31, parameter_32, parameter_36, parameter_33, parameter_35, parameter_34, parameter_37, parameter_41, parameter_38, parameter_40, parameter_39, parameter_42, parameter_46, parameter_43, parameter_45, parameter_44, parameter_47, parameter_51, parameter_48, parameter_50, parameter_49, parameter_52, parameter_56, parameter_53, parameter_55, parameter_54, parameter_57, parameter_61, parameter_58, parameter_60, parameter_59, parameter_62, parameter_66, parameter_63, parameter_65, parameter_64, parameter_67, parameter_68, feed_0):
        return self.builtin_module_171_0_0(parameter_0, parameter_1, parameter_5, parameter_2, parameter_4, parameter_3, parameter_6, parameter_10, parameter_7, parameter_9, parameter_8, parameter_11, parameter_15, parameter_12, parameter_14, parameter_13, parameter_16, parameter_20, parameter_17, parameter_19, parameter_18, parameter_21, parameter_25, parameter_22, parameter_24, parameter_23, parameter_26, parameter_30, parameter_27, parameter_29, parameter_28, parameter_31, parameter_32, parameter_36, parameter_33, parameter_35, parameter_34, parameter_37, parameter_41, parameter_38, parameter_40, parameter_39, parameter_42, parameter_46, parameter_43, parameter_45, parameter_44, parameter_47, parameter_51, parameter_48, parameter_50, parameter_49, parameter_52, parameter_56, parameter_53, parameter_55, parameter_54, parameter_57, parameter_61, parameter_58, parameter_60, parameter_59, parameter_62, parameter_66, parameter_63, parameter_65, parameter_64, parameter_67, parameter_68, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_171_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([64, 3, 4, 4], dtype='float16', min=0, max=0.5),
            # parameter_1
            paddle.uniform([128, 64, 4, 4], dtype='float16', min=0, max=0.5),
            # parameter_5
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([256, 128, 4, 4], dtype='float16', min=0, max=0.5),
            # parameter_10
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([512, 256, 4, 4], dtype='float16', min=0, max=0.5),
            # parameter_15
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([512, 512, 4, 4], dtype='float16', min=0, max=0.5),
            # parameter_20
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([512, 512, 4, 4], dtype='float16', min=0, max=0.5),
            # parameter_25
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([512, 512, 4, 4], dtype='float16', min=0, max=0.5),
            # parameter_30
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([512, 512, 4, 4], dtype='float16', min=0, max=0.5),
            # parameter_32
            paddle.uniform([512, 512, 4, 4], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([1024, 512, 4, 4], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([1024, 512, 4, 4], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([1024, 512, 4, 4], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([1024, 256, 4, 4], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([512, 128, 4, 4], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([256, 64, 4, 4], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([128, 3, 4, 4], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([3], dtype='float16', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 3, 256, 256], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_0
            paddle.static.InputSpec(shape=[64, 3, 4, 4], dtype='float16'),
            # parameter_1
            paddle.static.InputSpec(shape=[128, 64, 4, 4], dtype='float16'),
            # parameter_5
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[256, 128, 4, 4], dtype='float16'),
            # parameter_10
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[512, 256, 4, 4], dtype='float16'),
            # parameter_15
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[512, 512, 4, 4], dtype='float16'),
            # parameter_20
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[512, 512, 4, 4], dtype='float16'),
            # parameter_25
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[512, 512, 4, 4], dtype='float16'),
            # parameter_30
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[512, 512, 4, 4], dtype='float16'),
            # parameter_32
            paddle.static.InputSpec(shape=[512, 512, 4, 4], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[1024, 512, 4, 4], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[1024, 512, 4, 4], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[1024, 512, 4, 4], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[1024, 256, 4, 4], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[512, 128, 4, 4], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[256, 64, 4, 4], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[128, 3, 4, 4], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[3], dtype='float16'),
            # feed_0
            paddle.static.InputSpec(shape=[None, 3, None, None], dtype='float32'),
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