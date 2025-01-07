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
    return [265][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_366_0_0(self, parameter_0, parameter_1, parameter_3, parameter_2, parameter_4, parameter_5, parameter_7, parameter_6, parameter_8, parameter_9, parameter_11, parameter_10, parameter_12, parameter_13, parameter_15, parameter_14, parameter_16, parameter_17, parameter_19, parameter_18, parameter_20, parameter_21, parameter_23, parameter_22, parameter_24, parameter_25, parameter_27, parameter_26, parameter_28, parameter_29, parameter_31, parameter_30, parameter_32, parameter_33, parameter_35, parameter_34, parameter_36, parameter_37, parameter_39, parameter_38, parameter_40, parameter_41, parameter_43, parameter_42, parameter_44, parameter_45, parameter_47, parameter_46, parameter_48, parameter_49, parameter_51, parameter_50, parameter_52, parameter_53, parameter_55, parameter_54, parameter_56, parameter_57, parameter_59, parameter_58, parameter_60, parameter_61, parameter_63, parameter_62, parameter_64, parameter_65, parameter_67, parameter_66, parameter_68, parameter_69, parameter_71, parameter_70, parameter_72, parameter_73, parameter_75, parameter_74, parameter_76, parameter_77, parameter_79, parameter_78, parameter_80, parameter_81, parameter_83, parameter_82, parameter_84, parameter_85, parameter_87, parameter_86, parameter_88, parameter_89, parameter_91, parameter_90, parameter_92, parameter_93, feed_0):

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [2]

        # pd_op.unsqueeze: (-1x3x1x-1x-1xf32, None) <- (-1x3x-1x-1xf32, 1xi64)
        unsqueeze_0, unsqueeze_1 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(feed_0, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_1 = [3, 3, 3, 3, 0, 0]

        # pd_op.pad3d: (-1x3x1x-1x-1xf32) <- (-1x3x1x-1x-1xf32, 6xi64)
        pad3d_0 = paddle._C_ops.pad3d(unsqueeze_0, full_int_array_1, 'reflect', float('0'), 'NCDHW')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [2]

        # pd_op.squeeze_: (-1x3x-1x-1xf32, None) <- (-1x3x1x-1x-1xf32, 1xi64)
        squeeze__0, squeeze__1 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pad3d_0, full_int_array_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x3x-1x-1xf32, 64x3x7x7xf32)
        conv2d_0 = paddle._C_ops.conv2d(squeeze__0, parameter_0, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_1, full_int_array_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 1x64x1x1xf32)
        add_0 = paddle._C_ops.add(conv2d_0, reshape_0)

        # pd_op.instance_norm: (-1x64x-1x-1xf32, None, None) <- (-1x64x-1x-1xf32, 64xf32, 64xf32)
        instance_norm_0, instance_norm_1, instance_norm_2 = (lambda x, f: f(x))(paddle._C_ops.instance_norm(add_0, parameter_2, parameter_3, float('1e-05')), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_0 = paddle._C_ops.relu(instance_norm_0)

        # pd_op.conv2d: (-1x128x-1x-1xf32) <- (-1x64x-1x-1xf32, 128x64x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(relu_0, parameter_4, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_4 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_5, full_int_array_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32, 1x128x1x1xf32)
        add_1 = paddle._C_ops.add(conv2d_1, reshape_2)

        # pd_op.instance_norm: (-1x128x-1x-1xf32, None, None) <- (-1x128x-1x-1xf32, 128xf32, 128xf32)
        instance_norm_3, instance_norm_4, instance_norm_5 = (lambda x, f: f(x))(paddle._C_ops.instance_norm(add_1, parameter_6, parameter_7, float('1e-05')), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_1 = paddle._C_ops.relu(instance_norm_3)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x128x-1x-1xf32, 256x128x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(relu_1, parameter_8, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_5 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_9, full_int_array_5), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_2 = paddle._C_ops.add(conv2d_2, reshape_4)

        # pd_op.instance_norm: (-1x256x-1x-1xf32, None, None) <- (-1x256x-1x-1xf32, 256xf32, 256xf32)
        instance_norm_6, instance_norm_7, instance_norm_8 = (lambda x, f: f(x))(paddle._C_ops.instance_norm(add_2, parameter_10, parameter_11, float('1e-05')), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_2 = paddle._C_ops.relu(instance_norm_6)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [2]

        # pd_op.unsqueeze: (-1x256x1x-1x-1xf32, None) <- (-1x256x-1x-1xf32, 1xi64)
        unsqueeze_2, unsqueeze_3 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(relu_2, full_int_array_6), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_7 = [1, 1, 1, 1, 0, 0]

        # pd_op.pad3d: (-1x256x1x-1x-1xf32) <- (-1x256x1x-1x-1xf32, 6xi64)
        pad3d_1 = paddle._C_ops.pad3d(unsqueeze_2, full_int_array_7, 'reflect', float('0'), 'NCDHW')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [2]

        # pd_op.squeeze_: (-1x256x-1x-1xf32, None) <- (-1x256x1x-1x-1xf32, 1xi64)
        squeeze__2, squeeze__3 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pad3d_1, full_int_array_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(squeeze__2, parameter_12, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_9 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_13, full_int_array_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_3 = paddle._C_ops.add(conv2d_3, reshape_6)

        # pd_op.instance_norm: (-1x256x-1x-1xf32, None, None) <- (-1x256x-1x-1xf32, 256xf32, 256xf32)
        instance_norm_9, instance_norm_10, instance_norm_11 = (lambda x, f: f(x))(paddle._C_ops.instance_norm(add_3, parameter_14, parameter_15, float('1e-05')), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_3 = paddle._C_ops.relu(instance_norm_9)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_10 = [2]

        # pd_op.unsqueeze_: (-1x256x1x-1x-1xf32, None) <- (-1x256x-1x-1xf32, 1xi64)
        unsqueeze__0, unsqueeze__1 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(relu_3, full_int_array_10), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_11 = [1, 1, 1, 1, 0, 0]

        # pd_op.pad3d: (-1x256x1x-1x-1xf32) <- (-1x256x1x-1x-1xf32, 6xi64)
        pad3d_2 = paddle._C_ops.pad3d(unsqueeze__0, full_int_array_11, 'reflect', float('0'), 'NCDHW')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_12 = [2]

        # pd_op.squeeze_: (-1x256x-1x-1xf32, None) <- (-1x256x1x-1x-1xf32, 1xi64)
        squeeze__4, squeeze__5 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pad3d_2, full_int_array_12), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(squeeze__4, parameter_16, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_13 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_17, full_int_array_13), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_4 = paddle._C_ops.add(conv2d_4, reshape_8)

        # pd_op.instance_norm: (-1x256x-1x-1xf32, None, None) <- (-1x256x-1x-1xf32, 256xf32, 256xf32)
        instance_norm_12, instance_norm_13, instance_norm_14 = (lambda x, f: f(x))(paddle._C_ops.instance_norm(add_4, parameter_18, parameter_19, float('1e-05')), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, -1x256x-1x-1xf32)
        add_5 = paddle._C_ops.add(relu_2, instance_norm_12)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_14 = [2]

        # pd_op.unsqueeze: (-1x256x1x-1x-1xf32, None) <- (-1x256x-1x-1xf32, 1xi64)
        unsqueeze_4, unsqueeze_5 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add_5, full_int_array_14), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_15 = [1, 1, 1, 1, 0, 0]

        # pd_op.pad3d: (-1x256x1x-1x-1xf32) <- (-1x256x1x-1x-1xf32, 6xi64)
        pad3d_3 = paddle._C_ops.pad3d(unsqueeze_4, full_int_array_15, 'reflect', float('0'), 'NCDHW')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_16 = [2]

        # pd_op.squeeze_: (-1x256x-1x-1xf32, None) <- (-1x256x1x-1x-1xf32, 1xi64)
        squeeze__6, squeeze__7 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pad3d_3, full_int_array_16), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(squeeze__6, parameter_20, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_17 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_21, full_int_array_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_6 = paddle._C_ops.add(conv2d_5, reshape_10)

        # pd_op.instance_norm: (-1x256x-1x-1xf32, None, None) <- (-1x256x-1x-1xf32, 256xf32, 256xf32)
        instance_norm_15, instance_norm_16, instance_norm_17 = (lambda x, f: f(x))(paddle._C_ops.instance_norm(add_6, parameter_22, parameter_23, float('1e-05')), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_4 = paddle._C_ops.relu(instance_norm_15)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_18 = [2]

        # pd_op.unsqueeze_: (-1x256x1x-1x-1xf32, None) <- (-1x256x-1x-1xf32, 1xi64)
        unsqueeze__2, unsqueeze__3 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(relu_4, full_int_array_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_19 = [1, 1, 1, 1, 0, 0]

        # pd_op.pad3d: (-1x256x1x-1x-1xf32) <- (-1x256x1x-1x-1xf32, 6xi64)
        pad3d_4 = paddle._C_ops.pad3d(unsqueeze__2, full_int_array_19, 'reflect', float('0'), 'NCDHW')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_20 = [2]

        # pd_op.squeeze_: (-1x256x-1x-1xf32, None) <- (-1x256x1x-1x-1xf32, 1xi64)
        squeeze__8, squeeze__9 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pad3d_4, full_int_array_20), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(squeeze__8, parameter_24, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_21 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_12, reshape_13 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_25, full_int_array_21), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_7 = paddle._C_ops.add(conv2d_6, reshape_12)

        # pd_op.instance_norm: (-1x256x-1x-1xf32, None, None) <- (-1x256x-1x-1xf32, 256xf32, 256xf32)
        instance_norm_18, instance_norm_19, instance_norm_20 = (lambda x, f: f(x))(paddle._C_ops.instance_norm(add_7, parameter_26, parameter_27, float('1e-05')), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, -1x256x-1x-1xf32)
        add_8 = paddle._C_ops.add(add_5, instance_norm_18)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_22 = [2]

        # pd_op.unsqueeze: (-1x256x1x-1x-1xf32, None) <- (-1x256x-1x-1xf32, 1xi64)
        unsqueeze_6, unsqueeze_7 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add_8, full_int_array_22), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_23 = [1, 1, 1, 1, 0, 0]

        # pd_op.pad3d: (-1x256x1x-1x-1xf32) <- (-1x256x1x-1x-1xf32, 6xi64)
        pad3d_5 = paddle._C_ops.pad3d(unsqueeze_6, full_int_array_23, 'reflect', float('0'), 'NCDHW')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_24 = [2]

        # pd_op.squeeze_: (-1x256x-1x-1xf32, None) <- (-1x256x1x-1x-1xf32, 1xi64)
        squeeze__10, squeeze__11 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pad3d_5, full_int_array_24), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_7 = paddle._C_ops.conv2d(squeeze__10, parameter_28, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_25 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_14, reshape_15 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_29, full_int_array_25), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_9 = paddle._C_ops.add(conv2d_7, reshape_14)

        # pd_op.instance_norm: (-1x256x-1x-1xf32, None, None) <- (-1x256x-1x-1xf32, 256xf32, 256xf32)
        instance_norm_21, instance_norm_22, instance_norm_23 = (lambda x, f: f(x))(paddle._C_ops.instance_norm(add_9, parameter_30, parameter_31, float('1e-05')), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_5 = paddle._C_ops.relu(instance_norm_21)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_26 = [2]

        # pd_op.unsqueeze_: (-1x256x1x-1x-1xf32, None) <- (-1x256x-1x-1xf32, 1xi64)
        unsqueeze__4, unsqueeze__5 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(relu_5, full_int_array_26), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_27 = [1, 1, 1, 1, 0, 0]

        # pd_op.pad3d: (-1x256x1x-1x-1xf32) <- (-1x256x1x-1x-1xf32, 6xi64)
        pad3d_6 = paddle._C_ops.pad3d(unsqueeze__4, full_int_array_27, 'reflect', float('0'), 'NCDHW')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_28 = [2]

        # pd_op.squeeze_: (-1x256x-1x-1xf32, None) <- (-1x256x1x-1x-1xf32, 1xi64)
        squeeze__12, squeeze__13 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pad3d_6, full_int_array_28), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(squeeze__12, parameter_32, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_29 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_16, reshape_17 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_33, full_int_array_29), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_10 = paddle._C_ops.add(conv2d_8, reshape_16)

        # pd_op.instance_norm: (-1x256x-1x-1xf32, None, None) <- (-1x256x-1x-1xf32, 256xf32, 256xf32)
        instance_norm_24, instance_norm_25, instance_norm_26 = (lambda x, f: f(x))(paddle._C_ops.instance_norm(add_10, parameter_34, parameter_35, float('1e-05')), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, -1x256x-1x-1xf32)
        add_11 = paddle._C_ops.add(add_8, instance_norm_24)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_30 = [2]

        # pd_op.unsqueeze: (-1x256x1x-1x-1xf32, None) <- (-1x256x-1x-1xf32, 1xi64)
        unsqueeze_8, unsqueeze_9 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add_11, full_int_array_30), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_31 = [1, 1, 1, 1, 0, 0]

        # pd_op.pad3d: (-1x256x1x-1x-1xf32) <- (-1x256x1x-1x-1xf32, 6xi64)
        pad3d_7 = paddle._C_ops.pad3d(unsqueeze_8, full_int_array_31, 'reflect', float('0'), 'NCDHW')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_32 = [2]

        # pd_op.squeeze_: (-1x256x-1x-1xf32, None) <- (-1x256x1x-1x-1xf32, 1xi64)
        squeeze__14, squeeze__15 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pad3d_7, full_int_array_32), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_9 = paddle._C_ops.conv2d(squeeze__14, parameter_36, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_33 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_18, reshape_19 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_37, full_int_array_33), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_12 = paddle._C_ops.add(conv2d_9, reshape_18)

        # pd_op.instance_norm: (-1x256x-1x-1xf32, None, None) <- (-1x256x-1x-1xf32, 256xf32, 256xf32)
        instance_norm_27, instance_norm_28, instance_norm_29 = (lambda x, f: f(x))(paddle._C_ops.instance_norm(add_12, parameter_38, parameter_39, float('1e-05')), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_6 = paddle._C_ops.relu(instance_norm_27)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_34 = [2]

        # pd_op.unsqueeze_: (-1x256x1x-1x-1xf32, None) <- (-1x256x-1x-1xf32, 1xi64)
        unsqueeze__6, unsqueeze__7 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(relu_6, full_int_array_34), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_35 = [1, 1, 1, 1, 0, 0]

        # pd_op.pad3d: (-1x256x1x-1x-1xf32) <- (-1x256x1x-1x-1xf32, 6xi64)
        pad3d_8 = paddle._C_ops.pad3d(unsqueeze__6, full_int_array_35, 'reflect', float('0'), 'NCDHW')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_36 = [2]

        # pd_op.squeeze_: (-1x256x-1x-1xf32, None) <- (-1x256x1x-1x-1xf32, 1xi64)
        squeeze__16, squeeze__17 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pad3d_8, full_int_array_36), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_10 = paddle._C_ops.conv2d(squeeze__16, parameter_40, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_37 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_20, reshape_21 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_41, full_int_array_37), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_13 = paddle._C_ops.add(conv2d_10, reshape_20)

        # pd_op.instance_norm: (-1x256x-1x-1xf32, None, None) <- (-1x256x-1x-1xf32, 256xf32, 256xf32)
        instance_norm_30, instance_norm_31, instance_norm_32 = (lambda x, f: f(x))(paddle._C_ops.instance_norm(add_13, parameter_42, parameter_43, float('1e-05')), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, -1x256x-1x-1xf32)
        add_14 = paddle._C_ops.add(add_11, instance_norm_30)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_38 = [2]

        # pd_op.unsqueeze: (-1x256x1x-1x-1xf32, None) <- (-1x256x-1x-1xf32, 1xi64)
        unsqueeze_10, unsqueeze_11 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add_14, full_int_array_38), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_39 = [1, 1, 1, 1, 0, 0]

        # pd_op.pad3d: (-1x256x1x-1x-1xf32) <- (-1x256x1x-1x-1xf32, 6xi64)
        pad3d_9 = paddle._C_ops.pad3d(unsqueeze_10, full_int_array_39, 'reflect', float('0'), 'NCDHW')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_40 = [2]

        # pd_op.squeeze_: (-1x256x-1x-1xf32, None) <- (-1x256x1x-1x-1xf32, 1xi64)
        squeeze__18, squeeze__19 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pad3d_9, full_int_array_40), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(squeeze__18, parameter_44, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_41 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_22, reshape_23 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_45, full_int_array_41), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_15 = paddle._C_ops.add(conv2d_11, reshape_22)

        # pd_op.instance_norm: (-1x256x-1x-1xf32, None, None) <- (-1x256x-1x-1xf32, 256xf32, 256xf32)
        instance_norm_33, instance_norm_34, instance_norm_35 = (lambda x, f: f(x))(paddle._C_ops.instance_norm(add_15, parameter_46, parameter_47, float('1e-05')), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_7 = paddle._C_ops.relu(instance_norm_33)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_42 = [2]

        # pd_op.unsqueeze_: (-1x256x1x-1x-1xf32, None) <- (-1x256x-1x-1xf32, 1xi64)
        unsqueeze__8, unsqueeze__9 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(relu_7, full_int_array_42), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_43 = [1, 1, 1, 1, 0, 0]

        # pd_op.pad3d: (-1x256x1x-1x-1xf32) <- (-1x256x1x-1x-1xf32, 6xi64)
        pad3d_10 = paddle._C_ops.pad3d(unsqueeze__8, full_int_array_43, 'reflect', float('0'), 'NCDHW')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_44 = [2]

        # pd_op.squeeze_: (-1x256x-1x-1xf32, None) <- (-1x256x1x-1x-1xf32, 1xi64)
        squeeze__20, squeeze__21 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pad3d_10, full_int_array_44), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_12 = paddle._C_ops.conv2d(squeeze__20, parameter_48, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_45 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_24, reshape_25 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_49, full_int_array_45), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_16 = paddle._C_ops.add(conv2d_12, reshape_24)

        # pd_op.instance_norm: (-1x256x-1x-1xf32, None, None) <- (-1x256x-1x-1xf32, 256xf32, 256xf32)
        instance_norm_36, instance_norm_37, instance_norm_38 = (lambda x, f: f(x))(paddle._C_ops.instance_norm(add_16, parameter_50, parameter_51, float('1e-05')), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, -1x256x-1x-1xf32)
        add_17 = paddle._C_ops.add(add_14, instance_norm_36)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_46 = [2]

        # pd_op.unsqueeze: (-1x256x1x-1x-1xf32, None) <- (-1x256x-1x-1xf32, 1xi64)
        unsqueeze_12, unsqueeze_13 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add_17, full_int_array_46), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_47 = [1, 1, 1, 1, 0, 0]

        # pd_op.pad3d: (-1x256x1x-1x-1xf32) <- (-1x256x1x-1x-1xf32, 6xi64)
        pad3d_11 = paddle._C_ops.pad3d(unsqueeze_12, full_int_array_47, 'reflect', float('0'), 'NCDHW')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_48 = [2]

        # pd_op.squeeze_: (-1x256x-1x-1xf32, None) <- (-1x256x1x-1x-1xf32, 1xi64)
        squeeze__22, squeeze__23 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pad3d_11, full_int_array_48), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_13 = paddle._C_ops.conv2d(squeeze__22, parameter_52, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_49 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_26, reshape_27 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_53, full_int_array_49), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_18 = paddle._C_ops.add(conv2d_13, reshape_26)

        # pd_op.instance_norm: (-1x256x-1x-1xf32, None, None) <- (-1x256x-1x-1xf32, 256xf32, 256xf32)
        instance_norm_39, instance_norm_40, instance_norm_41 = (lambda x, f: f(x))(paddle._C_ops.instance_norm(add_18, parameter_54, parameter_55, float('1e-05')), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_8 = paddle._C_ops.relu(instance_norm_39)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_50 = [2]

        # pd_op.unsqueeze_: (-1x256x1x-1x-1xf32, None) <- (-1x256x-1x-1xf32, 1xi64)
        unsqueeze__10, unsqueeze__11 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(relu_8, full_int_array_50), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_51 = [1, 1, 1, 1, 0, 0]

        # pd_op.pad3d: (-1x256x1x-1x-1xf32) <- (-1x256x1x-1x-1xf32, 6xi64)
        pad3d_12 = paddle._C_ops.pad3d(unsqueeze__10, full_int_array_51, 'reflect', float('0'), 'NCDHW')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_52 = [2]

        # pd_op.squeeze_: (-1x256x-1x-1xf32, None) <- (-1x256x1x-1x-1xf32, 1xi64)
        squeeze__24, squeeze__25 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pad3d_12, full_int_array_52), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_14 = paddle._C_ops.conv2d(squeeze__24, parameter_56, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_53 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_28, reshape_29 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_57, full_int_array_53), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_19 = paddle._C_ops.add(conv2d_14, reshape_28)

        # pd_op.instance_norm: (-1x256x-1x-1xf32, None, None) <- (-1x256x-1x-1xf32, 256xf32, 256xf32)
        instance_norm_42, instance_norm_43, instance_norm_44 = (lambda x, f: f(x))(paddle._C_ops.instance_norm(add_19, parameter_58, parameter_59, float('1e-05')), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, -1x256x-1x-1xf32)
        add_20 = paddle._C_ops.add(add_17, instance_norm_42)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_54 = [2]

        # pd_op.unsqueeze: (-1x256x1x-1x-1xf32, None) <- (-1x256x-1x-1xf32, 1xi64)
        unsqueeze_14, unsqueeze_15 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add_20, full_int_array_54), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_55 = [1, 1, 1, 1, 0, 0]

        # pd_op.pad3d: (-1x256x1x-1x-1xf32) <- (-1x256x1x-1x-1xf32, 6xi64)
        pad3d_13 = paddle._C_ops.pad3d(unsqueeze_14, full_int_array_55, 'reflect', float('0'), 'NCDHW')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_56 = [2]

        # pd_op.squeeze_: (-1x256x-1x-1xf32, None) <- (-1x256x1x-1x-1xf32, 1xi64)
        squeeze__26, squeeze__27 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pad3d_13, full_int_array_56), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_15 = paddle._C_ops.conv2d(squeeze__26, parameter_60, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_57 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_30, reshape_31 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_61, full_int_array_57), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_21 = paddle._C_ops.add(conv2d_15, reshape_30)

        # pd_op.instance_norm: (-1x256x-1x-1xf32, None, None) <- (-1x256x-1x-1xf32, 256xf32, 256xf32)
        instance_norm_45, instance_norm_46, instance_norm_47 = (lambda x, f: f(x))(paddle._C_ops.instance_norm(add_21, parameter_62, parameter_63, float('1e-05')), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_9 = paddle._C_ops.relu(instance_norm_45)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_58 = [2]

        # pd_op.unsqueeze_: (-1x256x1x-1x-1xf32, None) <- (-1x256x-1x-1xf32, 1xi64)
        unsqueeze__12, unsqueeze__13 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(relu_9, full_int_array_58), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_59 = [1, 1, 1, 1, 0, 0]

        # pd_op.pad3d: (-1x256x1x-1x-1xf32) <- (-1x256x1x-1x-1xf32, 6xi64)
        pad3d_14 = paddle._C_ops.pad3d(unsqueeze__12, full_int_array_59, 'reflect', float('0'), 'NCDHW')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_60 = [2]

        # pd_op.squeeze_: (-1x256x-1x-1xf32, None) <- (-1x256x1x-1x-1xf32, 1xi64)
        squeeze__28, squeeze__29 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pad3d_14, full_int_array_60), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_16 = paddle._C_ops.conv2d(squeeze__28, parameter_64, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_61 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_32, reshape_33 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_65, full_int_array_61), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_22 = paddle._C_ops.add(conv2d_16, reshape_32)

        # pd_op.instance_norm: (-1x256x-1x-1xf32, None, None) <- (-1x256x-1x-1xf32, 256xf32, 256xf32)
        instance_norm_48, instance_norm_49, instance_norm_50 = (lambda x, f: f(x))(paddle._C_ops.instance_norm(add_22, parameter_66, parameter_67, float('1e-05')), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, -1x256x-1x-1xf32)
        add_23 = paddle._C_ops.add(add_20, instance_norm_48)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_62 = [2]

        # pd_op.unsqueeze: (-1x256x1x-1x-1xf32, None) <- (-1x256x-1x-1xf32, 1xi64)
        unsqueeze_16, unsqueeze_17 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add_23, full_int_array_62), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_63 = [1, 1, 1, 1, 0, 0]

        # pd_op.pad3d: (-1x256x1x-1x-1xf32) <- (-1x256x1x-1x-1xf32, 6xi64)
        pad3d_15 = paddle._C_ops.pad3d(unsqueeze_16, full_int_array_63, 'reflect', float('0'), 'NCDHW')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_64 = [2]

        # pd_op.squeeze_: (-1x256x-1x-1xf32, None) <- (-1x256x1x-1x-1xf32, 1xi64)
        squeeze__30, squeeze__31 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pad3d_15, full_int_array_64), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_17 = paddle._C_ops.conv2d(squeeze__30, parameter_68, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_65 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_34, reshape_35 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_69, full_int_array_65), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_24 = paddle._C_ops.add(conv2d_17, reshape_34)

        # pd_op.instance_norm: (-1x256x-1x-1xf32, None, None) <- (-1x256x-1x-1xf32, 256xf32, 256xf32)
        instance_norm_51, instance_norm_52, instance_norm_53 = (lambda x, f: f(x))(paddle._C_ops.instance_norm(add_24, parameter_70, parameter_71, float('1e-05')), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_10 = paddle._C_ops.relu(instance_norm_51)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_66 = [2]

        # pd_op.unsqueeze_: (-1x256x1x-1x-1xf32, None) <- (-1x256x-1x-1xf32, 1xi64)
        unsqueeze__14, unsqueeze__15 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(relu_10, full_int_array_66), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_67 = [1, 1, 1, 1, 0, 0]

        # pd_op.pad3d: (-1x256x1x-1x-1xf32) <- (-1x256x1x-1x-1xf32, 6xi64)
        pad3d_16 = paddle._C_ops.pad3d(unsqueeze__14, full_int_array_67, 'reflect', float('0'), 'NCDHW')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_68 = [2]

        # pd_op.squeeze_: (-1x256x-1x-1xf32, None) <- (-1x256x1x-1x-1xf32, 1xi64)
        squeeze__32, squeeze__33 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pad3d_16, full_int_array_68), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_18 = paddle._C_ops.conv2d(squeeze__32, parameter_72, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_69 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_36, reshape_37 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_73, full_int_array_69), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_25 = paddle._C_ops.add(conv2d_18, reshape_36)

        # pd_op.instance_norm: (-1x256x-1x-1xf32, None, None) <- (-1x256x-1x-1xf32, 256xf32, 256xf32)
        instance_norm_54, instance_norm_55, instance_norm_56 = (lambda x, f: f(x))(paddle._C_ops.instance_norm(add_25, parameter_74, parameter_75, float('1e-05')), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, -1x256x-1x-1xf32)
        add_26 = paddle._C_ops.add(add_23, instance_norm_54)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_70 = [2]

        # pd_op.unsqueeze: (-1x256x1x-1x-1xf32, None) <- (-1x256x-1x-1xf32, 1xi64)
        unsqueeze_18, unsqueeze_19 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(add_26, full_int_array_70), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_71 = [1, 1, 1, 1, 0, 0]

        # pd_op.pad3d: (-1x256x1x-1x-1xf32) <- (-1x256x1x-1x-1xf32, 6xi64)
        pad3d_17 = paddle._C_ops.pad3d(unsqueeze_18, full_int_array_71, 'reflect', float('0'), 'NCDHW')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_72 = [2]

        # pd_op.squeeze_: (-1x256x-1x-1xf32, None) <- (-1x256x1x-1x-1xf32, 1xi64)
        squeeze__34, squeeze__35 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pad3d_17, full_int_array_72), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_19 = paddle._C_ops.conv2d(squeeze__34, parameter_76, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_73 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_38, reshape_39 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_77, full_int_array_73), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_27 = paddle._C_ops.add(conv2d_19, reshape_38)

        # pd_op.instance_norm: (-1x256x-1x-1xf32, None, None) <- (-1x256x-1x-1xf32, 256xf32, 256xf32)
        instance_norm_57, instance_norm_58, instance_norm_59 = (lambda x, f: f(x))(paddle._C_ops.instance_norm(add_27, parameter_78, parameter_79, float('1e-05')), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_11 = paddle._C_ops.relu(instance_norm_57)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_74 = [2]

        # pd_op.unsqueeze_: (-1x256x1x-1x-1xf32, None) <- (-1x256x-1x-1xf32, 1xi64)
        unsqueeze__16, unsqueeze__17 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(relu_11, full_int_array_74), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_75 = [1, 1, 1, 1, 0, 0]

        # pd_op.pad3d: (-1x256x1x-1x-1xf32) <- (-1x256x1x-1x-1xf32, 6xi64)
        pad3d_18 = paddle._C_ops.pad3d(unsqueeze__16, full_int_array_75, 'reflect', float('0'), 'NCDHW')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_76 = [2]

        # pd_op.squeeze_: (-1x256x-1x-1xf32, None) <- (-1x256x1x-1x-1xf32, 1xi64)
        squeeze__36, squeeze__37 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pad3d_18, full_int_array_76), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_20 = paddle._C_ops.conv2d(squeeze__36, parameter_80, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_77 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_40, reshape_41 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_81, full_int_array_77), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_28 = paddle._C_ops.add(conv2d_20, reshape_40)

        # pd_op.instance_norm: (-1x256x-1x-1xf32, None, None) <- (-1x256x-1x-1xf32, 256xf32, 256xf32)
        instance_norm_60, instance_norm_61, instance_norm_62 = (lambda x, f: f(x))(paddle._C_ops.instance_norm(add_28, parameter_82, parameter_83, float('1e-05')), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, -1x256x-1x-1xf32)
        add_29 = paddle._C_ops.add(add_26, instance_norm_60)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_78 = []

        # pd_op.conv2d_transpose: (-1x128x0x0xf32) <- (-1x256x-1x-1xf32, 256x128x3x3xf32, 0xi64)
        conv2d_transpose_0 = paddle._C_ops.conv2d_transpose(add_29, parameter_84, [2, 2], [1, 1], [1, 1], full_int_array_78, 'EXPLICIT', 1, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_79 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_42, reshape_43 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_85, full_int_array_79), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x128x-1x-1xf32) <- (-1x128x0x0xf32, 1x128x1x1xf32)
        add_30 = paddle._C_ops.add(conv2d_transpose_0, reshape_42)

        # pd_op.instance_norm: (-1x128x-1x-1xf32, None, None) <- (-1x128x-1x-1xf32, 128xf32, 128xf32)
        instance_norm_63, instance_norm_64, instance_norm_65 = (lambda x, f: f(x))(paddle._C_ops.instance_norm(add_30, parameter_86, parameter_87, float('1e-05')), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x128x-1x-1xf32) <- (-1x128x-1x-1xf32)
        relu_12 = paddle._C_ops.relu(instance_norm_63)

        # pd_op.full_int_array: (0xi64) <- ()
        full_int_array_80 = []

        # pd_op.conv2d_transpose: (-1x64x0x0xf32) <- (-1x128x-1x-1xf32, 128x64x3x3xf32, 0xi64)
        conv2d_transpose_1 = paddle._C_ops.conv2d_transpose(relu_12, parameter_88, [2, 2], [1, 1], [1, 1], full_int_array_80, 'EXPLICIT', 1, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_81 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf32, 0x64xf32) <- (64xf32, 4xi64)
        reshape_44, reshape_45 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_89, full_int_array_81), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x64x-1x-1xf32) <- (-1x64x0x0xf32, 1x64x1x1xf32)
        add_31 = paddle._C_ops.add(conv2d_transpose_1, reshape_44)

        # pd_op.instance_norm: (-1x64x-1x-1xf32, None, None) <- (-1x64x-1x-1xf32, 64xf32, 64xf32)
        instance_norm_66, instance_norm_67, instance_norm_68 = (lambda x, f: f(x))(paddle._C_ops.instance_norm(add_31, parameter_90, parameter_91, float('1e-05')), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_13 = paddle._C_ops.relu(instance_norm_66)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_82 = [2]

        # pd_op.unsqueeze_: (-1x64x1x-1x-1xf32, None) <- (-1x64x-1x-1xf32, 1xi64)
        unsqueeze__18, unsqueeze__19 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(relu_13, full_int_array_82), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_83 = [3, 3, 3, 3, 0, 0]

        # pd_op.pad3d: (-1x64x1x-1x-1xf32) <- (-1x64x1x-1x-1xf32, 6xi64)
        pad3d_19 = paddle._C_ops.pad3d(unsqueeze__18, full_int_array_83, 'reflect', float('0'), 'NCDHW')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_84 = [2]

        # pd_op.squeeze_: (-1x64x-1x-1xf32, None) <- (-1x64x1x-1x-1xf32, 1xi64)
        squeeze__38, squeeze__39 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pad3d_19, full_int_array_84), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x3x-1x-1xf32) <- (-1x64x-1x-1xf32, 3x64x7x7xf32)
        conv2d_21 = paddle._C_ops.conv2d(squeeze__38, parameter_92, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_85 = [1, 3, 1, 1]

        # pd_op.reshape: (1x3x1x1xf32, 0x3xf32) <- (3xf32, 4xi64)
        reshape_46, reshape_47 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_93, full_int_array_85), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x3x-1x-1xf32) <- (-1x3x-1x-1xf32, 1x3x1x1xf32)
        add_32 = paddle._C_ops.add(conv2d_21, reshape_46)

        # pd_op.tanh: (-1x3x-1x-1xf32) <- (-1x3x-1x-1xf32)
        tanh_0 = paddle._C_ops.tanh(add_32)
        return tanh_0



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

    def forward(self, parameter_0, parameter_1, parameter_3, parameter_2, parameter_4, parameter_5, parameter_7, parameter_6, parameter_8, parameter_9, parameter_11, parameter_10, parameter_12, parameter_13, parameter_15, parameter_14, parameter_16, parameter_17, parameter_19, parameter_18, parameter_20, parameter_21, parameter_23, parameter_22, parameter_24, parameter_25, parameter_27, parameter_26, parameter_28, parameter_29, parameter_31, parameter_30, parameter_32, parameter_33, parameter_35, parameter_34, parameter_36, parameter_37, parameter_39, parameter_38, parameter_40, parameter_41, parameter_43, parameter_42, parameter_44, parameter_45, parameter_47, parameter_46, parameter_48, parameter_49, parameter_51, parameter_50, parameter_52, parameter_53, parameter_55, parameter_54, parameter_56, parameter_57, parameter_59, parameter_58, parameter_60, parameter_61, parameter_63, parameter_62, parameter_64, parameter_65, parameter_67, parameter_66, parameter_68, parameter_69, parameter_71, parameter_70, parameter_72, parameter_73, parameter_75, parameter_74, parameter_76, parameter_77, parameter_79, parameter_78, parameter_80, parameter_81, parameter_83, parameter_82, parameter_84, parameter_85, parameter_87, parameter_86, parameter_88, parameter_89, parameter_91, parameter_90, parameter_92, parameter_93, feed_0):
        return self.builtin_module_366_0_0(parameter_0, parameter_1, parameter_3, parameter_2, parameter_4, parameter_5, parameter_7, parameter_6, parameter_8, parameter_9, parameter_11, parameter_10, parameter_12, parameter_13, parameter_15, parameter_14, parameter_16, parameter_17, parameter_19, parameter_18, parameter_20, parameter_21, parameter_23, parameter_22, parameter_24, parameter_25, parameter_27, parameter_26, parameter_28, parameter_29, parameter_31, parameter_30, parameter_32, parameter_33, parameter_35, parameter_34, parameter_36, parameter_37, parameter_39, parameter_38, parameter_40, parameter_41, parameter_43, parameter_42, parameter_44, parameter_45, parameter_47, parameter_46, parameter_48, parameter_49, parameter_51, parameter_50, parameter_52, parameter_53, parameter_55, parameter_54, parameter_56, parameter_57, parameter_59, parameter_58, parameter_60, parameter_61, parameter_63, parameter_62, parameter_64, parameter_65, parameter_67, parameter_66, parameter_68, parameter_69, parameter_71, parameter_70, parameter_72, parameter_73, parameter_75, parameter_74, parameter_76, parameter_77, parameter_79, parameter_78, parameter_80, parameter_81, parameter_83, parameter_82, parameter_84, parameter_85, parameter_87, parameter_86, parameter_88, parameter_89, parameter_91, parameter_90, parameter_92, parameter_93, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_366_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([64, 3, 7, 7], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([128, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([256, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([128, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([3, 64, 7, 7], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([3], dtype='float32', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 3, 256, 256], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_0
            paddle.static.InputSpec(shape=[64, 3, 7, 7], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[128, 64, 3, 3], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[256, 128, 3, 3], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[256, 128, 3, 3], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[128, 64, 3, 3], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[3, 64, 7, 7], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[3], dtype='float32'),
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