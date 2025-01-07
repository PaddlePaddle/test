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
    return [206][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_381_0_0(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_56, parameter_57, parameter_58, parameter_59, parameter_60, parameter_61, parameter_65, parameter_62, parameter_64, parameter_63, parameter_66, parameter_70, parameter_67, parameter_69, parameter_68, parameter_71, parameter_72, parameter_73, parameter_74, parameter_75, parameter_76, parameter_77, parameter_81, parameter_78, parameter_80, parameter_79, parameter_82, parameter_86, parameter_83, parameter_85, parameter_84, parameter_87, parameter_88, parameter_89, parameter_90, parameter_91, parameter_92, parameter_93, parameter_97, parameter_94, parameter_96, parameter_95, parameter_98, parameter_102, parameter_99, parameter_101, parameter_100, parameter_103, parameter_104, parameter_105, parameter_106, parameter_107, parameter_108, parameter_109, parameter_113, parameter_110, parameter_112, parameter_111, parameter_114, parameter_118, parameter_115, parameter_117, parameter_116, parameter_119, parameter_120, parameter_121, parameter_122, parameter_123, parameter_124, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_136, parameter_137, parameter_138, parameter_139, parameter_140, parameter_141, parameter_145, parameter_142, parameter_144, parameter_143, parameter_146, parameter_150, parameter_147, parameter_149, parameter_148, parameter_151, parameter_152, parameter_153, parameter_157, parameter_154, parameter_156, parameter_155, parameter_158, parameter_159, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_166, parameter_167, feed_0):

        # pd_op.cast: (-1x3x224x224xf16) <- (-1x3x224x224xf32)
        cast_0 = paddle._C_ops.cast(feed_0, paddle.float16)

        # pd_op.conv2d: (-1x32x112x112xf16) <- (-1x3x224x224xf16, 32x3x3x3xf16)
        conv2d_0 = paddle._C_ops.conv2d(cast_0, parameter_0, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x112x112xf16, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x112x112xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_0, parameter_1, parameter_2, parameter_3, parameter_4, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x32x112x112xf16) <- (-1x32x112x112xf16)
        relu__0 = paddle._C_ops.relu_(batch_norm__0)

        # pd_op.depthwise_conv2d: (-1x32x112x112xf16) <- (-1x32x112x112xf16, 32x1x3x3xf16)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(relu__0, parameter_5, [1, 1], [1, 1], 'EXPLICIT', 32, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x32x112x112xf16, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x112x112xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_0, parameter_6, parameter_7, parameter_8, parameter_9, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x32x112x112xf16) <- (-1x32x112x112xf16)
        relu__1 = paddle._C_ops.relu_(batch_norm__6)

        # pd_op.conv2d: (-1x64x112x112xf16) <- (-1x32x112x112xf16, 64x32x1x1xf16)
        conv2d_1 = paddle._C_ops.conv2d(relu__1, parameter_10, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x112x112xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x112x112xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_1, parameter_11, parameter_12, parameter_13, parameter_14, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x112x112xf16) <- (-1x64x112x112xf16)
        relu__2 = paddle._C_ops.relu_(batch_norm__12)

        # pd_op.depthwise_conv2d: (-1x64x56x56xf16) <- (-1x64x112x112xf16, 64x1x3x3xf16)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(relu__2, parameter_15, [2, 2], [1, 1], 'EXPLICIT', 64, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x64x56x56xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x56x56xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_1, parameter_16, parameter_17, parameter_18, parameter_19, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x56x56xf16) <- (-1x64x56x56xf16)
        relu__3 = paddle._C_ops.relu_(batch_norm__18)

        # pd_op.conv2d: (-1x128x56x56xf16) <- (-1x64x56x56xf16, 128x64x1x1xf16)
        conv2d_2 = paddle._C_ops.conv2d(relu__3, parameter_20, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x56x56xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x56x56xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_2, parameter_21, parameter_22, parameter_23, parameter_24, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x56x56xf16) <- (-1x128x56x56xf16)
        relu__4 = paddle._C_ops.relu_(batch_norm__24)

        # pd_op.depthwise_conv2d: (-1x128x56x56xf16) <- (-1x128x56x56xf16, 128x1x3x3xf16)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(relu__4, parameter_25, [1, 1], [1, 1], 'EXPLICIT', 128, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x128x56x56xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x56x56xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_2, parameter_26, parameter_27, parameter_28, parameter_29, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x56x56xf16) <- (-1x128x56x56xf16)
        relu__5 = paddle._C_ops.relu_(batch_norm__30)

        # pd_op.conv2d: (-1x128x56x56xf16) <- (-1x128x56x56xf16, 128x128x1x1xf16)
        conv2d_3 = paddle._C_ops.conv2d(relu__5, parameter_30, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x56x56xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x56x56xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_3, parameter_31, parameter_32, parameter_33, parameter_34, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x56x56xf16) <- (-1x128x56x56xf16)
        relu__6 = paddle._C_ops.relu_(batch_norm__36)

        # pd_op.depthwise_conv2d: (-1x128x28x28xf16) <- (-1x128x56x56xf16, 128x1x3x3xf16)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(relu__6, parameter_35, [2, 2], [1, 1], 'EXPLICIT', 128, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x128x28x28xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x28x28xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_3, parameter_36, parameter_37, parameter_38, parameter_39, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x28x28xf16) <- (-1x128x28x28xf16)
        relu__7 = paddle._C_ops.relu_(batch_norm__42)

        # pd_op.conv2d: (-1x256x28x28xf16) <- (-1x128x28x28xf16, 256x128x1x1xf16)
        conv2d_4 = paddle._C_ops.conv2d(relu__7, parameter_40, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x28x28xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x28x28xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_4, parameter_41, parameter_42, parameter_43, parameter_44, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x28x28xf16) <- (-1x256x28x28xf16)
        relu__8 = paddle._C_ops.relu_(batch_norm__48)

        # pd_op.depthwise_conv2d: (-1x256x28x28xf16) <- (-1x256x28x28xf16, 256x1x3x3xf16)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(relu__8, parameter_45, [1, 1], [1, 1], 'EXPLICIT', 256, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x256x28x28xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x28x28xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_4, parameter_46, parameter_47, parameter_48, parameter_49, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x28x28xf16) <- (-1x256x28x28xf16)
        relu__9 = paddle._C_ops.relu_(batch_norm__54)

        # pd_op.conv2d: (-1x256x28x28xf16) <- (-1x256x28x28xf16, 256x256x1x1xf16)
        conv2d_5 = paddle._C_ops.conv2d(relu__9, parameter_50, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x28x28xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x28x28xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_5, parameter_51, parameter_52, parameter_53, parameter_54, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x28x28xf16) <- (-1x256x28x28xf16)
        relu__10 = paddle._C_ops.relu_(batch_norm__60)

        # pd_op.depthwise_conv2d: (-1x256x14x14xf16) <- (-1x256x28x28xf16, 256x1x5x5xf16)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(relu__10, parameter_55, [2, 2], [2, 2], 'EXPLICIT', 256, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_56, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x14x14xf16) <- (-1x256x14x14xf16, 1x256x1x1xf16)
        add__0 = paddle._C_ops.add_(depthwise_conv2d_5, reshape_0)

        # pd_op.relu_: (-1x256x14x14xf16) <- (-1x256x14x14xf16)
        relu__11 = paddle._C_ops.relu_(add__0)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf16) <- (-1x256x14x14xf16, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(relu__11, full_int_array_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x64x1x1xf16) <- (-1x256x1x1xf16, 64x256x1x1xf16)
        conv2d_6 = paddle._C_ops.conv2d(pool2d_0, parameter_57, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_2 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf16, 0x64xf16) <- (64xf16, 4xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_58, full_int_array_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x1x1xf16) <- (-1x64x1x1xf16, 1x64x1x1xf16)
        add__1 = paddle._C_ops.add_(conv2d_6, reshape_2)

        # pd_op.relu_: (-1x64x1x1xf16) <- (-1x64x1x1xf16)
        relu__12 = paddle._C_ops.relu_(add__1)

        # pd_op.conv2d: (-1x256x1x1xf16) <- (-1x64x1x1xf16, 256x64x1x1xf16)
        conv2d_7 = paddle._C_ops.conv2d(relu__12, parameter_59, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_60, full_int_array_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x1x1xf16) <- (-1x256x1x1xf16, 1x256x1x1xf16)
        add__2 = paddle._C_ops.add_(conv2d_7, reshape_4)

        # pd_op.sigmoid_: (-1x256x1x1xf16) <- (-1x256x1x1xf16)
        sigmoid__0 = paddle._C_ops.sigmoid_(add__2)

        # pd_op.multiply_: (-1x256x14x14xf16) <- (-1x256x14x14xf16, -1x256x1x1xf16)
        multiply__0 = paddle._C_ops.multiply_(relu__11, sigmoid__0)

        # pd_op.conv2d: (-1x256x14x14xf16) <- (-1x256x14x14xf16, 256x256x1x1xf16)
        conv2d_8 = paddle._C_ops.conv2d(multiply__0, parameter_61, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x14x14xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_8, parameter_62, parameter_63, parameter_64, parameter_65, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf16) <- (-1x256x14x14xf16)
        relu__13 = paddle._C_ops.relu_(batch_norm__66)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x256x14x14xf16, 512x256x1x1xf16)
        conv2d_9 = paddle._C_ops.conv2d(relu__13, parameter_66, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_9, parameter_67, parameter_68, parameter_69, parameter_70, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__14 = paddle._C_ops.relu_(batch_norm__72)

        # pd_op.depthwise_conv2d: (-1x512x14x14xf16) <- (-1x512x14x14xf16, 512x1x5x5xf16)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(relu__14, parameter_71, [1, 1], [2, 2], 'EXPLICIT', 512, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_4 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf16, 0x512xf16) <- (512xf16, 4xi64)
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_72, full_int_array_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x14x14xf16) <- (-1x512x14x14xf16, 1x512x1x1xf16)
        add__3 = paddle._C_ops.add_(depthwise_conv2d_6, reshape_6)

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__15 = paddle._C_ops.relu_(add__3)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [1, 1]

        # pd_op.pool2d: (-1x512x1x1xf16) <- (-1x512x14x14xf16, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(relu__15, full_int_array_5, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x1x1xf16) <- (-1x512x1x1xf16, 128x512x1x1xf16)
        conv2d_10 = paddle._C_ops.conv2d(pool2d_1, parameter_73, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_6 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf16, 0x128xf16) <- (128xf16, 4xi64)
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_74, full_int_array_6), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x1x1xf16) <- (-1x128x1x1xf16, 1x128x1x1xf16)
        add__4 = paddle._C_ops.add_(conv2d_10, reshape_8)

        # pd_op.relu_: (-1x128x1x1xf16) <- (-1x128x1x1xf16)
        relu__16 = paddle._C_ops.relu_(add__4)

        # pd_op.conv2d: (-1x512x1x1xf16) <- (-1x128x1x1xf16, 512x128x1x1xf16)
        conv2d_11 = paddle._C_ops.conv2d(relu__16, parameter_75, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_7 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf16, 0x512xf16) <- (512xf16, 4xi64)
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_76, full_int_array_7), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x1x1xf16) <- (-1x512x1x1xf16, 1x512x1x1xf16)
        add__5 = paddle._C_ops.add_(conv2d_11, reshape_10)

        # pd_op.sigmoid_: (-1x512x1x1xf16) <- (-1x512x1x1xf16)
        sigmoid__1 = paddle._C_ops.sigmoid_(add__5)

        # pd_op.multiply_: (-1x512x14x14xf16) <- (-1x512x14x14xf16, -1x512x1x1xf16)
        multiply__1 = paddle._C_ops.multiply_(relu__15, sigmoid__1)

        # pd_op.conv2d: (-1x256x14x14xf16) <- (-1x512x14x14xf16, 256x512x1x1xf16)
        conv2d_12 = paddle._C_ops.conv2d(multiply__1, parameter_77, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x14x14xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_12, parameter_78, parameter_79, parameter_80, parameter_81, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf16) <- (-1x256x14x14xf16)
        relu__17 = paddle._C_ops.relu_(batch_norm__78)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x256x14x14xf16, 512x256x1x1xf16)
        conv2d_13 = paddle._C_ops.conv2d(relu__17, parameter_82, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_13, parameter_83, parameter_84, parameter_85, parameter_86, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__18 = paddle._C_ops.relu_(batch_norm__84)

        # pd_op.depthwise_conv2d: (-1x512x14x14xf16) <- (-1x512x14x14xf16, 512x1x5x5xf16)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(relu__18, parameter_87, [1, 1], [2, 2], 'EXPLICIT', 512, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_8 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf16, 0x512xf16) <- (512xf16, 4xi64)
        reshape_12, reshape_13 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_88, full_int_array_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x14x14xf16) <- (-1x512x14x14xf16, 1x512x1x1xf16)
        add__6 = paddle._C_ops.add_(depthwise_conv2d_7, reshape_12)

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__19 = paddle._C_ops.relu_(add__6)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_9 = [1, 1]

        # pd_op.pool2d: (-1x512x1x1xf16) <- (-1x512x14x14xf16, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(relu__19, full_int_array_9, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x1x1xf16) <- (-1x512x1x1xf16, 128x512x1x1xf16)
        conv2d_14 = paddle._C_ops.conv2d(pool2d_2, parameter_89, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_10 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf16, 0x128xf16) <- (128xf16, 4xi64)
        reshape_14, reshape_15 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_90, full_int_array_10), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x1x1xf16) <- (-1x128x1x1xf16, 1x128x1x1xf16)
        add__7 = paddle._C_ops.add_(conv2d_14, reshape_14)

        # pd_op.relu_: (-1x128x1x1xf16) <- (-1x128x1x1xf16)
        relu__20 = paddle._C_ops.relu_(add__7)

        # pd_op.conv2d: (-1x512x1x1xf16) <- (-1x128x1x1xf16, 512x128x1x1xf16)
        conv2d_15 = paddle._C_ops.conv2d(relu__20, parameter_91, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_11 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf16, 0x512xf16) <- (512xf16, 4xi64)
        reshape_16, reshape_17 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_92, full_int_array_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x1x1xf16) <- (-1x512x1x1xf16, 1x512x1x1xf16)
        add__8 = paddle._C_ops.add_(conv2d_15, reshape_16)

        # pd_op.sigmoid_: (-1x512x1x1xf16) <- (-1x512x1x1xf16)
        sigmoid__2 = paddle._C_ops.sigmoid_(add__8)

        # pd_op.multiply_: (-1x512x14x14xf16) <- (-1x512x14x14xf16, -1x512x1x1xf16)
        multiply__2 = paddle._C_ops.multiply_(relu__19, sigmoid__2)

        # pd_op.conv2d: (-1x256x14x14xf16) <- (-1x512x14x14xf16, 256x512x1x1xf16)
        conv2d_16 = paddle._C_ops.conv2d(multiply__2, parameter_93, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x14x14xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_16, parameter_94, parameter_95, parameter_96, parameter_97, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf16) <- (-1x256x14x14xf16)
        relu__21 = paddle._C_ops.relu_(batch_norm__90)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x256x14x14xf16, 512x256x1x1xf16)
        conv2d_17 = paddle._C_ops.conv2d(relu__21, parameter_98, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_17, parameter_99, parameter_100, parameter_101, parameter_102, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__22 = paddle._C_ops.relu_(batch_norm__96)

        # pd_op.depthwise_conv2d: (-1x512x14x14xf16) <- (-1x512x14x14xf16, 512x1x5x5xf16)
        depthwise_conv2d_8 = paddle._C_ops.depthwise_conv2d(relu__22, parameter_103, [1, 1], [2, 2], 'EXPLICIT', 512, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_12 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf16, 0x512xf16) <- (512xf16, 4xi64)
        reshape_18, reshape_19 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_104, full_int_array_12), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x14x14xf16) <- (-1x512x14x14xf16, 1x512x1x1xf16)
        add__9 = paddle._C_ops.add_(depthwise_conv2d_8, reshape_18)

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__23 = paddle._C_ops.relu_(add__9)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_13 = [1, 1]

        # pd_op.pool2d: (-1x512x1x1xf16) <- (-1x512x14x14xf16, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(relu__23, full_int_array_13, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x1x1xf16) <- (-1x512x1x1xf16, 128x512x1x1xf16)
        conv2d_18 = paddle._C_ops.conv2d(pool2d_3, parameter_105, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_14 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf16, 0x128xf16) <- (128xf16, 4xi64)
        reshape_20, reshape_21 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_106, full_int_array_14), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x1x1xf16) <- (-1x128x1x1xf16, 1x128x1x1xf16)
        add__10 = paddle._C_ops.add_(conv2d_18, reshape_20)

        # pd_op.relu_: (-1x128x1x1xf16) <- (-1x128x1x1xf16)
        relu__24 = paddle._C_ops.relu_(add__10)

        # pd_op.conv2d: (-1x512x1x1xf16) <- (-1x128x1x1xf16, 512x128x1x1xf16)
        conv2d_19 = paddle._C_ops.conv2d(relu__24, parameter_107, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_15 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf16, 0x512xf16) <- (512xf16, 4xi64)
        reshape_22, reshape_23 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_108, full_int_array_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x1x1xf16) <- (-1x512x1x1xf16, 1x512x1x1xf16)
        add__11 = paddle._C_ops.add_(conv2d_19, reshape_22)

        # pd_op.sigmoid_: (-1x512x1x1xf16) <- (-1x512x1x1xf16)
        sigmoid__3 = paddle._C_ops.sigmoid_(add__11)

        # pd_op.multiply_: (-1x512x14x14xf16) <- (-1x512x14x14xf16, -1x512x1x1xf16)
        multiply__3 = paddle._C_ops.multiply_(relu__23, sigmoid__3)

        # pd_op.conv2d: (-1x256x14x14xf16) <- (-1x512x14x14xf16, 256x512x1x1xf16)
        conv2d_20 = paddle._C_ops.conv2d(multiply__3, parameter_109, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x14x14xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_20, parameter_110, parameter_111, parameter_112, parameter_113, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf16) <- (-1x256x14x14xf16)
        relu__25 = paddle._C_ops.relu_(batch_norm__102)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x256x14x14xf16, 512x256x1x1xf16)
        conv2d_21 = paddle._C_ops.conv2d(relu__25, parameter_114, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_21, parameter_115, parameter_116, parameter_117, parameter_118, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__26 = paddle._C_ops.relu_(batch_norm__108)

        # pd_op.depthwise_conv2d: (-1x512x14x14xf16) <- (-1x512x14x14xf16, 512x1x5x5xf16)
        depthwise_conv2d_9 = paddle._C_ops.depthwise_conv2d(relu__26, parameter_119, [1, 1], [2, 2], 'EXPLICIT', 512, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_16 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf16, 0x512xf16) <- (512xf16, 4xi64)
        reshape_24, reshape_25 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_120, full_int_array_16), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x14x14xf16) <- (-1x512x14x14xf16, 1x512x1x1xf16)
        add__12 = paddle._C_ops.add_(depthwise_conv2d_9, reshape_24)

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__27 = paddle._C_ops.relu_(add__12)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_17 = [1, 1]

        # pd_op.pool2d: (-1x512x1x1xf16) <- (-1x512x14x14xf16, 2xi64)
        pool2d_4 = paddle._C_ops.pool2d(relu__27, full_int_array_17, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x1x1xf16) <- (-1x512x1x1xf16, 128x512x1x1xf16)
        conv2d_22 = paddle._C_ops.conv2d(pool2d_4, parameter_121, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_18 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf16, 0x128xf16) <- (128xf16, 4xi64)
        reshape_26, reshape_27 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_122, full_int_array_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x1x1xf16) <- (-1x128x1x1xf16, 1x128x1x1xf16)
        add__13 = paddle._C_ops.add_(conv2d_22, reshape_26)

        # pd_op.relu_: (-1x128x1x1xf16) <- (-1x128x1x1xf16)
        relu__28 = paddle._C_ops.relu_(add__13)

        # pd_op.conv2d: (-1x512x1x1xf16) <- (-1x128x1x1xf16, 512x128x1x1xf16)
        conv2d_23 = paddle._C_ops.conv2d(relu__28, parameter_123, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_19 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf16, 0x512xf16) <- (512xf16, 4xi64)
        reshape_28, reshape_29 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_124, full_int_array_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x1x1xf16) <- (-1x512x1x1xf16, 1x512x1x1xf16)
        add__14 = paddle._C_ops.add_(conv2d_23, reshape_28)

        # pd_op.sigmoid_: (-1x512x1x1xf16) <- (-1x512x1x1xf16)
        sigmoid__4 = paddle._C_ops.sigmoid_(add__14)

        # pd_op.multiply_: (-1x512x14x14xf16) <- (-1x512x14x14xf16, -1x512x1x1xf16)
        multiply__4 = paddle._C_ops.multiply_(relu__27, sigmoid__4)

        # pd_op.conv2d: (-1x256x14x14xf16) <- (-1x512x14x14xf16, 256x512x1x1xf16)
        conv2d_24 = paddle._C_ops.conv2d(multiply__4, parameter_125, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x14x14xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_24, parameter_126, parameter_127, parameter_128, parameter_129, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf16) <- (-1x256x14x14xf16)
        relu__29 = paddle._C_ops.relu_(batch_norm__114)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x256x14x14xf16, 512x256x1x1xf16)
        conv2d_25 = paddle._C_ops.conv2d(relu__29, parameter_130, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_25, parameter_131, parameter_132, parameter_133, parameter_134, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__30 = paddle._C_ops.relu_(batch_norm__120)

        # pd_op.depthwise_conv2d: (-1x512x14x14xf16) <- (-1x512x14x14xf16, 512x1x5x5xf16)
        depthwise_conv2d_10 = paddle._C_ops.depthwise_conv2d(relu__30, parameter_135, [1, 1], [2, 2], 'EXPLICIT', 512, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_20 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf16, 0x512xf16) <- (512xf16, 4xi64)
        reshape_30, reshape_31 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_136, full_int_array_20), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x14x14xf16) <- (-1x512x14x14xf16, 1x512x1x1xf16)
        add__15 = paddle._C_ops.add_(depthwise_conv2d_10, reshape_30)

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__31 = paddle._C_ops.relu_(add__15)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_21 = [1, 1]

        # pd_op.pool2d: (-1x512x1x1xf16) <- (-1x512x14x14xf16, 2xi64)
        pool2d_5 = paddle._C_ops.pool2d(relu__31, full_int_array_21, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x1x1xf16) <- (-1x512x1x1xf16, 128x512x1x1xf16)
        conv2d_26 = paddle._C_ops.conv2d(pool2d_5, parameter_137, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_22 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf16, 0x128xf16) <- (128xf16, 4xi64)
        reshape_32, reshape_33 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_138, full_int_array_22), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x1x1xf16) <- (-1x128x1x1xf16, 1x128x1x1xf16)
        add__16 = paddle._C_ops.add_(conv2d_26, reshape_32)

        # pd_op.relu_: (-1x128x1x1xf16) <- (-1x128x1x1xf16)
        relu__32 = paddle._C_ops.relu_(add__16)

        # pd_op.conv2d: (-1x512x1x1xf16) <- (-1x128x1x1xf16, 512x128x1x1xf16)
        conv2d_27 = paddle._C_ops.conv2d(relu__32, parameter_139, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_23 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf16, 0x512xf16) <- (512xf16, 4xi64)
        reshape_34, reshape_35 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_140, full_int_array_23), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x1x1xf16) <- (-1x512x1x1xf16, 1x512x1x1xf16)
        add__17 = paddle._C_ops.add_(conv2d_27, reshape_34)

        # pd_op.sigmoid_: (-1x512x1x1xf16) <- (-1x512x1x1xf16)
        sigmoid__5 = paddle._C_ops.sigmoid_(add__17)

        # pd_op.multiply_: (-1x512x14x14xf16) <- (-1x512x14x14xf16, -1x512x1x1xf16)
        multiply__5 = paddle._C_ops.multiply_(relu__31, sigmoid__5)

        # pd_op.conv2d: (-1x256x14x14xf16) <- (-1x512x14x14xf16, 256x512x1x1xf16)
        conv2d_28 = paddle._C_ops.conv2d(multiply__5, parameter_141, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x14x14xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_28, parameter_142, parameter_143, parameter_144, parameter_145, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf16) <- (-1x256x14x14xf16)
        relu__33 = paddle._C_ops.relu_(batch_norm__126)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x256x14x14xf16, 512x256x1x1xf16)
        conv2d_29 = paddle._C_ops.conv2d(relu__33, parameter_146, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_29, parameter_147, parameter_148, parameter_149, parameter_150, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__34 = paddle._C_ops.relu_(batch_norm__132)

        # pd_op.depthwise_conv2d: (-1x512x7x7xf16) <- (-1x512x14x14xf16, 512x1x5x5xf16)
        depthwise_conv2d_11 = paddle._C_ops.depthwise_conv2d(relu__34, parameter_151, [2, 2], [2, 2], 'EXPLICIT', 512, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_24 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf16, 0x512xf16) <- (512xf16, 4xi64)
        reshape_36, reshape_37 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_152, full_int_array_24), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x7x7xf16) <- (-1x512x7x7xf16, 1x512x1x1xf16)
        add__18 = paddle._C_ops.add_(depthwise_conv2d_11, reshape_36)

        # pd_op.relu_: (-1x512x7x7xf16) <- (-1x512x7x7xf16)
        relu__35 = paddle._C_ops.relu_(add__18)

        # pd_op.conv2d: (-1x1024x7x7xf16) <- (-1x512x7x7xf16, 1024x512x1x1xf16)
        conv2d_30 = paddle._C_ops.conv2d(relu__35, parameter_153, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x7x7xf16, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x7x7xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_30, parameter_154, parameter_155, parameter_156, parameter_157, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1024x7x7xf16) <- (-1x1024x7x7xf16)
        relu__36 = paddle._C_ops.relu_(batch_norm__138)

        # pd_op.depthwise_conv2d: (-1x1024x7x7xf16) <- (-1x1024x7x7xf16, 1024x1x5x5xf16)
        depthwise_conv2d_12 = paddle._C_ops.depthwise_conv2d(relu__36, parameter_158, [1, 1], [2, 2], 'EXPLICIT', 1024, [1, 1], 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_25 = [1, 1024, 1, 1]

        # pd_op.reshape: (1x1024x1x1xf16, 0x1024xf16) <- (1024xf16, 4xi64)
        reshape_38, reshape_39 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_159, full_int_array_25), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1024x7x7xf16) <- (-1x1024x7x7xf16, 1x1024x1x1xf16)
        add__19 = paddle._C_ops.add_(depthwise_conv2d_12, reshape_38)

        # pd_op.relu_: (-1x1024x7x7xf16) <- (-1x1024x7x7xf16)
        relu__37 = paddle._C_ops.relu_(add__19)

        # pd_op.conv2d: (-1x1024x7x7xf16) <- (-1x1024x7x7xf16, 1024x1024x1x1xf16)
        conv2d_31 = paddle._C_ops.conv2d(relu__37, parameter_160, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x7x7xf16, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x7x7xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_31, parameter_161, parameter_162, parameter_163, parameter_164, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1024x7x7xf16) <- (-1x1024x7x7xf16)
        relu__38 = paddle._C_ops.relu_(batch_norm__144)

        # pd_op.add_: (-1x1024x7x7xf16) <- (-1x1024x7x7xf16, -1x1024x7x7xf16)
        add__20 = paddle._C_ops.add_(relu__38, relu__36)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_26 = [1, 1]

        # pd_op.pool2d: (-1x1024x1x1xf16) <- (-1x1024x7x7xf16, 2xi64)
        pool2d_6 = paddle._C_ops.pool2d(add__20, full_int_array_26, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x1280x1x1xf16) <- (-1x1024x1x1xf16, 1280x1024x1x1xf16)
        conv2d_32 = paddle._C_ops.conv2d(pool2d_6, parameter_165, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.relu_: (-1x1280x1x1xf16) <- (-1x1280x1x1xf16)
        relu__39 = paddle._C_ops.relu_(conv2d_32)

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('0.2'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.dropout: (-1x1280x1x1xf16, None) <- (-1x1280x1x1xf16, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(paddle._C_ops.dropout(relu__39, None, full_0, True, 'downgrade_in_infer', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.flatten_: (-1x1280xf16, None) <- (-1x1280x1x1xf16)
        flatten__0, flatten__1 = (lambda x, f: f(x))(paddle._C_ops.flatten_(dropout_0, 1, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x1000xf16) <- (-1x1280xf16, 1280x1000xf16)
        matmul_0 = paddle._C_ops.matmul(flatten__0, parameter_166, False, False)

        # pd_op.add_: (-1x1000xf16) <- (-1x1000xf16, 1000xf16)
        add__21 = paddle._C_ops.add_(matmul_0, parameter_167)

        # pd_op.softmax_: (-1x1000xf16) <- (-1x1000xf16)
        softmax__0 = paddle._C_ops.softmax_(add__21, -1)

        # pd_op.cast: (-1x1000xf32) <- (-1x1000xf16)
        cast_1 = paddle._C_ops.cast(softmax__0, paddle.float32)
        return cast_1



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

    def forward(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_56, parameter_57, parameter_58, parameter_59, parameter_60, parameter_61, parameter_65, parameter_62, parameter_64, parameter_63, parameter_66, parameter_70, parameter_67, parameter_69, parameter_68, parameter_71, parameter_72, parameter_73, parameter_74, parameter_75, parameter_76, parameter_77, parameter_81, parameter_78, parameter_80, parameter_79, parameter_82, parameter_86, parameter_83, parameter_85, parameter_84, parameter_87, parameter_88, parameter_89, parameter_90, parameter_91, parameter_92, parameter_93, parameter_97, parameter_94, parameter_96, parameter_95, parameter_98, parameter_102, parameter_99, parameter_101, parameter_100, parameter_103, parameter_104, parameter_105, parameter_106, parameter_107, parameter_108, parameter_109, parameter_113, parameter_110, parameter_112, parameter_111, parameter_114, parameter_118, parameter_115, parameter_117, parameter_116, parameter_119, parameter_120, parameter_121, parameter_122, parameter_123, parameter_124, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_136, parameter_137, parameter_138, parameter_139, parameter_140, parameter_141, parameter_145, parameter_142, parameter_144, parameter_143, parameter_146, parameter_150, parameter_147, parameter_149, parameter_148, parameter_151, parameter_152, parameter_153, parameter_157, parameter_154, parameter_156, parameter_155, parameter_158, parameter_159, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_166, parameter_167, feed_0):
        return self.builtin_module_381_0_0(parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_56, parameter_57, parameter_58, parameter_59, parameter_60, parameter_61, parameter_65, parameter_62, parameter_64, parameter_63, parameter_66, parameter_70, parameter_67, parameter_69, parameter_68, parameter_71, parameter_72, parameter_73, parameter_74, parameter_75, parameter_76, parameter_77, parameter_81, parameter_78, parameter_80, parameter_79, parameter_82, parameter_86, parameter_83, parameter_85, parameter_84, parameter_87, parameter_88, parameter_89, parameter_90, parameter_91, parameter_92, parameter_93, parameter_97, parameter_94, parameter_96, parameter_95, parameter_98, parameter_102, parameter_99, parameter_101, parameter_100, parameter_103, parameter_104, parameter_105, parameter_106, parameter_107, parameter_108, parameter_109, parameter_113, parameter_110, parameter_112, parameter_111, parameter_114, parameter_118, parameter_115, parameter_117, parameter_116, parameter_119, parameter_120, parameter_121, parameter_122, parameter_123, parameter_124, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_136, parameter_137, parameter_138, parameter_139, parameter_140, parameter_141, parameter_145, parameter_142, parameter_144, parameter_143, parameter_146, parameter_150, parameter_147, parameter_149, parameter_148, parameter_151, parameter_152, parameter_153, parameter_157, parameter_154, parameter_156, parameter_155, parameter_158, parameter_159, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_166, parameter_167, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_381_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([32, 3, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_4
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([32, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_9
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([64, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_14
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([64, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_19
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([128, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_24
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([128, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_29
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_34
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([128, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_39
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([256, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_44
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([256, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_49
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([256, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_54
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([256, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_56
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_57
            paddle.uniform([64, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_58
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_59
            paddle.uniform([256, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_60
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_61
            paddle.uniform([256, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_65
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([512, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_70
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([512, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_72
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_73
            paddle.uniform([128, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_74
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_75
            paddle.uniform([512, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_76
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_77
            paddle.uniform([256, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_81
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([512, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_86
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([512, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_88
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_89
            paddle.uniform([128, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_90
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_91
            paddle.uniform([512, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_92
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_93
            paddle.uniform([256, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_97
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([512, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_102
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([512, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_104
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_105
            paddle.uniform([128, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_106
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_107
            paddle.uniform([512, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_108
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_109
            paddle.uniform([256, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_113
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([512, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_118
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([512, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_120
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_121
            paddle.uniform([128, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_122
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_123
            paddle.uniform([512, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_124
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_125
            paddle.uniform([256, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_129
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([512, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_134
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([512, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_136
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_137
            paddle.uniform([128, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_138
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_139
            paddle.uniform([512, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_140
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_141
            paddle.uniform([256, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_145
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([512, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_150
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_149
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([512, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_152
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_153
            paddle.uniform([1024, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_157
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_154
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([1024, 1, 5, 5], dtype='float16', min=0, max=0.5),
            # parameter_159
            paddle.uniform([1024], dtype='float16', min=0, max=0.5),
            # parameter_160
            paddle.uniform([1024, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_164
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([1280, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_166
            paddle.uniform([1280, 1000], dtype='float16', min=0, max=0.5),
            # parameter_167
            paddle.uniform([1000], dtype='float16', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 3, 224, 224], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_0
            paddle.static.InputSpec(shape=[32, 3, 3, 3], dtype='float16'),
            # parameter_4
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[32, 1, 3, 3], dtype='float16'),
            # parameter_9
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[64, 32, 1, 1], dtype='float16'),
            # parameter_14
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[64, 1, 3, 3], dtype='float16'),
            # parameter_19
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[128, 64, 1, 1], dtype='float16'),
            # parameter_24
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float16'),
            # parameter_29
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_34
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float16'),
            # parameter_39
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float16'),
            # parameter_44
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[256, 1, 3, 3], dtype='float16'),
            # parameter_49
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float16'),
            # parameter_54
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[256, 1, 5, 5], dtype='float16'),
            # parameter_56
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_57
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float16'),
            # parameter_58
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_59
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float16'),
            # parameter_60
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_61
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float16'),
            # parameter_65
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[512, 256, 1, 1], dtype='float16'),
            # parameter_70
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[512, 1, 5, 5], dtype='float16'),
            # parameter_72
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_73
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float16'),
            # parameter_74
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_75
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float16'),
            # parameter_76
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_77
            paddle.static.InputSpec(shape=[256, 512, 1, 1], dtype='float16'),
            # parameter_81
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[512, 256, 1, 1], dtype='float16'),
            # parameter_86
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[512, 1, 5, 5], dtype='float16'),
            # parameter_88
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_89
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float16'),
            # parameter_90
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_91
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float16'),
            # parameter_92
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_93
            paddle.static.InputSpec(shape=[256, 512, 1, 1], dtype='float16'),
            # parameter_97
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[512, 256, 1, 1], dtype='float16'),
            # parameter_102
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[512, 1, 5, 5], dtype='float16'),
            # parameter_104
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_105
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float16'),
            # parameter_106
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_107
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float16'),
            # parameter_108
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_109
            paddle.static.InputSpec(shape=[256, 512, 1, 1], dtype='float16'),
            # parameter_113
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[512, 256, 1, 1], dtype='float16'),
            # parameter_118
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[512, 1, 5, 5], dtype='float16'),
            # parameter_120
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_121
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float16'),
            # parameter_122
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_123
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float16'),
            # parameter_124
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_125
            paddle.static.InputSpec(shape=[256, 512, 1, 1], dtype='float16'),
            # parameter_129
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[512, 256, 1, 1], dtype='float16'),
            # parameter_134
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[512, 1, 5, 5], dtype='float16'),
            # parameter_136
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_137
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float16'),
            # parameter_138
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_139
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float16'),
            # parameter_140
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_141
            paddle.static.InputSpec(shape=[256, 512, 1, 1], dtype='float16'),
            # parameter_145
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[512, 256, 1, 1], dtype='float16'),
            # parameter_150
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_149
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[512, 1, 5, 5], dtype='float16'),
            # parameter_152
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_153
            paddle.static.InputSpec(shape=[1024, 512, 1, 1], dtype='float16'),
            # parameter_157
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_154
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[1024, 1, 5, 5], dtype='float16'),
            # parameter_159
            paddle.static.InputSpec(shape=[1024], dtype='float16'),
            # parameter_160
            paddle.static.InputSpec(shape=[1024, 1024, 1, 1], dtype='float16'),
            # parameter_164
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[1280, 1024, 1, 1], dtype='float16'),
            # parameter_166
            paddle.static.InputSpec(shape=[1280, 1000], dtype='float16'),
            # parameter_167
            paddle.static.InputSpec(shape=[1000], dtype='float16'),
            # feed_0
            paddle.static.InputSpec(shape=[None, 3, 224, 224], dtype='float32'),
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