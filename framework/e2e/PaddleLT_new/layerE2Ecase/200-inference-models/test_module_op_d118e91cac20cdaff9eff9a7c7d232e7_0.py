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
    return [459][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_710_0_0(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_72, parameter_71, parameter_73, parameter_74, parameter_75, parameter_76, parameter_78, parameter_77, parameter_79, parameter_80, parameter_81, parameter_82, parameter_84, parameter_83, parameter_85, parameter_86, parameter_87, parameter_88, parameter_90, parameter_89, parameter_91, parameter_92, parameter_93, parameter_94, parameter_96, parameter_95, parameter_97, parameter_101, parameter_98, parameter_100, parameter_99, parameter_102, parameter_106, parameter_103, parameter_105, parameter_104, parameter_107, parameter_111, parameter_108, parameter_110, parameter_109, parameter_112, parameter_116, parameter_113, parameter_115, parameter_114, parameter_117, parameter_121, parameter_118, parameter_120, parameter_119, parameter_122, parameter_124, parameter_123, parameter_125, parameter_126, parameter_127, parameter_128, parameter_130, parameter_129, parameter_131, parameter_132, parameter_133, parameter_134, parameter_136, parameter_135, parameter_137, parameter_138, parameter_139, parameter_140, parameter_142, parameter_141, parameter_143, parameter_144, parameter_145, parameter_146, parameter_148, parameter_147, parameter_149, parameter_150, parameter_151, parameter_152, parameter_154, parameter_153, parameter_155, parameter_156, parameter_157, parameter_158, parameter_160, parameter_159, parameter_161, parameter_162, parameter_163, parameter_164, parameter_166, parameter_165, parameter_167, parameter_168, parameter_169, parameter_170, parameter_172, parameter_171, parameter_173, parameter_177, parameter_174, parameter_176, parameter_175, parameter_178, parameter_182, parameter_179, parameter_181, parameter_180, parameter_183, parameter_187, parameter_184, parameter_186, parameter_185, parameter_188, parameter_192, parameter_189, parameter_191, parameter_190, parameter_193, parameter_197, parameter_194, parameter_196, parameter_195, parameter_198, parameter_200, parameter_199, parameter_201, parameter_202, parameter_203, parameter_204, parameter_206, parameter_205, parameter_207, parameter_208, parameter_209, parameter_210, parameter_212, parameter_211, parameter_213, parameter_214, parameter_215, parameter_216, parameter_218, parameter_217, parameter_219, parameter_220, parameter_221, parameter_222, parameter_224, parameter_223, parameter_225, parameter_226, parameter_227, parameter_228, parameter_230, parameter_229, parameter_231, parameter_232, parameter_233, parameter_234, parameter_236, parameter_235, parameter_237, parameter_241, parameter_238, parameter_240, parameter_239, parameter_242, parameter_243, feed_0):

        # pd_op.cast: (-1x3x256x256xf16) <- (-1x3x256x256xf32)
        cast_0 = paddle._C_ops.cast(feed_0, paddle.float16)

        # pd_op.conv2d: (-1x16x128x128xf16) <- (-1x3x256x256xf16, 16x3x3x3xf16)
        conv2d_0 = paddle._C_ops.conv2d(cast_0, parameter_0, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x16x128x128xf16, 16xf32, 16xf32, xf32, xf32, None) <- (-1x16x128x128xf16, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_0, parameter_1, parameter_2, parameter_3, parameter_4, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x16x128x128xf16) <- (-1x16x128x128xf16)
        silu_0 = paddle._C_ops.silu(batch_norm__0)

        # pd_op.conv2d: (-1x32x128x128xf16) <- (-1x16x128x128xf16, 32x16x1x1xf16)
        conv2d_1 = paddle._C_ops.conv2d(silu_0, parameter_5, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x128x128xf16, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x128x128xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_1, parameter_6, parameter_7, parameter_8, parameter_9, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x32x128x128xf16) <- (-1x32x128x128xf16)
        silu_1 = paddle._C_ops.silu(batch_norm__6)

        # pd_op.depthwise_conv2d: (-1x32x128x128xf16) <- (-1x32x128x128xf16, 32x1x3x3xf16)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(silu_1, parameter_10, [1, 1], [1, 1], 'EXPLICIT', 32, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x32x128x128xf16, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x128x128xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_0, parameter_11, parameter_12, parameter_13, parameter_14, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x32x128x128xf16) <- (-1x32x128x128xf16)
        silu_2 = paddle._C_ops.silu(batch_norm__12)

        # pd_op.conv2d: (-1x32x128x128xf16) <- (-1x32x128x128xf16, 32x32x1x1xf16)
        conv2d_2 = paddle._C_ops.conv2d(silu_2, parameter_15, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x128x128xf16, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x128x128xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_2, parameter_16, parameter_17, parameter_18, parameter_19, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x64x128x128xf16) <- (-1x32x128x128xf16, 64x32x1x1xf16)
        conv2d_3 = paddle._C_ops.conv2d(batch_norm__18, parameter_20, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x128x128xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x128x128xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_3, parameter_21, parameter_22, parameter_23, parameter_24, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x64x128x128xf16) <- (-1x64x128x128xf16)
        silu_3 = paddle._C_ops.silu(batch_norm__24)

        # pd_op.depthwise_conv2d: (-1x64x64x64xf16) <- (-1x64x128x128xf16, 64x1x3x3xf16)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(silu_3, parameter_25, [2, 2], [1, 1], 'EXPLICIT', 64, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x64x64x64xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x64x64xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_1, parameter_26, parameter_27, parameter_28, parameter_29, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x64x64x64xf16) <- (-1x64x64x64xf16)
        silu_4 = paddle._C_ops.silu(batch_norm__30)

        # pd_op.conv2d: (-1x64x64x64xf16) <- (-1x64x64x64xf16, 64x64x1x1xf16)
        conv2d_4 = paddle._C_ops.conv2d(silu_4, parameter_30, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x64x64xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x64x64xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_4, parameter_31, parameter_32, parameter_33, parameter_34, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x128x64x64xf16) <- (-1x64x64x64xf16, 128x64x1x1xf16)
        conv2d_5 = paddle._C_ops.conv2d(batch_norm__36, parameter_35, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x64x64xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x64x64xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_5, parameter_36, parameter_37, parameter_38, parameter_39, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x128x64x64xf16) <- (-1x128x64x64xf16)
        silu_5 = paddle._C_ops.silu(batch_norm__42)

        # pd_op.depthwise_conv2d: (-1x128x64x64xf16) <- (-1x128x64x64xf16, 128x1x3x3xf16)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(silu_5, parameter_40, [1, 1], [1, 1], 'EXPLICIT', 128, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x128x64x64xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x64x64xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_2, parameter_41, parameter_42, parameter_43, parameter_44, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x128x64x64xf16) <- (-1x128x64x64xf16)
        silu_6 = paddle._C_ops.silu(batch_norm__48)

        # pd_op.conv2d: (-1x64x64x64xf16) <- (-1x128x64x64xf16, 64x128x1x1xf16)
        conv2d_6 = paddle._C_ops.conv2d(silu_6, parameter_45, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x64x64xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x64x64xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_6, parameter_46, parameter_47, parameter_48, parameter_49, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x64x64x64xf16) <- (-1x64x64x64xf16, -1x64x64x64xf16)
        add__0 = paddle._C_ops.add_(batch_norm__36, batch_norm__54)

        # pd_op.conv2d: (-1x128x64x64xf16) <- (-1x64x64x64xf16, 128x64x1x1xf16)
        conv2d_7 = paddle._C_ops.conv2d(add__0, parameter_50, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x64x64xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x64x64xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_7, parameter_51, parameter_52, parameter_53, parameter_54, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x128x64x64xf16) <- (-1x128x64x64xf16)
        silu_7 = paddle._C_ops.silu(batch_norm__60)

        # pd_op.depthwise_conv2d: (-1x128x32x32xf16) <- (-1x128x64x64xf16, 128x1x3x3xf16)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(silu_7, parameter_55, [2, 2], [1, 1], 'EXPLICIT', 128, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x128x32x32xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x32x32xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_3, parameter_56, parameter_57, parameter_58, parameter_59, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x128x32x32xf16) <- (-1x128x32x32xf16)
        silu_8 = paddle._C_ops.silu(batch_norm__66)

        # pd_op.conv2d: (-1x128x32x32xf16) <- (-1x128x32x32xf16, 128x128x1x1xf16)
        conv2d_8 = paddle._C_ops.conv2d(silu_8, parameter_60, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x32x32xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x32x32xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_8, parameter_61, parameter_62, parameter_63, parameter_64, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.depthwise_conv2d: (-1x128x32x32xf16) <- (-1x128x32x32xf16, 128x1x3x3xf16)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(batch_norm__72, parameter_65, [1, 1], [1, 1], 'EXPLICIT', 128, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x128x32x32xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x32x32xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_4, parameter_66, parameter_67, parameter_68, parameter_69, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x128x32x32xf16) <- (-1x128x32x32xf16)
        silu_9 = paddle._C_ops.silu(batch_norm__78)

        # pd_op.conv2d: (-1x64x32x32xf16) <- (-1x128x32x32xf16, 64x128x1x1xf16)
        conv2d_9 = paddle._C_ops.conv2d(silu_9, parameter_70, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.shape: (4xi32) <- (-1x64x32x32xf16)
        shape_0 = paddle._C_ops.shape(paddle.cast(conv2d_9, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], full_int_array_0, full_int_array_1, [1], [0])

        # pd_op.unfold: (-1x256x256xf16) <- (-1x64x32x32xf16)
        unfold_0 = paddle._C_ops.unfold(conv2d_9, [2, 2], [2, 2], [0, 0, 0, 0], [1, 1])

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_0 = [slice_0, full_0, full_1, full_2]

        # pd_op.reshape_: (-1x64x4x256xf16, 0x-1x256x256xf16) <- (-1x256x256xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_0, [x.reshape([]) for x in combine_0]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.group_norm: (-1x64x4x256xf16, -1x1xf32, -1x1xf32) <- (-1x64x4x256xf16, 64xf16, 64xf16)
        group_norm_0, group_norm_1, group_norm_2 = (lambda x, f: f(x))(paddle._C_ops.group_norm(reshape__0, parameter_71, parameter_72, float('1e-05'), 1, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.conv2d: (-1x129x4x256xf16) <- (-1x64x4x256xf16, 129x64x1x1xf16)
        conv2d_10 = paddle._C_ops.conv2d(group_norm_0, parameter_73, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_2 = [1, 129, 1, 1]

        # pd_op.reshape: (1x129x1x1xf16, 0x129xf16) <- (129xf16, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_74, full_int_array_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x129x4x256xf16) <- (-1x129x4x256xf16, 1x129x1x1xf16)
        add__1 = paddle._C_ops.add_(conv2d_10, reshape_0)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_3 = [1, 64, 64]

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1x4x256xf16, -1x64x4x256xf16, -1x64x4x256xf16]) <- (-1x129x4x256xf16, 3xi64, 1xi32)
        split_0 = paddle._C_ops.split(add__1, full_int_array_3, full_3)

        # builtin.slice: (-1x1x4x256xf16) <- ([-1x1x4x256xf16, -1x64x4x256xf16, -1x64x4x256xf16])
        slice_1 = split_0[0]

        # pd_op.softmax_: (-1x1x4x256xf16) <- (-1x1x4x256xf16)
        softmax__0 = paddle._C_ops.softmax_(slice_1, -1)

        # builtin.slice: (-1x64x4x256xf16) <- ([-1x1x4x256xf16, -1x64x4x256xf16, -1x64x4x256xf16])
        slice_2 = split_0[1]

        # pd_op.multiply_: (-1x64x4x256xf16) <- (-1x64x4x256xf16, -1x1x4x256xf16)
        multiply__0 = paddle._C_ops.multiply_(slice_2, softmax__0)

        # pd_op.cast: (-1x64x4x256xf32) <- (-1x64x4x256xf16)
        cast_1 = paddle._C_ops.cast(multiply__0, paddle.float32)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [-1]

        # pd_op.sum: (-1x64x4x1xf32) <- (-1x64x4x256xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(cast_1, full_int_array_4, None, True)

        # builtin.slice: (-1x64x4x256xf16) <- ([-1x1x4x256xf16, -1x64x4x256xf16, -1x64x4x256xf16])
        slice_3 = split_0[2]

        # pd_op.relu_: (-1x64x4x256xf16) <- (-1x64x4x256xf16)
        relu__0 = paddle._C_ops.relu_(slice_3)

        # pd_op.cast: (-1x64x4x1xf16) <- (-1x64x4x1xf32)
        cast_2 = paddle._C_ops.cast(sum_0, paddle.float16)

        # pd_op.multiply_: (-1x64x4x256xf16) <- (-1x64x4x256xf16, -1x64x4x1xf16)
        multiply__1 = paddle._C_ops.multiply_(relu__0, cast_2)

        # pd_op.conv2d: (-1x64x4x256xf16) <- (-1x64x4x256xf16, 64x64x1x1xf16)
        conv2d_11 = paddle._C_ops.conv2d(multiply__1, parameter_75, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_5 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf16, 0x64xf16) <- (64xf16, 4xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_76, full_int_array_5), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x4x256xf16) <- (-1x64x4x256xf16, 1x64x1x1xf16)
        add__2 = paddle._C_ops.add_(conv2d_11, reshape_2)

        # pd_op.add_: (-1x64x4x256xf16) <- (-1x64x4x256xf16, -1x64x4x256xf16)
        add__3 = paddle._C_ops.add_(reshape__0, add__2)

        # pd_op.group_norm: (-1x64x4x256xf16, -1x1xf32, -1x1xf32) <- (-1x64x4x256xf16, 64xf16, 64xf16)
        group_norm_3, group_norm_4, group_norm_5 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add__3, parameter_77, parameter_78, float('1e-05'), 1, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.conv2d: (-1x128x4x256xf16) <- (-1x64x4x256xf16, 128x64x1x1xf16)
        conv2d_12 = paddle._C_ops.conv2d(group_norm_3, parameter_79, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_6 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf16, 0x128xf16) <- (128xf16, 4xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_80, full_int_array_6), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x4x256xf16) <- (-1x128x4x256xf16, 1x128x1x1xf16)
        add__4 = paddle._C_ops.add_(conv2d_12, reshape_4)

        # pd_op.silu: (-1x128x4x256xf16) <- (-1x128x4x256xf16)
        silu_10 = paddle._C_ops.silu(add__4)

        # pd_op.conv2d: (-1x64x4x256xf16) <- (-1x128x4x256xf16, 64x128x1x1xf16)
        conv2d_13 = paddle._C_ops.conv2d(silu_10, parameter_81, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_7 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf16, 0x64xf16) <- (64xf16, 4xi64)
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_82, full_int_array_7), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x4x256xf16) <- (-1x64x4x256xf16, 1x64x1x1xf16)
        add__5 = paddle._C_ops.add_(conv2d_13, reshape_6)

        # pd_op.add_: (-1x64x4x256xf16) <- (-1x64x4x256xf16, -1x64x4x256xf16)
        add__6 = paddle._C_ops.add_(add__3, add__5)

        # pd_op.group_norm: (-1x64x4x256xf16, -1x1xf32, -1x1xf32) <- (-1x64x4x256xf16, 64xf16, 64xf16)
        group_norm_6, group_norm_7, group_norm_8 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add__6, parameter_83, parameter_84, float('1e-05'), 1, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.conv2d: (-1x129x4x256xf16) <- (-1x64x4x256xf16, 129x64x1x1xf16)
        conv2d_14 = paddle._C_ops.conv2d(group_norm_6, parameter_85, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_8 = [1, 129, 1, 1]

        # pd_op.reshape: (1x129x1x1xf16, 0x129xf16) <- (129xf16, 4xi64)
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_86, full_int_array_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x129x4x256xf16) <- (-1x129x4x256xf16, 1x129x1x1xf16)
        add__7 = paddle._C_ops.add_(conv2d_14, reshape_8)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_9 = [1, 64, 64]

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1x4x256xf16, -1x64x4x256xf16, -1x64x4x256xf16]) <- (-1x129x4x256xf16, 3xi64, 1xi32)
        split_1 = paddle._C_ops.split(add__7, full_int_array_9, full_4)

        # builtin.slice: (-1x1x4x256xf16) <- ([-1x1x4x256xf16, -1x64x4x256xf16, -1x64x4x256xf16])
        slice_4 = split_1[0]

        # pd_op.softmax_: (-1x1x4x256xf16) <- (-1x1x4x256xf16)
        softmax__1 = paddle._C_ops.softmax_(slice_4, -1)

        # builtin.slice: (-1x64x4x256xf16) <- ([-1x1x4x256xf16, -1x64x4x256xf16, -1x64x4x256xf16])
        slice_5 = split_1[1]

        # pd_op.multiply_: (-1x64x4x256xf16) <- (-1x64x4x256xf16, -1x1x4x256xf16)
        multiply__2 = paddle._C_ops.multiply_(slice_5, softmax__1)

        # pd_op.cast: (-1x64x4x256xf32) <- (-1x64x4x256xf16)
        cast_3 = paddle._C_ops.cast(multiply__2, paddle.float32)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_10 = [-1]

        # pd_op.sum: (-1x64x4x1xf32) <- (-1x64x4x256xf32, 1xi64)
        sum_1 = paddle._C_ops.sum(cast_3, full_int_array_10, None, True)

        # builtin.slice: (-1x64x4x256xf16) <- ([-1x1x4x256xf16, -1x64x4x256xf16, -1x64x4x256xf16])
        slice_6 = split_1[2]

        # pd_op.relu_: (-1x64x4x256xf16) <- (-1x64x4x256xf16)
        relu__1 = paddle._C_ops.relu_(slice_6)

        # pd_op.cast: (-1x64x4x1xf16) <- (-1x64x4x1xf32)
        cast_4 = paddle._C_ops.cast(sum_1, paddle.float16)

        # pd_op.multiply_: (-1x64x4x256xf16) <- (-1x64x4x256xf16, -1x64x4x1xf16)
        multiply__3 = paddle._C_ops.multiply_(relu__1, cast_4)

        # pd_op.conv2d: (-1x64x4x256xf16) <- (-1x64x4x256xf16, 64x64x1x1xf16)
        conv2d_15 = paddle._C_ops.conv2d(multiply__3, parameter_87, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_11 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf16, 0x64xf16) <- (64xf16, 4xi64)
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_88, full_int_array_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x4x256xf16) <- (-1x64x4x256xf16, 1x64x1x1xf16)
        add__8 = paddle._C_ops.add_(conv2d_15, reshape_10)

        # pd_op.add_: (-1x64x4x256xf16) <- (-1x64x4x256xf16, -1x64x4x256xf16)
        add__9 = paddle._C_ops.add_(add__6, add__8)

        # pd_op.group_norm: (-1x64x4x256xf16, -1x1xf32, -1x1xf32) <- (-1x64x4x256xf16, 64xf16, 64xf16)
        group_norm_9, group_norm_10, group_norm_11 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add__9, parameter_89, parameter_90, float('1e-05'), 1, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.conv2d: (-1x128x4x256xf16) <- (-1x64x4x256xf16, 128x64x1x1xf16)
        conv2d_16 = paddle._C_ops.conv2d(group_norm_9, parameter_91, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_12 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf16, 0x128xf16) <- (128xf16, 4xi64)
        reshape_12, reshape_13 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_92, full_int_array_12), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x4x256xf16) <- (-1x128x4x256xf16, 1x128x1x1xf16)
        add__10 = paddle._C_ops.add_(conv2d_16, reshape_12)

        # pd_op.silu: (-1x128x4x256xf16) <- (-1x128x4x256xf16)
        silu_11 = paddle._C_ops.silu(add__10)

        # pd_op.conv2d: (-1x64x4x256xf16) <- (-1x128x4x256xf16, 64x128x1x1xf16)
        conv2d_17 = paddle._C_ops.conv2d(silu_11, parameter_93, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_13 = [1, 64, 1, 1]

        # pd_op.reshape: (1x64x1x1xf16, 0x64xf16) <- (64xf16, 4xi64)
        reshape_14, reshape_15 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_94, full_int_array_13), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x4x256xf16) <- (-1x64x4x256xf16, 1x64x1x1xf16)
        add__11 = paddle._C_ops.add_(conv2d_17, reshape_14)

        # pd_op.add_: (-1x64x4x256xf16) <- (-1x64x4x256xf16, -1x64x4x256xf16)
        add__12 = paddle._C_ops.add_(add__9, add__11)

        # pd_op.group_norm: (-1x64x4x256xf16, -1x1xf32, -1x1xf32) <- (-1x64x4x256xf16, 64xf16, 64xf16)
        group_norm_12, group_norm_13, group_norm_14 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add__12, parameter_95, parameter_96, float('1e-05'), 1, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (4xi32) <- (-1x64x4x256xf16)
        shape_1 = paddle._C_ops.shape(paddle.cast(group_norm_12, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_14 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_15 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(shape_1, [0], full_int_array_14, full_int_array_15, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_1 = [slice_7, full_5, full_6]

        # pd_op.reshape_: (-1x256x256xf16, 0x-1x64x4x256xf16) <- (-1x64x4x256xf16, [1xi32, 1xi32, 1xi32])
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape_(group_norm_12, [x.reshape([]) for x in combine_1]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.cast: (-1x256x256xf32) <- (-1x256x256xf16)
        cast_5 = paddle._C_ops.cast(reshape__2, paddle.float32)

        # pd_op.fold: (-1x64x32x32xf32) <- (-1x256x256xf32)
        fold_0 = paddle._C_ops.fold(cast_5, [32, 32], [2, 2], [2, 2], [0, 0, 0, 0], [1, 1])

        # pd_op.cast: (-1x64x32x32xf16) <- (-1x64x32x32xf32)
        cast_6 = paddle._C_ops.cast(fold_0, paddle.float16)

        # pd_op.conv2d: (-1x128x32x32xf16) <- (-1x64x32x32xf16, 128x64x1x1xf16)
        conv2d_18 = paddle._C_ops.conv2d(cast_6, parameter_97, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x32x32xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x32x32xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_18, parameter_98, parameter_99, parameter_100, parameter_101, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x256x32x32xf16) <- (-1x128x32x32xf16, 256x128x1x1xf16)
        conv2d_19 = paddle._C_ops.conv2d(batch_norm__84, parameter_102, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x32x32xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x32x32xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_19, parameter_103, parameter_104, parameter_105, parameter_106, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x256x32x32xf16) <- (-1x256x32x32xf16)
        silu_12 = paddle._C_ops.silu(batch_norm__90)

        # pd_op.depthwise_conv2d: (-1x256x16x16xf16) <- (-1x256x32x32xf16, 256x1x3x3xf16)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(silu_12, parameter_107, [2, 2], [1, 1], 'EXPLICIT', 256, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x256x16x16xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x16x16xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_5, parameter_108, parameter_109, parameter_110, parameter_111, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x256x16x16xf16) <- (-1x256x16x16xf16)
        silu_13 = paddle._C_ops.silu(batch_norm__96)

        # pd_op.conv2d: (-1x192x16x16xf16) <- (-1x256x16x16xf16, 192x256x1x1xf16)
        conv2d_20 = paddle._C_ops.conv2d(silu_13, parameter_112, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x192x16x16xf16, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x16x16xf16, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_20, parameter_113, parameter_114, parameter_115, parameter_116, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.depthwise_conv2d: (-1x192x16x16xf16) <- (-1x192x16x16xf16, 192x1x3x3xf16)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(batch_norm__102, parameter_117, [1, 1], [1, 1], 'EXPLICIT', 192, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x192x16x16xf16, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x16x16xf16, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_6, parameter_118, parameter_119, parameter_120, parameter_121, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x192x16x16xf16) <- (-1x192x16x16xf16)
        silu_14 = paddle._C_ops.silu(batch_norm__108)

        # pd_op.conv2d: (-1x96x16x16xf16) <- (-1x192x16x16xf16, 96x192x1x1xf16)
        conv2d_21 = paddle._C_ops.conv2d(silu_14, parameter_122, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.shape: (4xi32) <- (-1x96x16x16xf16)
        shape_2 = paddle._C_ops.shape(paddle.cast(conv2d_21, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_16 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_17 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(shape_2, [0], full_int_array_16, full_int_array_17, [1], [0])

        # pd_op.unfold: (-1x384x64xf16) <- (-1x96x16x16xf16)
        unfold_1 = paddle._C_ops.unfold(conv2d_21, [2, 2], [2, 2], [0, 0, 0, 0], [1, 1])

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full([1], float('96'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_8 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_9 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_2 = [slice_8, full_7, full_8, full_9]

        # pd_op.reshape_: (-1x96x4x64xf16, 0x-1x384x64xf16) <- (-1x384x64xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_1, [x.reshape([]) for x in combine_2]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.group_norm: (-1x96x4x64xf16, -1x1xf32, -1x1xf32) <- (-1x96x4x64xf16, 96xf16, 96xf16)
        group_norm_15, group_norm_16, group_norm_17 = (lambda x, f: f(x))(paddle._C_ops.group_norm(reshape__4, parameter_123, parameter_124, float('1e-05'), 1, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.conv2d: (-1x193x4x64xf16) <- (-1x96x4x64xf16, 193x96x1x1xf16)
        conv2d_22 = paddle._C_ops.conv2d(group_norm_15, parameter_125, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_18 = [1, 193, 1, 1]

        # pd_op.reshape: (1x193x1x1xf16, 0x193xf16) <- (193xf16, 4xi64)
        reshape_16, reshape_17 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_126, full_int_array_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x193x4x64xf16) <- (-1x193x4x64xf16, 1x193x1x1xf16)
        add__13 = paddle._C_ops.add_(conv2d_22, reshape_16)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_19 = [1, 96, 96]

        # pd_op.full: (1xi32) <- ()
        full_10 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1x4x64xf16, -1x96x4x64xf16, -1x96x4x64xf16]) <- (-1x193x4x64xf16, 3xi64, 1xi32)
        split_2 = paddle._C_ops.split(add__13, full_int_array_19, full_10)

        # builtin.slice: (-1x1x4x64xf16) <- ([-1x1x4x64xf16, -1x96x4x64xf16, -1x96x4x64xf16])
        slice_9 = split_2[0]

        # pd_op.softmax_: (-1x1x4x64xf16) <- (-1x1x4x64xf16)
        softmax__2 = paddle._C_ops.softmax_(slice_9, -1)

        # builtin.slice: (-1x96x4x64xf16) <- ([-1x1x4x64xf16, -1x96x4x64xf16, -1x96x4x64xf16])
        slice_10 = split_2[1]

        # pd_op.multiply_: (-1x96x4x64xf16) <- (-1x96x4x64xf16, -1x1x4x64xf16)
        multiply__4 = paddle._C_ops.multiply_(slice_10, softmax__2)

        # pd_op.cast: (-1x96x4x64xf32) <- (-1x96x4x64xf16)
        cast_7 = paddle._C_ops.cast(multiply__4, paddle.float32)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_20 = [-1]

        # pd_op.sum: (-1x96x4x1xf32) <- (-1x96x4x64xf32, 1xi64)
        sum_2 = paddle._C_ops.sum(cast_7, full_int_array_20, None, True)

        # builtin.slice: (-1x96x4x64xf16) <- ([-1x1x4x64xf16, -1x96x4x64xf16, -1x96x4x64xf16])
        slice_11 = split_2[2]

        # pd_op.relu_: (-1x96x4x64xf16) <- (-1x96x4x64xf16)
        relu__2 = paddle._C_ops.relu_(slice_11)

        # pd_op.cast: (-1x96x4x1xf16) <- (-1x96x4x1xf32)
        cast_8 = paddle._C_ops.cast(sum_2, paddle.float16)

        # pd_op.multiply_: (-1x96x4x64xf16) <- (-1x96x4x64xf16, -1x96x4x1xf16)
        multiply__5 = paddle._C_ops.multiply_(relu__2, cast_8)

        # pd_op.conv2d: (-1x96x4x64xf16) <- (-1x96x4x64xf16, 96x96x1x1xf16)
        conv2d_23 = paddle._C_ops.conv2d(multiply__5, parameter_127, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_21 = [1, 96, 1, 1]

        # pd_op.reshape: (1x96x1x1xf16, 0x96xf16) <- (96xf16, 4xi64)
        reshape_18, reshape_19 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_128, full_int_array_21), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x96x4x64xf16) <- (-1x96x4x64xf16, 1x96x1x1xf16)
        add__14 = paddle._C_ops.add_(conv2d_23, reshape_18)

        # pd_op.add_: (-1x96x4x64xf16) <- (-1x96x4x64xf16, -1x96x4x64xf16)
        add__15 = paddle._C_ops.add_(reshape__4, add__14)

        # pd_op.group_norm: (-1x96x4x64xf16, -1x1xf32, -1x1xf32) <- (-1x96x4x64xf16, 96xf16, 96xf16)
        group_norm_18, group_norm_19, group_norm_20 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add__15, parameter_129, parameter_130, float('1e-05'), 1, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.conv2d: (-1x192x4x64xf16) <- (-1x96x4x64xf16, 192x96x1x1xf16)
        conv2d_24 = paddle._C_ops.conv2d(group_norm_18, parameter_131, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_22 = [1, 192, 1, 1]

        # pd_op.reshape: (1x192x1x1xf16, 0x192xf16) <- (192xf16, 4xi64)
        reshape_20, reshape_21 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_132, full_int_array_22), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x192x4x64xf16) <- (-1x192x4x64xf16, 1x192x1x1xf16)
        add__16 = paddle._C_ops.add_(conv2d_24, reshape_20)

        # pd_op.silu: (-1x192x4x64xf16) <- (-1x192x4x64xf16)
        silu_15 = paddle._C_ops.silu(add__16)

        # pd_op.conv2d: (-1x96x4x64xf16) <- (-1x192x4x64xf16, 96x192x1x1xf16)
        conv2d_25 = paddle._C_ops.conv2d(silu_15, parameter_133, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_23 = [1, 96, 1, 1]

        # pd_op.reshape: (1x96x1x1xf16, 0x96xf16) <- (96xf16, 4xi64)
        reshape_22, reshape_23 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_134, full_int_array_23), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x96x4x64xf16) <- (-1x96x4x64xf16, 1x96x1x1xf16)
        add__17 = paddle._C_ops.add_(conv2d_25, reshape_22)

        # pd_op.add_: (-1x96x4x64xf16) <- (-1x96x4x64xf16, -1x96x4x64xf16)
        add__18 = paddle._C_ops.add_(add__15, add__17)

        # pd_op.group_norm: (-1x96x4x64xf16, -1x1xf32, -1x1xf32) <- (-1x96x4x64xf16, 96xf16, 96xf16)
        group_norm_21, group_norm_22, group_norm_23 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add__18, parameter_135, parameter_136, float('1e-05'), 1, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.conv2d: (-1x193x4x64xf16) <- (-1x96x4x64xf16, 193x96x1x1xf16)
        conv2d_26 = paddle._C_ops.conv2d(group_norm_21, parameter_137, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_24 = [1, 193, 1, 1]

        # pd_op.reshape: (1x193x1x1xf16, 0x193xf16) <- (193xf16, 4xi64)
        reshape_24, reshape_25 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_138, full_int_array_24), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x193x4x64xf16) <- (-1x193x4x64xf16, 1x193x1x1xf16)
        add__19 = paddle._C_ops.add_(conv2d_26, reshape_24)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_25 = [1, 96, 96]

        # pd_op.full: (1xi32) <- ()
        full_11 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1x4x64xf16, -1x96x4x64xf16, -1x96x4x64xf16]) <- (-1x193x4x64xf16, 3xi64, 1xi32)
        split_3 = paddle._C_ops.split(add__19, full_int_array_25, full_11)

        # builtin.slice: (-1x1x4x64xf16) <- ([-1x1x4x64xf16, -1x96x4x64xf16, -1x96x4x64xf16])
        slice_12 = split_3[0]

        # pd_op.softmax_: (-1x1x4x64xf16) <- (-1x1x4x64xf16)
        softmax__3 = paddle._C_ops.softmax_(slice_12, -1)

        # builtin.slice: (-1x96x4x64xf16) <- ([-1x1x4x64xf16, -1x96x4x64xf16, -1x96x4x64xf16])
        slice_13 = split_3[1]

        # pd_op.multiply_: (-1x96x4x64xf16) <- (-1x96x4x64xf16, -1x1x4x64xf16)
        multiply__6 = paddle._C_ops.multiply_(slice_13, softmax__3)

        # pd_op.cast: (-1x96x4x64xf32) <- (-1x96x4x64xf16)
        cast_9 = paddle._C_ops.cast(multiply__6, paddle.float32)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_26 = [-1]

        # pd_op.sum: (-1x96x4x1xf32) <- (-1x96x4x64xf32, 1xi64)
        sum_3 = paddle._C_ops.sum(cast_9, full_int_array_26, None, True)

        # builtin.slice: (-1x96x4x64xf16) <- ([-1x1x4x64xf16, -1x96x4x64xf16, -1x96x4x64xf16])
        slice_14 = split_3[2]

        # pd_op.relu_: (-1x96x4x64xf16) <- (-1x96x4x64xf16)
        relu__3 = paddle._C_ops.relu_(slice_14)

        # pd_op.cast: (-1x96x4x1xf16) <- (-1x96x4x1xf32)
        cast_10 = paddle._C_ops.cast(sum_3, paddle.float16)

        # pd_op.multiply_: (-1x96x4x64xf16) <- (-1x96x4x64xf16, -1x96x4x1xf16)
        multiply__7 = paddle._C_ops.multiply_(relu__3, cast_10)

        # pd_op.conv2d: (-1x96x4x64xf16) <- (-1x96x4x64xf16, 96x96x1x1xf16)
        conv2d_27 = paddle._C_ops.conv2d(multiply__7, parameter_139, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_27 = [1, 96, 1, 1]

        # pd_op.reshape: (1x96x1x1xf16, 0x96xf16) <- (96xf16, 4xi64)
        reshape_26, reshape_27 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_140, full_int_array_27), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x96x4x64xf16) <- (-1x96x4x64xf16, 1x96x1x1xf16)
        add__20 = paddle._C_ops.add_(conv2d_27, reshape_26)

        # pd_op.add_: (-1x96x4x64xf16) <- (-1x96x4x64xf16, -1x96x4x64xf16)
        add__21 = paddle._C_ops.add_(add__18, add__20)

        # pd_op.group_norm: (-1x96x4x64xf16, -1x1xf32, -1x1xf32) <- (-1x96x4x64xf16, 96xf16, 96xf16)
        group_norm_24, group_norm_25, group_norm_26 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add__21, parameter_141, parameter_142, float('1e-05'), 1, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.conv2d: (-1x192x4x64xf16) <- (-1x96x4x64xf16, 192x96x1x1xf16)
        conv2d_28 = paddle._C_ops.conv2d(group_norm_24, parameter_143, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_28 = [1, 192, 1, 1]

        # pd_op.reshape: (1x192x1x1xf16, 0x192xf16) <- (192xf16, 4xi64)
        reshape_28, reshape_29 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_144, full_int_array_28), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x192x4x64xf16) <- (-1x192x4x64xf16, 1x192x1x1xf16)
        add__22 = paddle._C_ops.add_(conv2d_28, reshape_28)

        # pd_op.silu: (-1x192x4x64xf16) <- (-1x192x4x64xf16)
        silu_16 = paddle._C_ops.silu(add__22)

        # pd_op.conv2d: (-1x96x4x64xf16) <- (-1x192x4x64xf16, 96x192x1x1xf16)
        conv2d_29 = paddle._C_ops.conv2d(silu_16, parameter_145, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_29 = [1, 96, 1, 1]

        # pd_op.reshape: (1x96x1x1xf16, 0x96xf16) <- (96xf16, 4xi64)
        reshape_30, reshape_31 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_146, full_int_array_29), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x96x4x64xf16) <- (-1x96x4x64xf16, 1x96x1x1xf16)
        add__23 = paddle._C_ops.add_(conv2d_29, reshape_30)

        # pd_op.add_: (-1x96x4x64xf16) <- (-1x96x4x64xf16, -1x96x4x64xf16)
        add__24 = paddle._C_ops.add_(add__21, add__23)

        # pd_op.group_norm: (-1x96x4x64xf16, -1x1xf32, -1x1xf32) <- (-1x96x4x64xf16, 96xf16, 96xf16)
        group_norm_27, group_norm_28, group_norm_29 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add__24, parameter_147, parameter_148, float('1e-05'), 1, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.conv2d: (-1x193x4x64xf16) <- (-1x96x4x64xf16, 193x96x1x1xf16)
        conv2d_30 = paddle._C_ops.conv2d(group_norm_27, parameter_149, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_30 = [1, 193, 1, 1]

        # pd_op.reshape: (1x193x1x1xf16, 0x193xf16) <- (193xf16, 4xi64)
        reshape_32, reshape_33 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_150, full_int_array_30), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x193x4x64xf16) <- (-1x193x4x64xf16, 1x193x1x1xf16)
        add__25 = paddle._C_ops.add_(conv2d_30, reshape_32)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_31 = [1, 96, 96]

        # pd_op.full: (1xi32) <- ()
        full_12 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1x4x64xf16, -1x96x4x64xf16, -1x96x4x64xf16]) <- (-1x193x4x64xf16, 3xi64, 1xi32)
        split_4 = paddle._C_ops.split(add__25, full_int_array_31, full_12)

        # builtin.slice: (-1x1x4x64xf16) <- ([-1x1x4x64xf16, -1x96x4x64xf16, -1x96x4x64xf16])
        slice_15 = split_4[0]

        # pd_op.softmax_: (-1x1x4x64xf16) <- (-1x1x4x64xf16)
        softmax__4 = paddle._C_ops.softmax_(slice_15, -1)

        # builtin.slice: (-1x96x4x64xf16) <- ([-1x1x4x64xf16, -1x96x4x64xf16, -1x96x4x64xf16])
        slice_16 = split_4[1]

        # pd_op.multiply_: (-1x96x4x64xf16) <- (-1x96x4x64xf16, -1x1x4x64xf16)
        multiply__8 = paddle._C_ops.multiply_(slice_16, softmax__4)

        # pd_op.cast: (-1x96x4x64xf32) <- (-1x96x4x64xf16)
        cast_11 = paddle._C_ops.cast(multiply__8, paddle.float32)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_32 = [-1]

        # pd_op.sum: (-1x96x4x1xf32) <- (-1x96x4x64xf32, 1xi64)
        sum_4 = paddle._C_ops.sum(cast_11, full_int_array_32, None, True)

        # builtin.slice: (-1x96x4x64xf16) <- ([-1x1x4x64xf16, -1x96x4x64xf16, -1x96x4x64xf16])
        slice_17 = split_4[2]

        # pd_op.relu_: (-1x96x4x64xf16) <- (-1x96x4x64xf16)
        relu__4 = paddle._C_ops.relu_(slice_17)

        # pd_op.cast: (-1x96x4x1xf16) <- (-1x96x4x1xf32)
        cast_12 = paddle._C_ops.cast(sum_4, paddle.float16)

        # pd_op.multiply_: (-1x96x4x64xf16) <- (-1x96x4x64xf16, -1x96x4x1xf16)
        multiply__9 = paddle._C_ops.multiply_(relu__4, cast_12)

        # pd_op.conv2d: (-1x96x4x64xf16) <- (-1x96x4x64xf16, 96x96x1x1xf16)
        conv2d_31 = paddle._C_ops.conv2d(multiply__9, parameter_151, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_33 = [1, 96, 1, 1]

        # pd_op.reshape: (1x96x1x1xf16, 0x96xf16) <- (96xf16, 4xi64)
        reshape_34, reshape_35 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_152, full_int_array_33), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x96x4x64xf16) <- (-1x96x4x64xf16, 1x96x1x1xf16)
        add__26 = paddle._C_ops.add_(conv2d_31, reshape_34)

        # pd_op.add_: (-1x96x4x64xf16) <- (-1x96x4x64xf16, -1x96x4x64xf16)
        add__27 = paddle._C_ops.add_(add__24, add__26)

        # pd_op.group_norm: (-1x96x4x64xf16, -1x1xf32, -1x1xf32) <- (-1x96x4x64xf16, 96xf16, 96xf16)
        group_norm_30, group_norm_31, group_norm_32 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add__27, parameter_153, parameter_154, float('1e-05'), 1, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.conv2d: (-1x192x4x64xf16) <- (-1x96x4x64xf16, 192x96x1x1xf16)
        conv2d_32 = paddle._C_ops.conv2d(group_norm_30, parameter_155, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_34 = [1, 192, 1, 1]

        # pd_op.reshape: (1x192x1x1xf16, 0x192xf16) <- (192xf16, 4xi64)
        reshape_36, reshape_37 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_156, full_int_array_34), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x192x4x64xf16) <- (-1x192x4x64xf16, 1x192x1x1xf16)
        add__28 = paddle._C_ops.add_(conv2d_32, reshape_36)

        # pd_op.silu: (-1x192x4x64xf16) <- (-1x192x4x64xf16)
        silu_17 = paddle._C_ops.silu(add__28)

        # pd_op.conv2d: (-1x96x4x64xf16) <- (-1x192x4x64xf16, 96x192x1x1xf16)
        conv2d_33 = paddle._C_ops.conv2d(silu_17, parameter_157, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_35 = [1, 96, 1, 1]

        # pd_op.reshape: (1x96x1x1xf16, 0x96xf16) <- (96xf16, 4xi64)
        reshape_38, reshape_39 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_158, full_int_array_35), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x96x4x64xf16) <- (-1x96x4x64xf16, 1x96x1x1xf16)
        add__29 = paddle._C_ops.add_(conv2d_33, reshape_38)

        # pd_op.add_: (-1x96x4x64xf16) <- (-1x96x4x64xf16, -1x96x4x64xf16)
        add__30 = paddle._C_ops.add_(add__27, add__29)

        # pd_op.group_norm: (-1x96x4x64xf16, -1x1xf32, -1x1xf32) <- (-1x96x4x64xf16, 96xf16, 96xf16)
        group_norm_33, group_norm_34, group_norm_35 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add__30, parameter_159, parameter_160, float('1e-05'), 1, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.conv2d: (-1x193x4x64xf16) <- (-1x96x4x64xf16, 193x96x1x1xf16)
        conv2d_34 = paddle._C_ops.conv2d(group_norm_33, parameter_161, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_36 = [1, 193, 1, 1]

        # pd_op.reshape: (1x193x1x1xf16, 0x193xf16) <- (193xf16, 4xi64)
        reshape_40, reshape_41 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_162, full_int_array_36), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x193x4x64xf16) <- (-1x193x4x64xf16, 1x193x1x1xf16)
        add__31 = paddle._C_ops.add_(conv2d_34, reshape_40)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_37 = [1, 96, 96]

        # pd_op.full: (1xi32) <- ()
        full_13 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1x4x64xf16, -1x96x4x64xf16, -1x96x4x64xf16]) <- (-1x193x4x64xf16, 3xi64, 1xi32)
        split_5 = paddle._C_ops.split(add__31, full_int_array_37, full_13)

        # builtin.slice: (-1x1x4x64xf16) <- ([-1x1x4x64xf16, -1x96x4x64xf16, -1x96x4x64xf16])
        slice_18 = split_5[0]

        # pd_op.softmax_: (-1x1x4x64xf16) <- (-1x1x4x64xf16)
        softmax__5 = paddle._C_ops.softmax_(slice_18, -1)

        # builtin.slice: (-1x96x4x64xf16) <- ([-1x1x4x64xf16, -1x96x4x64xf16, -1x96x4x64xf16])
        slice_19 = split_5[1]

        # pd_op.multiply_: (-1x96x4x64xf16) <- (-1x96x4x64xf16, -1x1x4x64xf16)
        multiply__10 = paddle._C_ops.multiply_(slice_19, softmax__5)

        # pd_op.cast: (-1x96x4x64xf32) <- (-1x96x4x64xf16)
        cast_13 = paddle._C_ops.cast(multiply__10, paddle.float32)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_38 = [-1]

        # pd_op.sum: (-1x96x4x1xf32) <- (-1x96x4x64xf32, 1xi64)
        sum_5 = paddle._C_ops.sum(cast_13, full_int_array_38, None, True)

        # builtin.slice: (-1x96x4x64xf16) <- ([-1x1x4x64xf16, -1x96x4x64xf16, -1x96x4x64xf16])
        slice_20 = split_5[2]

        # pd_op.relu_: (-1x96x4x64xf16) <- (-1x96x4x64xf16)
        relu__5 = paddle._C_ops.relu_(slice_20)

        # pd_op.cast: (-1x96x4x1xf16) <- (-1x96x4x1xf32)
        cast_14 = paddle._C_ops.cast(sum_5, paddle.float16)

        # pd_op.multiply_: (-1x96x4x64xf16) <- (-1x96x4x64xf16, -1x96x4x1xf16)
        multiply__11 = paddle._C_ops.multiply_(relu__5, cast_14)

        # pd_op.conv2d: (-1x96x4x64xf16) <- (-1x96x4x64xf16, 96x96x1x1xf16)
        conv2d_35 = paddle._C_ops.conv2d(multiply__11, parameter_163, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_39 = [1, 96, 1, 1]

        # pd_op.reshape: (1x96x1x1xf16, 0x96xf16) <- (96xf16, 4xi64)
        reshape_42, reshape_43 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_164, full_int_array_39), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x96x4x64xf16) <- (-1x96x4x64xf16, 1x96x1x1xf16)
        add__32 = paddle._C_ops.add_(conv2d_35, reshape_42)

        # pd_op.add_: (-1x96x4x64xf16) <- (-1x96x4x64xf16, -1x96x4x64xf16)
        add__33 = paddle._C_ops.add_(add__30, add__32)

        # pd_op.group_norm: (-1x96x4x64xf16, -1x1xf32, -1x1xf32) <- (-1x96x4x64xf16, 96xf16, 96xf16)
        group_norm_36, group_norm_37, group_norm_38 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add__33, parameter_165, parameter_166, float('1e-05'), 1, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.conv2d: (-1x192x4x64xf16) <- (-1x96x4x64xf16, 192x96x1x1xf16)
        conv2d_36 = paddle._C_ops.conv2d(group_norm_36, parameter_167, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_40 = [1, 192, 1, 1]

        # pd_op.reshape: (1x192x1x1xf16, 0x192xf16) <- (192xf16, 4xi64)
        reshape_44, reshape_45 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_168, full_int_array_40), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x192x4x64xf16) <- (-1x192x4x64xf16, 1x192x1x1xf16)
        add__34 = paddle._C_ops.add_(conv2d_36, reshape_44)

        # pd_op.silu: (-1x192x4x64xf16) <- (-1x192x4x64xf16)
        silu_18 = paddle._C_ops.silu(add__34)

        # pd_op.conv2d: (-1x96x4x64xf16) <- (-1x192x4x64xf16, 96x192x1x1xf16)
        conv2d_37 = paddle._C_ops.conv2d(silu_18, parameter_169, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_41 = [1, 96, 1, 1]

        # pd_op.reshape: (1x96x1x1xf16, 0x96xf16) <- (96xf16, 4xi64)
        reshape_46, reshape_47 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_170, full_int_array_41), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x96x4x64xf16) <- (-1x96x4x64xf16, 1x96x1x1xf16)
        add__35 = paddle._C_ops.add_(conv2d_37, reshape_46)

        # pd_op.add_: (-1x96x4x64xf16) <- (-1x96x4x64xf16, -1x96x4x64xf16)
        add__36 = paddle._C_ops.add_(add__33, add__35)

        # pd_op.group_norm: (-1x96x4x64xf16, -1x1xf32, -1x1xf32) <- (-1x96x4x64xf16, 96xf16, 96xf16)
        group_norm_39, group_norm_40, group_norm_41 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add__36, parameter_171, parameter_172, float('1e-05'), 1, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (4xi32) <- (-1x96x4x64xf16)
        shape_3 = paddle._C_ops.shape(paddle.cast(group_norm_39, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_42 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_43 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(shape_3, [0], full_int_array_42, full_int_array_43, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_14 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_15 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_3 = [slice_21, full_14, full_15]

        # pd_op.reshape_: (-1x384x64xf16, 0x-1x96x4x64xf16) <- (-1x96x4x64xf16, [1xi32, 1xi32, 1xi32])
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape_(group_norm_39, [x.reshape([]) for x in combine_3]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.cast: (-1x384x64xf32) <- (-1x384x64xf16)
        cast_15 = paddle._C_ops.cast(reshape__6, paddle.float32)

        # pd_op.fold: (-1x96x16x16xf32) <- (-1x384x64xf32)
        fold_1 = paddle._C_ops.fold(cast_15, [16, 16], [2, 2], [2, 2], [0, 0, 0, 0], [1, 1])

        # pd_op.cast: (-1x96x16x16xf16) <- (-1x96x16x16xf32)
        cast_16 = paddle._C_ops.cast(fold_1, paddle.float16)

        # pd_op.conv2d: (-1x192x16x16xf16) <- (-1x96x16x16xf16, 192x96x1x1xf16)
        conv2d_38 = paddle._C_ops.conv2d(cast_16, parameter_173, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x192x16x16xf16, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x16x16xf16, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_38, parameter_174, parameter_175, parameter_176, parameter_177, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x384x16x16xf16) <- (-1x192x16x16xf16, 384x192x1x1xf16)
        conv2d_39 = paddle._C_ops.conv2d(batch_norm__114, parameter_178, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x384x16x16xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x16x16xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_39, parameter_179, parameter_180, parameter_181, parameter_182, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x384x16x16xf16) <- (-1x384x16x16xf16)
        silu_19 = paddle._C_ops.silu(batch_norm__120)

        # pd_op.depthwise_conv2d: (-1x384x8x8xf16) <- (-1x384x16x16xf16, 384x1x3x3xf16)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(silu_19, parameter_183, [2, 2], [1, 1], 'EXPLICIT', 384, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x384x8x8xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x8x8xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_7, parameter_184, parameter_185, parameter_186, parameter_187, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x384x8x8xf16) <- (-1x384x8x8xf16)
        silu_20 = paddle._C_ops.silu(batch_norm__126)

        # pd_op.conv2d: (-1x256x8x8xf16) <- (-1x384x8x8xf16, 256x384x1x1xf16)
        conv2d_40 = paddle._C_ops.conv2d(silu_20, parameter_188, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x8x8xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x8x8xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_40, parameter_189, parameter_190, parameter_191, parameter_192, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.depthwise_conv2d: (-1x256x8x8xf16) <- (-1x256x8x8xf16, 256x1x3x3xf16)
        depthwise_conv2d_8 = paddle._C_ops.depthwise_conv2d(batch_norm__132, parameter_193, [1, 1], [1, 1], 'EXPLICIT', 256, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x256x8x8xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x8x8xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_8, parameter_194, parameter_195, parameter_196, parameter_197, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x256x8x8xf16) <- (-1x256x8x8xf16)
        silu_21 = paddle._C_ops.silu(batch_norm__138)

        # pd_op.conv2d: (-1x128x8x8xf16) <- (-1x256x8x8xf16, 128x256x1x1xf16)
        conv2d_41 = paddle._C_ops.conv2d(silu_21, parameter_198, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.shape: (4xi32) <- (-1x128x8x8xf16)
        shape_4 = paddle._C_ops.shape(paddle.cast(conv2d_41, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_44 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_45 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(shape_4, [0], full_int_array_44, full_int_array_45, [1], [0])

        # pd_op.unfold: (-1x512x16xf16) <- (-1x128x8x8xf16)
        unfold_2 = paddle._C_ops.unfold(conv2d_41, [2, 2], [2, 2], [0, 0, 0, 0], [1, 1])

        # pd_op.full: (1xi32) <- ()
        full_16 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_17 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_18 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_4 = [slice_22, full_16, full_17, full_18]

        # pd_op.reshape_: (-1x128x4x16xf16, 0x-1x512x16xf16) <- (-1x512x16xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_2, [x.reshape([]) for x in combine_4]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.group_norm: (-1x128x4x16xf16, -1x1xf32, -1x1xf32) <- (-1x128x4x16xf16, 128xf16, 128xf16)
        group_norm_42, group_norm_43, group_norm_44 = (lambda x, f: f(x))(paddle._C_ops.group_norm(reshape__8, parameter_199, parameter_200, float('1e-05'), 1, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.conv2d: (-1x257x4x16xf16) <- (-1x128x4x16xf16, 257x128x1x1xf16)
        conv2d_42 = paddle._C_ops.conv2d(group_norm_42, parameter_201, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_46 = [1, 257, 1, 1]

        # pd_op.reshape: (1x257x1x1xf16, 0x257xf16) <- (257xf16, 4xi64)
        reshape_48, reshape_49 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_202, full_int_array_46), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x257x4x16xf16) <- (-1x257x4x16xf16, 1x257x1x1xf16)
        add__37 = paddle._C_ops.add_(conv2d_42, reshape_48)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_47 = [1, 128, 128]

        # pd_op.full: (1xi32) <- ()
        full_19 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1x4x16xf16, -1x128x4x16xf16, -1x128x4x16xf16]) <- (-1x257x4x16xf16, 3xi64, 1xi32)
        split_6 = paddle._C_ops.split(add__37, full_int_array_47, full_19)

        # builtin.slice: (-1x1x4x16xf16) <- ([-1x1x4x16xf16, -1x128x4x16xf16, -1x128x4x16xf16])
        slice_23 = split_6[0]

        # pd_op.softmax_: (-1x1x4x16xf16) <- (-1x1x4x16xf16)
        softmax__6 = paddle._C_ops.softmax_(slice_23, -1)

        # builtin.slice: (-1x128x4x16xf16) <- ([-1x1x4x16xf16, -1x128x4x16xf16, -1x128x4x16xf16])
        slice_24 = split_6[1]

        # pd_op.multiply_: (-1x128x4x16xf16) <- (-1x128x4x16xf16, -1x1x4x16xf16)
        multiply__12 = paddle._C_ops.multiply_(slice_24, softmax__6)

        # pd_op.cast: (-1x128x4x16xf32) <- (-1x128x4x16xf16)
        cast_17 = paddle._C_ops.cast(multiply__12, paddle.float32)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_48 = [-1]

        # pd_op.sum: (-1x128x4x1xf32) <- (-1x128x4x16xf32, 1xi64)
        sum_6 = paddle._C_ops.sum(cast_17, full_int_array_48, None, True)

        # builtin.slice: (-1x128x4x16xf16) <- ([-1x1x4x16xf16, -1x128x4x16xf16, -1x128x4x16xf16])
        slice_25 = split_6[2]

        # pd_op.relu_: (-1x128x4x16xf16) <- (-1x128x4x16xf16)
        relu__6 = paddle._C_ops.relu_(slice_25)

        # pd_op.cast: (-1x128x4x1xf16) <- (-1x128x4x1xf32)
        cast_18 = paddle._C_ops.cast(sum_6, paddle.float16)

        # pd_op.multiply_: (-1x128x4x16xf16) <- (-1x128x4x16xf16, -1x128x4x1xf16)
        multiply__13 = paddle._C_ops.multiply_(relu__6, cast_18)

        # pd_op.conv2d: (-1x128x4x16xf16) <- (-1x128x4x16xf16, 128x128x1x1xf16)
        conv2d_43 = paddle._C_ops.conv2d(multiply__13, parameter_203, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_49 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf16, 0x128xf16) <- (128xf16, 4xi64)
        reshape_50, reshape_51 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_204, full_int_array_49), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x4x16xf16) <- (-1x128x4x16xf16, 1x128x1x1xf16)
        add__38 = paddle._C_ops.add_(conv2d_43, reshape_50)

        # pd_op.add_: (-1x128x4x16xf16) <- (-1x128x4x16xf16, -1x128x4x16xf16)
        add__39 = paddle._C_ops.add_(reshape__8, add__38)

        # pd_op.group_norm: (-1x128x4x16xf16, -1x1xf32, -1x1xf32) <- (-1x128x4x16xf16, 128xf16, 128xf16)
        group_norm_45, group_norm_46, group_norm_47 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add__39, parameter_205, parameter_206, float('1e-05'), 1, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.conv2d: (-1x256x4x16xf16) <- (-1x128x4x16xf16, 256x128x1x1xf16)
        conv2d_44 = paddle._C_ops.conv2d(group_norm_45, parameter_207, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_50 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_52, reshape_53 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_208, full_int_array_50), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x4x16xf16) <- (-1x256x4x16xf16, 1x256x1x1xf16)
        add__40 = paddle._C_ops.add_(conv2d_44, reshape_52)

        # pd_op.silu: (-1x256x4x16xf16) <- (-1x256x4x16xf16)
        silu_22 = paddle._C_ops.silu(add__40)

        # pd_op.conv2d: (-1x128x4x16xf16) <- (-1x256x4x16xf16, 128x256x1x1xf16)
        conv2d_45 = paddle._C_ops.conv2d(silu_22, parameter_209, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_51 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf16, 0x128xf16) <- (128xf16, 4xi64)
        reshape_54, reshape_55 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_210, full_int_array_51), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x4x16xf16) <- (-1x128x4x16xf16, 1x128x1x1xf16)
        add__41 = paddle._C_ops.add_(conv2d_45, reshape_54)

        # pd_op.add_: (-1x128x4x16xf16) <- (-1x128x4x16xf16, -1x128x4x16xf16)
        add__42 = paddle._C_ops.add_(add__39, add__41)

        # pd_op.group_norm: (-1x128x4x16xf16, -1x1xf32, -1x1xf32) <- (-1x128x4x16xf16, 128xf16, 128xf16)
        group_norm_48, group_norm_49, group_norm_50 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add__42, parameter_211, parameter_212, float('1e-05'), 1, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.conv2d: (-1x257x4x16xf16) <- (-1x128x4x16xf16, 257x128x1x1xf16)
        conv2d_46 = paddle._C_ops.conv2d(group_norm_48, parameter_213, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_52 = [1, 257, 1, 1]

        # pd_op.reshape: (1x257x1x1xf16, 0x257xf16) <- (257xf16, 4xi64)
        reshape_56, reshape_57 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_214, full_int_array_52), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x257x4x16xf16) <- (-1x257x4x16xf16, 1x257x1x1xf16)
        add__43 = paddle._C_ops.add_(conv2d_46, reshape_56)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_53 = [1, 128, 128]

        # pd_op.full: (1xi32) <- ()
        full_20 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1x4x16xf16, -1x128x4x16xf16, -1x128x4x16xf16]) <- (-1x257x4x16xf16, 3xi64, 1xi32)
        split_7 = paddle._C_ops.split(add__43, full_int_array_53, full_20)

        # builtin.slice: (-1x1x4x16xf16) <- ([-1x1x4x16xf16, -1x128x4x16xf16, -1x128x4x16xf16])
        slice_26 = split_7[0]

        # pd_op.softmax_: (-1x1x4x16xf16) <- (-1x1x4x16xf16)
        softmax__7 = paddle._C_ops.softmax_(slice_26, -1)

        # builtin.slice: (-1x128x4x16xf16) <- ([-1x1x4x16xf16, -1x128x4x16xf16, -1x128x4x16xf16])
        slice_27 = split_7[1]

        # pd_op.multiply_: (-1x128x4x16xf16) <- (-1x128x4x16xf16, -1x1x4x16xf16)
        multiply__14 = paddle._C_ops.multiply_(slice_27, softmax__7)

        # pd_op.cast: (-1x128x4x16xf32) <- (-1x128x4x16xf16)
        cast_19 = paddle._C_ops.cast(multiply__14, paddle.float32)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_54 = [-1]

        # pd_op.sum: (-1x128x4x1xf32) <- (-1x128x4x16xf32, 1xi64)
        sum_7 = paddle._C_ops.sum(cast_19, full_int_array_54, None, True)

        # builtin.slice: (-1x128x4x16xf16) <- ([-1x1x4x16xf16, -1x128x4x16xf16, -1x128x4x16xf16])
        slice_28 = split_7[2]

        # pd_op.relu_: (-1x128x4x16xf16) <- (-1x128x4x16xf16)
        relu__7 = paddle._C_ops.relu_(slice_28)

        # pd_op.cast: (-1x128x4x1xf16) <- (-1x128x4x1xf32)
        cast_20 = paddle._C_ops.cast(sum_7, paddle.float16)

        # pd_op.multiply_: (-1x128x4x16xf16) <- (-1x128x4x16xf16, -1x128x4x1xf16)
        multiply__15 = paddle._C_ops.multiply_(relu__7, cast_20)

        # pd_op.conv2d: (-1x128x4x16xf16) <- (-1x128x4x16xf16, 128x128x1x1xf16)
        conv2d_47 = paddle._C_ops.conv2d(multiply__15, parameter_215, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_55 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf16, 0x128xf16) <- (128xf16, 4xi64)
        reshape_58, reshape_59 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_216, full_int_array_55), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x4x16xf16) <- (-1x128x4x16xf16, 1x128x1x1xf16)
        add__44 = paddle._C_ops.add_(conv2d_47, reshape_58)

        # pd_op.add_: (-1x128x4x16xf16) <- (-1x128x4x16xf16, -1x128x4x16xf16)
        add__45 = paddle._C_ops.add_(add__42, add__44)

        # pd_op.group_norm: (-1x128x4x16xf16, -1x1xf32, -1x1xf32) <- (-1x128x4x16xf16, 128xf16, 128xf16)
        group_norm_51, group_norm_52, group_norm_53 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add__45, parameter_217, parameter_218, float('1e-05'), 1, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.conv2d: (-1x256x4x16xf16) <- (-1x128x4x16xf16, 256x128x1x1xf16)
        conv2d_48 = paddle._C_ops.conv2d(group_norm_51, parameter_219, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_56 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_60, reshape_61 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_220, full_int_array_56), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x4x16xf16) <- (-1x256x4x16xf16, 1x256x1x1xf16)
        add__46 = paddle._C_ops.add_(conv2d_48, reshape_60)

        # pd_op.silu: (-1x256x4x16xf16) <- (-1x256x4x16xf16)
        silu_23 = paddle._C_ops.silu(add__46)

        # pd_op.conv2d: (-1x128x4x16xf16) <- (-1x256x4x16xf16, 128x256x1x1xf16)
        conv2d_49 = paddle._C_ops.conv2d(silu_23, parameter_221, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_57 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf16, 0x128xf16) <- (128xf16, 4xi64)
        reshape_62, reshape_63 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_222, full_int_array_57), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x4x16xf16) <- (-1x128x4x16xf16, 1x128x1x1xf16)
        add__47 = paddle._C_ops.add_(conv2d_49, reshape_62)

        # pd_op.add_: (-1x128x4x16xf16) <- (-1x128x4x16xf16, -1x128x4x16xf16)
        add__48 = paddle._C_ops.add_(add__45, add__47)

        # pd_op.group_norm: (-1x128x4x16xf16, -1x1xf32, -1x1xf32) <- (-1x128x4x16xf16, 128xf16, 128xf16)
        group_norm_54, group_norm_55, group_norm_56 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add__48, parameter_223, parameter_224, float('1e-05'), 1, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.conv2d: (-1x257x4x16xf16) <- (-1x128x4x16xf16, 257x128x1x1xf16)
        conv2d_50 = paddle._C_ops.conv2d(group_norm_54, parameter_225, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_58 = [1, 257, 1, 1]

        # pd_op.reshape: (1x257x1x1xf16, 0x257xf16) <- (257xf16, 4xi64)
        reshape_64, reshape_65 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_226, full_int_array_58), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x257x4x16xf16) <- (-1x257x4x16xf16, 1x257x1x1xf16)
        add__49 = paddle._C_ops.add_(conv2d_50, reshape_64)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_59 = [1, 128, 128]

        # pd_op.full: (1xi32) <- ()
        full_21 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x1x4x16xf16, -1x128x4x16xf16, -1x128x4x16xf16]) <- (-1x257x4x16xf16, 3xi64, 1xi32)
        split_8 = paddle._C_ops.split(add__49, full_int_array_59, full_21)

        # builtin.slice: (-1x1x4x16xf16) <- ([-1x1x4x16xf16, -1x128x4x16xf16, -1x128x4x16xf16])
        slice_29 = split_8[0]

        # pd_op.softmax_: (-1x1x4x16xf16) <- (-1x1x4x16xf16)
        softmax__8 = paddle._C_ops.softmax_(slice_29, -1)

        # builtin.slice: (-1x128x4x16xf16) <- ([-1x1x4x16xf16, -1x128x4x16xf16, -1x128x4x16xf16])
        slice_30 = split_8[1]

        # pd_op.multiply_: (-1x128x4x16xf16) <- (-1x128x4x16xf16, -1x1x4x16xf16)
        multiply__16 = paddle._C_ops.multiply_(slice_30, softmax__8)

        # pd_op.cast: (-1x128x4x16xf32) <- (-1x128x4x16xf16)
        cast_21 = paddle._C_ops.cast(multiply__16, paddle.float32)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_60 = [-1]

        # pd_op.sum: (-1x128x4x1xf32) <- (-1x128x4x16xf32, 1xi64)
        sum_8 = paddle._C_ops.sum(cast_21, full_int_array_60, None, True)

        # builtin.slice: (-1x128x4x16xf16) <- ([-1x1x4x16xf16, -1x128x4x16xf16, -1x128x4x16xf16])
        slice_31 = split_8[2]

        # pd_op.relu_: (-1x128x4x16xf16) <- (-1x128x4x16xf16)
        relu__8 = paddle._C_ops.relu_(slice_31)

        # pd_op.cast: (-1x128x4x1xf16) <- (-1x128x4x1xf32)
        cast_22 = paddle._C_ops.cast(sum_8, paddle.float16)

        # pd_op.multiply_: (-1x128x4x16xf16) <- (-1x128x4x16xf16, -1x128x4x1xf16)
        multiply__17 = paddle._C_ops.multiply_(relu__8, cast_22)

        # pd_op.conv2d: (-1x128x4x16xf16) <- (-1x128x4x16xf16, 128x128x1x1xf16)
        conv2d_51 = paddle._C_ops.conv2d(multiply__17, parameter_227, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_61 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf16, 0x128xf16) <- (128xf16, 4xi64)
        reshape_66, reshape_67 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_228, full_int_array_61), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x4x16xf16) <- (-1x128x4x16xf16, 1x128x1x1xf16)
        add__50 = paddle._C_ops.add_(conv2d_51, reshape_66)

        # pd_op.add_: (-1x128x4x16xf16) <- (-1x128x4x16xf16, -1x128x4x16xf16)
        add__51 = paddle._C_ops.add_(add__48, add__50)

        # pd_op.group_norm: (-1x128x4x16xf16, -1x1xf32, -1x1xf32) <- (-1x128x4x16xf16, 128xf16, 128xf16)
        group_norm_57, group_norm_58, group_norm_59 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add__51, parameter_229, parameter_230, float('1e-05'), 1, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.conv2d: (-1x256x4x16xf16) <- (-1x128x4x16xf16, 256x128x1x1xf16)
        conv2d_52 = paddle._C_ops.conv2d(group_norm_57, parameter_231, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_62 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_68, reshape_69 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_232, full_int_array_62), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x4x16xf16) <- (-1x256x4x16xf16, 1x256x1x1xf16)
        add__52 = paddle._C_ops.add_(conv2d_52, reshape_68)

        # pd_op.silu: (-1x256x4x16xf16) <- (-1x256x4x16xf16)
        silu_24 = paddle._C_ops.silu(add__52)

        # pd_op.conv2d: (-1x128x4x16xf16) <- (-1x256x4x16xf16, 128x256x1x1xf16)
        conv2d_53 = paddle._C_ops.conv2d(silu_24, parameter_233, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_63 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf16, 0x128xf16) <- (128xf16, 4xi64)
        reshape_70, reshape_71 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_234, full_int_array_63), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x4x16xf16) <- (-1x128x4x16xf16, 1x128x1x1xf16)
        add__53 = paddle._C_ops.add_(conv2d_53, reshape_70)

        # pd_op.add_: (-1x128x4x16xf16) <- (-1x128x4x16xf16, -1x128x4x16xf16)
        add__54 = paddle._C_ops.add_(add__51, add__53)

        # pd_op.group_norm: (-1x128x4x16xf16, -1x1xf32, -1x1xf32) <- (-1x128x4x16xf16, 128xf16, 128xf16)
        group_norm_60, group_norm_61, group_norm_62 = (lambda x, f: f(x))(paddle._C_ops.group_norm(add__54, parameter_235, parameter_236, float('1e-05'), 1, 'NCHW'), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (4xi32) <- (-1x128x4x16xf16)
        shape_5 = paddle._C_ops.shape(paddle.cast(group_norm_60, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_64 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_65 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(shape_5, [0], full_int_array_64, full_int_array_65, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_22 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_23 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_5 = [slice_32, full_22, full_23]

        # pd_op.reshape_: (-1x512x16xf16, 0x-1x128x4x16xf16) <- (-1x128x4x16xf16, [1xi32, 1xi32, 1xi32])
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape_(group_norm_60, [x.reshape([]) for x in combine_5]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.cast: (-1x512x16xf32) <- (-1x512x16xf16)
        cast_23 = paddle._C_ops.cast(reshape__10, paddle.float32)

        # pd_op.fold: (-1x128x8x8xf32) <- (-1x512x16xf32)
        fold_2 = paddle._C_ops.fold(cast_23, [8, 8], [2, 2], [2, 2], [0, 0, 0, 0], [1, 1])

        # pd_op.cast: (-1x128x8x8xf16) <- (-1x128x8x8xf32)
        cast_24 = paddle._C_ops.cast(fold_2, paddle.float16)

        # pd_op.conv2d: (-1x256x8x8xf16) <- (-1x128x8x8xf16, 256x128x1x1xf16)
        conv2d_54 = paddle._C_ops.conv2d(cast_24, parameter_237, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x8x8xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x8x8xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_54, parameter_238, parameter_239, parameter_240, parameter_241, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_66 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf16) <- (-1x256x8x8xf16, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(batch_norm__144, full_int_array_66, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.flatten_: (-1x256xf16, None) <- (-1x256x1x1xf16)
        flatten__0, flatten__1 = (lambda x, f: f(x))(paddle._C_ops.flatten_(pool2d_0, 1, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x1000xf16) <- (-1x256xf16, 256x1000xf16)
        matmul_0 = paddle._C_ops.matmul(flatten__0, parameter_242, False, False)

        # pd_op.add_: (-1x1000xf16) <- (-1x1000xf16, 1000xf16)
        add__55 = paddle._C_ops.add_(matmul_0, parameter_243)

        # pd_op.softmax_: (-1x1000xf16) <- (-1x1000xf16)
        softmax__9 = paddle._C_ops.softmax_(add__55, -1)

        # pd_op.cast: (-1x1000xf32) <- (-1x1000xf16)
        cast_25 = paddle._C_ops.cast(softmax__9, paddle.float32)
        return cast_25



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

    def forward(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_72, parameter_71, parameter_73, parameter_74, parameter_75, parameter_76, parameter_78, parameter_77, parameter_79, parameter_80, parameter_81, parameter_82, parameter_84, parameter_83, parameter_85, parameter_86, parameter_87, parameter_88, parameter_90, parameter_89, parameter_91, parameter_92, parameter_93, parameter_94, parameter_96, parameter_95, parameter_97, parameter_101, parameter_98, parameter_100, parameter_99, parameter_102, parameter_106, parameter_103, parameter_105, parameter_104, parameter_107, parameter_111, parameter_108, parameter_110, parameter_109, parameter_112, parameter_116, parameter_113, parameter_115, parameter_114, parameter_117, parameter_121, parameter_118, parameter_120, parameter_119, parameter_122, parameter_124, parameter_123, parameter_125, parameter_126, parameter_127, parameter_128, parameter_130, parameter_129, parameter_131, parameter_132, parameter_133, parameter_134, parameter_136, parameter_135, parameter_137, parameter_138, parameter_139, parameter_140, parameter_142, parameter_141, parameter_143, parameter_144, parameter_145, parameter_146, parameter_148, parameter_147, parameter_149, parameter_150, parameter_151, parameter_152, parameter_154, parameter_153, parameter_155, parameter_156, parameter_157, parameter_158, parameter_160, parameter_159, parameter_161, parameter_162, parameter_163, parameter_164, parameter_166, parameter_165, parameter_167, parameter_168, parameter_169, parameter_170, parameter_172, parameter_171, parameter_173, parameter_177, parameter_174, parameter_176, parameter_175, parameter_178, parameter_182, parameter_179, parameter_181, parameter_180, parameter_183, parameter_187, parameter_184, parameter_186, parameter_185, parameter_188, parameter_192, parameter_189, parameter_191, parameter_190, parameter_193, parameter_197, parameter_194, parameter_196, parameter_195, parameter_198, parameter_200, parameter_199, parameter_201, parameter_202, parameter_203, parameter_204, parameter_206, parameter_205, parameter_207, parameter_208, parameter_209, parameter_210, parameter_212, parameter_211, parameter_213, parameter_214, parameter_215, parameter_216, parameter_218, parameter_217, parameter_219, parameter_220, parameter_221, parameter_222, parameter_224, parameter_223, parameter_225, parameter_226, parameter_227, parameter_228, parameter_230, parameter_229, parameter_231, parameter_232, parameter_233, parameter_234, parameter_236, parameter_235, parameter_237, parameter_241, parameter_238, parameter_240, parameter_239, parameter_242, parameter_243, feed_0):
        return self.builtin_module_710_0_0(parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_72, parameter_71, parameter_73, parameter_74, parameter_75, parameter_76, parameter_78, parameter_77, parameter_79, parameter_80, parameter_81, parameter_82, parameter_84, parameter_83, parameter_85, parameter_86, parameter_87, parameter_88, parameter_90, parameter_89, parameter_91, parameter_92, parameter_93, parameter_94, parameter_96, parameter_95, parameter_97, parameter_101, parameter_98, parameter_100, parameter_99, parameter_102, parameter_106, parameter_103, parameter_105, parameter_104, parameter_107, parameter_111, parameter_108, parameter_110, parameter_109, parameter_112, parameter_116, parameter_113, parameter_115, parameter_114, parameter_117, parameter_121, parameter_118, parameter_120, parameter_119, parameter_122, parameter_124, parameter_123, parameter_125, parameter_126, parameter_127, parameter_128, parameter_130, parameter_129, parameter_131, parameter_132, parameter_133, parameter_134, parameter_136, parameter_135, parameter_137, parameter_138, parameter_139, parameter_140, parameter_142, parameter_141, parameter_143, parameter_144, parameter_145, parameter_146, parameter_148, parameter_147, parameter_149, parameter_150, parameter_151, parameter_152, parameter_154, parameter_153, parameter_155, parameter_156, parameter_157, parameter_158, parameter_160, parameter_159, parameter_161, parameter_162, parameter_163, parameter_164, parameter_166, parameter_165, parameter_167, parameter_168, parameter_169, parameter_170, parameter_172, parameter_171, parameter_173, parameter_177, parameter_174, parameter_176, parameter_175, parameter_178, parameter_182, parameter_179, parameter_181, parameter_180, parameter_183, parameter_187, parameter_184, parameter_186, parameter_185, parameter_188, parameter_192, parameter_189, parameter_191, parameter_190, parameter_193, parameter_197, parameter_194, parameter_196, parameter_195, parameter_198, parameter_200, parameter_199, parameter_201, parameter_202, parameter_203, parameter_204, parameter_206, parameter_205, parameter_207, parameter_208, parameter_209, parameter_210, parameter_212, parameter_211, parameter_213, parameter_214, parameter_215, parameter_216, parameter_218, parameter_217, parameter_219, parameter_220, parameter_221, parameter_222, parameter_224, parameter_223, parameter_225, parameter_226, parameter_227, parameter_228, parameter_230, parameter_229, parameter_231, parameter_232, parameter_233, parameter_234, parameter_236, parameter_235, parameter_237, parameter_241, parameter_238, parameter_240, parameter_239, parameter_242, parameter_243, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_710_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([16, 3, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_4
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([32, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_9
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([32, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_14
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([32, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_19
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([64, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_24
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([64, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_29
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_34
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([128, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_39
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([128, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_44
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([64, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_49
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([128, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_54
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([128, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_59
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_64
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([128, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_69
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([64, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_72
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_71
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_73
            paddle.uniform([129, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_74
            paddle.uniform([129], dtype='float16', min=0, max=0.5),
            # parameter_75
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_76
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_78
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_77
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_79
            paddle.uniform([128, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_80
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_81
            paddle.uniform([64, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_82
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_84
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_83
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_85
            paddle.uniform([129, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_86
            paddle.uniform([129], dtype='float16', min=0, max=0.5),
            # parameter_87
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_88
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_90
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_89
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_91
            paddle.uniform([128, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_92
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_93
            paddle.uniform([64, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_94
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_96
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_95
            paddle.uniform([64], dtype='float16', min=0, max=0.5),
            # parameter_97
            paddle.uniform([128, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_101
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([256, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_106
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([256, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_111
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([192, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_116
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([192, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_121
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([96, 192, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_124
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_123
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_125
            paddle.uniform([193, 96, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_126
            paddle.uniform([193], dtype='float16', min=0, max=0.5),
            # parameter_127
            paddle.uniform([96, 96, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_128
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_130
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_129
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_131
            paddle.uniform([192, 96, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_132
            paddle.uniform([192], dtype='float16', min=0, max=0.5),
            # parameter_133
            paddle.uniform([96, 192, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_134
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_136
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_135
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_137
            paddle.uniform([193, 96, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_138
            paddle.uniform([193], dtype='float16', min=0, max=0.5),
            # parameter_139
            paddle.uniform([96, 96, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_140
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_142
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_141
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_143
            paddle.uniform([192, 96, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_144
            paddle.uniform([192], dtype='float16', min=0, max=0.5),
            # parameter_145
            paddle.uniform([96, 192, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_146
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_148
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_147
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_149
            paddle.uniform([193, 96, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_150
            paddle.uniform([193], dtype='float16', min=0, max=0.5),
            # parameter_151
            paddle.uniform([96, 96, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_152
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_154
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_153
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_155
            paddle.uniform([192, 96, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_156
            paddle.uniform([192], dtype='float16', min=0, max=0.5),
            # parameter_157
            paddle.uniform([96, 192, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_158
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_160
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_159
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_161
            paddle.uniform([193, 96, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_162
            paddle.uniform([193], dtype='float16', min=0, max=0.5),
            # parameter_163
            paddle.uniform([96, 96, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_164
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_166
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_165
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_167
            paddle.uniform([192, 96, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_168
            paddle.uniform([192], dtype='float16', min=0, max=0.5),
            # parameter_169
            paddle.uniform([96, 192, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_170
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_172
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_171
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_173
            paddle.uniform([192, 96, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_177
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([384, 192, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_182
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([384, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_187
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([256, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_192
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_189
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([256, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_197
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([128, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_200
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_199
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_201
            paddle.uniform([257, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_202
            paddle.uniform([257], dtype='float16', min=0, max=0.5),
            # parameter_203
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_204
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_206
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_205
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_207
            paddle.uniform([256, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_208
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_209
            paddle.uniform([128, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_210
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_212
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_211
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_213
            paddle.uniform([257, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_214
            paddle.uniform([257], dtype='float16', min=0, max=0.5),
            # parameter_215
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_216
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_218
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_217
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_219
            paddle.uniform([256, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_220
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_221
            paddle.uniform([128, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_222
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_224
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_223
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_225
            paddle.uniform([257, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_226
            paddle.uniform([257], dtype='float16', min=0, max=0.5),
            # parameter_227
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_228
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_230
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_229
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_231
            paddle.uniform([256, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_232
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_233
            paddle.uniform([128, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_234
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_236
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_235
            paddle.uniform([128], dtype='float16', min=0, max=0.5),
            # parameter_237
            paddle.uniform([256, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_241
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_238
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_239
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([256, 1000], dtype='float16', min=0, max=0.5),
            # parameter_243
            paddle.uniform([1000], dtype='float16', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 3, 256, 256], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_0
            paddle.static.InputSpec(shape=[16, 3, 3, 3], dtype='float16'),
            # parameter_4
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[32, 16, 1, 1], dtype='float16'),
            # parameter_9
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[32, 1, 3, 3], dtype='float16'),
            # parameter_14
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[32, 32, 1, 1], dtype='float16'),
            # parameter_19
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[64, 32, 1, 1], dtype='float16'),
            # parameter_24
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[64, 1, 3, 3], dtype='float16'),
            # parameter_29
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_34
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[128, 64, 1, 1], dtype='float16'),
            # parameter_39
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float16'),
            # parameter_44
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[64, 128, 1, 1], dtype='float16'),
            # parameter_49
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[128, 64, 1, 1], dtype='float16'),
            # parameter_54
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float16'),
            # parameter_59
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_64
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float16'),
            # parameter_69
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[64, 128, 1, 1], dtype='float16'),
            # parameter_72
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_71
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_73
            paddle.static.InputSpec(shape=[129, 64, 1, 1], dtype='float16'),
            # parameter_74
            paddle.static.InputSpec(shape=[129], dtype='float16'),
            # parameter_75
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_76
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_78
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_77
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_79
            paddle.static.InputSpec(shape=[128, 64, 1, 1], dtype='float16'),
            # parameter_80
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_81
            paddle.static.InputSpec(shape=[64, 128, 1, 1], dtype='float16'),
            # parameter_82
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_84
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_83
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_85
            paddle.static.InputSpec(shape=[129, 64, 1, 1], dtype='float16'),
            # parameter_86
            paddle.static.InputSpec(shape=[129], dtype='float16'),
            # parameter_87
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_88
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_90
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_89
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_91
            paddle.static.InputSpec(shape=[128, 64, 1, 1], dtype='float16'),
            # parameter_92
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_93
            paddle.static.InputSpec(shape=[64, 128, 1, 1], dtype='float16'),
            # parameter_94
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_96
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_95
            paddle.static.InputSpec(shape=[64], dtype='float16'),
            # parameter_97
            paddle.static.InputSpec(shape=[128, 64, 1, 1], dtype='float16'),
            # parameter_101
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float16'),
            # parameter_106
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[256, 1, 3, 3], dtype='float16'),
            # parameter_111
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[192, 256, 1, 1], dtype='float16'),
            # parameter_116
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[192, 1, 3, 3], dtype='float16'),
            # parameter_121
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[96, 192, 1, 1], dtype='float16'),
            # parameter_124
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_123
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_125
            paddle.static.InputSpec(shape=[193, 96, 1, 1], dtype='float16'),
            # parameter_126
            paddle.static.InputSpec(shape=[193], dtype='float16'),
            # parameter_127
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float16'),
            # parameter_128
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_130
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_129
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_131
            paddle.static.InputSpec(shape=[192, 96, 1, 1], dtype='float16'),
            # parameter_132
            paddle.static.InputSpec(shape=[192], dtype='float16'),
            # parameter_133
            paddle.static.InputSpec(shape=[96, 192, 1, 1], dtype='float16'),
            # parameter_134
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_136
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_135
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_137
            paddle.static.InputSpec(shape=[193, 96, 1, 1], dtype='float16'),
            # parameter_138
            paddle.static.InputSpec(shape=[193], dtype='float16'),
            # parameter_139
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float16'),
            # parameter_140
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_142
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_141
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_143
            paddle.static.InputSpec(shape=[192, 96, 1, 1], dtype='float16'),
            # parameter_144
            paddle.static.InputSpec(shape=[192], dtype='float16'),
            # parameter_145
            paddle.static.InputSpec(shape=[96, 192, 1, 1], dtype='float16'),
            # parameter_146
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_148
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_147
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_149
            paddle.static.InputSpec(shape=[193, 96, 1, 1], dtype='float16'),
            # parameter_150
            paddle.static.InputSpec(shape=[193], dtype='float16'),
            # parameter_151
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float16'),
            # parameter_152
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_154
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_153
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_155
            paddle.static.InputSpec(shape=[192, 96, 1, 1], dtype='float16'),
            # parameter_156
            paddle.static.InputSpec(shape=[192], dtype='float16'),
            # parameter_157
            paddle.static.InputSpec(shape=[96, 192, 1, 1], dtype='float16'),
            # parameter_158
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_160
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_159
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_161
            paddle.static.InputSpec(shape=[193, 96, 1, 1], dtype='float16'),
            # parameter_162
            paddle.static.InputSpec(shape=[193], dtype='float16'),
            # parameter_163
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float16'),
            # parameter_164
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_166
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_165
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_167
            paddle.static.InputSpec(shape=[192, 96, 1, 1], dtype='float16'),
            # parameter_168
            paddle.static.InputSpec(shape=[192], dtype='float16'),
            # parameter_169
            paddle.static.InputSpec(shape=[96, 192, 1, 1], dtype='float16'),
            # parameter_170
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_172
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_171
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_173
            paddle.static.InputSpec(shape=[192, 96, 1, 1], dtype='float16'),
            # parameter_177
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[384, 192, 1, 1], dtype='float16'),
            # parameter_182
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[384, 1, 3, 3], dtype='float16'),
            # parameter_187
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[256, 384, 1, 1], dtype='float16'),
            # parameter_192
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_189
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[256, 1, 3, 3], dtype='float16'),
            # parameter_197
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float16'),
            # parameter_200
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_199
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_201
            paddle.static.InputSpec(shape=[257, 128, 1, 1], dtype='float16'),
            # parameter_202
            paddle.static.InputSpec(shape=[257], dtype='float16'),
            # parameter_203
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_204
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_206
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_205
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_207
            paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float16'),
            # parameter_208
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_209
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float16'),
            # parameter_210
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_212
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_211
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_213
            paddle.static.InputSpec(shape=[257, 128, 1, 1], dtype='float16'),
            # parameter_214
            paddle.static.InputSpec(shape=[257], dtype='float16'),
            # parameter_215
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_216
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_218
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_217
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_219
            paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float16'),
            # parameter_220
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_221
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float16'),
            # parameter_222
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_224
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_223
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_225
            paddle.static.InputSpec(shape=[257, 128, 1, 1], dtype='float16'),
            # parameter_226
            paddle.static.InputSpec(shape=[257], dtype='float16'),
            # parameter_227
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_228
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_230
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_229
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_231
            paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float16'),
            # parameter_232
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_233
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float16'),
            # parameter_234
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_236
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_235
            paddle.static.InputSpec(shape=[128], dtype='float16'),
            # parameter_237
            paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float16'),
            # parameter_241
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_238
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_239
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[256, 1000], dtype='float16'),
            # parameter_243
            paddle.static.InputSpec(shape=[1000], dtype='float16'),
            # feed_0
            paddle.static.InputSpec(shape=[None, 3, 256, 256], dtype='float32'),
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