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
    return [618][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_904_0_0(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_87, parameter_86, parameter_88, parameter_89, parameter_90, parameter_91, parameter_93, parameter_92, parameter_94, parameter_95, parameter_96, parameter_97, parameter_99, parameter_98, parameter_100, parameter_101, parameter_102, parameter_103, parameter_105, parameter_104, parameter_106, parameter_107, parameter_108, parameter_109, parameter_111, parameter_110, parameter_112, parameter_116, parameter_113, parameter_115, parameter_114, parameter_117, parameter_121, parameter_118, parameter_120, parameter_119, parameter_122, parameter_126, parameter_123, parameter_125, parameter_124, parameter_127, parameter_131, parameter_128, parameter_130, parameter_129, parameter_132, parameter_136, parameter_133, parameter_135, parameter_134, parameter_137, parameter_141, parameter_138, parameter_140, parameter_139, parameter_142, parameter_144, parameter_143, parameter_145, parameter_146, parameter_147, parameter_148, parameter_150, parameter_149, parameter_151, parameter_152, parameter_153, parameter_154, parameter_156, parameter_155, parameter_157, parameter_158, parameter_159, parameter_160, parameter_162, parameter_161, parameter_163, parameter_164, parameter_165, parameter_166, parameter_168, parameter_167, parameter_169, parameter_170, parameter_171, parameter_172, parameter_174, parameter_173, parameter_175, parameter_176, parameter_177, parameter_178, parameter_180, parameter_179, parameter_181, parameter_182, parameter_183, parameter_184, parameter_186, parameter_185, parameter_187, parameter_188, parameter_189, parameter_190, parameter_192, parameter_191, parameter_193, parameter_197, parameter_194, parameter_196, parameter_195, parameter_198, parameter_202, parameter_199, parameter_201, parameter_200, parameter_203, parameter_207, parameter_204, parameter_206, parameter_205, parameter_208, parameter_212, parameter_209, parameter_211, parameter_210, parameter_213, parameter_217, parameter_214, parameter_216, parameter_215, parameter_218, parameter_222, parameter_219, parameter_221, parameter_220, parameter_223, parameter_225, parameter_224, parameter_226, parameter_227, parameter_228, parameter_229, parameter_231, parameter_230, parameter_232, parameter_233, parameter_234, parameter_235, parameter_237, parameter_236, parameter_238, parameter_239, parameter_240, parameter_241, parameter_243, parameter_242, parameter_244, parameter_245, parameter_246, parameter_247, parameter_249, parameter_248, parameter_250, parameter_251, parameter_252, parameter_253, parameter_255, parameter_254, parameter_256, parameter_257, parameter_258, parameter_259, parameter_261, parameter_260, parameter_262, parameter_266, parameter_263, parameter_265, parameter_264, parameter_267, parameter_271, parameter_268, parameter_270, parameter_269, parameter_272, parameter_276, parameter_273, parameter_275, parameter_274, parameter_277, parameter_278, feed_0):

        # pd_op.conv2d: (-1x16x128x128xf32) <- (-1x3x256x256xf32, 16x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(feed_0, parameter_0, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x16x128x128xf32, 16xf32, 16xf32, xf32, xf32, None) <- (-1x16x128x128xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_0, parameter_1, parameter_2, parameter_3, parameter_4, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x16x128x128xf32) <- (-1x16x128x128xf32)
        silu_0 = paddle._C_ops.silu(batch_norm__0)

        # pd_op.conv2d: (-1x64x128x128xf32) <- (-1x16x128x128xf32, 64x16x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(silu_0, parameter_5, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x128x128xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x128x128xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_1, parameter_6, parameter_7, parameter_8, parameter_9, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x64x128x128xf32) <- (-1x64x128x128xf32)
        silu_1 = paddle._C_ops.silu(batch_norm__6)

        # pd_op.depthwise_conv2d: (-1x64x128x128xf32) <- (-1x64x128x128xf32, 64x1x3x3xf32)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(silu_1, parameter_10, [1, 1], [1, 1], 'EXPLICIT', 64, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x64x128x128xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x128x128xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_0, parameter_11, parameter_12, parameter_13, parameter_14, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x64x128x128xf32) <- (-1x64x128x128xf32)
        silu_2 = paddle._C_ops.silu(batch_norm__12)

        # pd_op.conv2d: (-1x32x128x128xf32) <- (-1x64x128x128xf32, 32x64x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(silu_2, parameter_15, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x128x128xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x128x128xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_2, parameter_16, parameter_17, parameter_18, parameter_19, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x128x128x128xf32) <- (-1x32x128x128xf32, 128x32x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(batch_norm__18, parameter_20, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x128x128xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x128x128xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_3, parameter_21, parameter_22, parameter_23, parameter_24, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x128x128x128xf32) <- (-1x128x128x128xf32)
        silu_3 = paddle._C_ops.silu(batch_norm__24)

        # pd_op.depthwise_conv2d: (-1x128x64x64xf32) <- (-1x128x128x128xf32, 128x1x3x3xf32)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(silu_3, parameter_25, [2, 2], [1, 1], 'EXPLICIT', 128, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x128x64x64xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_1, parameter_26, parameter_27, parameter_28, parameter_29, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x128x64x64xf32) <- (-1x128x64x64xf32)
        silu_4 = paddle._C_ops.silu(batch_norm__30)

        # pd_op.conv2d: (-1x64x64x64xf32) <- (-1x128x64x64xf32, 64x128x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(silu_4, parameter_30, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x64x64xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x64x64xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_4, parameter_31, parameter_32, parameter_33, parameter_34, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x256x64x64xf32) <- (-1x64x64x64xf32, 256x64x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(batch_norm__36, parameter_35, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x64x64xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x64x64xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_5, parameter_36, parameter_37, parameter_38, parameter_39, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x256x64x64xf32) <- (-1x256x64x64xf32)
        silu_5 = paddle._C_ops.silu(batch_norm__42)

        # pd_op.depthwise_conv2d: (-1x256x64x64xf32) <- (-1x256x64x64xf32, 256x1x3x3xf32)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(silu_5, parameter_40, [1, 1], [1, 1], 'EXPLICIT', 256, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x256x64x64xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x64x64xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_2, parameter_41, parameter_42, parameter_43, parameter_44, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x256x64x64xf32) <- (-1x256x64x64xf32)
        silu_6 = paddle._C_ops.silu(batch_norm__48)

        # pd_op.conv2d: (-1x64x64x64xf32) <- (-1x256x64x64xf32, 64x256x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(silu_6, parameter_45, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x64x64xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x64x64xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_6, parameter_46, parameter_47, parameter_48, parameter_49, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x64x64x64xf32) <- (-1x64x64x64xf32, -1x64x64x64xf32)
        add__0 = paddle._C_ops.add_(batch_norm__36, batch_norm__54)

        # pd_op.conv2d: (-1x256x64x64xf32) <- (-1x64x64x64xf32, 256x64x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(add__0, parameter_50, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x64x64xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x64x64xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_7, parameter_51, parameter_52, parameter_53, parameter_54, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x256x64x64xf32) <- (-1x256x64x64xf32)
        silu_7 = paddle._C_ops.silu(batch_norm__60)

        # pd_op.depthwise_conv2d: (-1x256x64x64xf32) <- (-1x256x64x64xf32, 256x1x3x3xf32)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(silu_7, parameter_55, [1, 1], [1, 1], 'EXPLICIT', 256, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x256x64x64xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x64x64xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_3, parameter_56, parameter_57, parameter_58, parameter_59, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x256x64x64xf32) <- (-1x256x64x64xf32)
        silu_8 = paddle._C_ops.silu(batch_norm__66)

        # pd_op.conv2d: (-1x64x64x64xf32) <- (-1x256x64x64xf32, 64x256x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(silu_8, parameter_60, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x64x64xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x64x64xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_8, parameter_61, parameter_62, parameter_63, parameter_64, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x64x64x64xf32) <- (-1x64x64x64xf32, -1x64x64x64xf32)
        add__1 = paddle._C_ops.add_(add__0, batch_norm__72)

        # pd_op.conv2d: (-1x256x64x64xf32) <- (-1x64x64x64xf32, 256x64x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(add__1, parameter_65, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x64x64xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x64x64xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_9, parameter_66, parameter_67, parameter_68, parameter_69, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x256x64x64xf32) <- (-1x256x64x64xf32)
        silu_9 = paddle._C_ops.silu(batch_norm__78)

        # pd_op.depthwise_conv2d: (-1x256x32x32xf32) <- (-1x256x64x64xf32, 256x1x3x3xf32)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(silu_9, parameter_70, [2, 2], [1, 1], 'EXPLICIT', 256, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x256x32x32xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_4, parameter_71, parameter_72, parameter_73, parameter_74, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x256x32x32xf32) <- (-1x256x32x32xf32)
        silu_10 = paddle._C_ops.silu(batch_norm__84)

        # pd_op.conv2d: (-1x96x32x32xf32) <- (-1x256x32x32xf32, 96x256x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(silu_10, parameter_75, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x32x32xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x32x32xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_10, parameter_76, parameter_77, parameter_78, parameter_79, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x96x32x32xf32) <- (-1x96x32x32xf32, 96x96x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(batch_norm__90, parameter_80, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x32x32xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x32x32xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_11, parameter_81, parameter_82, parameter_83, parameter_84, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x96x32x32xf32) <- (-1x96x32x32xf32)
        silu_11 = paddle._C_ops.silu(batch_norm__96)

        # pd_op.conv2d: (-1x144x32x32xf32) <- (-1x96x32x32xf32, 144x96x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(silu_11, parameter_85, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [-1, 2, 16, 2]

        # pd_op.reshape_: (-1x2x16x2xf32, 0x-1x144x32x32xf32) <- (-1x144x32x32xf32, 4xi64)
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(conv2d_12, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x16x2x2xf32) <- (-1x2x16x2xf32)
        transpose_0 = paddle._C_ops.transpose(reshape__0, [0, 2, 1, 3])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [-1, 144, 256, 4]

        # pd_op.reshape_: (-1x144x256x4xf32, 0x-1x16x2x2xf32) <- (-1x16x2x2xf32, 4xi64)
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_0, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x256x144xf32) <- (-1x144x256x4xf32)
        transpose_1 = paddle._C_ops.transpose(reshape__2, [0, 3, 2, 1])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_2 = [-1, 256, 144]

        # pd_op.reshape_: (-1x256x144xf32, 0x-1x4x256x144xf32) <- (-1x4x256x144xf32, 3xi64)
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_1, full_int_array_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.layer_norm: (-1x256x144xf32, -256xf32, -256xf32) <- (-1x256x144xf32, 144xf32, 144xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(reshape__4, parameter_86, parameter_87, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x256x144xf32)
        shape_0 = paddle._C_ops.shape(layer_norm_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], full_int_array_3, full_int_array_4, [1], [0])

        # pd_op.matmul: (-1x256x432xf32) <- (-1x256x144xf32, 144x432xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_0, parameter_88, False, False)

        # pd_op.add_: (-1x256x432xf32) <- (-1x256x432xf32, 432xf32)
        add__2 = paddle._C_ops.add_(matmul_0, parameter_89)

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full([1], float('36'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_0 = [slice_0, full_0, full_1, full_2, full_3]

        # pd_op.reshape_: (-1x256x3x4x36xf32, 0x-1x256x432xf32) <- (-1x256x432xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__2, [x.reshape([]) for x in combine_0]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x3x256x36xf32) <- (-1x256x3x4x36xf32)
        transpose_2 = paddle._C_ops.transpose(reshape__6, [0, 3, 2, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [1]

        # pd_op.slice: (-1x4x256x36xf32) <- (-1x4x3x256x36xf32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(transpose_2, [2], full_int_array_5, full_int_array_6, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [2]

        # pd_op.slice: (-1x4x256x36xf32) <- (-1x4x3x256x36xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(transpose_2, [2], full_int_array_7, full_int_array_8, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_10 = [3]

        # pd_op.slice: (-1x4x256x36xf32) <- (-1x4x3x256x36xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(transpose_2, [2], full_int_array_9, full_int_array_10, [1], [2])

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('0.166667'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x4x256x36xf32) <- (-1x4x256x36xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(slice_1, full_4, float('0'), True)

        # pd_op.transpose: (-1x4x36x256xf32) <- (-1x4x256x36xf32)
        transpose_3 = paddle._C_ops.transpose(slice_2, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x256x256xf32) <- (-1x4x256x36xf32, -1x4x36x256xf32)
        matmul_1 = paddle._C_ops.matmul(scale__0, transpose_3, False, False)

        # pd_op.softmax_: (-1x4x256x256xf32) <- (-1x4x256x256xf32)
        softmax__0 = paddle._C_ops.softmax_(matmul_1, -1)

        # pd_op.matmul: (-1x4x256x36xf32) <- (-1x4x256x256xf32, -1x4x256x36xf32)
        matmul_2 = paddle._C_ops.matmul(softmax__0, slice_3, False, False)

        # pd_op.transpose: (-1x256x4x36xf32) <- (-1x4x256x36xf32)
        transpose_4 = paddle._C_ops.transpose(matmul_2, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_1 = [slice_0, full_5, full_6]

        # pd_op.reshape_: (-1x256x144xf32, 0x-1x256x4x36xf32) <- (-1x256x4x36xf32, [1xi32, 1xi32, 1xi32])
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_4, [x.reshape([]) for x in combine_1]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x256x144xf32) <- (-1x256x144xf32, 144x144xf32)
        matmul_3 = paddle._C_ops.matmul(reshape__8, parameter_90, False, False)

        # pd_op.add_: (-1x256x144xf32) <- (-1x256x144xf32, 144xf32)
        add__3 = paddle._C_ops.add_(matmul_3, parameter_91)

        # pd_op.full: (1xf32) <- ()
        full_7 = paddle._C_ops.full([1], float('0.1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.dropout: (-1x256x144xf32, None) <- (-1x256x144xf32, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__3, None, full_7, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x144xf32) <- (-1x256x144xf32, -1x256x144xf32)
        add__4 = paddle._C_ops.add_(reshape__4, dropout_0)

        # pd_op.layer_norm: (-1x256x144xf32, -256xf32, -256xf32) <- (-1x256x144xf32, 144xf32, 144xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__4, parameter_92, parameter_93, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x256x288xf32) <- (-1x256x144xf32, 144x288xf32)
        matmul_4 = paddle._C_ops.matmul(layer_norm_3, parameter_94, False, False)

        # pd_op.add_: (-1x256x288xf32) <- (-1x256x288xf32, 288xf32)
        add__5 = paddle._C_ops.add_(matmul_4, parameter_95)

        # pd_op.silu: (-1x256x288xf32) <- (-1x256x288xf32)
        silu_12 = paddle._C_ops.silu(add__5)

        # pd_op.full: (1xf32) <- ()
        full_8 = paddle._C_ops.full([1], float('0.1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.dropout: (-1x256x288xf32, None) <- (-1x256x288xf32, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(paddle._C_ops.dropout(silu_12, None, full_8, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x256x144xf32) <- (-1x256x288xf32, 288x144xf32)
        matmul_5 = paddle._C_ops.matmul(dropout_2, parameter_96, False, False)

        # pd_op.add_: (-1x256x144xf32) <- (-1x256x144xf32, 144xf32)
        add__6 = paddle._C_ops.add_(matmul_5, parameter_97)

        # pd_op.full: (1xf32) <- ()
        full_9 = paddle._C_ops.full([1], float('0.1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.dropout: (-1x256x144xf32, None) <- (-1x256x144xf32, None, 1xf32)
        dropout_4, dropout_5 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__6, None, full_9, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x144xf32) <- (-1x256x144xf32, -1x256x144xf32)
        add__7 = paddle._C_ops.add_(dropout_4, add__4)

        # pd_op.layer_norm: (-1x256x144xf32, -256xf32, -256xf32) <- (-1x256x144xf32, 144xf32, 144xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__7, parameter_98, parameter_99, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x256x144xf32)
        shape_1 = paddle._C_ops.shape(layer_norm_6)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_11 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_12 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(shape_1, [0], full_int_array_11, full_int_array_12, [1], [0])

        # pd_op.matmul: (-1x256x432xf32) <- (-1x256x144xf32, 144x432xf32)
        matmul_6 = paddle._C_ops.matmul(layer_norm_6, parameter_100, False, False)

        # pd_op.add_: (-1x256x432xf32) <- (-1x256x432xf32, 432xf32)
        add__8 = paddle._C_ops.add_(matmul_6, parameter_101)

        # pd_op.full: (1xi32) <- ()
        full_10 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_11 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_12 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_13 = paddle._C_ops.full([1], float('36'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_2 = [slice_4, full_10, full_11, full_12, full_13]

        # pd_op.reshape_: (-1x256x3x4x36xf32, 0x-1x256x432xf32) <- (-1x256x432xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__8, [x.reshape([]) for x in combine_2]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x3x256x36xf32) <- (-1x256x3x4x36xf32)
        transpose_5 = paddle._C_ops.transpose(reshape__10, [0, 3, 2, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_13 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_14 = [1]

        # pd_op.slice: (-1x4x256x36xf32) <- (-1x4x3x256x36xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(transpose_5, [2], full_int_array_13, full_int_array_14, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_15 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_16 = [2]

        # pd_op.slice: (-1x4x256x36xf32) <- (-1x4x3x256x36xf32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(transpose_5, [2], full_int_array_15, full_int_array_16, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_17 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_18 = [3]

        # pd_op.slice: (-1x4x256x36xf32) <- (-1x4x3x256x36xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(transpose_5, [2], full_int_array_17, full_int_array_18, [1], [2])

        # pd_op.full: (1xf32) <- ()
        full_14 = paddle._C_ops.full([1], float('0.166667'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x4x256x36xf32) <- (-1x4x256x36xf32, 1xf32)
        scale__1 = paddle._C_ops.scale_(slice_5, full_14, float('0'), True)

        # pd_op.transpose: (-1x4x36x256xf32) <- (-1x4x256x36xf32)
        transpose_6 = paddle._C_ops.transpose(slice_6, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x256x256xf32) <- (-1x4x256x36xf32, -1x4x36x256xf32)
        matmul_7 = paddle._C_ops.matmul(scale__1, transpose_6, False, False)

        # pd_op.softmax_: (-1x4x256x256xf32) <- (-1x4x256x256xf32)
        softmax__1 = paddle._C_ops.softmax_(matmul_7, -1)

        # pd_op.matmul: (-1x4x256x36xf32) <- (-1x4x256x256xf32, -1x4x256x36xf32)
        matmul_8 = paddle._C_ops.matmul(softmax__1, slice_7, False, False)

        # pd_op.transpose: (-1x256x4x36xf32) <- (-1x4x256x36xf32)
        transpose_7 = paddle._C_ops.transpose(matmul_8, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_15 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_16 = paddle._C_ops.full([1], float('144'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_3 = [slice_4, full_15, full_16]

        # pd_op.reshape_: (-1x256x144xf32, 0x-1x256x4x36xf32) <- (-1x256x4x36xf32, [1xi32, 1xi32, 1xi32])
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_7, [x.reshape([]) for x in combine_3]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x256x144xf32) <- (-1x256x144xf32, 144x144xf32)
        matmul_9 = paddle._C_ops.matmul(reshape__12, parameter_102, False, False)

        # pd_op.add_: (-1x256x144xf32) <- (-1x256x144xf32, 144xf32)
        add__9 = paddle._C_ops.add_(matmul_9, parameter_103)

        # pd_op.full: (1xf32) <- ()
        full_17 = paddle._C_ops.full([1], float('0.1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.dropout: (-1x256x144xf32, None) <- (-1x256x144xf32, None, 1xf32)
        dropout_6, dropout_7 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__9, None, full_17, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x144xf32) <- (-1x256x144xf32, -1x256x144xf32)
        add__10 = paddle._C_ops.add_(add__7, dropout_6)

        # pd_op.layer_norm: (-1x256x144xf32, -256xf32, -256xf32) <- (-1x256x144xf32, 144xf32, 144xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__10, parameter_104, parameter_105, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x256x288xf32) <- (-1x256x144xf32, 144x288xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_9, parameter_106, False, False)

        # pd_op.add_: (-1x256x288xf32) <- (-1x256x288xf32, 288xf32)
        add__11 = paddle._C_ops.add_(matmul_10, parameter_107)

        # pd_op.silu: (-1x256x288xf32) <- (-1x256x288xf32)
        silu_13 = paddle._C_ops.silu(add__11)

        # pd_op.full: (1xf32) <- ()
        full_18 = paddle._C_ops.full([1], float('0.1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.dropout: (-1x256x288xf32, None) <- (-1x256x288xf32, None, 1xf32)
        dropout_8, dropout_9 = (lambda x, f: f(x))(paddle._C_ops.dropout(silu_13, None, full_18, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x256x144xf32) <- (-1x256x288xf32, 288x144xf32)
        matmul_11 = paddle._C_ops.matmul(dropout_8, parameter_108, False, False)

        # pd_op.add_: (-1x256x144xf32) <- (-1x256x144xf32, 144xf32)
        add__12 = paddle._C_ops.add_(matmul_11, parameter_109)

        # pd_op.full: (1xf32) <- ()
        full_19 = paddle._C_ops.full([1], float('0.1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.dropout: (-1x256x144xf32, None) <- (-1x256x144xf32, None, 1xf32)
        dropout_10, dropout_11 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__12, None, full_19, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x144xf32) <- (-1x256x144xf32, -1x256x144xf32)
        add__13 = paddle._C_ops.add_(dropout_10, add__10)

        # pd_op.layer_norm: (-1x256x144xf32, -256xf32, -256xf32) <- (-1x256x144xf32, 144xf32, 144xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__13, parameter_110, parameter_111, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_19 = [-1, 4, 256, 144]

        # pd_op.reshape_: (-1x4x256x144xf32, 0x-1x256x144xf32) <- (-1x256x144xf32, 4xi64)
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_12, full_int_array_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x144x256x4xf32) <- (-1x4x256x144xf32)
        transpose_8 = paddle._C_ops.transpose(reshape__14, [0, 3, 2, 1])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_20 = [-1, 16, 2, 2]

        # pd_op.reshape_: (-1x16x2x2xf32, 0x-1x144x256x4xf32) <- (-1x144x256x4xf32, 4xi64)
        reshape__16, reshape__17 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_8, full_int_array_20), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x16x2xf32) <- (-1x16x2x2xf32)
        transpose_9 = paddle._C_ops.transpose(reshape__16, [0, 2, 1, 3])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_21 = [-1, 144, 32, 32]

        # pd_op.reshape_: (-1x144x32x32xf32, 0x-1x2x16x2xf32) <- (-1x2x16x2xf32, 4xi64)
        reshape__18, reshape__19 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_9, full_int_array_21), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x96x32x32xf32) <- (-1x144x32x32xf32, 96x144x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(reshape__18, parameter_112, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x32x32xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x32x32xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_13, parameter_113, parameter_114, parameter_115, parameter_116, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x96x32x32xf32) <- (-1x96x32x32xf32)
        silu_14 = paddle._C_ops.silu(batch_norm__102)

        # builtin.combine: ([-1x96x32x32xf32, -1x96x32x32xf32]) <- (-1x96x32x32xf32, -1x96x32x32xf32)
        combine_4 = [batch_norm__90, silu_14]

        # pd_op.full: (1xi32) <- ()
        full_20 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x192x32x32xf32) <- ([-1x96x32x32xf32, -1x96x32x32xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_4, full_20)

        # pd_op.conv2d: (-1x96x32x32xf32) <- (-1x192x32x32xf32, 96x192x3x3xf32)
        conv2d_14 = paddle._C_ops.conv2d(concat_0, parameter_117, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x32x32xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x32x32xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_14, parameter_118, parameter_119, parameter_120, parameter_121, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x96x32x32xf32) <- (-1x96x32x32xf32)
        silu_15 = paddle._C_ops.silu(batch_norm__108)

        # pd_op.conv2d: (-1x384x32x32xf32) <- (-1x96x32x32xf32, 384x96x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(silu_15, parameter_122, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x384x32x32xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x32x32xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_15, parameter_123, parameter_124, parameter_125, parameter_126, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x384x32x32xf32) <- (-1x384x32x32xf32)
        silu_16 = paddle._C_ops.silu(batch_norm__114)

        # pd_op.depthwise_conv2d: (-1x384x16x16xf32) <- (-1x384x32x32xf32, 384x1x3x3xf32)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(silu_16, parameter_127, [2, 2], [1, 1], 'EXPLICIT', 384, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x384x16x16xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x16x16xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_5, parameter_128, parameter_129, parameter_130, parameter_131, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x384x16x16xf32) <- (-1x384x16x16xf32)
        silu_17 = paddle._C_ops.silu(batch_norm__120)

        # pd_op.conv2d: (-1x128x16x16xf32) <- (-1x384x16x16xf32, 128x384x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(silu_17, parameter_132, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x16x16xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x16x16xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_16, parameter_133, parameter_134, parameter_135, parameter_136, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x128x16x16xf32) <- (-1x128x16x16xf32, 128x128x3x3xf32)
        conv2d_17 = paddle._C_ops.conv2d(batch_norm__126, parameter_137, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x16x16xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x16x16xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_17, parameter_138, parameter_139, parameter_140, parameter_141, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x128x16x16xf32) <- (-1x128x16x16xf32)
        silu_18 = paddle._C_ops.silu(batch_norm__132)

        # pd_op.conv2d: (-1x192x16x16xf32) <- (-1x128x16x16xf32, 192x128x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(silu_18, parameter_142, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_22 = [-1, 2, 8, 2]

        # pd_op.reshape_: (-1x2x8x2xf32, 0x-1x192x16x16xf32) <- (-1x192x16x16xf32, 4xi64)
        reshape__20, reshape__21 = (lambda x, f: f(x))(paddle._C_ops.reshape_(conv2d_18, full_int_array_22), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x8x2x2xf32) <- (-1x2x8x2xf32)
        transpose_10 = paddle._C_ops.transpose(reshape__20, [0, 2, 1, 3])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_23 = [-1, 192, 64, 4]

        # pd_op.reshape_: (-1x192x64x4xf32, 0x-1x8x2x2xf32) <- (-1x8x2x2xf32, 4xi64)
        reshape__22, reshape__23 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_10, full_int_array_23), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x64x192xf32) <- (-1x192x64x4xf32)
        transpose_11 = paddle._C_ops.transpose(reshape__22, [0, 3, 2, 1])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_24 = [-1, 64, 192]

        # pd_op.reshape_: (-1x64x192xf32, 0x-1x4x64x192xf32) <- (-1x4x64x192xf32, 3xi64)
        reshape__24, reshape__25 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_11, full_int_array_24), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.layer_norm: (-1x64x192xf32, -64xf32, -64xf32) <- (-1x64x192xf32, 192xf32, 192xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(reshape__24, parameter_143, parameter_144, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x64x192xf32)
        shape_2 = paddle._C_ops.shape(layer_norm_15)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_25 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_26 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(shape_2, [0], full_int_array_25, full_int_array_26, [1], [0])

        # pd_op.matmul: (-1x64x576xf32) <- (-1x64x192xf32, 192x576xf32)
        matmul_12 = paddle._C_ops.matmul(layer_norm_15, parameter_145, False, False)

        # pd_op.add_: (-1x64x576xf32) <- (-1x64x576xf32, 576xf32)
        add__14 = paddle._C_ops.add_(matmul_12, parameter_146)

        # pd_op.full: (1xi32) <- ()
        full_21 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_22 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_23 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_24 = paddle._C_ops.full([1], float('48'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_5 = [slice_8, full_21, full_22, full_23, full_24]

        # pd_op.reshape_: (-1x64x3x4x48xf32, 0x-1x64x576xf32) <- (-1x64x576xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__26, reshape__27 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__14, [x.reshape([]) for x in combine_5]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x3x64x48xf32) <- (-1x64x3x4x48xf32)
        transpose_12 = paddle._C_ops.transpose(reshape__26, [0, 3, 2, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_27 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_28 = [1]

        # pd_op.slice: (-1x4x64x48xf32) <- (-1x4x3x64x48xf32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(transpose_12, [2], full_int_array_27, full_int_array_28, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_29 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_30 = [2]

        # pd_op.slice: (-1x4x64x48xf32) <- (-1x4x3x64x48xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(transpose_12, [2], full_int_array_29, full_int_array_30, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_31 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_32 = [3]

        # pd_op.slice: (-1x4x64x48xf32) <- (-1x4x3x64x48xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(transpose_12, [2], full_int_array_31, full_int_array_32, [1], [2])

        # pd_op.full: (1xf32) <- ()
        full_25 = paddle._C_ops.full([1], float('0.144338'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x4x64x48xf32) <- (-1x4x64x48xf32, 1xf32)
        scale__2 = paddle._C_ops.scale_(slice_9, full_25, float('0'), True)

        # pd_op.transpose: (-1x4x48x64xf32) <- (-1x4x64x48xf32)
        transpose_13 = paddle._C_ops.transpose(slice_10, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x64x64xf32) <- (-1x4x64x48xf32, -1x4x48x64xf32)
        matmul_13 = paddle._C_ops.matmul(scale__2, transpose_13, False, False)

        # pd_op.softmax_: (-1x4x64x64xf32) <- (-1x4x64x64xf32)
        softmax__2 = paddle._C_ops.softmax_(matmul_13, -1)

        # pd_op.matmul: (-1x4x64x48xf32) <- (-1x4x64x64xf32, -1x4x64x48xf32)
        matmul_14 = paddle._C_ops.matmul(softmax__2, slice_11, False, False)

        # pd_op.transpose: (-1x64x4x48xf32) <- (-1x4x64x48xf32)
        transpose_14 = paddle._C_ops.transpose(matmul_14, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_26 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_27 = paddle._C_ops.full([1], float('192'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_6 = [slice_8, full_26, full_27]

        # pd_op.reshape_: (-1x64x192xf32, 0x-1x64x4x48xf32) <- (-1x64x4x48xf32, [1xi32, 1xi32, 1xi32])
        reshape__28, reshape__29 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_14, [x.reshape([]) for x in combine_6]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64x192xf32) <- (-1x64x192xf32, 192x192xf32)
        matmul_15 = paddle._C_ops.matmul(reshape__28, parameter_147, False, False)

        # pd_op.add_: (-1x64x192xf32) <- (-1x64x192xf32, 192xf32)
        add__15 = paddle._C_ops.add_(matmul_15, parameter_148)

        # pd_op.full: (1xf32) <- ()
        full_28 = paddle._C_ops.full([1], float('0.1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.dropout: (-1x64x192xf32, None) <- (-1x64x192xf32, None, 1xf32)
        dropout_12, dropout_13 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__15, None, full_28, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x192xf32) <- (-1x64x192xf32, -1x64x192xf32)
        add__16 = paddle._C_ops.add_(reshape__24, dropout_12)

        # pd_op.layer_norm: (-1x64x192xf32, -64xf32, -64xf32) <- (-1x64x192xf32, 192xf32, 192xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__16, parameter_149, parameter_150, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x64x384xf32) <- (-1x64x192xf32, 192x384xf32)
        matmul_16 = paddle._C_ops.matmul(layer_norm_18, parameter_151, False, False)

        # pd_op.add_: (-1x64x384xf32) <- (-1x64x384xf32, 384xf32)
        add__17 = paddle._C_ops.add_(matmul_16, parameter_152)

        # pd_op.silu: (-1x64x384xf32) <- (-1x64x384xf32)
        silu_19 = paddle._C_ops.silu(add__17)

        # pd_op.full: (1xf32) <- ()
        full_29 = paddle._C_ops.full([1], float('0.1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.dropout: (-1x64x384xf32, None) <- (-1x64x384xf32, None, 1xf32)
        dropout_14, dropout_15 = (lambda x, f: f(x))(paddle._C_ops.dropout(silu_19, None, full_29, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64x192xf32) <- (-1x64x384xf32, 384x192xf32)
        matmul_17 = paddle._C_ops.matmul(dropout_14, parameter_153, False, False)

        # pd_op.add_: (-1x64x192xf32) <- (-1x64x192xf32, 192xf32)
        add__18 = paddle._C_ops.add_(matmul_17, parameter_154)

        # pd_op.full: (1xf32) <- ()
        full_30 = paddle._C_ops.full([1], float('0.1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.dropout: (-1x64x192xf32, None) <- (-1x64x192xf32, None, 1xf32)
        dropout_16, dropout_17 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__18, None, full_30, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x192xf32) <- (-1x64x192xf32, -1x64x192xf32)
        add__19 = paddle._C_ops.add_(dropout_16, add__16)

        # pd_op.layer_norm: (-1x64x192xf32, -64xf32, -64xf32) <- (-1x64x192xf32, 192xf32, 192xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__19, parameter_155, parameter_156, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x64x192xf32)
        shape_3 = paddle._C_ops.shape(layer_norm_21)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_33 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_34 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(shape_3, [0], full_int_array_33, full_int_array_34, [1], [0])

        # pd_op.matmul: (-1x64x576xf32) <- (-1x64x192xf32, 192x576xf32)
        matmul_18 = paddle._C_ops.matmul(layer_norm_21, parameter_157, False, False)

        # pd_op.add_: (-1x64x576xf32) <- (-1x64x576xf32, 576xf32)
        add__20 = paddle._C_ops.add_(matmul_18, parameter_158)

        # pd_op.full: (1xi32) <- ()
        full_31 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_32 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_33 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_34 = paddle._C_ops.full([1], float('48'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_7 = [slice_12, full_31, full_32, full_33, full_34]

        # pd_op.reshape_: (-1x64x3x4x48xf32, 0x-1x64x576xf32) <- (-1x64x576xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__30, reshape__31 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__20, [x.reshape([]) for x in combine_7]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x3x64x48xf32) <- (-1x64x3x4x48xf32)
        transpose_15 = paddle._C_ops.transpose(reshape__30, [0, 3, 2, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_35 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_36 = [1]

        # pd_op.slice: (-1x4x64x48xf32) <- (-1x4x3x64x48xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(transpose_15, [2], full_int_array_35, full_int_array_36, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_37 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_38 = [2]

        # pd_op.slice: (-1x4x64x48xf32) <- (-1x4x3x64x48xf32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(transpose_15, [2], full_int_array_37, full_int_array_38, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_39 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_40 = [3]

        # pd_op.slice: (-1x4x64x48xf32) <- (-1x4x3x64x48xf32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(transpose_15, [2], full_int_array_39, full_int_array_40, [1], [2])

        # pd_op.full: (1xf32) <- ()
        full_35 = paddle._C_ops.full([1], float('0.144338'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x4x64x48xf32) <- (-1x4x64x48xf32, 1xf32)
        scale__3 = paddle._C_ops.scale_(slice_13, full_35, float('0'), True)

        # pd_op.transpose: (-1x4x48x64xf32) <- (-1x4x64x48xf32)
        transpose_16 = paddle._C_ops.transpose(slice_14, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x64x64xf32) <- (-1x4x64x48xf32, -1x4x48x64xf32)
        matmul_19 = paddle._C_ops.matmul(scale__3, transpose_16, False, False)

        # pd_op.softmax_: (-1x4x64x64xf32) <- (-1x4x64x64xf32)
        softmax__3 = paddle._C_ops.softmax_(matmul_19, -1)

        # pd_op.matmul: (-1x4x64x48xf32) <- (-1x4x64x64xf32, -1x4x64x48xf32)
        matmul_20 = paddle._C_ops.matmul(softmax__3, slice_15, False, False)

        # pd_op.transpose: (-1x64x4x48xf32) <- (-1x4x64x48xf32)
        transpose_17 = paddle._C_ops.transpose(matmul_20, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_36 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_37 = paddle._C_ops.full([1], float('192'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_8 = [slice_12, full_36, full_37]

        # pd_op.reshape_: (-1x64x192xf32, 0x-1x64x4x48xf32) <- (-1x64x4x48xf32, [1xi32, 1xi32, 1xi32])
        reshape__32, reshape__33 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_17, [x.reshape([]) for x in combine_8]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64x192xf32) <- (-1x64x192xf32, 192x192xf32)
        matmul_21 = paddle._C_ops.matmul(reshape__32, parameter_159, False, False)

        # pd_op.add_: (-1x64x192xf32) <- (-1x64x192xf32, 192xf32)
        add__21 = paddle._C_ops.add_(matmul_21, parameter_160)

        # pd_op.full: (1xf32) <- ()
        full_38 = paddle._C_ops.full([1], float('0.1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.dropout: (-1x64x192xf32, None) <- (-1x64x192xf32, None, 1xf32)
        dropout_18, dropout_19 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__21, None, full_38, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x192xf32) <- (-1x64x192xf32, -1x64x192xf32)
        add__22 = paddle._C_ops.add_(add__19, dropout_18)

        # pd_op.layer_norm: (-1x64x192xf32, -64xf32, -64xf32) <- (-1x64x192xf32, 192xf32, 192xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__22, parameter_161, parameter_162, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x64x384xf32) <- (-1x64x192xf32, 192x384xf32)
        matmul_22 = paddle._C_ops.matmul(layer_norm_24, parameter_163, False, False)

        # pd_op.add_: (-1x64x384xf32) <- (-1x64x384xf32, 384xf32)
        add__23 = paddle._C_ops.add_(matmul_22, parameter_164)

        # pd_op.silu: (-1x64x384xf32) <- (-1x64x384xf32)
        silu_20 = paddle._C_ops.silu(add__23)

        # pd_op.full: (1xf32) <- ()
        full_39 = paddle._C_ops.full([1], float('0.1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.dropout: (-1x64x384xf32, None) <- (-1x64x384xf32, None, 1xf32)
        dropout_20, dropout_21 = (lambda x, f: f(x))(paddle._C_ops.dropout(silu_20, None, full_39, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64x192xf32) <- (-1x64x384xf32, 384x192xf32)
        matmul_23 = paddle._C_ops.matmul(dropout_20, parameter_165, False, False)

        # pd_op.add_: (-1x64x192xf32) <- (-1x64x192xf32, 192xf32)
        add__24 = paddle._C_ops.add_(matmul_23, parameter_166)

        # pd_op.full: (1xf32) <- ()
        full_40 = paddle._C_ops.full([1], float('0.1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.dropout: (-1x64x192xf32, None) <- (-1x64x192xf32, None, 1xf32)
        dropout_22, dropout_23 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__24, None, full_40, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x192xf32) <- (-1x64x192xf32, -1x64x192xf32)
        add__25 = paddle._C_ops.add_(dropout_22, add__22)

        # pd_op.layer_norm: (-1x64x192xf32, -64xf32, -64xf32) <- (-1x64x192xf32, 192xf32, 192xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__25, parameter_167, parameter_168, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x64x192xf32)
        shape_4 = paddle._C_ops.shape(layer_norm_27)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_41 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_42 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(shape_4, [0], full_int_array_41, full_int_array_42, [1], [0])

        # pd_op.matmul: (-1x64x576xf32) <- (-1x64x192xf32, 192x576xf32)
        matmul_24 = paddle._C_ops.matmul(layer_norm_27, parameter_169, False, False)

        # pd_op.add_: (-1x64x576xf32) <- (-1x64x576xf32, 576xf32)
        add__26 = paddle._C_ops.add_(matmul_24, parameter_170)

        # pd_op.full: (1xi32) <- ()
        full_41 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_42 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_43 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_44 = paddle._C_ops.full([1], float('48'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_9 = [slice_16, full_41, full_42, full_43, full_44]

        # pd_op.reshape_: (-1x64x3x4x48xf32, 0x-1x64x576xf32) <- (-1x64x576xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__34, reshape__35 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__26, [x.reshape([]) for x in combine_9]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x3x64x48xf32) <- (-1x64x3x4x48xf32)
        transpose_18 = paddle._C_ops.transpose(reshape__34, [0, 3, 2, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_43 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_44 = [1]

        # pd_op.slice: (-1x4x64x48xf32) <- (-1x4x3x64x48xf32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(transpose_18, [2], full_int_array_43, full_int_array_44, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_45 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_46 = [2]

        # pd_op.slice: (-1x4x64x48xf32) <- (-1x4x3x64x48xf32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(transpose_18, [2], full_int_array_45, full_int_array_46, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_47 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_48 = [3]

        # pd_op.slice: (-1x4x64x48xf32) <- (-1x4x3x64x48xf32, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(transpose_18, [2], full_int_array_47, full_int_array_48, [1], [2])

        # pd_op.full: (1xf32) <- ()
        full_45 = paddle._C_ops.full([1], float('0.144338'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x4x64x48xf32) <- (-1x4x64x48xf32, 1xf32)
        scale__4 = paddle._C_ops.scale_(slice_17, full_45, float('0'), True)

        # pd_op.transpose: (-1x4x48x64xf32) <- (-1x4x64x48xf32)
        transpose_19 = paddle._C_ops.transpose(slice_18, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x64x64xf32) <- (-1x4x64x48xf32, -1x4x48x64xf32)
        matmul_25 = paddle._C_ops.matmul(scale__4, transpose_19, False, False)

        # pd_op.softmax_: (-1x4x64x64xf32) <- (-1x4x64x64xf32)
        softmax__4 = paddle._C_ops.softmax_(matmul_25, -1)

        # pd_op.matmul: (-1x4x64x48xf32) <- (-1x4x64x64xf32, -1x4x64x48xf32)
        matmul_26 = paddle._C_ops.matmul(softmax__4, slice_19, False, False)

        # pd_op.transpose: (-1x64x4x48xf32) <- (-1x4x64x48xf32)
        transpose_20 = paddle._C_ops.transpose(matmul_26, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_46 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_47 = paddle._C_ops.full([1], float('192'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_10 = [slice_16, full_46, full_47]

        # pd_op.reshape_: (-1x64x192xf32, 0x-1x64x4x48xf32) <- (-1x64x4x48xf32, [1xi32, 1xi32, 1xi32])
        reshape__36, reshape__37 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_20, [x.reshape([]) for x in combine_10]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64x192xf32) <- (-1x64x192xf32, 192x192xf32)
        matmul_27 = paddle._C_ops.matmul(reshape__36, parameter_171, False, False)

        # pd_op.add_: (-1x64x192xf32) <- (-1x64x192xf32, 192xf32)
        add__27 = paddle._C_ops.add_(matmul_27, parameter_172)

        # pd_op.full: (1xf32) <- ()
        full_48 = paddle._C_ops.full([1], float('0.1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.dropout: (-1x64x192xf32, None) <- (-1x64x192xf32, None, 1xf32)
        dropout_24, dropout_25 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__27, None, full_48, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x192xf32) <- (-1x64x192xf32, -1x64x192xf32)
        add__28 = paddle._C_ops.add_(add__25, dropout_24)

        # pd_op.layer_norm: (-1x64x192xf32, -64xf32, -64xf32) <- (-1x64x192xf32, 192xf32, 192xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__28, parameter_173, parameter_174, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x64x384xf32) <- (-1x64x192xf32, 192x384xf32)
        matmul_28 = paddle._C_ops.matmul(layer_norm_30, parameter_175, False, False)

        # pd_op.add_: (-1x64x384xf32) <- (-1x64x384xf32, 384xf32)
        add__29 = paddle._C_ops.add_(matmul_28, parameter_176)

        # pd_op.silu: (-1x64x384xf32) <- (-1x64x384xf32)
        silu_21 = paddle._C_ops.silu(add__29)

        # pd_op.full: (1xf32) <- ()
        full_49 = paddle._C_ops.full([1], float('0.1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.dropout: (-1x64x384xf32, None) <- (-1x64x384xf32, None, 1xf32)
        dropout_26, dropout_27 = (lambda x, f: f(x))(paddle._C_ops.dropout(silu_21, None, full_49, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64x192xf32) <- (-1x64x384xf32, 384x192xf32)
        matmul_29 = paddle._C_ops.matmul(dropout_26, parameter_177, False, False)

        # pd_op.add_: (-1x64x192xf32) <- (-1x64x192xf32, 192xf32)
        add__30 = paddle._C_ops.add_(matmul_29, parameter_178)

        # pd_op.full: (1xf32) <- ()
        full_50 = paddle._C_ops.full([1], float('0.1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.dropout: (-1x64x192xf32, None) <- (-1x64x192xf32, None, 1xf32)
        dropout_28, dropout_29 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__30, None, full_50, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x192xf32) <- (-1x64x192xf32, -1x64x192xf32)
        add__31 = paddle._C_ops.add_(dropout_28, add__28)

        # pd_op.layer_norm: (-1x64x192xf32, -64xf32, -64xf32) <- (-1x64x192xf32, 192xf32, 192xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__31, parameter_179, parameter_180, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x64x192xf32)
        shape_5 = paddle._C_ops.shape(layer_norm_33)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_49 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_50 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(shape_5, [0], full_int_array_49, full_int_array_50, [1], [0])

        # pd_op.matmul: (-1x64x576xf32) <- (-1x64x192xf32, 192x576xf32)
        matmul_30 = paddle._C_ops.matmul(layer_norm_33, parameter_181, False, False)

        # pd_op.add_: (-1x64x576xf32) <- (-1x64x576xf32, 576xf32)
        add__32 = paddle._C_ops.add_(matmul_30, parameter_182)

        # pd_op.full: (1xi32) <- ()
        full_51 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_52 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_53 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_54 = paddle._C_ops.full([1], float('48'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_11 = [slice_20, full_51, full_52, full_53, full_54]

        # pd_op.reshape_: (-1x64x3x4x48xf32, 0x-1x64x576xf32) <- (-1x64x576xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__38, reshape__39 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__32, [x.reshape([]) for x in combine_11]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x3x64x48xf32) <- (-1x64x3x4x48xf32)
        transpose_21 = paddle._C_ops.transpose(reshape__38, [0, 3, 2, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_51 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_52 = [1]

        # pd_op.slice: (-1x4x64x48xf32) <- (-1x4x3x64x48xf32, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(transpose_21, [2], full_int_array_51, full_int_array_52, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_53 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_54 = [2]

        # pd_op.slice: (-1x4x64x48xf32) <- (-1x4x3x64x48xf32, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(transpose_21, [2], full_int_array_53, full_int_array_54, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_55 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_56 = [3]

        # pd_op.slice: (-1x4x64x48xf32) <- (-1x4x3x64x48xf32, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(transpose_21, [2], full_int_array_55, full_int_array_56, [1], [2])

        # pd_op.full: (1xf32) <- ()
        full_55 = paddle._C_ops.full([1], float('0.144338'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x4x64x48xf32) <- (-1x4x64x48xf32, 1xf32)
        scale__5 = paddle._C_ops.scale_(slice_21, full_55, float('0'), True)

        # pd_op.transpose: (-1x4x48x64xf32) <- (-1x4x64x48xf32)
        transpose_22 = paddle._C_ops.transpose(slice_22, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x64x64xf32) <- (-1x4x64x48xf32, -1x4x48x64xf32)
        matmul_31 = paddle._C_ops.matmul(scale__5, transpose_22, False, False)

        # pd_op.softmax_: (-1x4x64x64xf32) <- (-1x4x64x64xf32)
        softmax__5 = paddle._C_ops.softmax_(matmul_31, -1)

        # pd_op.matmul: (-1x4x64x48xf32) <- (-1x4x64x64xf32, -1x4x64x48xf32)
        matmul_32 = paddle._C_ops.matmul(softmax__5, slice_23, False, False)

        # pd_op.transpose: (-1x64x4x48xf32) <- (-1x4x64x48xf32)
        transpose_23 = paddle._C_ops.transpose(matmul_32, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_56 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_57 = paddle._C_ops.full([1], float('192'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_12 = [slice_20, full_56, full_57]

        # pd_op.reshape_: (-1x64x192xf32, 0x-1x64x4x48xf32) <- (-1x64x4x48xf32, [1xi32, 1xi32, 1xi32])
        reshape__40, reshape__41 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_23, [x.reshape([]) for x in combine_12]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64x192xf32) <- (-1x64x192xf32, 192x192xf32)
        matmul_33 = paddle._C_ops.matmul(reshape__40, parameter_183, False, False)

        # pd_op.add_: (-1x64x192xf32) <- (-1x64x192xf32, 192xf32)
        add__33 = paddle._C_ops.add_(matmul_33, parameter_184)

        # pd_op.full: (1xf32) <- ()
        full_58 = paddle._C_ops.full([1], float('0.1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.dropout: (-1x64x192xf32, None) <- (-1x64x192xf32, None, 1xf32)
        dropout_30, dropout_31 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__33, None, full_58, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x192xf32) <- (-1x64x192xf32, -1x64x192xf32)
        add__34 = paddle._C_ops.add_(add__31, dropout_30)

        # pd_op.layer_norm: (-1x64x192xf32, -64xf32, -64xf32) <- (-1x64x192xf32, 192xf32, 192xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__34, parameter_185, parameter_186, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x64x384xf32) <- (-1x64x192xf32, 192x384xf32)
        matmul_34 = paddle._C_ops.matmul(layer_norm_36, parameter_187, False, False)

        # pd_op.add_: (-1x64x384xf32) <- (-1x64x384xf32, 384xf32)
        add__35 = paddle._C_ops.add_(matmul_34, parameter_188)

        # pd_op.silu: (-1x64x384xf32) <- (-1x64x384xf32)
        silu_22 = paddle._C_ops.silu(add__35)

        # pd_op.full: (1xf32) <- ()
        full_59 = paddle._C_ops.full([1], float('0.1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.dropout: (-1x64x384xf32, None) <- (-1x64x384xf32, None, 1xf32)
        dropout_32, dropout_33 = (lambda x, f: f(x))(paddle._C_ops.dropout(silu_22, None, full_59, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x64x192xf32) <- (-1x64x384xf32, 384x192xf32)
        matmul_35 = paddle._C_ops.matmul(dropout_32, parameter_189, False, False)

        # pd_op.add_: (-1x64x192xf32) <- (-1x64x192xf32, 192xf32)
        add__36 = paddle._C_ops.add_(matmul_35, parameter_190)

        # pd_op.full: (1xf32) <- ()
        full_60 = paddle._C_ops.full([1], float('0.1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.dropout: (-1x64x192xf32, None) <- (-1x64x192xf32, None, 1xf32)
        dropout_34, dropout_35 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__36, None, full_60, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x64x192xf32) <- (-1x64x192xf32, -1x64x192xf32)
        add__37 = paddle._C_ops.add_(dropout_34, add__34)

        # pd_op.layer_norm: (-1x64x192xf32, -64xf32, -64xf32) <- (-1x64x192xf32, 192xf32, 192xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__37, parameter_191, parameter_192, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_57 = [-1, 4, 64, 192]

        # pd_op.reshape_: (-1x4x64x192xf32, 0x-1x64x192xf32) <- (-1x64x192xf32, 4xi64)
        reshape__42, reshape__43 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_39, full_int_array_57), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x192x64x4xf32) <- (-1x4x64x192xf32)
        transpose_24 = paddle._C_ops.transpose(reshape__42, [0, 3, 2, 1])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_58 = [-1, 8, 2, 2]

        # pd_op.reshape_: (-1x8x2x2xf32, 0x-1x192x64x4xf32) <- (-1x192x64x4xf32, 4xi64)
        reshape__44, reshape__45 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_24, full_int_array_58), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x8x2xf32) <- (-1x8x2x2xf32)
        transpose_25 = paddle._C_ops.transpose(reshape__44, [0, 2, 1, 3])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_59 = [-1, 192, 16, 16]

        # pd_op.reshape_: (-1x192x16x16xf32, 0x-1x2x8x2xf32) <- (-1x2x8x2xf32, 4xi64)
        reshape__46, reshape__47 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_25, full_int_array_59), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x128x16x16xf32) <- (-1x192x16x16xf32, 128x192x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(reshape__46, parameter_193, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x16x16xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x16x16xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_19, parameter_194, parameter_195, parameter_196, parameter_197, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x128x16x16xf32) <- (-1x128x16x16xf32)
        silu_23 = paddle._C_ops.silu(batch_norm__138)

        # builtin.combine: ([-1x128x16x16xf32, -1x128x16x16xf32]) <- (-1x128x16x16xf32, -1x128x16x16xf32)
        combine_13 = [batch_norm__126, silu_23]

        # pd_op.full: (1xi32) <- ()
        full_61 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x256x16x16xf32) <- ([-1x128x16x16xf32, -1x128x16x16xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_13, full_61)

        # pd_op.conv2d: (-1x128x16x16xf32) <- (-1x256x16x16xf32, 128x256x3x3xf32)
        conv2d_20 = paddle._C_ops.conv2d(concat_1, parameter_198, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x16x16xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x16x16xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_20, parameter_199, parameter_200, parameter_201, parameter_202, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x128x16x16xf32) <- (-1x128x16x16xf32)
        silu_24 = paddle._C_ops.silu(batch_norm__144)

        # pd_op.conv2d: (-1x512x16x16xf32) <- (-1x128x16x16xf32, 512x128x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(silu_24, parameter_203, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x16x16xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_21, parameter_204, parameter_205, parameter_206, parameter_207, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x512x16x16xf32) <- (-1x512x16x16xf32)
        silu_25 = paddle._C_ops.silu(batch_norm__150)

        # pd_op.depthwise_conv2d: (-1x512x8x8xf32) <- (-1x512x16x16xf32, 512x1x3x3xf32)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(silu_25, parameter_208, [2, 2], [1, 1], 'EXPLICIT', 512, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x512x8x8xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x8x8xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_6, parameter_209, parameter_210, parameter_211, parameter_212, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x512x8x8xf32) <- (-1x512x8x8xf32)
        silu_26 = paddle._C_ops.silu(batch_norm__156)

        # pd_op.conv2d: (-1x160x8x8xf32) <- (-1x512x8x8xf32, 160x512x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(silu_26, parameter_213, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x160x8x8xf32, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x8x8xf32, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_22, parameter_214, parameter_215, parameter_216, parameter_217, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x160x8x8xf32) <- (-1x160x8x8xf32, 160x160x3x3xf32)
        conv2d_23 = paddle._C_ops.conv2d(batch_norm__162, parameter_218, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x160x8x8xf32, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x8x8xf32, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_23, parameter_219, parameter_220, parameter_221, parameter_222, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x160x8x8xf32) <- (-1x160x8x8xf32)
        silu_27 = paddle._C_ops.silu(batch_norm__168)

        # pd_op.conv2d: (-1x240x8x8xf32) <- (-1x160x8x8xf32, 240x160x1x1xf32)
        conv2d_24 = paddle._C_ops.conv2d(silu_27, parameter_223, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_60 = [-1, 2, 4, 2]

        # pd_op.reshape_: (-1x2x4x2xf32, 0x-1x240x8x8xf32) <- (-1x240x8x8xf32, 4xi64)
        reshape__48, reshape__49 = (lambda x, f: f(x))(paddle._C_ops.reshape_(conv2d_24, full_int_array_60), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x2x2xf32) <- (-1x2x4x2xf32)
        transpose_26 = paddle._C_ops.transpose(reshape__48, [0, 2, 1, 3])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_61 = [-1, 240, 16, 4]

        # pd_op.reshape_: (-1x240x16x4xf32, 0x-1x4x2x2xf32) <- (-1x4x2x2xf32, 4xi64)
        reshape__50, reshape__51 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_26, full_int_array_61), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x16x240xf32) <- (-1x240x16x4xf32)
        transpose_27 = paddle._C_ops.transpose(reshape__50, [0, 3, 2, 1])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_62 = [-1, 16, 240]

        # pd_op.reshape_: (-1x16x240xf32, 0x-1x4x16x240xf32) <- (-1x4x16x240xf32, 3xi64)
        reshape__52, reshape__53 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_27, full_int_array_62), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.layer_norm: (-1x16x240xf32, -16xf32, -16xf32) <- (-1x16x240xf32, 240xf32, 240xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(reshape__52, parameter_224, parameter_225, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x16x240xf32)
        shape_6 = paddle._C_ops.shape(layer_norm_42)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_63 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_64 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(shape_6, [0], full_int_array_63, full_int_array_64, [1], [0])

        # pd_op.matmul: (-1x16x720xf32) <- (-1x16x240xf32, 240x720xf32)
        matmul_36 = paddle._C_ops.matmul(layer_norm_42, parameter_226, False, False)

        # pd_op.add_: (-1x16x720xf32) <- (-1x16x720xf32, 720xf32)
        add__38 = paddle._C_ops.add_(matmul_36, parameter_227)

        # pd_op.full: (1xi32) <- ()
        full_62 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_63 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_64 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_65 = paddle._C_ops.full([1], float('60'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_14 = [slice_24, full_62, full_63, full_64, full_65]

        # pd_op.reshape_: (-1x16x3x4x60xf32, 0x-1x16x720xf32) <- (-1x16x720xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__54, reshape__55 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__38, [x.reshape([]) for x in combine_14]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x3x16x60xf32) <- (-1x16x3x4x60xf32)
        transpose_28 = paddle._C_ops.transpose(reshape__54, [0, 3, 2, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_65 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_66 = [1]

        # pd_op.slice: (-1x4x16x60xf32) <- (-1x4x3x16x60xf32, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(transpose_28, [2], full_int_array_65, full_int_array_66, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_67 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_68 = [2]

        # pd_op.slice: (-1x4x16x60xf32) <- (-1x4x3x16x60xf32, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(transpose_28, [2], full_int_array_67, full_int_array_68, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_69 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_70 = [3]

        # pd_op.slice: (-1x4x16x60xf32) <- (-1x4x3x16x60xf32, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(transpose_28, [2], full_int_array_69, full_int_array_70, [1], [2])

        # pd_op.full: (1xf32) <- ()
        full_66 = paddle._C_ops.full([1], float('0.129099'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x4x16x60xf32) <- (-1x4x16x60xf32, 1xf32)
        scale__6 = paddle._C_ops.scale_(slice_25, full_66, float('0'), True)

        # pd_op.transpose: (-1x4x60x16xf32) <- (-1x4x16x60xf32)
        transpose_29 = paddle._C_ops.transpose(slice_26, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x16x16xf32) <- (-1x4x16x60xf32, -1x4x60x16xf32)
        matmul_37 = paddle._C_ops.matmul(scale__6, transpose_29, False, False)

        # pd_op.softmax_: (-1x4x16x16xf32) <- (-1x4x16x16xf32)
        softmax__6 = paddle._C_ops.softmax_(matmul_37, -1)

        # pd_op.matmul: (-1x4x16x60xf32) <- (-1x4x16x16xf32, -1x4x16x60xf32)
        matmul_38 = paddle._C_ops.matmul(softmax__6, slice_27, False, False)

        # pd_op.transpose: (-1x16x4x60xf32) <- (-1x4x16x60xf32)
        transpose_30 = paddle._C_ops.transpose(matmul_38, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_67 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_68 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_15 = [slice_24, full_67, full_68]

        # pd_op.reshape_: (-1x16x240xf32, 0x-1x16x4x60xf32) <- (-1x16x4x60xf32, [1xi32, 1xi32, 1xi32])
        reshape__56, reshape__57 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_30, [x.reshape([]) for x in combine_15]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16x240xf32) <- (-1x16x240xf32, 240x240xf32)
        matmul_39 = paddle._C_ops.matmul(reshape__56, parameter_228, False, False)

        # pd_op.add_: (-1x16x240xf32) <- (-1x16x240xf32, 240xf32)
        add__39 = paddle._C_ops.add_(matmul_39, parameter_229)

        # pd_op.full: (1xf32) <- ()
        full_69 = paddle._C_ops.full([1], float('0.1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.dropout: (-1x16x240xf32, None) <- (-1x16x240xf32, None, 1xf32)
        dropout_36, dropout_37 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__39, None, full_69, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x16x240xf32) <- (-1x16x240xf32, -1x16x240xf32)
        add__40 = paddle._C_ops.add_(reshape__52, dropout_36)

        # pd_op.layer_norm: (-1x16x240xf32, -16xf32, -16xf32) <- (-1x16x240xf32, 240xf32, 240xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__40, parameter_230, parameter_231, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x16x480xf32) <- (-1x16x240xf32, 240x480xf32)
        matmul_40 = paddle._C_ops.matmul(layer_norm_45, parameter_232, False, False)

        # pd_op.add_: (-1x16x480xf32) <- (-1x16x480xf32, 480xf32)
        add__41 = paddle._C_ops.add_(matmul_40, parameter_233)

        # pd_op.silu: (-1x16x480xf32) <- (-1x16x480xf32)
        silu_28 = paddle._C_ops.silu(add__41)

        # pd_op.full: (1xf32) <- ()
        full_70 = paddle._C_ops.full([1], float('0.1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.dropout: (-1x16x480xf32, None) <- (-1x16x480xf32, None, 1xf32)
        dropout_38, dropout_39 = (lambda x, f: f(x))(paddle._C_ops.dropout(silu_28, None, full_70, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16x240xf32) <- (-1x16x480xf32, 480x240xf32)
        matmul_41 = paddle._C_ops.matmul(dropout_38, parameter_234, False, False)

        # pd_op.add_: (-1x16x240xf32) <- (-1x16x240xf32, 240xf32)
        add__42 = paddle._C_ops.add_(matmul_41, parameter_235)

        # pd_op.full: (1xf32) <- ()
        full_71 = paddle._C_ops.full([1], float('0.1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.dropout: (-1x16x240xf32, None) <- (-1x16x240xf32, None, 1xf32)
        dropout_40, dropout_41 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__42, None, full_71, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x16x240xf32) <- (-1x16x240xf32, -1x16x240xf32)
        add__43 = paddle._C_ops.add_(dropout_40, add__40)

        # pd_op.layer_norm: (-1x16x240xf32, -16xf32, -16xf32) <- (-1x16x240xf32, 240xf32, 240xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__43, parameter_236, parameter_237, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x16x240xf32)
        shape_7 = paddle._C_ops.shape(layer_norm_48)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_71 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_72 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(shape_7, [0], full_int_array_71, full_int_array_72, [1], [0])

        # pd_op.matmul: (-1x16x720xf32) <- (-1x16x240xf32, 240x720xf32)
        matmul_42 = paddle._C_ops.matmul(layer_norm_48, parameter_238, False, False)

        # pd_op.add_: (-1x16x720xf32) <- (-1x16x720xf32, 720xf32)
        add__44 = paddle._C_ops.add_(matmul_42, parameter_239)

        # pd_op.full: (1xi32) <- ()
        full_72 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_73 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_74 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_75 = paddle._C_ops.full([1], float('60'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_16 = [slice_28, full_72, full_73, full_74, full_75]

        # pd_op.reshape_: (-1x16x3x4x60xf32, 0x-1x16x720xf32) <- (-1x16x720xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__58, reshape__59 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__44, [x.reshape([]) for x in combine_16]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x3x16x60xf32) <- (-1x16x3x4x60xf32)
        transpose_31 = paddle._C_ops.transpose(reshape__58, [0, 3, 2, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_73 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_74 = [1]

        # pd_op.slice: (-1x4x16x60xf32) <- (-1x4x3x16x60xf32, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(transpose_31, [2], full_int_array_73, full_int_array_74, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_75 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_76 = [2]

        # pd_op.slice: (-1x4x16x60xf32) <- (-1x4x3x16x60xf32, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(transpose_31, [2], full_int_array_75, full_int_array_76, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_77 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_78 = [3]

        # pd_op.slice: (-1x4x16x60xf32) <- (-1x4x3x16x60xf32, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(transpose_31, [2], full_int_array_77, full_int_array_78, [1], [2])

        # pd_op.full: (1xf32) <- ()
        full_76 = paddle._C_ops.full([1], float('0.129099'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x4x16x60xf32) <- (-1x4x16x60xf32, 1xf32)
        scale__7 = paddle._C_ops.scale_(slice_29, full_76, float('0'), True)

        # pd_op.transpose: (-1x4x60x16xf32) <- (-1x4x16x60xf32)
        transpose_32 = paddle._C_ops.transpose(slice_30, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x16x16xf32) <- (-1x4x16x60xf32, -1x4x60x16xf32)
        matmul_43 = paddle._C_ops.matmul(scale__7, transpose_32, False, False)

        # pd_op.softmax_: (-1x4x16x16xf32) <- (-1x4x16x16xf32)
        softmax__7 = paddle._C_ops.softmax_(matmul_43, -1)

        # pd_op.matmul: (-1x4x16x60xf32) <- (-1x4x16x16xf32, -1x4x16x60xf32)
        matmul_44 = paddle._C_ops.matmul(softmax__7, slice_31, False, False)

        # pd_op.transpose: (-1x16x4x60xf32) <- (-1x4x16x60xf32)
        transpose_33 = paddle._C_ops.transpose(matmul_44, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_77 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_78 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_17 = [slice_28, full_77, full_78]

        # pd_op.reshape_: (-1x16x240xf32, 0x-1x16x4x60xf32) <- (-1x16x4x60xf32, [1xi32, 1xi32, 1xi32])
        reshape__60, reshape__61 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_33, [x.reshape([]) for x in combine_17]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16x240xf32) <- (-1x16x240xf32, 240x240xf32)
        matmul_45 = paddle._C_ops.matmul(reshape__60, parameter_240, False, False)

        # pd_op.add_: (-1x16x240xf32) <- (-1x16x240xf32, 240xf32)
        add__45 = paddle._C_ops.add_(matmul_45, parameter_241)

        # pd_op.full: (1xf32) <- ()
        full_79 = paddle._C_ops.full([1], float('0.1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.dropout: (-1x16x240xf32, None) <- (-1x16x240xf32, None, 1xf32)
        dropout_42, dropout_43 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__45, None, full_79, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x16x240xf32) <- (-1x16x240xf32, -1x16x240xf32)
        add__46 = paddle._C_ops.add_(add__43, dropout_42)

        # pd_op.layer_norm: (-1x16x240xf32, -16xf32, -16xf32) <- (-1x16x240xf32, 240xf32, 240xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__46, parameter_242, parameter_243, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x16x480xf32) <- (-1x16x240xf32, 240x480xf32)
        matmul_46 = paddle._C_ops.matmul(layer_norm_51, parameter_244, False, False)

        # pd_op.add_: (-1x16x480xf32) <- (-1x16x480xf32, 480xf32)
        add__47 = paddle._C_ops.add_(matmul_46, parameter_245)

        # pd_op.silu: (-1x16x480xf32) <- (-1x16x480xf32)
        silu_29 = paddle._C_ops.silu(add__47)

        # pd_op.full: (1xf32) <- ()
        full_80 = paddle._C_ops.full([1], float('0.1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.dropout: (-1x16x480xf32, None) <- (-1x16x480xf32, None, 1xf32)
        dropout_44, dropout_45 = (lambda x, f: f(x))(paddle._C_ops.dropout(silu_29, None, full_80, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16x240xf32) <- (-1x16x480xf32, 480x240xf32)
        matmul_47 = paddle._C_ops.matmul(dropout_44, parameter_246, False, False)

        # pd_op.add_: (-1x16x240xf32) <- (-1x16x240xf32, 240xf32)
        add__48 = paddle._C_ops.add_(matmul_47, parameter_247)

        # pd_op.full: (1xf32) <- ()
        full_81 = paddle._C_ops.full([1], float('0.1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.dropout: (-1x16x240xf32, None) <- (-1x16x240xf32, None, 1xf32)
        dropout_46, dropout_47 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__48, None, full_81, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x16x240xf32) <- (-1x16x240xf32, -1x16x240xf32)
        add__49 = paddle._C_ops.add_(dropout_46, add__46)

        # pd_op.layer_norm: (-1x16x240xf32, -16xf32, -16xf32) <- (-1x16x240xf32, 240xf32, 240xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__49, parameter_248, parameter_249, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x16x240xf32)
        shape_8 = paddle._C_ops.shape(layer_norm_54)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_79 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_80 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(shape_8, [0], full_int_array_79, full_int_array_80, [1], [0])

        # pd_op.matmul: (-1x16x720xf32) <- (-1x16x240xf32, 240x720xf32)
        matmul_48 = paddle._C_ops.matmul(layer_norm_54, parameter_250, False, False)

        # pd_op.add_: (-1x16x720xf32) <- (-1x16x720xf32, 720xf32)
        add__50 = paddle._C_ops.add_(matmul_48, parameter_251)

        # pd_op.full: (1xi32) <- ()
        full_82 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_83 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_84 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_85 = paddle._C_ops.full([1], float('60'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_18 = [slice_32, full_82, full_83, full_84, full_85]

        # pd_op.reshape_: (-1x16x3x4x60xf32, 0x-1x16x720xf32) <- (-1x16x720xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__62, reshape__63 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__50, [x.reshape([]) for x in combine_18]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x3x16x60xf32) <- (-1x16x3x4x60xf32)
        transpose_34 = paddle._C_ops.transpose(reshape__62, [0, 3, 2, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_81 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_82 = [1]

        # pd_op.slice: (-1x4x16x60xf32) <- (-1x4x3x16x60xf32, 1xi64, 1xi64)
        slice_33 = paddle._C_ops.slice(transpose_34, [2], full_int_array_81, full_int_array_82, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_83 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_84 = [2]

        # pd_op.slice: (-1x4x16x60xf32) <- (-1x4x3x16x60xf32, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(transpose_34, [2], full_int_array_83, full_int_array_84, [1], [2])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_85 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_86 = [3]

        # pd_op.slice: (-1x4x16x60xf32) <- (-1x4x3x16x60xf32, 1xi64, 1xi64)
        slice_35 = paddle._C_ops.slice(transpose_34, [2], full_int_array_85, full_int_array_86, [1], [2])

        # pd_op.full: (1xf32) <- ()
        full_86 = paddle._C_ops.full([1], float('0.129099'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x4x16x60xf32) <- (-1x4x16x60xf32, 1xf32)
        scale__8 = paddle._C_ops.scale_(slice_33, full_86, float('0'), True)

        # pd_op.transpose: (-1x4x60x16xf32) <- (-1x4x16x60xf32)
        transpose_35 = paddle._C_ops.transpose(slice_34, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x16x16xf32) <- (-1x4x16x60xf32, -1x4x60x16xf32)
        matmul_49 = paddle._C_ops.matmul(scale__8, transpose_35, False, False)

        # pd_op.softmax_: (-1x4x16x16xf32) <- (-1x4x16x16xf32)
        softmax__8 = paddle._C_ops.softmax_(matmul_49, -1)

        # pd_op.matmul: (-1x4x16x60xf32) <- (-1x4x16x16xf32, -1x4x16x60xf32)
        matmul_50 = paddle._C_ops.matmul(softmax__8, slice_35, False, False)

        # pd_op.transpose: (-1x16x4x60xf32) <- (-1x4x16x60xf32)
        transpose_36 = paddle._C_ops.transpose(matmul_50, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_87 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_88 = paddle._C_ops.full([1], float('240'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_19 = [slice_32, full_87, full_88]

        # pd_op.reshape_: (-1x16x240xf32, 0x-1x16x4x60xf32) <- (-1x16x4x60xf32, [1xi32, 1xi32, 1xi32])
        reshape__64, reshape__65 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_36, [x.reshape([]) for x in combine_19]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16x240xf32) <- (-1x16x240xf32, 240x240xf32)
        matmul_51 = paddle._C_ops.matmul(reshape__64, parameter_252, False, False)

        # pd_op.add_: (-1x16x240xf32) <- (-1x16x240xf32, 240xf32)
        add__51 = paddle._C_ops.add_(matmul_51, parameter_253)

        # pd_op.full: (1xf32) <- ()
        full_89 = paddle._C_ops.full([1], float('0.1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.dropout: (-1x16x240xf32, None) <- (-1x16x240xf32, None, 1xf32)
        dropout_48, dropout_49 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__51, None, full_89, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x16x240xf32) <- (-1x16x240xf32, -1x16x240xf32)
        add__52 = paddle._C_ops.add_(add__49, dropout_48)

        # pd_op.layer_norm: (-1x16x240xf32, -16xf32, -16xf32) <- (-1x16x240xf32, 240xf32, 240xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__52, parameter_254, parameter_255, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x16x480xf32) <- (-1x16x240xf32, 240x480xf32)
        matmul_52 = paddle._C_ops.matmul(layer_norm_57, parameter_256, False, False)

        # pd_op.add_: (-1x16x480xf32) <- (-1x16x480xf32, 480xf32)
        add__53 = paddle._C_ops.add_(matmul_52, parameter_257)

        # pd_op.silu: (-1x16x480xf32) <- (-1x16x480xf32)
        silu_30 = paddle._C_ops.silu(add__53)

        # pd_op.full: (1xf32) <- ()
        full_90 = paddle._C_ops.full([1], float('0.1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.dropout: (-1x16x480xf32, None) <- (-1x16x480xf32, None, 1xf32)
        dropout_50, dropout_51 = (lambda x, f: f(x))(paddle._C_ops.dropout(silu_30, None, full_90, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16x240xf32) <- (-1x16x480xf32, 480x240xf32)
        matmul_53 = paddle._C_ops.matmul(dropout_50, parameter_258, False, False)

        # pd_op.add_: (-1x16x240xf32) <- (-1x16x240xf32, 240xf32)
        add__54 = paddle._C_ops.add_(matmul_53, parameter_259)

        # pd_op.full: (1xf32) <- ()
        full_91 = paddle._C_ops.full([1], float('0.1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.dropout: (-1x16x240xf32, None) <- (-1x16x240xf32, None, 1xf32)
        dropout_52, dropout_53 = (lambda x, f: f(x))(paddle._C_ops.dropout(add__54, None, full_91, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x16x240xf32) <- (-1x16x240xf32, -1x16x240xf32)
        add__55 = paddle._C_ops.add_(dropout_52, add__52)

        # pd_op.layer_norm: (-1x16x240xf32, -16xf32, -16xf32) <- (-1x16x240xf32, 240xf32, 240xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__55, parameter_260, parameter_261, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_87 = [-1, 4, 16, 240]

        # pd_op.reshape_: (-1x4x16x240xf32, 0x-1x16x240xf32) <- (-1x16x240xf32, 4xi64)
        reshape__66, reshape__67 = (lambda x, f: f(x))(paddle._C_ops.reshape_(layer_norm_60, full_int_array_87), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x240x16x4xf32) <- (-1x4x16x240xf32)
        transpose_37 = paddle._C_ops.transpose(reshape__66, [0, 3, 2, 1])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_88 = [-1, 4, 2, 2]

        # pd_op.reshape_: (-1x4x2x2xf32, 0x-1x240x16x4xf32) <- (-1x240x16x4xf32, 4xi64)
        reshape__68, reshape__69 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_37, full_int_array_88), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x4x2xf32) <- (-1x4x2x2xf32)
        transpose_38 = paddle._C_ops.transpose(reshape__68, [0, 2, 1, 3])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_89 = [-1, 240, 8, 8]

        # pd_op.reshape_: (-1x240x8x8xf32, 0x-1x2x4x2xf32) <- (-1x2x4x2xf32, 4xi64)
        reshape__70, reshape__71 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_38, full_int_array_89), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x160x8x8xf32) <- (-1x240x8x8xf32, 160x240x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(reshape__70, parameter_262, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x160x8x8xf32, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x8x8xf32, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_25, parameter_263, parameter_264, parameter_265, parameter_266, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x160x8x8xf32) <- (-1x160x8x8xf32)
        silu_31 = paddle._C_ops.silu(batch_norm__174)

        # builtin.combine: ([-1x160x8x8xf32, -1x160x8x8xf32]) <- (-1x160x8x8xf32, -1x160x8x8xf32)
        combine_20 = [batch_norm__162, silu_31]

        # pd_op.full: (1xi32) <- ()
        full_92 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x320x8x8xf32) <- ([-1x160x8x8xf32, -1x160x8x8xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_20, full_92)

        # pd_op.conv2d: (-1x160x8x8xf32) <- (-1x320x8x8xf32, 160x320x3x3xf32)
        conv2d_26 = paddle._C_ops.conv2d(concat_2, parameter_267, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x160x8x8xf32, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x8x8xf32, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__180, batch_norm__181, batch_norm__182, batch_norm__183, batch_norm__184, batch_norm__185 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_26, parameter_268, parameter_269, parameter_270, parameter_271, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x160x8x8xf32) <- (-1x160x8x8xf32)
        silu_32 = paddle._C_ops.silu(batch_norm__180)

        # pd_op.conv2d: (-1x640x8x8xf32) <- (-1x160x8x8xf32, 640x160x1x1xf32)
        conv2d_27 = paddle._C_ops.conv2d(silu_32, parameter_272, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x640x8x8xf32, 640xf32, 640xf32, xf32, xf32, None) <- (-1x640x8x8xf32, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__186, batch_norm__187, batch_norm__188, batch_norm__189, batch_norm__190, batch_norm__191 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_27, parameter_273, parameter_274, parameter_275, parameter_276, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.silu: (-1x640x8x8xf32) <- (-1x640x8x8xf32)
        silu_33 = paddle._C_ops.silu(batch_norm__186)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_90 = [1, 1]

        # pd_op.pool2d: (-1x640x1x1xf32) <- (-1x640x8x8xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(silu_33, full_int_array_90, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.shape: (4xi32) <- (-1x640x1x1xf32)
        shape_9 = paddle._C_ops.shape(pool2d_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_91 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_92 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(shape_9, [0], full_int_array_91, full_int_array_92, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_93 = paddle._C_ops.full([1], float('640'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32]) <- (1xi32, 1xi32)
        combine_21 = [slice_36, full_93]

        # pd_op.reshape_: (-1x640xf32, 0x-1x640x1x1xf32) <- (-1x640x1x1xf32, [1xi32, 1xi32])
        reshape__72, reshape__73 = (lambda x, f: f(x))(paddle._C_ops.reshape_(pool2d_0, [x.reshape([]) for x in combine_21]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xf32) <- ()
        full_94 = paddle._C_ops.full([1], float('0.1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.dropout: (-1x640xf32, None) <- (-1x640xf32, None, 1xf32)
        dropout_54, dropout_55 = (lambda x, f: f(x))(paddle._C_ops.dropout(reshape__72, None, full_94, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x1000xf32) <- (-1x640xf32, 640x1000xf32)
        matmul_54 = paddle._C_ops.matmul(dropout_54, parameter_277, False, False)

        # pd_op.add_: (-1x1000xf32) <- (-1x1000xf32, 1000xf32)
        add__56 = paddle._C_ops.add_(matmul_54, parameter_278)

        # pd_op.softmax_: (-1x1000xf32) <- (-1x1000xf32)
        softmax__9 = paddle._C_ops.softmax_(add__56, -1)
        return softmax__9



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

    def forward(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_87, parameter_86, parameter_88, parameter_89, parameter_90, parameter_91, parameter_93, parameter_92, parameter_94, parameter_95, parameter_96, parameter_97, parameter_99, parameter_98, parameter_100, parameter_101, parameter_102, parameter_103, parameter_105, parameter_104, parameter_106, parameter_107, parameter_108, parameter_109, parameter_111, parameter_110, parameter_112, parameter_116, parameter_113, parameter_115, parameter_114, parameter_117, parameter_121, parameter_118, parameter_120, parameter_119, parameter_122, parameter_126, parameter_123, parameter_125, parameter_124, parameter_127, parameter_131, parameter_128, parameter_130, parameter_129, parameter_132, parameter_136, parameter_133, parameter_135, parameter_134, parameter_137, parameter_141, parameter_138, parameter_140, parameter_139, parameter_142, parameter_144, parameter_143, parameter_145, parameter_146, parameter_147, parameter_148, parameter_150, parameter_149, parameter_151, parameter_152, parameter_153, parameter_154, parameter_156, parameter_155, parameter_157, parameter_158, parameter_159, parameter_160, parameter_162, parameter_161, parameter_163, parameter_164, parameter_165, parameter_166, parameter_168, parameter_167, parameter_169, parameter_170, parameter_171, parameter_172, parameter_174, parameter_173, parameter_175, parameter_176, parameter_177, parameter_178, parameter_180, parameter_179, parameter_181, parameter_182, parameter_183, parameter_184, parameter_186, parameter_185, parameter_187, parameter_188, parameter_189, parameter_190, parameter_192, parameter_191, parameter_193, parameter_197, parameter_194, parameter_196, parameter_195, parameter_198, parameter_202, parameter_199, parameter_201, parameter_200, parameter_203, parameter_207, parameter_204, parameter_206, parameter_205, parameter_208, parameter_212, parameter_209, parameter_211, parameter_210, parameter_213, parameter_217, parameter_214, parameter_216, parameter_215, parameter_218, parameter_222, parameter_219, parameter_221, parameter_220, parameter_223, parameter_225, parameter_224, parameter_226, parameter_227, parameter_228, parameter_229, parameter_231, parameter_230, parameter_232, parameter_233, parameter_234, parameter_235, parameter_237, parameter_236, parameter_238, parameter_239, parameter_240, parameter_241, parameter_243, parameter_242, parameter_244, parameter_245, parameter_246, parameter_247, parameter_249, parameter_248, parameter_250, parameter_251, parameter_252, parameter_253, parameter_255, parameter_254, parameter_256, parameter_257, parameter_258, parameter_259, parameter_261, parameter_260, parameter_262, parameter_266, parameter_263, parameter_265, parameter_264, parameter_267, parameter_271, parameter_268, parameter_270, parameter_269, parameter_272, parameter_276, parameter_273, parameter_275, parameter_274, parameter_277, parameter_278, feed_0):
        return self.builtin_module_904_0_0(parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_87, parameter_86, parameter_88, parameter_89, parameter_90, parameter_91, parameter_93, parameter_92, parameter_94, parameter_95, parameter_96, parameter_97, parameter_99, parameter_98, parameter_100, parameter_101, parameter_102, parameter_103, parameter_105, parameter_104, parameter_106, parameter_107, parameter_108, parameter_109, parameter_111, parameter_110, parameter_112, parameter_116, parameter_113, parameter_115, parameter_114, parameter_117, parameter_121, parameter_118, parameter_120, parameter_119, parameter_122, parameter_126, parameter_123, parameter_125, parameter_124, parameter_127, parameter_131, parameter_128, parameter_130, parameter_129, parameter_132, parameter_136, parameter_133, parameter_135, parameter_134, parameter_137, parameter_141, parameter_138, parameter_140, parameter_139, parameter_142, parameter_144, parameter_143, parameter_145, parameter_146, parameter_147, parameter_148, parameter_150, parameter_149, parameter_151, parameter_152, parameter_153, parameter_154, parameter_156, parameter_155, parameter_157, parameter_158, parameter_159, parameter_160, parameter_162, parameter_161, parameter_163, parameter_164, parameter_165, parameter_166, parameter_168, parameter_167, parameter_169, parameter_170, parameter_171, parameter_172, parameter_174, parameter_173, parameter_175, parameter_176, parameter_177, parameter_178, parameter_180, parameter_179, parameter_181, parameter_182, parameter_183, parameter_184, parameter_186, parameter_185, parameter_187, parameter_188, parameter_189, parameter_190, parameter_192, parameter_191, parameter_193, parameter_197, parameter_194, parameter_196, parameter_195, parameter_198, parameter_202, parameter_199, parameter_201, parameter_200, parameter_203, parameter_207, parameter_204, parameter_206, parameter_205, parameter_208, parameter_212, parameter_209, parameter_211, parameter_210, parameter_213, parameter_217, parameter_214, parameter_216, parameter_215, parameter_218, parameter_222, parameter_219, parameter_221, parameter_220, parameter_223, parameter_225, parameter_224, parameter_226, parameter_227, parameter_228, parameter_229, parameter_231, parameter_230, parameter_232, parameter_233, parameter_234, parameter_235, parameter_237, parameter_236, parameter_238, parameter_239, parameter_240, parameter_241, parameter_243, parameter_242, parameter_244, parameter_245, parameter_246, parameter_247, parameter_249, parameter_248, parameter_250, parameter_251, parameter_252, parameter_253, parameter_255, parameter_254, parameter_256, parameter_257, parameter_258, parameter_259, parameter_261, parameter_260, parameter_262, parameter_266, parameter_263, parameter_265, parameter_264, parameter_267, parameter_271, parameter_268, parameter_270, parameter_269, parameter_272, parameter_276, parameter_273, parameter_275, parameter_274, parameter_277, parameter_278, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_904_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([16, 3, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([64, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([32, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([128, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([64, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([256, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([256, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([256, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([96, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([96, 96, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([144, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([144, 432], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([432], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([144, 144], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([144, 288], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([288], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([288, 144], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([144, 432], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([432], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([144, 144], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([144, 288], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([288], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([288, 144], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([96, 144, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([96, 192, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([384, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([384, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([128, 384, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_139
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([192, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_149
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([192, 384], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([384, 192], dtype='float32', min=0, max=0.5),
            # parameter_154
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([192, 384], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([384, 192], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([192, 384], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([384, 192], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([192, 384], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_189
            paddle.uniform([384, 192], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([128, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([512, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([512, 1, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_209
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_213
            paddle.uniform([160, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_217
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_214
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_215
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_218
            paddle.uniform([160, 160, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_219
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_221
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_223
            paddle.uniform([240, 160, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_225
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_224
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([240, 720], dtype='float32', min=0, max=0.5),
            # parameter_227
            paddle.uniform([720], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([240, 240], dtype='float32', min=0, max=0.5),
            # parameter_229
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([240, 480], dtype='float32', min=0, max=0.5),
            # parameter_233
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_234
            paddle.uniform([480, 240], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_238
            paddle.uniform([240, 720], dtype='float32', min=0, max=0.5),
            # parameter_239
            paddle.uniform([720], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([240, 240], dtype='float32', min=0, max=0.5),
            # parameter_241
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_243
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_244
            paddle.uniform([240, 480], dtype='float32', min=0, max=0.5),
            # parameter_245
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_246
            paddle.uniform([480, 240], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_249
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_248
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_250
            paddle.uniform([240, 720], dtype='float32', min=0, max=0.5),
            # parameter_251
            paddle.uniform([720], dtype='float32', min=0, max=0.5),
            # parameter_252
            paddle.uniform([240, 240], dtype='float32', min=0, max=0.5),
            # parameter_253
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_255
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_254
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_256
            paddle.uniform([240, 480], dtype='float32', min=0, max=0.5),
            # parameter_257
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_258
            paddle.uniform([480, 240], dtype='float32', min=0, max=0.5),
            # parameter_259
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_261
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            # parameter_262
            paddle.uniform([160, 240, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_266
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_263
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_265
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_264
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_267
            paddle.uniform([160, 320, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_271
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_268
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_270
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_269
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_272
            paddle.uniform([640, 160, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_276
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_273
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_275
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_274
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_277
            paddle.uniform([640, 1000], dtype='float32', min=0, max=0.5),
            # parameter_278
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 3, 256, 256], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_0
            paddle.static.InputSpec(shape=[16, 3, 3, 3], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[64, 16, 1, 1], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[64, 1, 3, 3], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[32, 64, 1, 1], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[128, 32, 1, 1], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[64, 128, 1, 1], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[256, 1, 3, 3], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[256, 1, 3, 3], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[256, 1, 3, 3], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[96, 256, 1, 1], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[96, 96, 3, 3], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[144, 96, 1, 1], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[144, 432], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[432], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[144, 144], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[144, 288], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[288], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[288, 144], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[144, 432], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[432], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[144, 144], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[144, 288], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[288], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[288, 144], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[96, 144, 1, 1], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[96, 192, 3, 3], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[384, 96, 1, 1], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[384, 1, 3, 3], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[128, 384, 1, 1], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_139
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[192, 128, 1, 1], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[576], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[192, 192], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_149
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[192, 384], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[384, 192], dtype='float32'),
            # parameter_154
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[576], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[192, 192], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[192, 384], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[384, 192], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[576], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[192, 192], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[192, 384], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[384, 192], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[576], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[192, 192], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[192, 384], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_189
            paddle.static.InputSpec(shape=[384, 192], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[128, 192, 1, 1], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[128, 256, 3, 3], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[512, 1, 3, 3], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_209
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_213
            paddle.static.InputSpec(shape=[160, 512, 1, 1], dtype='float32'),
            # parameter_217
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_214
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_215
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_218
            paddle.static.InputSpec(shape=[160, 160, 3, 3], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_219
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_221
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_223
            paddle.static.InputSpec(shape=[240, 160, 1, 1], dtype='float32'),
            # parameter_225
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_224
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[240, 720], dtype='float32'),
            # parameter_227
            paddle.static.InputSpec(shape=[720], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[240, 240], dtype='float32'),
            # parameter_229
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[240, 480], dtype='float32'),
            # parameter_233
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_234
            paddle.static.InputSpec(shape=[480, 240], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_238
            paddle.static.InputSpec(shape=[240, 720], dtype='float32'),
            # parameter_239
            paddle.static.InputSpec(shape=[720], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[240, 240], dtype='float32'),
            # parameter_241
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_243
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_244
            paddle.static.InputSpec(shape=[240, 480], dtype='float32'),
            # parameter_245
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_246
            paddle.static.InputSpec(shape=[480, 240], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_249
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_248
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_250
            paddle.static.InputSpec(shape=[240, 720], dtype='float32'),
            # parameter_251
            paddle.static.InputSpec(shape=[720], dtype='float32'),
            # parameter_252
            paddle.static.InputSpec(shape=[240, 240], dtype='float32'),
            # parameter_253
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_255
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_254
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_256
            paddle.static.InputSpec(shape=[240, 480], dtype='float32'),
            # parameter_257
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_258
            paddle.static.InputSpec(shape=[480, 240], dtype='float32'),
            # parameter_259
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_261
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            # parameter_262
            paddle.static.InputSpec(shape=[160, 240, 1, 1], dtype='float32'),
            # parameter_266
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_263
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_265
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_264
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_267
            paddle.static.InputSpec(shape=[160, 320, 3, 3], dtype='float32'),
            # parameter_271
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_268
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_270
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_269
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_272
            paddle.static.InputSpec(shape=[640, 160, 1, 1], dtype='float32'),
            # parameter_276
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_273
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_275
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_274
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_277
            paddle.static.InputSpec(shape=[640, 1000], dtype='float32'),
            # parameter_278
            paddle.static.InputSpec(shape=[1000], dtype='float32'),
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