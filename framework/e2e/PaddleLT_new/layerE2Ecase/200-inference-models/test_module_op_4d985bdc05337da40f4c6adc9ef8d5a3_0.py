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
    return [435][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_782_0_0(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_36, parameter_37, parameter_38, parameter_39, parameter_43, parameter_40, parameter_42, parameter_41, parameter_44, parameter_48, parameter_45, parameter_47, parameter_46, parameter_49, parameter_53, parameter_50, parameter_52, parameter_51, parameter_54, parameter_58, parameter_55, parameter_57, parameter_56, parameter_59, parameter_63, parameter_60, parameter_62, parameter_61, parameter_64, parameter_65, parameter_66, parameter_70, parameter_67, parameter_69, parameter_68, parameter_71, parameter_72, parameter_73, parameter_74, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_101, parameter_102, parameter_106, parameter_103, parameter_105, parameter_104, parameter_107, parameter_108, parameter_109, parameter_110, parameter_111, parameter_115, parameter_112, parameter_114, parameter_113, parameter_116, parameter_120, parameter_117, parameter_119, parameter_118, parameter_121, parameter_125, parameter_122, parameter_124, parameter_123, parameter_126, parameter_130, parameter_127, parameter_129, parameter_128, parameter_131, parameter_135, parameter_132, parameter_134, parameter_133, parameter_136, parameter_137, parameter_138, parameter_139, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_154, parameter_151, parameter_153, parameter_152, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_161, parameter_162, parameter_166, parameter_163, parameter_165, parameter_164, parameter_167, parameter_171, parameter_168, parameter_170, parameter_169, parameter_172, parameter_176, parameter_173, parameter_175, parameter_174, parameter_177, parameter_181, parameter_178, parameter_180, parameter_179, parameter_182, parameter_186, parameter_183, parameter_185, parameter_184, parameter_187, parameter_191, parameter_188, parameter_190, parameter_189, parameter_192, parameter_193, parameter_194, parameter_198, parameter_195, parameter_197, parameter_196, parameter_199, parameter_203, parameter_200, parameter_202, parameter_201, parameter_204, parameter_208, parameter_205, parameter_207, parameter_206, parameter_209, parameter_213, parameter_210, parameter_212, parameter_211, parameter_214, parameter_218, parameter_215, parameter_217, parameter_216, parameter_219, parameter_220, parameter_221, parameter_225, parameter_222, parameter_224, parameter_223, parameter_226, parameter_230, parameter_227, parameter_229, parameter_228, parameter_231, parameter_235, parameter_232, parameter_234, parameter_233, parameter_236, parameter_240, parameter_237, parameter_239, parameter_238, parameter_241, parameter_245, parameter_242, parameter_244, parameter_243, parameter_246, parameter_247, parameter_248, parameter_252, parameter_249, parameter_251, parameter_250, parameter_253, parameter_257, parameter_254, parameter_256, parameter_255, parameter_258, parameter_262, parameter_259, parameter_261, parameter_260, parameter_263, parameter_267, parameter_264, parameter_266, parameter_265, parameter_268, parameter_272, parameter_269, parameter_271, parameter_270, parameter_273, parameter_274, parameter_275, parameter_279, parameter_276, parameter_278, parameter_277, parameter_280, parameter_281, parameter_282, parameter_286, parameter_283, parameter_285, parameter_284, parameter_287, parameter_288, parameter_289, parameter_290, parameter_291, parameter_295, parameter_292, parameter_294, parameter_293, parameter_296, parameter_297, parameter_298, parameter_299, parameter_300, parameter_301, parameter_305, parameter_302, parameter_304, parameter_303, parameter_306, parameter_307, parameter_308, parameter_309, parameter_310, parameter_314, parameter_311, parameter_313, parameter_312, parameter_315, parameter_316, parameter_317, parameter_318, parameter_319, parameter_323, parameter_320, parameter_322, parameter_321, parameter_324, parameter_325, parameter_326, parameter_327, parameter_328, parameter_332, parameter_329, parameter_331, parameter_330, parameter_333, parameter_334, parameter_335, parameter_336, feed_1, feed_0):

        # pd_op.conv2d: (-1x16x320x320xf32) <- (-1x3x640x640xf32, 16x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(feed_0, parameter_0, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x16x320x320xf32, 16xf32, 16xf32, xf32, xf32, None) <- (-1x16x320x320xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_0, parameter_1, parameter_2, parameter_3, parameter_4, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x16x320x320xf32) <- (-1x16x320x320xf32)
        swish_0 = paddle._C_ops.swish(batch_norm__0)

        # pd_op.conv2d: (-1x16x320x320xf32) <- (-1x16x320x320xf32, 16x16x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(swish_0, parameter_5, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x16x320x320xf32, 16xf32, 16xf32, xf32, xf32, None) <- (-1x16x320x320xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_1, parameter_6, parameter_7, parameter_8, parameter_9, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x16x320x320xf32) <- (-1x16x320x320xf32)
        swish_1 = paddle._C_ops.swish(batch_norm__6)

        # pd_op.conv2d: (-1x32x320x320xf32) <- (-1x16x320x320xf32, 32x16x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(swish_1, parameter_10, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x320x320xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x320x320xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_2, parameter_11, parameter_12, parameter_13, parameter_14, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x32x320x320xf32) <- (-1x32x320x320xf32)
        swish_2 = paddle._C_ops.swish(batch_norm__12)

        # pd_op.conv2d: (-1x48x160x160xf32) <- (-1x32x320x320xf32, 48x32x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(swish_2, parameter_15, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x160x160xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x160x160xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_3, parameter_16, parameter_17, parameter_18, parameter_19, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x48x160x160xf32) <- (-1x48x160x160xf32)
        swish_3 = paddle._C_ops.swish(batch_norm__18)

        # pd_op.conv2d: (-1x24x160x160xf32) <- (-1x48x160x160xf32, 24x48x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(swish_3, parameter_20, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x24x160x160xf32, 24xf32, 24xf32, xf32, xf32, None) <- (-1x24x160x160xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_4, parameter_21, parameter_22, parameter_23, parameter_24, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x24x160x160xf32) <- (-1x24x160x160xf32)
        swish_4 = paddle._C_ops.swish(batch_norm__24)

        # pd_op.conv2d: (-1x24x160x160xf32) <- (-1x48x160x160xf32, 24x48x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(swish_3, parameter_25, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x24x160x160xf32, 24xf32, 24xf32, xf32, xf32, None) <- (-1x24x160x160xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_5, parameter_26, parameter_27, parameter_28, parameter_29, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x24x160x160xf32) <- (-1x24x160x160xf32)
        swish_5 = paddle._C_ops.swish(batch_norm__30)

        # pd_op.conv2d: (-1x24x160x160xf32) <- (-1x24x160x160xf32, 24x24x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(swish_5, parameter_30, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x24x160x160xf32, 24xf32, 24xf32, xf32, xf32, None) <- (-1x24x160x160xf32, 24xf32, 24xf32, 24xf32, 24xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_6, parameter_31, parameter_32, parameter_33, parameter_34, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x24x160x160xf32) <- (-1x24x160x160xf32)
        swish_6 = paddle._C_ops.swish(batch_norm__36)

        # pd_op.conv2d: (-1x24x160x160xf32) <- (-1x24x160x160xf32, 24x24x3x3xf32)
        conv2d_7 = paddle._C_ops.conv2d(swish_6, parameter_35, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, 24, 1, 1]

        # pd_op.reshape: (1x24x1x1xf32, 0x24xf32) <- (24xf32, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_36, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x24x160x160xf32) <- (-1x24x160x160xf32, 1x24x1x1xf32)
        add__0 = paddle._C_ops.add_(conv2d_7, reshape_0)

        # pd_op.swish: (-1x24x160x160xf32) <- (-1x24x160x160xf32)
        swish_7 = paddle._C_ops.swish(add__0)

        # pd_op.add_: (-1x24x160x160xf32) <- (-1x24x160x160xf32, -1x24x160x160xf32)
        add__1 = paddle._C_ops.add_(swish_5, swish_7)

        # builtin.combine: ([-1x24x160x160xf32, -1x24x160x160xf32]) <- (-1x24x160x160xf32, -1x24x160x160xf32)
        combine_0 = [swish_4, add__1]

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x48x160x160xf32) <- ([-1x24x160x160xf32, -1x24x160x160xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        # pd_op.mean: (-1x48x1x1xf32) <- (-1x48x160x160xf32)
        mean_0 = paddle._C_ops.mean(concat_0, [2, 3], True)

        # pd_op.conv2d: (-1x48x1x1xf32) <- (-1x48x1x1xf32, 48x48x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(mean_0, parameter_37, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [1, 48, 1, 1]

        # pd_op.reshape: (1x48x1x1xf32, 0x48xf32) <- (48xf32, 4xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_38, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x48x1x1xf32) <- (-1x48x1x1xf32, 1x48x1x1xf32)
        add__2 = paddle._C_ops.add_(conv2d_8, reshape_2)

        # pd_op.hardsigmoid: (-1x48x1x1xf32) <- (-1x48x1x1xf32)
        hardsigmoid_0 = paddle._C_ops.hardsigmoid(add__2, float('0.166667'), float('0.5'))

        # pd_op.multiply_: (-1x48x160x160xf32) <- (-1x48x160x160xf32, -1x48x1x1xf32)
        multiply__0 = paddle._C_ops.multiply_(concat_0, hardsigmoid_0)

        # pd_op.conv2d: (-1x64x160x160xf32) <- (-1x48x160x160xf32, 64x48x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(multiply__0, parameter_39, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x160x160xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x160x160xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_9, parameter_40, parameter_41, parameter_42, parameter_43, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x64x160x160xf32) <- (-1x64x160x160xf32)
        swish_8 = paddle._C_ops.swish(batch_norm__42)

        # pd_op.conv2d: (-1x96x80x80xf32) <- (-1x64x160x160xf32, 96x64x3x3xf32)
        conv2d_10 = paddle._C_ops.conv2d(swish_8, parameter_44, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x80x80xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x80x80xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_10, parameter_45, parameter_46, parameter_47, parameter_48, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x96x80x80xf32) <- (-1x96x80x80xf32)
        swish_9 = paddle._C_ops.swish(batch_norm__48)

        # pd_op.conv2d: (-1x48x80x80xf32) <- (-1x96x80x80xf32, 48x96x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(swish_9, parameter_49, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x80x80xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x80x80xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_11, parameter_50, parameter_51, parameter_52, parameter_53, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x48x80x80xf32) <- (-1x48x80x80xf32)
        swish_10 = paddle._C_ops.swish(batch_norm__54)

        # pd_op.conv2d: (-1x48x80x80xf32) <- (-1x96x80x80xf32, 48x96x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(swish_9, parameter_54, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x80x80xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x80x80xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_12, parameter_55, parameter_56, parameter_57, parameter_58, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x48x80x80xf32) <- (-1x48x80x80xf32)
        swish_11 = paddle._C_ops.swish(batch_norm__60)

        # pd_op.conv2d: (-1x48x80x80xf32) <- (-1x48x80x80xf32, 48x48x3x3xf32)
        conv2d_13 = paddle._C_ops.conv2d(swish_11, parameter_59, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x80x80xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x80x80xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_13, parameter_60, parameter_61, parameter_62, parameter_63, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x48x80x80xf32) <- (-1x48x80x80xf32)
        swish_12 = paddle._C_ops.swish(batch_norm__66)

        # pd_op.conv2d: (-1x48x80x80xf32) <- (-1x48x80x80xf32, 48x48x3x3xf32)
        conv2d_14 = paddle._C_ops.conv2d(swish_12, parameter_64, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_2 = [1, 48, 1, 1]

        # pd_op.reshape: (1x48x1x1xf32, 0x48xf32) <- (48xf32, 4xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_65, full_int_array_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x48x80x80xf32) <- (-1x48x80x80xf32, 1x48x1x1xf32)
        add__3 = paddle._C_ops.add_(conv2d_14, reshape_4)

        # pd_op.swish: (-1x48x80x80xf32) <- (-1x48x80x80xf32)
        swish_13 = paddle._C_ops.swish(add__3)

        # pd_op.add_: (-1x48x80x80xf32) <- (-1x48x80x80xf32, -1x48x80x80xf32)
        add__4 = paddle._C_ops.add_(swish_11, swish_13)

        # pd_op.conv2d: (-1x48x80x80xf32) <- (-1x48x80x80xf32, 48x48x3x3xf32)
        conv2d_15 = paddle._C_ops.conv2d(add__4, parameter_66, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x80x80xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x80x80xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_15, parameter_67, parameter_68, parameter_69, parameter_70, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x48x80x80xf32) <- (-1x48x80x80xf32)
        swish_14 = paddle._C_ops.swish(batch_norm__72)

        # pd_op.conv2d: (-1x48x80x80xf32) <- (-1x48x80x80xf32, 48x48x3x3xf32)
        conv2d_16 = paddle._C_ops.conv2d(swish_14, parameter_71, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [1, 48, 1, 1]

        # pd_op.reshape: (1x48x1x1xf32, 0x48xf32) <- (48xf32, 4xi64)
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_72, full_int_array_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x48x80x80xf32) <- (-1x48x80x80xf32, 1x48x1x1xf32)
        add__5 = paddle._C_ops.add_(conv2d_16, reshape_6)

        # pd_op.swish: (-1x48x80x80xf32) <- (-1x48x80x80xf32)
        swish_15 = paddle._C_ops.swish(add__5)

        # pd_op.add_: (-1x48x80x80xf32) <- (-1x48x80x80xf32, -1x48x80x80xf32)
        add__6 = paddle._C_ops.add_(add__4, swish_15)

        # builtin.combine: ([-1x48x80x80xf32, -1x48x80x80xf32]) <- (-1x48x80x80xf32, -1x48x80x80xf32)
        combine_1 = [swish_10, add__6]

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x96x80x80xf32) <- ([-1x48x80x80xf32, -1x48x80x80xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_1)

        # pd_op.mean: (-1x96x1x1xf32) <- (-1x96x80x80xf32)
        mean_1 = paddle._C_ops.mean(concat_1, [2, 3], True)

        # pd_op.conv2d: (-1x96x1x1xf32) <- (-1x96x1x1xf32, 96x96x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(mean_1, parameter_73, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_4 = [1, 96, 1, 1]

        # pd_op.reshape: (1x96x1x1xf32, 0x96xf32) <- (96xf32, 4xi64)
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_74, full_int_array_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x96x1x1xf32) <- (-1x96x1x1xf32, 1x96x1x1xf32)
        add__7 = paddle._C_ops.add_(conv2d_17, reshape_8)

        # pd_op.hardsigmoid: (-1x96x1x1xf32) <- (-1x96x1x1xf32)
        hardsigmoid_1 = paddle._C_ops.hardsigmoid(add__7, float('0.166667'), float('0.5'))

        # pd_op.multiply_: (-1x96x80x80xf32) <- (-1x96x80x80xf32, -1x96x1x1xf32)
        multiply__1 = paddle._C_ops.multiply_(concat_1, hardsigmoid_1)

        # pd_op.conv2d: (-1x128x80x80xf32) <- (-1x96x80x80xf32, 128x96x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(multiply__1, parameter_75, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x80x80xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x80x80xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_18, parameter_76, parameter_77, parameter_78, parameter_79, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x128x80x80xf32) <- (-1x128x80x80xf32)
        swish_16 = paddle._C_ops.swish(batch_norm__78)

        # pd_op.conv2d: (-1x192x40x40xf32) <- (-1x128x80x80xf32, 192x128x3x3xf32)
        conv2d_19 = paddle._C_ops.conv2d(swish_16, parameter_80, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x192x40x40xf32, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_19, parameter_81, parameter_82, parameter_83, parameter_84, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x192x40x40xf32) <- (-1x192x40x40xf32)
        swish_17 = paddle._C_ops.swish(batch_norm__84)

        # pd_op.conv2d: (-1x96x40x40xf32) <- (-1x192x40x40xf32, 96x192x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(swish_17, parameter_85, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x40x40xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x40x40xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_20, parameter_86, parameter_87, parameter_88, parameter_89, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x96x40x40xf32) <- (-1x96x40x40xf32)
        swish_18 = paddle._C_ops.swish(batch_norm__90)

        # pd_op.conv2d: (-1x96x40x40xf32) <- (-1x192x40x40xf32, 96x192x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(swish_17, parameter_90, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x40x40xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x40x40xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_21, parameter_91, parameter_92, parameter_93, parameter_94, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x96x40x40xf32) <- (-1x96x40x40xf32)
        swish_19 = paddle._C_ops.swish(batch_norm__96)

        # pd_op.conv2d: (-1x96x40x40xf32) <- (-1x96x40x40xf32, 96x96x3x3xf32)
        conv2d_22 = paddle._C_ops.conv2d(swish_19, parameter_95, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x40x40xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x40x40xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_22, parameter_96, parameter_97, parameter_98, parameter_99, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x96x40x40xf32) <- (-1x96x40x40xf32)
        swish_20 = paddle._C_ops.swish(batch_norm__102)

        # pd_op.conv2d: (-1x96x40x40xf32) <- (-1x96x40x40xf32, 96x96x3x3xf32)
        conv2d_23 = paddle._C_ops.conv2d(swish_20, parameter_100, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_5 = [1, 96, 1, 1]

        # pd_op.reshape: (1x96x1x1xf32, 0x96xf32) <- (96xf32, 4xi64)
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_101, full_int_array_5), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x96x40x40xf32) <- (-1x96x40x40xf32, 1x96x1x1xf32)
        add__8 = paddle._C_ops.add_(conv2d_23, reshape_10)

        # pd_op.swish: (-1x96x40x40xf32) <- (-1x96x40x40xf32)
        swish_21 = paddle._C_ops.swish(add__8)

        # pd_op.add_: (-1x96x40x40xf32) <- (-1x96x40x40xf32, -1x96x40x40xf32)
        add__9 = paddle._C_ops.add_(swish_19, swish_21)

        # pd_op.conv2d: (-1x96x40x40xf32) <- (-1x96x40x40xf32, 96x96x3x3xf32)
        conv2d_24 = paddle._C_ops.conv2d(add__9, parameter_102, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x40x40xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x40x40xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_24, parameter_103, parameter_104, parameter_105, parameter_106, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x96x40x40xf32) <- (-1x96x40x40xf32)
        swish_22 = paddle._C_ops.swish(batch_norm__108)

        # pd_op.conv2d: (-1x96x40x40xf32) <- (-1x96x40x40xf32, 96x96x3x3xf32)
        conv2d_25 = paddle._C_ops.conv2d(swish_22, parameter_107, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_6 = [1, 96, 1, 1]

        # pd_op.reshape: (1x96x1x1xf32, 0x96xf32) <- (96xf32, 4xi64)
        reshape_12, reshape_13 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_108, full_int_array_6), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x96x40x40xf32) <- (-1x96x40x40xf32, 1x96x1x1xf32)
        add__10 = paddle._C_ops.add_(conv2d_25, reshape_12)

        # pd_op.swish: (-1x96x40x40xf32) <- (-1x96x40x40xf32)
        swish_23 = paddle._C_ops.swish(add__10)

        # pd_op.add_: (-1x96x40x40xf32) <- (-1x96x40x40xf32, -1x96x40x40xf32)
        add__11 = paddle._C_ops.add_(add__9, swish_23)

        # builtin.combine: ([-1x96x40x40xf32, -1x96x40x40xf32]) <- (-1x96x40x40xf32, -1x96x40x40xf32)
        combine_2 = [swish_18, add__11]

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x192x40x40xf32) <- ([-1x96x40x40xf32, -1x96x40x40xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_2)

        # pd_op.mean: (-1x192x1x1xf32) <- (-1x192x40x40xf32)
        mean_2 = paddle._C_ops.mean(concat_2, [2, 3], True)

        # pd_op.conv2d: (-1x192x1x1xf32) <- (-1x192x1x1xf32, 192x192x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(mean_2, parameter_109, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_7 = [1, 192, 1, 1]

        # pd_op.reshape: (1x192x1x1xf32, 0x192xf32) <- (192xf32, 4xi64)
        reshape_14, reshape_15 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_110, full_int_array_7), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x192x1x1xf32) <- (-1x192x1x1xf32, 1x192x1x1xf32)
        add__12 = paddle._C_ops.add_(conv2d_26, reshape_14)

        # pd_op.hardsigmoid: (-1x192x1x1xf32) <- (-1x192x1x1xf32)
        hardsigmoid_2 = paddle._C_ops.hardsigmoid(add__12, float('0.166667'), float('0.5'))

        # pd_op.multiply_: (-1x192x40x40xf32) <- (-1x192x40x40xf32, -1x192x1x1xf32)
        multiply__2 = paddle._C_ops.multiply_(concat_2, hardsigmoid_2)

        # pd_op.conv2d: (-1x256x40x40xf32) <- (-1x192x40x40xf32, 256x192x1x1xf32)
        conv2d_27 = paddle._C_ops.conv2d(multiply__2, parameter_111, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x40x40xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x40x40xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_27, parameter_112, parameter_113, parameter_114, parameter_115, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x256x40x40xf32) <- (-1x256x40x40xf32)
        swish_24 = paddle._C_ops.swish(batch_norm__114)

        # pd_op.conv2d: (-1x384x20x20xf32) <- (-1x256x40x40xf32, 384x256x3x3xf32)
        conv2d_28 = paddle._C_ops.conv2d(swish_24, parameter_116, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x384x20x20xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_28, parameter_117, parameter_118, parameter_119, parameter_120, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x384x20x20xf32) <- (-1x384x20x20xf32)
        swish_25 = paddle._C_ops.swish(batch_norm__120)

        # pd_op.conv2d: (-1x192x20x20xf32) <- (-1x384x20x20xf32, 192x384x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(swish_25, parameter_121, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x192x20x20xf32, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x20x20xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_29, parameter_122, parameter_123, parameter_124, parameter_125, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x192x20x20xf32) <- (-1x192x20x20xf32)
        swish_26 = paddle._C_ops.swish(batch_norm__126)

        # pd_op.conv2d: (-1x192x20x20xf32) <- (-1x384x20x20xf32, 192x384x1x1xf32)
        conv2d_30 = paddle._C_ops.conv2d(swish_25, parameter_126, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x192x20x20xf32, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x20x20xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_30, parameter_127, parameter_128, parameter_129, parameter_130, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x192x20x20xf32) <- (-1x192x20x20xf32)
        swish_27 = paddle._C_ops.swish(batch_norm__132)

        # pd_op.conv2d: (-1x192x20x20xf32) <- (-1x192x20x20xf32, 192x192x3x3xf32)
        conv2d_31 = paddle._C_ops.conv2d(swish_27, parameter_131, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x192x20x20xf32, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x20x20xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_31, parameter_132, parameter_133, parameter_134, parameter_135, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x192x20x20xf32) <- (-1x192x20x20xf32)
        swish_28 = paddle._C_ops.swish(batch_norm__138)

        # pd_op.conv2d: (-1x192x20x20xf32) <- (-1x192x20x20xf32, 192x192x3x3xf32)
        conv2d_32 = paddle._C_ops.conv2d(swish_28, parameter_136, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_8 = [1, 192, 1, 1]

        # pd_op.reshape: (1x192x1x1xf32, 0x192xf32) <- (192xf32, 4xi64)
        reshape_16, reshape_17 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_137, full_int_array_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x192x20x20xf32) <- (-1x192x20x20xf32, 1x192x1x1xf32)
        add__13 = paddle._C_ops.add_(conv2d_32, reshape_16)

        # pd_op.swish: (-1x192x20x20xf32) <- (-1x192x20x20xf32)
        swish_29 = paddle._C_ops.swish(add__13)

        # pd_op.add_: (-1x192x20x20xf32) <- (-1x192x20x20xf32, -1x192x20x20xf32)
        add__14 = paddle._C_ops.add_(swish_27, swish_29)

        # builtin.combine: ([-1x192x20x20xf32, -1x192x20x20xf32]) <- (-1x192x20x20xf32, -1x192x20x20xf32)
        combine_3 = [swish_26, add__14]

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x384x20x20xf32) <- ([-1x192x20x20xf32, -1x192x20x20xf32], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_3, full_3)

        # pd_op.mean: (-1x384x1x1xf32) <- (-1x384x20x20xf32)
        mean_3 = paddle._C_ops.mean(concat_3, [2, 3], True)

        # pd_op.conv2d: (-1x384x1x1xf32) <- (-1x384x1x1xf32, 384x384x1x1xf32)
        conv2d_33 = paddle._C_ops.conv2d(mean_3, parameter_138, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_9 = [1, 384, 1, 1]

        # pd_op.reshape: (1x384x1x1xf32, 0x384xf32) <- (384xf32, 4xi64)
        reshape_18, reshape_19 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_139, full_int_array_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x384x1x1xf32) <- (-1x384x1x1xf32, 1x384x1x1xf32)
        add__15 = paddle._C_ops.add_(conv2d_33, reshape_18)

        # pd_op.hardsigmoid: (-1x384x1x1xf32) <- (-1x384x1x1xf32)
        hardsigmoid_3 = paddle._C_ops.hardsigmoid(add__15, float('0.166667'), float('0.5'))

        # pd_op.multiply_: (-1x384x20x20xf32) <- (-1x384x20x20xf32, -1x384x1x1xf32)
        multiply__3 = paddle._C_ops.multiply_(concat_3, hardsigmoid_3)

        # pd_op.conv2d: (-1x512x20x20xf32) <- (-1x384x20x20xf32, 512x384x1x1xf32)
        conv2d_34 = paddle._C_ops.conv2d(multiply__3, parameter_140, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x20x20xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x20x20xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_34, parameter_141, parameter_142, parameter_143, parameter_144, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x512x20x20xf32) <- (-1x512x20x20xf32)
        swish_30 = paddle._C_ops.swish(batch_norm__144)

        # pd_op.conv2d: (-1x192x20x20xf32) <- (-1x512x20x20xf32, 192x512x1x1xf32)
        conv2d_35 = paddle._C_ops.conv2d(swish_30, parameter_145, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x192x20x20xf32, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x20x20xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_35, parameter_146, parameter_147, parameter_148, parameter_149, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x192x20x20xf32) <- (-1x192x20x20xf32)
        swish_31 = paddle._C_ops.swish(batch_norm__150)

        # pd_op.conv2d: (-1x192x20x20xf32) <- (-1x512x20x20xf32, 192x512x1x1xf32)
        conv2d_36 = paddle._C_ops.conv2d(swish_30, parameter_150, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x192x20x20xf32, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x20x20xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_36, parameter_151, parameter_152, parameter_153, parameter_154, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x192x20x20xf32) <- (-1x192x20x20xf32)
        swish_32 = paddle._C_ops.swish(batch_norm__156)

        # pd_op.conv2d: (-1x192x20x20xf32) <- (-1x192x20x20xf32, 192x192x3x3xf32)
        conv2d_37 = paddle._C_ops.conv2d(swish_32, parameter_155, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x192x20x20xf32, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x20x20xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_37, parameter_156, parameter_157, parameter_158, parameter_159, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x192x20x20xf32) <- (-1x192x20x20xf32)
        swish_33 = paddle._C_ops.swish(batch_norm__162)

        # pd_op.conv2d: (-1x192x20x20xf32) <- (-1x192x20x20xf32, 192x192x3x3xf32)
        conv2d_38 = paddle._C_ops.conv2d(swish_33, parameter_160, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_10 = [1, 192, 1, 1]

        # pd_op.reshape: (1x192x1x1xf32, 0x192xf32) <- (192xf32, 4xi64)
        reshape_20, reshape_21 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_161, full_int_array_10), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x192x20x20xf32) <- (-1x192x20x20xf32, 1x192x1x1xf32)
        add__16 = paddle._C_ops.add_(conv2d_38, reshape_20)

        # pd_op.swish: (-1x192x20x20xf32) <- (-1x192x20x20xf32)
        swish_34 = paddle._C_ops.swish(add__16)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_11 = [5, 5]

        # pd_op.pool2d: (-1x192x20x20xf32) <- (-1x192x20x20xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(swish_34, full_int_array_11, [1, 1], [2, 2], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_12 = [9, 9]

        # pd_op.pool2d: (-1x192x20x20xf32) <- (-1x192x20x20xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(swish_34, full_int_array_12, [1, 1], [4, 4], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_13 = [13, 13]

        # pd_op.pool2d: (-1x192x20x20xf32) <- (-1x192x20x20xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(swish_34, full_int_array_13, [1, 1], [6, 6], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # builtin.combine: ([-1x192x20x20xf32, -1x192x20x20xf32, -1x192x20x20xf32, -1x192x20x20xf32]) <- (-1x192x20x20xf32, -1x192x20x20xf32, -1x192x20x20xf32, -1x192x20x20xf32)
        combine_4 = [swish_34, pool2d_0, pool2d_1, pool2d_2]

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x768x20x20xf32) <- ([-1x192x20x20xf32, -1x192x20x20xf32, -1x192x20x20xf32, -1x192x20x20xf32], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_4, full_4)

        # pd_op.conv2d: (-1x192x20x20xf32) <- (-1x768x20x20xf32, 192x768x1x1xf32)
        conv2d_39 = paddle._C_ops.conv2d(concat_4, parameter_162, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x192x20x20xf32, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x20x20xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_39, parameter_163, parameter_164, parameter_165, parameter_166, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x192x20x20xf32) <- (-1x192x20x20xf32)
        swish_35 = paddle._C_ops.swish(batch_norm__168)

        # builtin.combine: ([-1x192x20x20xf32, -1x192x20x20xf32]) <- (-1x192x20x20xf32, -1x192x20x20xf32)
        combine_5 = [swish_31, swish_35]

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x384x20x20xf32) <- ([-1x192x20x20xf32, -1x192x20x20xf32], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_5, full_5)

        # pd_op.conv2d: (-1x384x20x20xf32) <- (-1x384x20x20xf32, 384x384x1x1xf32)
        conv2d_40 = paddle._C_ops.conv2d(concat_5, parameter_167, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x384x20x20xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_40, parameter_168, parameter_169, parameter_170, parameter_171, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x384x20x20xf32) <- (-1x384x20x20xf32)
        swish_36 = paddle._C_ops.swish(batch_norm__174)

        # pd_op.conv2d: (-1x192x20x20xf32) <- (-1x384x20x20xf32, 192x384x1x1xf32)
        conv2d_41 = paddle._C_ops.conv2d(swish_36, parameter_172, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x192x20x20xf32, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x20x20xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__180, batch_norm__181, batch_norm__182, batch_norm__183, batch_norm__184, batch_norm__185 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_41, parameter_173, parameter_174, parameter_175, parameter_176, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x192x20x20xf32) <- (-1x192x20x20xf32)
        swish_37 = paddle._C_ops.swish(batch_norm__180)

        # pd_op.nearest_interp: (-1x192x40x40xf32) <- (-1x192x20x20xf32, None, None, None)
        nearest_interp_0 = paddle._C_ops.nearest_interp(swish_37, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 0)

        # builtin.combine: ([-1x192x40x40xf32, -1x256x40x40xf32]) <- (-1x192x40x40xf32, -1x256x40x40xf32)
        combine_6 = [nearest_interp_0, swish_24]

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x448x40x40xf32) <- ([-1x192x40x40xf32, -1x256x40x40xf32], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_6, full_6)

        # pd_op.conv2d: (-1x96x40x40xf32) <- (-1x448x40x40xf32, 96x448x1x1xf32)
        conv2d_42 = paddle._C_ops.conv2d(concat_6, parameter_177, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x40x40xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x40x40xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__186, batch_norm__187, batch_norm__188, batch_norm__189, batch_norm__190, batch_norm__191 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_42, parameter_178, parameter_179, parameter_180, parameter_181, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x96x40x40xf32) <- (-1x96x40x40xf32)
        swish_38 = paddle._C_ops.swish(batch_norm__186)

        # pd_op.conv2d: (-1x96x40x40xf32) <- (-1x448x40x40xf32, 96x448x1x1xf32)
        conv2d_43 = paddle._C_ops.conv2d(concat_6, parameter_182, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x40x40xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x40x40xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__192, batch_norm__193, batch_norm__194, batch_norm__195, batch_norm__196, batch_norm__197 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_43, parameter_183, parameter_184, parameter_185, parameter_186, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x96x40x40xf32) <- (-1x96x40x40xf32)
        swish_39 = paddle._C_ops.swish(batch_norm__192)

        # pd_op.conv2d: (-1x96x40x40xf32) <- (-1x96x40x40xf32, 96x96x3x3xf32)
        conv2d_44 = paddle._C_ops.conv2d(swish_39, parameter_187, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x40x40xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x40x40xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__198, batch_norm__199, batch_norm__200, batch_norm__201, batch_norm__202, batch_norm__203 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_44, parameter_188, parameter_189, parameter_190, parameter_191, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x96x40x40xf32) <- (-1x96x40x40xf32)
        swish_40 = paddle._C_ops.swish(batch_norm__198)

        # pd_op.conv2d: (-1x96x40x40xf32) <- (-1x96x40x40xf32, 96x96x3x3xf32)
        conv2d_45 = paddle._C_ops.conv2d(swish_40, parameter_192, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_14 = [1, 96, 1, 1]

        # pd_op.reshape: (1x96x1x1xf32, 0x96xf32) <- (96xf32, 4xi64)
        reshape_22, reshape_23 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_193, full_int_array_14), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x96x40x40xf32) <- (-1x96x40x40xf32, 1x96x1x1xf32)
        add__17 = paddle._C_ops.add_(conv2d_45, reshape_22)

        # pd_op.swish: (-1x96x40x40xf32) <- (-1x96x40x40xf32)
        swish_41 = paddle._C_ops.swish(add__17)

        # builtin.combine: ([-1x96x40x40xf32, -1x96x40x40xf32]) <- (-1x96x40x40xf32, -1x96x40x40xf32)
        combine_7 = [swish_38, swish_41]

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x192x40x40xf32) <- ([-1x96x40x40xf32, -1x96x40x40xf32], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_7, full_7)

        # pd_op.conv2d: (-1x192x40x40xf32) <- (-1x192x40x40xf32, 192x192x1x1xf32)
        conv2d_46 = paddle._C_ops.conv2d(concat_7, parameter_194, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x192x40x40xf32, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__204, batch_norm__205, batch_norm__206, batch_norm__207, batch_norm__208, batch_norm__209 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_46, parameter_195, parameter_196, parameter_197, parameter_198, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x192x40x40xf32) <- (-1x192x40x40xf32)
        swish_42 = paddle._C_ops.swish(batch_norm__204)

        # pd_op.conv2d: (-1x96x40x40xf32) <- (-1x192x40x40xf32, 96x192x1x1xf32)
        conv2d_47 = paddle._C_ops.conv2d(swish_42, parameter_199, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x40x40xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x40x40xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__210, batch_norm__211, batch_norm__212, batch_norm__213, batch_norm__214, batch_norm__215 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_47, parameter_200, parameter_201, parameter_202, parameter_203, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x96x40x40xf32) <- (-1x96x40x40xf32)
        swish_43 = paddle._C_ops.swish(batch_norm__210)

        # pd_op.nearest_interp: (-1x96x80x80xf32) <- (-1x96x40x40xf32, None, None, None)
        nearest_interp_1 = paddle._C_ops.nearest_interp(swish_43, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 0)

        # builtin.combine: ([-1x96x80x80xf32, -1x128x80x80xf32]) <- (-1x96x80x80xf32, -1x128x80x80xf32)
        combine_8 = [nearest_interp_1, swish_16]

        # pd_op.full: (1xi32) <- ()
        full_8 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x224x80x80xf32) <- ([-1x96x80x80xf32, -1x128x80x80xf32], 1xi32)
        concat_8 = paddle._C_ops.concat(combine_8, full_8)

        # pd_op.conv2d: (-1x48x80x80xf32) <- (-1x224x80x80xf32, 48x224x1x1xf32)
        conv2d_48 = paddle._C_ops.conv2d(concat_8, parameter_204, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x80x80xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x80x80xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__216, batch_norm__217, batch_norm__218, batch_norm__219, batch_norm__220, batch_norm__221 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_48, parameter_205, parameter_206, parameter_207, parameter_208, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x48x80x80xf32) <- (-1x48x80x80xf32)
        swish_44 = paddle._C_ops.swish(batch_norm__216)

        # pd_op.conv2d: (-1x48x80x80xf32) <- (-1x224x80x80xf32, 48x224x1x1xf32)
        conv2d_49 = paddle._C_ops.conv2d(concat_8, parameter_209, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x80x80xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x80x80xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__222, batch_norm__223, batch_norm__224, batch_norm__225, batch_norm__226, batch_norm__227 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_49, parameter_210, parameter_211, parameter_212, parameter_213, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x48x80x80xf32) <- (-1x48x80x80xf32)
        swish_45 = paddle._C_ops.swish(batch_norm__222)

        # pd_op.conv2d: (-1x48x80x80xf32) <- (-1x48x80x80xf32, 48x48x3x3xf32)
        conv2d_50 = paddle._C_ops.conv2d(swish_45, parameter_214, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x80x80xf32, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x80x80xf32, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__228, batch_norm__229, batch_norm__230, batch_norm__231, batch_norm__232, batch_norm__233 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_50, parameter_215, parameter_216, parameter_217, parameter_218, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x48x80x80xf32) <- (-1x48x80x80xf32)
        swish_46 = paddle._C_ops.swish(batch_norm__228)

        # pd_op.conv2d: (-1x48x80x80xf32) <- (-1x48x80x80xf32, 48x48x3x3xf32)
        conv2d_51 = paddle._C_ops.conv2d(swish_46, parameter_219, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_15 = [1, 48, 1, 1]

        # pd_op.reshape: (1x48x1x1xf32, 0x48xf32) <- (48xf32, 4xi64)
        reshape_24, reshape_25 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_220, full_int_array_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x48x80x80xf32) <- (-1x48x80x80xf32, 1x48x1x1xf32)
        add__18 = paddle._C_ops.add_(conv2d_51, reshape_24)

        # pd_op.swish: (-1x48x80x80xf32) <- (-1x48x80x80xf32)
        swish_47 = paddle._C_ops.swish(add__18)

        # builtin.combine: ([-1x48x80x80xf32, -1x48x80x80xf32]) <- (-1x48x80x80xf32, -1x48x80x80xf32)
        combine_9 = [swish_44, swish_47]

        # pd_op.full: (1xi32) <- ()
        full_9 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x96x80x80xf32) <- ([-1x48x80x80xf32, -1x48x80x80xf32], 1xi32)
        concat_9 = paddle._C_ops.concat(combine_9, full_9)

        # pd_op.conv2d: (-1x96x80x80xf32) <- (-1x96x80x80xf32, 96x96x1x1xf32)
        conv2d_52 = paddle._C_ops.conv2d(concat_9, parameter_221, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x80x80xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x80x80xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__234, batch_norm__235, batch_norm__236, batch_norm__237, batch_norm__238, batch_norm__239 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_52, parameter_222, parameter_223, parameter_224, parameter_225, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x96x80x80xf32) <- (-1x96x80x80xf32)
        swish_48 = paddle._C_ops.swish(batch_norm__234)

        # pd_op.conv2d: (-1x96x40x40xf32) <- (-1x96x80x80xf32, 96x96x3x3xf32)
        conv2d_53 = paddle._C_ops.conv2d(swish_48, parameter_226, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x40x40xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x40x40xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__240, batch_norm__241, batch_norm__242, batch_norm__243, batch_norm__244, batch_norm__245 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_53, parameter_227, parameter_228, parameter_229, parameter_230, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x96x40x40xf32) <- (-1x96x40x40xf32)
        swish_49 = paddle._C_ops.swish(batch_norm__240)

        # builtin.combine: ([-1x96x40x40xf32, -1x192x40x40xf32]) <- (-1x96x40x40xf32, -1x192x40x40xf32)
        combine_10 = [swish_49, swish_42]

        # pd_op.full: (1xi32) <- ()
        full_10 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x288x40x40xf32) <- ([-1x96x40x40xf32, -1x192x40x40xf32], 1xi32)
        concat_10 = paddle._C_ops.concat(combine_10, full_10)

        # pd_op.conv2d: (-1x96x40x40xf32) <- (-1x288x40x40xf32, 96x288x1x1xf32)
        conv2d_54 = paddle._C_ops.conv2d(concat_10, parameter_231, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x40x40xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x40x40xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__246, batch_norm__247, batch_norm__248, batch_norm__249, batch_norm__250, batch_norm__251 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_54, parameter_232, parameter_233, parameter_234, parameter_235, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x96x40x40xf32) <- (-1x96x40x40xf32)
        swish_50 = paddle._C_ops.swish(batch_norm__246)

        # pd_op.conv2d: (-1x96x40x40xf32) <- (-1x288x40x40xf32, 96x288x1x1xf32)
        conv2d_55 = paddle._C_ops.conv2d(concat_10, parameter_236, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x40x40xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x40x40xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__252, batch_norm__253, batch_norm__254, batch_norm__255, batch_norm__256, batch_norm__257 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_55, parameter_237, parameter_238, parameter_239, parameter_240, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x96x40x40xf32) <- (-1x96x40x40xf32)
        swish_51 = paddle._C_ops.swish(batch_norm__252)

        # pd_op.conv2d: (-1x96x40x40xf32) <- (-1x96x40x40xf32, 96x96x3x3xf32)
        conv2d_56 = paddle._C_ops.conv2d(swish_51, parameter_241, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x40x40xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x40x40xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__258, batch_norm__259, batch_norm__260, batch_norm__261, batch_norm__262, batch_norm__263 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_56, parameter_242, parameter_243, parameter_244, parameter_245, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x96x40x40xf32) <- (-1x96x40x40xf32)
        swish_52 = paddle._C_ops.swish(batch_norm__258)

        # pd_op.conv2d: (-1x96x40x40xf32) <- (-1x96x40x40xf32, 96x96x3x3xf32)
        conv2d_57 = paddle._C_ops.conv2d(swish_52, parameter_246, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_16 = [1, 96, 1, 1]

        # pd_op.reshape: (1x96x1x1xf32, 0x96xf32) <- (96xf32, 4xi64)
        reshape_26, reshape_27 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_247, full_int_array_16), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x96x40x40xf32) <- (-1x96x40x40xf32, 1x96x1x1xf32)
        add__19 = paddle._C_ops.add_(conv2d_57, reshape_26)

        # pd_op.swish: (-1x96x40x40xf32) <- (-1x96x40x40xf32)
        swish_53 = paddle._C_ops.swish(add__19)

        # builtin.combine: ([-1x96x40x40xf32, -1x96x40x40xf32]) <- (-1x96x40x40xf32, -1x96x40x40xf32)
        combine_11 = [swish_50, swish_53]

        # pd_op.full: (1xi32) <- ()
        full_11 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x192x40x40xf32) <- ([-1x96x40x40xf32, -1x96x40x40xf32], 1xi32)
        concat_11 = paddle._C_ops.concat(combine_11, full_11)

        # pd_op.conv2d: (-1x192x40x40xf32) <- (-1x192x40x40xf32, 192x192x1x1xf32)
        conv2d_58 = paddle._C_ops.conv2d(concat_11, parameter_248, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x192x40x40xf32, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__264, batch_norm__265, batch_norm__266, batch_norm__267, batch_norm__268, batch_norm__269 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_58, parameter_249, parameter_250, parameter_251, parameter_252, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x192x40x40xf32) <- (-1x192x40x40xf32)
        swish_54 = paddle._C_ops.swish(batch_norm__264)

        # pd_op.conv2d: (-1x192x20x20xf32) <- (-1x192x40x40xf32, 192x192x3x3xf32)
        conv2d_59 = paddle._C_ops.conv2d(swish_54, parameter_253, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x192x20x20xf32, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x20x20xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__270, batch_norm__271, batch_norm__272, batch_norm__273, batch_norm__274, batch_norm__275 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_59, parameter_254, parameter_255, parameter_256, parameter_257, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x192x20x20xf32) <- (-1x192x20x20xf32)
        swish_55 = paddle._C_ops.swish(batch_norm__270)

        # builtin.combine: ([-1x192x20x20xf32, -1x384x20x20xf32]) <- (-1x192x20x20xf32, -1x384x20x20xf32)
        combine_12 = [swish_55, swish_36]

        # pd_op.full: (1xi32) <- ()
        full_12 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x576x20x20xf32) <- ([-1x192x20x20xf32, -1x384x20x20xf32], 1xi32)
        concat_12 = paddle._C_ops.concat(combine_12, full_12)

        # pd_op.conv2d: (-1x192x20x20xf32) <- (-1x576x20x20xf32, 192x576x1x1xf32)
        conv2d_60 = paddle._C_ops.conv2d(concat_12, parameter_258, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x192x20x20xf32, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x20x20xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__276, batch_norm__277, batch_norm__278, batch_norm__279, batch_norm__280, batch_norm__281 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_60, parameter_259, parameter_260, parameter_261, parameter_262, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x192x20x20xf32) <- (-1x192x20x20xf32)
        swish_56 = paddle._C_ops.swish(batch_norm__276)

        # pd_op.conv2d: (-1x192x20x20xf32) <- (-1x576x20x20xf32, 192x576x1x1xf32)
        conv2d_61 = paddle._C_ops.conv2d(concat_12, parameter_263, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x192x20x20xf32, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x20x20xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__282, batch_norm__283, batch_norm__284, batch_norm__285, batch_norm__286, batch_norm__287 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_61, parameter_264, parameter_265, parameter_266, parameter_267, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x192x20x20xf32) <- (-1x192x20x20xf32)
        swish_57 = paddle._C_ops.swish(batch_norm__282)

        # pd_op.conv2d: (-1x192x20x20xf32) <- (-1x192x20x20xf32, 192x192x3x3xf32)
        conv2d_62 = paddle._C_ops.conv2d(swish_57, parameter_268, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x192x20x20xf32, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x20x20xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__288, batch_norm__289, batch_norm__290, batch_norm__291, batch_norm__292, batch_norm__293 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_62, parameter_269, parameter_270, parameter_271, parameter_272, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x192x20x20xf32) <- (-1x192x20x20xf32)
        swish_58 = paddle._C_ops.swish(batch_norm__288)

        # pd_op.conv2d: (-1x192x20x20xf32) <- (-1x192x20x20xf32, 192x192x3x3xf32)
        conv2d_63 = paddle._C_ops.conv2d(swish_58, parameter_273, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_17 = [1, 192, 1, 1]

        # pd_op.reshape: (1x192x1x1xf32, 0x192xf32) <- (192xf32, 4xi64)
        reshape_28, reshape_29 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_274, full_int_array_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x192x20x20xf32) <- (-1x192x20x20xf32, 1x192x1x1xf32)
        add__20 = paddle._C_ops.add_(conv2d_63, reshape_28)

        # pd_op.swish: (-1x192x20x20xf32) <- (-1x192x20x20xf32)
        swish_59 = paddle._C_ops.swish(add__20)

        # builtin.combine: ([-1x192x20x20xf32, -1x192x20x20xf32]) <- (-1x192x20x20xf32, -1x192x20x20xf32)
        combine_13 = [swish_56, swish_59]

        # pd_op.full: (1xi32) <- ()
        full_13 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x384x20x20xf32) <- ([-1x192x20x20xf32, -1x192x20x20xf32], 1xi32)
        concat_13 = paddle._C_ops.concat(combine_13, full_13)

        # pd_op.conv2d: (-1x384x20x20xf32) <- (-1x384x20x20xf32, 384x384x1x1xf32)
        conv2d_64 = paddle._C_ops.conv2d(concat_13, parameter_275, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x384x20x20xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__294, batch_norm__295, batch_norm__296, batch_norm__297, batch_norm__298, batch_norm__299 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_64, parameter_276, parameter_277, parameter_278, parameter_279, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x384x20x20xf32) <- (-1x384x20x20xf32)
        swish_60 = paddle._C_ops.swish(batch_norm__294)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_18 = [1, 1]

        # pd_op.pool2d: (-1x384x1x1xf32) <- (-1x384x20x20xf32, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(swish_60, full_int_array_18, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x384x1x1xf32) <- (-1x384x1x1xf32, 384x384x1x1xf32)
        conv2d_65 = paddle._C_ops.conv2d(pool2d_3, parameter_280, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_19 = [1, 384, 1, 1]

        # pd_op.reshape: (1x384x1x1xf32, 0x384xf32) <- (384xf32, 4xi64)
        reshape_30, reshape_31 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_281, full_int_array_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x384x1x1xf32) <- (-1x384x1x1xf32, 1x384x1x1xf32)
        add__21 = paddle._C_ops.add_(conv2d_65, reshape_30)

        # pd_op.sigmoid_: (-1x384x1x1xf32) <- (-1x384x1x1xf32)
        sigmoid__0 = paddle._C_ops.sigmoid_(add__21)

        # pd_op.multiply: (-1x384x20x20xf32) <- (-1x384x20x20xf32, -1x384x1x1xf32)
        multiply_0 = paddle._C_ops.multiply(swish_60, sigmoid__0)

        # pd_op.conv2d: (-1x384x20x20xf32) <- (-1x384x20x20xf32, 384x384x1x1xf32)
        conv2d_66 = paddle._C_ops.conv2d(multiply_0, parameter_282, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x384x20x20xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__300, batch_norm__301, batch_norm__302, batch_norm__303, batch_norm__304, batch_norm__305 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_66, parameter_283, parameter_284, parameter_285, parameter_286, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x384x20x20xf32) <- (-1x384x20x20xf32)
        swish_61 = paddle._C_ops.swish(batch_norm__300)

        # pd_op.add_: (-1x384x20x20xf32) <- (-1x384x20x20xf32, -1x384x20x20xf32)
        add__22 = paddle._C_ops.add_(swish_61, swish_60)

        # pd_op.conv2d: (-1x80x20x20xf32) <- (-1x384x20x20xf32, 80x384x3x3xf32)
        conv2d_67 = paddle._C_ops.conv2d(add__22, parameter_287, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_20 = [1, 80, 1, 1]

        # pd_op.reshape: (1x80x1x1xf32, 0x80xf32) <- (80xf32, 4xi64)
        reshape_32, reshape_33 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_288, full_int_array_20), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x80x20x20xf32) <- (-1x80x20x20xf32, 1x80x1x1xf32)
        add__23 = paddle._C_ops.add_(conv2d_67, reshape_32)

        # pd_op.conv2d: (-1x384x1x1xf32) <- (-1x384x1x1xf32, 384x384x1x1xf32)
        conv2d_68 = paddle._C_ops.conv2d(pool2d_3, parameter_289, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_21 = [1, 384, 1, 1]

        # pd_op.reshape: (1x384x1x1xf32, 0x384xf32) <- (384xf32, 4xi64)
        reshape_34, reshape_35 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_290, full_int_array_21), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x384x1x1xf32) <- (-1x384x1x1xf32, 1x384x1x1xf32)
        add__24 = paddle._C_ops.add_(conv2d_68, reshape_34)

        # pd_op.sigmoid_: (-1x384x1x1xf32) <- (-1x384x1x1xf32)
        sigmoid__1 = paddle._C_ops.sigmoid_(add__24)

        # pd_op.multiply_: (-1x384x20x20xf32) <- (-1x384x20x20xf32, -1x384x1x1xf32)
        multiply__4 = paddle._C_ops.multiply_(swish_60, sigmoid__1)

        # pd_op.conv2d: (-1x384x20x20xf32) <- (-1x384x20x20xf32, 384x384x1x1xf32)
        conv2d_69 = paddle._C_ops.conv2d(multiply__4, parameter_291, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x384x20x20xf32, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x20x20xf32, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__306, batch_norm__307, batch_norm__308, batch_norm__309, batch_norm__310, batch_norm__311 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_69, parameter_292, parameter_293, parameter_294, parameter_295, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x384x20x20xf32) <- (-1x384x20x20xf32)
        swish_62 = paddle._C_ops.swish(batch_norm__306)

        # pd_op.conv2d: (-1x68x20x20xf32) <- (-1x384x20x20xf32, 68x384x3x3xf32)
        conv2d_70 = paddle._C_ops.conv2d(swish_62, parameter_296, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_22 = [1, 68, 1, 1]

        # pd_op.reshape: (1x68x1x1xf32, 0x68xf32) <- (68xf32, 4xi64)
        reshape_36, reshape_37 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_297, full_int_array_22), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x68x20x20xf32) <- (-1x68x20x20xf32, 1x68x1x1xf32)
        add__25 = paddle._C_ops.add_(conv2d_70, reshape_36)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_23 = [-1, 4, 17, 400]

        # pd_op.reshape_: (-1x4x17x400xf32, 0x-1x68x20x20xf32) <- (-1x68x20x20xf32, 4xi64)
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__25, full_int_array_23), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x17x400x4xf32) <- (-1x4x17x400xf32)
        transpose_0 = paddle._C_ops.transpose(reshape__0, [0, 2, 3, 1])

        # pd_op.softmax_: (-1x17x400x4xf32) <- (-1x17x400x4xf32)
        softmax__0 = paddle._C_ops.softmax_(transpose_0, 1)

        # pd_op.conv2d: (-1x1x400x4xf32) <- (-1x17x400x4xf32, 1x17x1x1xf32)
        conv2d_71 = paddle._C_ops.conv2d(softmax__0, parameter_298, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_24 = [1]

        # pd_op.squeeze_: (-1x400x4xf32, None) <- (-1x1x400x4xf32, 1xi64)
        squeeze__0, squeeze__1 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(conv2d_71, full_int_array_24), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.sigmoid_: (-1x80x20x20xf32) <- (-1x80x20x20xf32)
        sigmoid__2 = paddle._C_ops.sigmoid_(add__23)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_25 = [-1, 80, 400]

        # pd_op.reshape_: (-1x80x400xf32, 0x-1x80x20x20xf32) <- (-1x80x20x20xf32, 3xi64)
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sigmoid__2, full_int_array_25), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_26 = [1, 1]

        # pd_op.pool2d: (-1x192x1x1xf32) <- (-1x192x40x40xf32, 2xi64)
        pool2d_4 = paddle._C_ops.pool2d(swish_54, full_int_array_26, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x192x1x1xf32) <- (-1x192x1x1xf32, 192x192x1x1xf32)
        conv2d_72 = paddle._C_ops.conv2d(pool2d_4, parameter_299, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_27 = [1, 192, 1, 1]

        # pd_op.reshape: (1x192x1x1xf32, 0x192xf32) <- (192xf32, 4xi64)
        reshape_38, reshape_39 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_300, full_int_array_27), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x192x1x1xf32) <- (-1x192x1x1xf32, 1x192x1x1xf32)
        add__26 = paddle._C_ops.add_(conv2d_72, reshape_38)

        # pd_op.sigmoid_: (-1x192x1x1xf32) <- (-1x192x1x1xf32)
        sigmoid__3 = paddle._C_ops.sigmoid_(add__26)

        # pd_op.multiply: (-1x192x40x40xf32) <- (-1x192x40x40xf32, -1x192x1x1xf32)
        multiply_1 = paddle._C_ops.multiply(swish_54, sigmoid__3)

        # pd_op.conv2d: (-1x192x40x40xf32) <- (-1x192x40x40xf32, 192x192x1x1xf32)
        conv2d_73 = paddle._C_ops.conv2d(multiply_1, parameter_301, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x192x40x40xf32, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__312, batch_norm__313, batch_norm__314, batch_norm__315, batch_norm__316, batch_norm__317 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_73, parameter_302, parameter_303, parameter_304, parameter_305, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x192x40x40xf32) <- (-1x192x40x40xf32)
        swish_63 = paddle._C_ops.swish(batch_norm__312)

        # pd_op.add_: (-1x192x40x40xf32) <- (-1x192x40x40xf32, -1x192x40x40xf32)
        add__27 = paddle._C_ops.add_(swish_63, swish_54)

        # pd_op.conv2d: (-1x80x40x40xf32) <- (-1x192x40x40xf32, 80x192x3x3xf32)
        conv2d_74 = paddle._C_ops.conv2d(add__27, parameter_306, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_28 = [1, 80, 1, 1]

        # pd_op.reshape: (1x80x1x1xf32, 0x80xf32) <- (80xf32, 4xi64)
        reshape_40, reshape_41 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_307, full_int_array_28), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x80x40x40xf32) <- (-1x80x40x40xf32, 1x80x1x1xf32)
        add__28 = paddle._C_ops.add_(conv2d_74, reshape_40)

        # pd_op.conv2d: (-1x192x1x1xf32) <- (-1x192x1x1xf32, 192x192x1x1xf32)
        conv2d_75 = paddle._C_ops.conv2d(pool2d_4, parameter_308, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_29 = [1, 192, 1, 1]

        # pd_op.reshape: (1x192x1x1xf32, 0x192xf32) <- (192xf32, 4xi64)
        reshape_42, reshape_43 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_309, full_int_array_29), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x192x1x1xf32) <- (-1x192x1x1xf32, 1x192x1x1xf32)
        add__29 = paddle._C_ops.add_(conv2d_75, reshape_42)

        # pd_op.sigmoid_: (-1x192x1x1xf32) <- (-1x192x1x1xf32)
        sigmoid__4 = paddle._C_ops.sigmoid_(add__29)

        # pd_op.multiply_: (-1x192x40x40xf32) <- (-1x192x40x40xf32, -1x192x1x1xf32)
        multiply__5 = paddle._C_ops.multiply_(swish_54, sigmoid__4)

        # pd_op.conv2d: (-1x192x40x40xf32) <- (-1x192x40x40xf32, 192x192x1x1xf32)
        conv2d_76 = paddle._C_ops.conv2d(multiply__5, parameter_310, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x192x40x40xf32, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x40x40xf32, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__318, batch_norm__319, batch_norm__320, batch_norm__321, batch_norm__322, batch_norm__323 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_76, parameter_311, parameter_312, parameter_313, parameter_314, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x192x40x40xf32) <- (-1x192x40x40xf32)
        swish_64 = paddle._C_ops.swish(batch_norm__318)

        # pd_op.conv2d: (-1x68x40x40xf32) <- (-1x192x40x40xf32, 68x192x3x3xf32)
        conv2d_77 = paddle._C_ops.conv2d(swish_64, parameter_315, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_30 = [1, 68, 1, 1]

        # pd_op.reshape: (1x68x1x1xf32, 0x68xf32) <- (68xf32, 4xi64)
        reshape_44, reshape_45 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_316, full_int_array_30), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x68x40x40xf32) <- (-1x68x40x40xf32, 1x68x1x1xf32)
        add__30 = paddle._C_ops.add_(conv2d_77, reshape_44)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_31 = [-1, 4, 17, 1600]

        # pd_op.reshape_: (-1x4x17x1600xf32, 0x-1x68x40x40xf32) <- (-1x68x40x40xf32, 4xi64)
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__30, full_int_array_31), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x17x1600x4xf32) <- (-1x4x17x1600xf32)
        transpose_1 = paddle._C_ops.transpose(reshape__4, [0, 2, 3, 1])

        # pd_op.softmax_: (-1x17x1600x4xf32) <- (-1x17x1600x4xf32)
        softmax__1 = paddle._C_ops.softmax_(transpose_1, 1)

        # pd_op.conv2d: (-1x1x1600x4xf32) <- (-1x17x1600x4xf32, 1x17x1x1xf32)
        conv2d_78 = paddle._C_ops.conv2d(softmax__1, parameter_298, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_32 = [1]

        # pd_op.squeeze_: (-1x1600x4xf32, None) <- (-1x1x1600x4xf32, 1xi64)
        squeeze__2, squeeze__3 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(conv2d_78, full_int_array_32), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.sigmoid_: (-1x80x40x40xf32) <- (-1x80x40x40xf32)
        sigmoid__5 = paddle._C_ops.sigmoid_(add__28)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_33 = [-1, 80, 1600]

        # pd_op.reshape_: (-1x80x1600xf32, 0x-1x80x40x40xf32) <- (-1x80x40x40xf32, 3xi64)
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sigmoid__5, full_int_array_33), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_34 = [1, 1]

        # pd_op.pool2d: (-1x96x1x1xf32) <- (-1x96x80x80xf32, 2xi64)
        pool2d_5 = paddle._C_ops.pool2d(swish_48, full_int_array_34, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x96x1x1xf32) <- (-1x96x1x1xf32, 96x96x1x1xf32)
        conv2d_79 = paddle._C_ops.conv2d(pool2d_5, parameter_317, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_35 = [1, 96, 1, 1]

        # pd_op.reshape: (1x96x1x1xf32, 0x96xf32) <- (96xf32, 4xi64)
        reshape_46, reshape_47 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_318, full_int_array_35), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x96x1x1xf32) <- (-1x96x1x1xf32, 1x96x1x1xf32)
        add__31 = paddle._C_ops.add_(conv2d_79, reshape_46)

        # pd_op.sigmoid_: (-1x96x1x1xf32) <- (-1x96x1x1xf32)
        sigmoid__6 = paddle._C_ops.sigmoid_(add__31)

        # pd_op.multiply: (-1x96x80x80xf32) <- (-1x96x80x80xf32, -1x96x1x1xf32)
        multiply_2 = paddle._C_ops.multiply(swish_48, sigmoid__6)

        # pd_op.conv2d: (-1x96x80x80xf32) <- (-1x96x80x80xf32, 96x96x1x1xf32)
        conv2d_80 = paddle._C_ops.conv2d(multiply_2, parameter_319, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x80x80xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x80x80xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__324, batch_norm__325, batch_norm__326, batch_norm__327, batch_norm__328, batch_norm__329 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_80, parameter_320, parameter_321, parameter_322, parameter_323, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x96x80x80xf32) <- (-1x96x80x80xf32)
        swish_65 = paddle._C_ops.swish(batch_norm__324)

        # pd_op.add_: (-1x96x80x80xf32) <- (-1x96x80x80xf32, -1x96x80x80xf32)
        add__32 = paddle._C_ops.add_(swish_65, swish_48)

        # pd_op.conv2d: (-1x80x80x80xf32) <- (-1x96x80x80xf32, 80x96x3x3xf32)
        conv2d_81 = paddle._C_ops.conv2d(add__32, parameter_324, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_36 = [1, 80, 1, 1]

        # pd_op.reshape: (1x80x1x1xf32, 0x80xf32) <- (80xf32, 4xi64)
        reshape_48, reshape_49 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_325, full_int_array_36), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x80x80x80xf32) <- (-1x80x80x80xf32, 1x80x1x1xf32)
        add__33 = paddle._C_ops.add_(conv2d_81, reshape_48)

        # pd_op.conv2d: (-1x96x1x1xf32) <- (-1x96x1x1xf32, 96x96x1x1xf32)
        conv2d_82 = paddle._C_ops.conv2d(pool2d_5, parameter_326, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_37 = [1, 96, 1, 1]

        # pd_op.reshape: (1x96x1x1xf32, 0x96xf32) <- (96xf32, 4xi64)
        reshape_50, reshape_51 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_327, full_int_array_37), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x96x1x1xf32) <- (-1x96x1x1xf32, 1x96x1x1xf32)
        add__34 = paddle._C_ops.add_(conv2d_82, reshape_50)

        # pd_op.sigmoid_: (-1x96x1x1xf32) <- (-1x96x1x1xf32)
        sigmoid__7 = paddle._C_ops.sigmoid_(add__34)

        # pd_op.multiply_: (-1x96x80x80xf32) <- (-1x96x80x80xf32, -1x96x1x1xf32)
        multiply__6 = paddle._C_ops.multiply_(swish_48, sigmoid__7)

        # pd_op.conv2d: (-1x96x80x80xf32) <- (-1x96x80x80xf32, 96x96x1x1xf32)
        conv2d_83 = paddle._C_ops.conv2d(multiply__6, parameter_328, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x80x80xf32, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x80x80xf32, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__330, batch_norm__331, batch_norm__332, batch_norm__333, batch_norm__334, batch_norm__335 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_83, parameter_329, parameter_330, parameter_331, parameter_332, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.swish: (-1x96x80x80xf32) <- (-1x96x80x80xf32)
        swish_66 = paddle._C_ops.swish(batch_norm__330)

        # pd_op.conv2d: (-1x68x80x80xf32) <- (-1x96x80x80xf32, 68x96x3x3xf32)
        conv2d_84 = paddle._C_ops.conv2d(swish_66, parameter_333, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_38 = [1, 68, 1, 1]

        # pd_op.reshape: (1x68x1x1xf32, 0x68xf32) <- (68xf32, 4xi64)
        reshape_52, reshape_53 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_334, full_int_array_38), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x68x80x80xf32) <- (-1x68x80x80xf32, 1x68x1x1xf32)
        add__35 = paddle._C_ops.add_(conv2d_84, reshape_52)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_39 = [-1, 4, 17, 6400]

        # pd_op.reshape_: (-1x4x17x6400xf32, 0x-1x68x80x80xf32) <- (-1x68x80x80xf32, 4xi64)
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__35, full_int_array_39), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x17x6400x4xf32) <- (-1x4x17x6400xf32)
        transpose_2 = paddle._C_ops.transpose(reshape__8, [0, 2, 3, 1])

        # pd_op.softmax_: (-1x17x6400x4xf32) <- (-1x17x6400x4xf32)
        softmax__2 = paddle._C_ops.softmax_(transpose_2, 1)

        # pd_op.conv2d: (-1x1x6400x4xf32) <- (-1x17x6400x4xf32, 1x17x1x1xf32)
        conv2d_85 = paddle._C_ops.conv2d(softmax__2, parameter_298, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_40 = [1]

        # pd_op.squeeze_: (-1x6400x4xf32, None) <- (-1x1x6400x4xf32, 1xi64)
        squeeze__4, squeeze__5 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(conv2d_85, full_int_array_40), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.sigmoid_: (-1x80x80x80xf32) <- (-1x80x80x80xf32)
        sigmoid__8 = paddle._C_ops.sigmoid_(add__33)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_41 = [-1, 80, 6400]

        # pd_op.reshape_: (-1x80x6400xf32, 0x-1x80x80x80xf32) <- (-1x80x80x80xf32, 3xi64)
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sigmoid__8, full_int_array_41), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x80x400xf32, -1x80x1600xf32, -1x80x6400xf32]) <- (-1x80x400xf32, -1x80x1600xf32, -1x80x6400xf32)
        combine_14 = [reshape__2, reshape__6, reshape__10]

        # pd_op.full: (1xi32) <- ()
        full_14 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x80x8400xf32) <- ([-1x80x400xf32, -1x80x1600xf32, -1x80x6400xf32], 1xi32)
        concat_14 = paddle._C_ops.concat(combine_14, full_14)

        # builtin.combine: ([-1x400x4xf32, -1x1600x4xf32, -1x6400x4xf32]) <- (-1x400x4xf32, -1x1600x4xf32, -1x6400x4xf32)
        combine_15 = [squeeze__0, squeeze__2, squeeze__4]

        # pd_op.full: (1xi32) <- ()
        full_15 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x8400x4xf32) <- ([-1x400x4xf32, -1x1600x4xf32, -1x6400x4xf32], 1xi32)
        concat_15 = paddle._C_ops.concat(combine_15, full_15)

        # pd_op.full: (1xi32) <- ()
        full_16 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x8400x2xf32, -1x8400x2xf32]) <- (-1x8400x4xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(concat_15, 2, full_16)

        # builtin.slice: (-1x8400x2xf32) <- ([-1x8400x2xf32, -1x8400x2xf32])
        slice_0 = split_with_num_0[0]

        # pd_op.full: (1xf32) <- ()
        full_17 = paddle._C_ops.full([1], float('-1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x8400x2xf32) <- (-1x8400x2xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(slice_0, full_17, float('0'), True)

        # pd_op.add_: (-1x8400x2xf32) <- (-1x8400x2xf32, 8400x2xf32)
        add__36 = paddle._C_ops.add_(scale__0, parameter_335)

        # builtin.slice: (-1x8400x2xf32) <- ([-1x8400x2xf32, -1x8400x2xf32])
        slice_1 = split_with_num_0[1]

        # pd_op.add_: (-1x8400x2xf32) <- (-1x8400x2xf32, 8400x2xf32)
        add__37 = paddle._C_ops.add_(slice_1, parameter_335)

        # builtin.combine: ([-1x8400x2xf32, -1x8400x2xf32]) <- (-1x8400x2xf32, -1x8400x2xf32)
        combine_16 = [add__36, add__37]

        # pd_op.full: (1xi32) <- ()
        full_18 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x8400x4xf32) <- ([-1x8400x2xf32, -1x8400x2xf32], 1xi32)
        concat_16 = paddle._C_ops.concat(combine_16, full_18)

        # pd_op.multiply_: (-1x8400x4xf32) <- (-1x8400x4xf32, 8400x1xf32)
        multiply__7 = paddle._C_ops.multiply_(concat_16, parameter_336)

        # pd_op.full: (1xi32) <- ()
        full_19 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x1xf32, -1x1xf32]) <- (-1x2xf32, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(feed_1, 2, full_19)

        # builtin.slice: (-1x1xf32) <- ([-1x1xf32, -1x1xf32])
        slice_2 = split_with_num_1[1]

        # builtin.slice: (-1x1xf32) <- ([-1x1xf32, -1x1xf32])
        slice_3 = split_with_num_1[0]

        # builtin.combine: ([-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32]) <- (-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32)
        combine_17 = [slice_2, slice_3, slice_2, slice_3]

        # pd_op.full: (1xi32) <- ()
        full_20 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x4xf32) <- ([-1x1xf32, -1x1xf32, -1x1xf32, -1x1xf32], 1xi32)
        concat_17 = paddle._C_ops.concat(combine_17, full_20)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_42 = [-1, 1, 4]

        # pd_op.reshape_: (-1x1x4xf32, 0x-1x4xf32) <- (-1x4xf32, 3xi64)
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_17, full_int_array_42), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.divide_: (-1x8400x4xf32) <- (-1x8400x4xf32, -1x1x4xf32)
        divide__0 = paddle._C_ops.divide_(multiply__7, reshape__12)

        # pd_op.multiclass_nms3: (-1x6xf32, -1x1xi32, -1xi32) <- (-1x8400x4xf32, -1x80x8400xf32, None)
        multiclass_nms3_0, multiclass_nms3_1, multiclass_nms3_2 = (lambda x, f: f(x))(paddle._C_ops.multiclass_nms3(divide__0, concat_14, None, float('0.01'), 1000, 300, float('0.7'), True, float('1'), -1), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))
        return multiclass_nms3_0, multiclass_nms3_2



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

    def forward(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_36, parameter_37, parameter_38, parameter_39, parameter_43, parameter_40, parameter_42, parameter_41, parameter_44, parameter_48, parameter_45, parameter_47, parameter_46, parameter_49, parameter_53, parameter_50, parameter_52, parameter_51, parameter_54, parameter_58, parameter_55, parameter_57, parameter_56, parameter_59, parameter_63, parameter_60, parameter_62, parameter_61, parameter_64, parameter_65, parameter_66, parameter_70, parameter_67, parameter_69, parameter_68, parameter_71, parameter_72, parameter_73, parameter_74, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_101, parameter_102, parameter_106, parameter_103, parameter_105, parameter_104, parameter_107, parameter_108, parameter_109, parameter_110, parameter_111, parameter_115, parameter_112, parameter_114, parameter_113, parameter_116, parameter_120, parameter_117, parameter_119, parameter_118, parameter_121, parameter_125, parameter_122, parameter_124, parameter_123, parameter_126, parameter_130, parameter_127, parameter_129, parameter_128, parameter_131, parameter_135, parameter_132, parameter_134, parameter_133, parameter_136, parameter_137, parameter_138, parameter_139, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_154, parameter_151, parameter_153, parameter_152, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_161, parameter_162, parameter_166, parameter_163, parameter_165, parameter_164, parameter_167, parameter_171, parameter_168, parameter_170, parameter_169, parameter_172, parameter_176, parameter_173, parameter_175, parameter_174, parameter_177, parameter_181, parameter_178, parameter_180, parameter_179, parameter_182, parameter_186, parameter_183, parameter_185, parameter_184, parameter_187, parameter_191, parameter_188, parameter_190, parameter_189, parameter_192, parameter_193, parameter_194, parameter_198, parameter_195, parameter_197, parameter_196, parameter_199, parameter_203, parameter_200, parameter_202, parameter_201, parameter_204, parameter_208, parameter_205, parameter_207, parameter_206, parameter_209, parameter_213, parameter_210, parameter_212, parameter_211, parameter_214, parameter_218, parameter_215, parameter_217, parameter_216, parameter_219, parameter_220, parameter_221, parameter_225, parameter_222, parameter_224, parameter_223, parameter_226, parameter_230, parameter_227, parameter_229, parameter_228, parameter_231, parameter_235, parameter_232, parameter_234, parameter_233, parameter_236, parameter_240, parameter_237, parameter_239, parameter_238, parameter_241, parameter_245, parameter_242, parameter_244, parameter_243, parameter_246, parameter_247, parameter_248, parameter_252, parameter_249, parameter_251, parameter_250, parameter_253, parameter_257, parameter_254, parameter_256, parameter_255, parameter_258, parameter_262, parameter_259, parameter_261, parameter_260, parameter_263, parameter_267, parameter_264, parameter_266, parameter_265, parameter_268, parameter_272, parameter_269, parameter_271, parameter_270, parameter_273, parameter_274, parameter_275, parameter_279, parameter_276, parameter_278, parameter_277, parameter_280, parameter_281, parameter_282, parameter_286, parameter_283, parameter_285, parameter_284, parameter_287, parameter_288, parameter_289, parameter_290, parameter_291, parameter_295, parameter_292, parameter_294, parameter_293, parameter_296, parameter_297, parameter_298, parameter_299, parameter_300, parameter_301, parameter_305, parameter_302, parameter_304, parameter_303, parameter_306, parameter_307, parameter_308, parameter_309, parameter_310, parameter_314, parameter_311, parameter_313, parameter_312, parameter_315, parameter_316, parameter_317, parameter_318, parameter_319, parameter_323, parameter_320, parameter_322, parameter_321, parameter_324, parameter_325, parameter_326, parameter_327, parameter_328, parameter_332, parameter_329, parameter_331, parameter_330, parameter_333, parameter_334, parameter_335, parameter_336, feed_1, feed_0):
        return self.builtin_module_782_0_0(parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_36, parameter_37, parameter_38, parameter_39, parameter_43, parameter_40, parameter_42, parameter_41, parameter_44, parameter_48, parameter_45, parameter_47, parameter_46, parameter_49, parameter_53, parameter_50, parameter_52, parameter_51, parameter_54, parameter_58, parameter_55, parameter_57, parameter_56, parameter_59, parameter_63, parameter_60, parameter_62, parameter_61, parameter_64, parameter_65, parameter_66, parameter_70, parameter_67, parameter_69, parameter_68, parameter_71, parameter_72, parameter_73, parameter_74, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_101, parameter_102, parameter_106, parameter_103, parameter_105, parameter_104, parameter_107, parameter_108, parameter_109, parameter_110, parameter_111, parameter_115, parameter_112, parameter_114, parameter_113, parameter_116, parameter_120, parameter_117, parameter_119, parameter_118, parameter_121, parameter_125, parameter_122, parameter_124, parameter_123, parameter_126, parameter_130, parameter_127, parameter_129, parameter_128, parameter_131, parameter_135, parameter_132, parameter_134, parameter_133, parameter_136, parameter_137, parameter_138, parameter_139, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_154, parameter_151, parameter_153, parameter_152, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_161, parameter_162, parameter_166, parameter_163, parameter_165, parameter_164, parameter_167, parameter_171, parameter_168, parameter_170, parameter_169, parameter_172, parameter_176, parameter_173, parameter_175, parameter_174, parameter_177, parameter_181, parameter_178, parameter_180, parameter_179, parameter_182, parameter_186, parameter_183, parameter_185, parameter_184, parameter_187, parameter_191, parameter_188, parameter_190, parameter_189, parameter_192, parameter_193, parameter_194, parameter_198, parameter_195, parameter_197, parameter_196, parameter_199, parameter_203, parameter_200, parameter_202, parameter_201, parameter_204, parameter_208, parameter_205, parameter_207, parameter_206, parameter_209, parameter_213, parameter_210, parameter_212, parameter_211, parameter_214, parameter_218, parameter_215, parameter_217, parameter_216, parameter_219, parameter_220, parameter_221, parameter_225, parameter_222, parameter_224, parameter_223, parameter_226, parameter_230, parameter_227, parameter_229, parameter_228, parameter_231, parameter_235, parameter_232, parameter_234, parameter_233, parameter_236, parameter_240, parameter_237, parameter_239, parameter_238, parameter_241, parameter_245, parameter_242, parameter_244, parameter_243, parameter_246, parameter_247, parameter_248, parameter_252, parameter_249, parameter_251, parameter_250, parameter_253, parameter_257, parameter_254, parameter_256, parameter_255, parameter_258, parameter_262, parameter_259, parameter_261, parameter_260, parameter_263, parameter_267, parameter_264, parameter_266, parameter_265, parameter_268, parameter_272, parameter_269, parameter_271, parameter_270, parameter_273, parameter_274, parameter_275, parameter_279, parameter_276, parameter_278, parameter_277, parameter_280, parameter_281, parameter_282, parameter_286, parameter_283, parameter_285, parameter_284, parameter_287, parameter_288, parameter_289, parameter_290, parameter_291, parameter_295, parameter_292, parameter_294, parameter_293, parameter_296, parameter_297, parameter_298, parameter_299, parameter_300, parameter_301, parameter_305, parameter_302, parameter_304, parameter_303, parameter_306, parameter_307, parameter_308, parameter_309, parameter_310, parameter_314, parameter_311, parameter_313, parameter_312, parameter_315, parameter_316, parameter_317, parameter_318, parameter_319, parameter_323, parameter_320, parameter_322, parameter_321, parameter_324, parameter_325, parameter_326, parameter_327, parameter_328, parameter_332, parameter_329, parameter_331, parameter_330, parameter_333, parameter_334, parameter_335, parameter_336, feed_1, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_782_0_0(CinnTestBase, unittest.TestCase):
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
            paddle.uniform([16, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([32, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([48, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([24, 48, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([24, 48, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([24, 24, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([24, 24, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([48, 48, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([64, 48, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([96, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([48, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([48, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([48, 48, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([48, 48, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([48, 48, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([48, 48, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([128, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([192, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([96, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([96, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([96, 96, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([96, 96, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([96, 96, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([96, 96, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([256, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([384, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([192, 384, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([192, 384, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([192, 192, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([192, 192, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_139
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([512, 384, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([192, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_149
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([192, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_154
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([192, 192, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([192, 192, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([192, 768, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([192, 384, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([96, 448, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([96, 448, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([96, 96, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_189
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([96, 96, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([96, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([48, 224, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_209
            paddle.uniform([48, 224, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_213
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_214
            paddle.uniform([48, 48, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_218
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_215
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_217
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_219
            paddle.uniform([48, 48, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_221
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_225
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_224
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_223
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([96, 96, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_227
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_229
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([96, 288, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_234
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_233
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([96, 288, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_239
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_238
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_241
            paddle.uniform([96, 96, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_245
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_244
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_243
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_246
            paddle.uniform([96, 96, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_248
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_252
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_249
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_251
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_250
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_253
            paddle.uniform([192, 192, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_257
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_254
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_256
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_255
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_258
            paddle.uniform([192, 576, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_262
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_259
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_261
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_263
            paddle.uniform([192, 576, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_267
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_264
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_266
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_265
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_268
            paddle.uniform([192, 192, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_272
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_269
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_271
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_270
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_273
            paddle.uniform([192, 192, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_274
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_275
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_279
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_276
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_278
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_277
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_280
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_281
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_282
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_286
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_283
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_285
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_284
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_287
            paddle.uniform([80, 384, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_288
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_289
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_290
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_291
            paddle.uniform([384, 384, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_295
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_292
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_294
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_293
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_296
            paddle.uniform([68, 384, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_297
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            # parameter_298
            paddle.uniform([1, 17, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_299
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_300
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_301
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_305
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_302
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_304
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_303
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_306
            paddle.uniform([80, 192, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_307
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_308
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_309
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_310
            paddle.uniform([192, 192, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_314
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_311
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_313
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_312
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_315
            paddle.uniform([68, 192, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_316
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            # parameter_317
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_318
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_319
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_323
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_320
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_322
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_321
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_324
            paddle.uniform([80, 96, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_325
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_326
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_327
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_328
            paddle.uniform([96, 96, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_332
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_329
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_331
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_330
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_333
            paddle.uniform([68, 96, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_334
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            # parameter_335
            paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
            # parameter_336
            paddle.uniform([8400, 1], dtype='float32', min=0, max=0.5),
            # feed_1
            paddle.to_tensor([1.0, 1.0], dtype='float32').reshape([1, 2]),
            # feed_0
            paddle.uniform([1, 3, 640, 640], dtype='float32', min=0, max=0.5),
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
            paddle.static.InputSpec(shape=[16, 16, 3, 3], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[32, 16, 3, 3], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[48, 32, 3, 3], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[24, 48, 1, 1], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[24, 48, 1, 1], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[24, 24, 3, 3], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[24, 24, 3, 3], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[48, 48, 1, 1], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[64, 48, 1, 1], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[96, 64, 3, 3], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[48, 96, 1, 1], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[48, 96, 1, 1], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[48, 48, 3, 3], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[48, 48, 3, 3], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[48, 48, 3, 3], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[48, 48, 3, 3], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[128, 96, 1, 1], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[192, 128, 3, 3], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[96, 192, 1, 1], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[96, 192, 1, 1], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[96, 96, 3, 3], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[96, 96, 3, 3], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[96, 96, 3, 3], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[96, 96, 3, 3], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[192, 192, 1, 1], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[256, 192, 1, 1], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[384, 256, 3, 3], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[192, 384, 1, 1], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[192, 384, 1, 1], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[192, 192, 3, 3], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[192, 192, 3, 3], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[384, 384, 1, 1], dtype='float32'),
            # parameter_139
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[512, 384, 1, 1], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[192, 512, 1, 1], dtype='float32'),
            # parameter_149
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[192, 512, 1, 1], dtype='float32'),
            # parameter_154
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[192, 192, 3, 3], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[192, 192, 3, 3], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[192, 768, 1, 1], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[384, 384, 1, 1], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[192, 384, 1, 1], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[96, 448, 1, 1], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[96, 448, 1, 1], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[96, 96, 3, 3], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_189
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[96, 96, 3, 3], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[192, 192, 1, 1], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[96, 192, 1, 1], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[48, 224, 1, 1], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_209
            paddle.static.InputSpec(shape=[48, 224, 1, 1], dtype='float32'),
            # parameter_213
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_214
            paddle.static.InputSpec(shape=[48, 48, 3, 3], dtype='float32'),
            # parameter_218
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_215
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_217
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_219
            paddle.static.InputSpec(shape=[48, 48, 3, 3], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_221
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float32'),
            # parameter_225
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_224
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_223
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[96, 96, 3, 3], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_227
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_229
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[96, 288, 1, 1], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_234
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_233
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[96, 288, 1, 1], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_239
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_238
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_241
            paddle.static.InputSpec(shape=[96, 96, 3, 3], dtype='float32'),
            # parameter_245
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_244
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_243
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_246
            paddle.static.InputSpec(shape=[96, 96, 3, 3], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_248
            paddle.static.InputSpec(shape=[192, 192, 1, 1], dtype='float32'),
            # parameter_252
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_249
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_251
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_250
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_253
            paddle.static.InputSpec(shape=[192, 192, 3, 3], dtype='float32'),
            # parameter_257
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_254
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_256
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_255
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_258
            paddle.static.InputSpec(shape=[192, 576, 1, 1], dtype='float32'),
            # parameter_262
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_259
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_261
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_263
            paddle.static.InputSpec(shape=[192, 576, 1, 1], dtype='float32'),
            # parameter_267
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_264
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_266
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_265
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_268
            paddle.static.InputSpec(shape=[192, 192, 3, 3], dtype='float32'),
            # parameter_272
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_269
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_271
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_270
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_273
            paddle.static.InputSpec(shape=[192, 192, 3, 3], dtype='float32'),
            # parameter_274
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_275
            paddle.static.InputSpec(shape=[384, 384, 1, 1], dtype='float32'),
            # parameter_279
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_276
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_278
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_277
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_280
            paddle.static.InputSpec(shape=[384, 384, 1, 1], dtype='float32'),
            # parameter_281
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_282
            paddle.static.InputSpec(shape=[384, 384, 1, 1], dtype='float32'),
            # parameter_286
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_283
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_285
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_284
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_287
            paddle.static.InputSpec(shape=[80, 384, 3, 3], dtype='float32'),
            # parameter_288
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_289
            paddle.static.InputSpec(shape=[384, 384, 1, 1], dtype='float32'),
            # parameter_290
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_291
            paddle.static.InputSpec(shape=[384, 384, 1, 1], dtype='float32'),
            # parameter_295
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_292
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_294
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_293
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_296
            paddle.static.InputSpec(shape=[68, 384, 3, 3], dtype='float32'),
            # parameter_297
            paddle.static.InputSpec(shape=[68], dtype='float32'),
            # parameter_298
            paddle.static.InputSpec(shape=[1, 17, 1, 1], dtype='float32'),
            # parameter_299
            paddle.static.InputSpec(shape=[192, 192, 1, 1], dtype='float32'),
            # parameter_300
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_301
            paddle.static.InputSpec(shape=[192, 192, 1, 1], dtype='float32'),
            # parameter_305
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_302
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_304
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_303
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_306
            paddle.static.InputSpec(shape=[80, 192, 3, 3], dtype='float32'),
            # parameter_307
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_308
            paddle.static.InputSpec(shape=[192, 192, 1, 1], dtype='float32'),
            # parameter_309
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_310
            paddle.static.InputSpec(shape=[192, 192, 1, 1], dtype='float32'),
            # parameter_314
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_311
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_313
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_312
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_315
            paddle.static.InputSpec(shape=[68, 192, 3, 3], dtype='float32'),
            # parameter_316
            paddle.static.InputSpec(shape=[68], dtype='float32'),
            # parameter_317
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float32'),
            # parameter_318
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_319
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float32'),
            # parameter_323
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_320
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_322
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_321
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_324
            paddle.static.InputSpec(shape=[80, 96, 3, 3], dtype='float32'),
            # parameter_325
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_326
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float32'),
            # parameter_327
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_328
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float32'),
            # parameter_332
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_329
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_331
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_330
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_333
            paddle.static.InputSpec(shape=[68, 96, 3, 3], dtype='float32'),
            # parameter_334
            paddle.static.InputSpec(shape=[68], dtype='float32'),
            # parameter_335
            paddle.static.InputSpec(shape=[8400, 2], dtype='float32'),
            # parameter_336
            paddle.static.InputSpec(shape=[8400, 1], dtype='float32'),
            # feed_1
            paddle.static.InputSpec(shape=[None, 2], dtype='float32'),
            # feed_0
            paddle.static.InputSpec(shape=[None, 3, 640, 640], dtype='float32'),
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