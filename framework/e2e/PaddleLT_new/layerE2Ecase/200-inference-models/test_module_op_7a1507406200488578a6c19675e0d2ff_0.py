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
    return [404][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_833_0_0(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_114, parameter_111, parameter_113, parameter_112, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_154, parameter_151, parameter_153, parameter_152, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_169, parameter_166, parameter_168, parameter_167, parameter_170, parameter_174, parameter_171, parameter_173, parameter_172, parameter_175, parameter_179, parameter_176, parameter_178, parameter_177, parameter_180, parameter_184, parameter_181, parameter_183, parameter_182, parameter_185, parameter_189, parameter_186, parameter_188, parameter_187, parameter_190, parameter_194, parameter_191, parameter_193, parameter_192, parameter_195, parameter_199, parameter_196, parameter_198, parameter_197, parameter_200, parameter_204, parameter_201, parameter_203, parameter_202, parameter_205, parameter_209, parameter_206, parameter_208, parameter_207, parameter_210, parameter_214, parameter_211, parameter_213, parameter_212, parameter_215, parameter_219, parameter_216, parameter_218, parameter_217, parameter_220, parameter_224, parameter_221, parameter_223, parameter_222, parameter_225, parameter_229, parameter_226, parameter_228, parameter_227, parameter_230, parameter_234, parameter_231, parameter_233, parameter_232, parameter_235, parameter_239, parameter_236, parameter_238, parameter_237, parameter_240, parameter_244, parameter_241, parameter_243, parameter_242, parameter_245, parameter_249, parameter_246, parameter_248, parameter_247, parameter_250, parameter_254, parameter_251, parameter_253, parameter_252, parameter_255, parameter_259, parameter_256, parameter_258, parameter_257, parameter_260, parameter_264, parameter_261, parameter_263, parameter_262, parameter_265, parameter_269, parameter_266, parameter_268, parameter_267, parameter_270, parameter_274, parameter_271, parameter_273, parameter_272, parameter_275, parameter_279, parameter_276, parameter_278, parameter_277, parameter_280, parameter_284, parameter_281, parameter_283, parameter_282, parameter_285, parameter_289, parameter_286, parameter_288, parameter_287, parameter_290, parameter_294, parameter_291, parameter_293, parameter_292, parameter_295, parameter_299, parameter_296, parameter_298, parameter_297, parameter_300, parameter_304, parameter_301, parameter_303, parameter_302, parameter_305, parameter_309, parameter_306, parameter_308, parameter_307, parameter_310, parameter_314, parameter_311, parameter_313, parameter_312, parameter_315, parameter_319, parameter_316, parameter_318, parameter_317, parameter_320, parameter_324, parameter_321, parameter_323, parameter_322, parameter_325, parameter_329, parameter_326, parameter_328, parameter_327, parameter_330, parameter_334, parameter_331, parameter_333, parameter_332, parameter_335, parameter_339, parameter_336, parameter_338, parameter_337, parameter_340, parameter_344, parameter_341, parameter_343, parameter_342, parameter_345, parameter_349, parameter_346, parameter_348, parameter_347, parameter_350, parameter_354, parameter_351, parameter_353, parameter_352, parameter_355, parameter_359, parameter_356, parameter_358, parameter_357, parameter_360, parameter_364, parameter_361, parameter_363, parameter_362, parameter_365, parameter_369, parameter_366, parameter_368, parameter_367, parameter_370, parameter_374, parameter_371, parameter_373, parameter_372, parameter_375, parameter_379, parameter_376, parameter_378, parameter_377, parameter_380, parameter_384, parameter_381, parameter_383, parameter_382, parameter_385, parameter_389, parameter_386, parameter_388, parameter_387, parameter_390, parameter_394, parameter_391, parameter_393, parameter_392, parameter_395, parameter_399, parameter_396, parameter_398, parameter_397, parameter_400, parameter_404, parameter_401, parameter_403, parameter_402, parameter_405, parameter_409, parameter_406, parameter_408, parameter_407, parameter_410, parameter_414, parameter_411, parameter_413, parameter_412, parameter_415, parameter_419, parameter_416, parameter_418, parameter_417, parameter_420, parameter_421, feed_0):

        # pd_op.cast: (-1x3x224x224xf16) <- (-1x3x224x224xf32)
        cast_0 = paddle._C_ops.cast(feed_0, paddle.float16)

        # pd_op.conv2d: (-1x48x112x112xf16) <- (-1x3x224x224xf16, 48x3x3x3xf16)
        conv2d_0 = paddle._C_ops.conv2d(cast_0, parameter_0, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x112x112xf16, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x112x112xf16, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_0, parameter_1, parameter_2, parameter_3, parameter_4, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x48x112x112xf16) <- (-1x48x112x112xf16)
        relu6_0 = paddle._C_ops.relu6(batch_norm__0)

        # pd_op.conv2d: (-1x96x112x112xf16) <- (-1x48x112x112xf16, 96x48x3x3xf16)
        conv2d_1 = paddle._C_ops.conv2d(relu6_0, parameter_5, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x112x112xf16, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x112x112xf16, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_1, parameter_6, parameter_7, parameter_8, parameter_9, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x96x112x112xf16) <- (-1x96x112x112xf16)
        relu6_1 = paddle._C_ops.relu6(batch_norm__6)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [3, 3]

        # pd_op.pool2d: (-1x96x56x56xf16) <- (-1x96x112x112xf16, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(relu6_1, full_int_array_0, [2, 2], [1, 1], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x24x56x56xf16) <- (-1x96x56x56xf16, 24x96x3x3xf16)
        conv2d_2 = paddle._C_ops.conv2d(pool2d_0, parameter_10, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x24x56x56xf16, 24xf32, 24xf32, xf32, xf32, None) <- (-1x24x56x56xf16, 24xf32, 24xf32, 24xf32, 24xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_2, parameter_11, parameter_12, parameter_13, parameter_14, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x24x56x56xf16) <- (-1x24x56x56xf16)
        relu6_2 = paddle._C_ops.relu6(batch_norm__12)

        # builtin.combine: ([-1x24x56x56xf16, -1x96x56x56xf16]) <- (-1x24x56x56xf16, -1x96x56x56xf16)
        combine_0 = [relu6_2, pool2d_0]

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x120x56x56xf16) <- ([-1x24x56x56xf16, -1x96x56x56xf16], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        # pd_op.conv2d: (-1x40x56x56xf16) <- (-1x120x56x56xf16, 40x120x3x3xf16)
        conv2d_3 = paddle._C_ops.conv2d(concat_0, parameter_15, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x56x56xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x56x56xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_3, parameter_16, parameter_17, parameter_18, parameter_19, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x40x56x56xf16) <- (-1x40x56x56xf16)
        relu6_3 = paddle._C_ops.relu6(batch_norm__18)

        # pd_op.conv2d: (-1x24x56x56xf16) <- (-1x40x56x56xf16, 24x40x3x3xf16)
        conv2d_4 = paddle._C_ops.conv2d(relu6_3, parameter_20, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x24x56x56xf16, 24xf32, 24xf32, xf32, xf32, None) <- (-1x24x56x56xf16, 24xf32, 24xf32, 24xf32, 24xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_4, parameter_21, parameter_22, parameter_23, parameter_24, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x24x56x56xf16) <- (-1x24x56x56xf16)
        relu6_4 = paddle._C_ops.relu6(batch_norm__24)

        # builtin.combine: ([-1x24x56x56xf16, -1x40x56x56xf16, -1x96x56x56xf16]) <- (-1x24x56x56xf16, -1x40x56x56xf16, -1x96x56x56xf16)
        combine_1 = [relu6_4, relu6_3, pool2d_0]

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x160x56x56xf16) <- ([-1x24x56x56xf16, -1x40x56x56xf16, -1x96x56x56xf16], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_1)

        # pd_op.conv2d: (-1x70x56x56xf16) <- (-1x160x56x56xf16, 70x160x3x3xf16)
        conv2d_5 = paddle._C_ops.conv2d(concat_1, parameter_25, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x70x56x56xf16, 70xf32, 70xf32, xf32, xf32, None) <- (-1x70x56x56xf16, 70xf32, 70xf32, 70xf32, 70xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_5, parameter_26, parameter_27, parameter_28, parameter_29, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x70x56x56xf16) <- (-1x70x56x56xf16)
        relu6_5 = paddle._C_ops.relu6(batch_norm__30)

        # pd_op.conv2d: (-1x24x56x56xf16) <- (-1x70x56x56xf16, 24x70x3x3xf16)
        conv2d_6 = paddle._C_ops.conv2d(relu6_5, parameter_30, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x24x56x56xf16, 24xf32, 24xf32, xf32, xf32, None) <- (-1x24x56x56xf16, 24xf32, 24xf32, 24xf32, 24xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_6, parameter_31, parameter_32, parameter_33, parameter_34, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x24x56x56xf16) <- (-1x24x56x56xf16)
        relu6_6 = paddle._C_ops.relu6(batch_norm__36)

        # builtin.combine: ([-1x24x56x56xf16, -1x70x56x56xf16]) <- (-1x24x56x56xf16, -1x70x56x56xf16)
        combine_2 = [relu6_6, relu6_5]

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x94x56x56xf16) <- ([-1x24x56x56xf16, -1x70x56x56xf16], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_2)

        # pd_op.conv2d: (-1x40x56x56xf16) <- (-1x94x56x56xf16, 40x94x3x3xf16)
        conv2d_7 = paddle._C_ops.conv2d(concat_2, parameter_35, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x56x56xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x56x56xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_7, parameter_36, parameter_37, parameter_38, parameter_39, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x40x56x56xf16) <- (-1x40x56x56xf16)
        relu6_7 = paddle._C_ops.relu6(batch_norm__42)

        # pd_op.conv2d: (-1x24x56x56xf16) <- (-1x40x56x56xf16, 24x40x3x3xf16)
        conv2d_8 = paddle._C_ops.conv2d(relu6_7, parameter_40, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x24x56x56xf16, 24xf32, 24xf32, xf32, xf32, None) <- (-1x24x56x56xf16, 24xf32, 24xf32, 24xf32, 24xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_8, parameter_41, parameter_42, parameter_43, parameter_44, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x24x56x56xf16) <- (-1x24x56x56xf16)
        relu6_8 = paddle._C_ops.relu6(batch_norm__48)

        # builtin.combine: ([-1x24x56x56xf16, -1x40x56x56xf16, -1x70x56x56xf16, -1x96x56x56xf16]) <- (-1x24x56x56xf16, -1x40x56x56xf16, -1x70x56x56xf16, -1x96x56x56xf16)
        combine_3 = [relu6_8, relu6_7, relu6_5, pool2d_0]

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x230x56x56xf16) <- ([-1x24x56x56xf16, -1x40x56x56xf16, -1x70x56x56xf16, -1x96x56x56xf16], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_3, full_3)

        # pd_op.conv2d: (-1x118x56x56xf16) <- (-1x230x56x56xf16, 118x230x3x3xf16)
        conv2d_9 = paddle._C_ops.conv2d(concat_3, parameter_45, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x118x56x56xf16, 118xf32, 118xf32, xf32, xf32, None) <- (-1x118x56x56xf16, 118xf32, 118xf32, 118xf32, 118xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_9, parameter_46, parameter_47, parameter_48, parameter_49, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x118x56x56xf16) <- (-1x118x56x56xf16)
        relu6_9 = paddle._C_ops.relu6(batch_norm__54)

        # builtin.combine: ([-1x24x56x56xf16, -1x24x56x56xf16, -1x24x56x56xf16, -1x24x56x56xf16, -1x118x56x56xf16]) <- (-1x24x56x56xf16, -1x24x56x56xf16, -1x24x56x56xf16, -1x24x56x56xf16, -1x118x56x56xf16)
        combine_4 = [relu6_2, relu6_4, relu6_6, relu6_8, relu6_9]

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x214x56x56xf16) <- ([-1x24x56x56xf16, -1x24x56x56xf16, -1x24x56x56xf16, -1x24x56x56xf16, -1x118x56x56xf16], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_4, full_4)

        # pd_op.conv2d: (-1x192x56x56xf16) <- (-1x214x56x56xf16, 192x214x1x1xf16)
        conv2d_10 = paddle._C_ops.conv2d(concat_4, parameter_50, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x192x56x56xf16, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x56x56xf16, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_10, parameter_51, parameter_52, parameter_53, parameter_54, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x192x56x56xf16) <- (-1x192x56x56xf16)
        relu6_10 = paddle._C_ops.relu6(batch_norm__60)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [2, 2]

        # pd_op.pool2d: (-1x192x28x28xf16) <- (-1x192x56x56xf16, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(relu6_10, full_int_array_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x24x28x28xf16) <- (-1x192x28x28xf16, 24x192x3x3xf16)
        conv2d_11 = paddle._C_ops.conv2d(pool2d_1, parameter_55, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x24x28x28xf16, 24xf32, 24xf32, xf32, xf32, None) <- (-1x24x28x28xf16, 24xf32, 24xf32, 24xf32, 24xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_11, parameter_56, parameter_57, parameter_58, parameter_59, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x24x28x28xf16) <- (-1x24x28x28xf16)
        relu6_11 = paddle._C_ops.relu6(batch_norm__66)

        # builtin.combine: ([-1x24x28x28xf16, -1x192x28x28xf16]) <- (-1x24x28x28xf16, -1x192x28x28xf16)
        combine_5 = [relu6_11, pool2d_1]

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x216x28x28xf16) <- ([-1x24x28x28xf16, -1x192x28x28xf16], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_5, full_5)

        # pd_op.conv2d: (-1x40x28x28xf16) <- (-1x216x28x28xf16, 40x216x3x3xf16)
        conv2d_12 = paddle._C_ops.conv2d(concat_5, parameter_60, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x28x28xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x28x28xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_12, parameter_61, parameter_62, parameter_63, parameter_64, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x40x28x28xf16) <- (-1x40x28x28xf16)
        relu6_12 = paddle._C_ops.relu6(batch_norm__72)

        # pd_op.conv2d: (-1x24x28x28xf16) <- (-1x40x28x28xf16, 24x40x3x3xf16)
        conv2d_13 = paddle._C_ops.conv2d(relu6_12, parameter_65, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x24x28x28xf16, 24xf32, 24xf32, xf32, xf32, None) <- (-1x24x28x28xf16, 24xf32, 24xf32, 24xf32, 24xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_13, parameter_66, parameter_67, parameter_68, parameter_69, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x24x28x28xf16) <- (-1x24x28x28xf16)
        relu6_13 = paddle._C_ops.relu6(batch_norm__78)

        # builtin.combine: ([-1x24x28x28xf16, -1x40x28x28xf16, -1x192x28x28xf16]) <- (-1x24x28x28xf16, -1x40x28x28xf16, -1x192x28x28xf16)
        combine_6 = [relu6_13, relu6_12, pool2d_1]

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x256x28x28xf16) <- ([-1x24x28x28xf16, -1x40x28x28xf16, -1x192x28x28xf16], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_6, full_6)

        # pd_op.conv2d: (-1x70x28x28xf16) <- (-1x256x28x28xf16, 70x256x3x3xf16)
        conv2d_14 = paddle._C_ops.conv2d(concat_6, parameter_70, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x70x28x28xf16, 70xf32, 70xf32, xf32, xf32, None) <- (-1x70x28x28xf16, 70xf32, 70xf32, 70xf32, 70xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_14, parameter_71, parameter_72, parameter_73, parameter_74, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x70x28x28xf16) <- (-1x70x28x28xf16)
        relu6_14 = paddle._C_ops.relu6(batch_norm__84)

        # pd_op.conv2d: (-1x24x28x28xf16) <- (-1x70x28x28xf16, 24x70x3x3xf16)
        conv2d_15 = paddle._C_ops.conv2d(relu6_14, parameter_75, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x24x28x28xf16, 24xf32, 24xf32, xf32, xf32, None) <- (-1x24x28x28xf16, 24xf32, 24xf32, 24xf32, 24xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_15, parameter_76, parameter_77, parameter_78, parameter_79, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x24x28x28xf16) <- (-1x24x28x28xf16)
        relu6_15 = paddle._C_ops.relu6(batch_norm__90)

        # builtin.combine: ([-1x24x28x28xf16, -1x70x28x28xf16]) <- (-1x24x28x28xf16, -1x70x28x28xf16)
        combine_7 = [relu6_15, relu6_14]

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x94x28x28xf16) <- ([-1x24x28x28xf16, -1x70x28x28xf16], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_7, full_7)

        # pd_op.conv2d: (-1x40x28x28xf16) <- (-1x94x28x28xf16, 40x94x3x3xf16)
        conv2d_16 = paddle._C_ops.conv2d(concat_7, parameter_80, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x28x28xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x28x28xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_16, parameter_81, parameter_82, parameter_83, parameter_84, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x40x28x28xf16) <- (-1x40x28x28xf16)
        relu6_16 = paddle._C_ops.relu6(batch_norm__96)

        # pd_op.conv2d: (-1x24x28x28xf16) <- (-1x40x28x28xf16, 24x40x3x3xf16)
        conv2d_17 = paddle._C_ops.conv2d(relu6_16, parameter_85, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x24x28x28xf16, 24xf32, 24xf32, xf32, xf32, None) <- (-1x24x28x28xf16, 24xf32, 24xf32, 24xf32, 24xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_17, parameter_86, parameter_87, parameter_88, parameter_89, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x24x28x28xf16) <- (-1x24x28x28xf16)
        relu6_17 = paddle._C_ops.relu6(batch_norm__102)

        # builtin.combine: ([-1x24x28x28xf16, -1x40x28x28xf16, -1x70x28x28xf16, -1x192x28x28xf16]) <- (-1x24x28x28xf16, -1x40x28x28xf16, -1x70x28x28xf16, -1x192x28x28xf16)
        combine_8 = [relu6_17, relu6_16, relu6_14, pool2d_1]

        # pd_op.full: (1xi32) <- ()
        full_8 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x326x28x28xf16) <- ([-1x24x28x28xf16, -1x40x28x28xf16, -1x70x28x28xf16, -1x192x28x28xf16], 1xi32)
        concat_8 = paddle._C_ops.concat(combine_8, full_8)

        # pd_op.conv2d: (-1x118x28x28xf16) <- (-1x326x28x28xf16, 118x326x3x3xf16)
        conv2d_18 = paddle._C_ops.conv2d(concat_8, parameter_90, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x118x28x28xf16, 118xf32, 118xf32, xf32, xf32, None) <- (-1x118x28x28xf16, 118xf32, 118xf32, 118xf32, 118xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_18, parameter_91, parameter_92, parameter_93, parameter_94, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x118x28x28xf16) <- (-1x118x28x28xf16)
        relu6_18 = paddle._C_ops.relu6(batch_norm__108)

        # pd_op.conv2d: (-1x24x28x28xf16) <- (-1x118x28x28xf16, 24x118x3x3xf16)
        conv2d_19 = paddle._C_ops.conv2d(relu6_18, parameter_95, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x24x28x28xf16, 24xf32, 24xf32, xf32, xf32, None) <- (-1x24x28x28xf16, 24xf32, 24xf32, 24xf32, 24xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_19, parameter_96, parameter_97, parameter_98, parameter_99, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x24x28x28xf16) <- (-1x24x28x28xf16)
        relu6_19 = paddle._C_ops.relu6(batch_norm__114)

        # builtin.combine: ([-1x24x28x28xf16, -1x118x28x28xf16]) <- (-1x24x28x28xf16, -1x118x28x28xf16)
        combine_9 = [relu6_19, relu6_18]

        # pd_op.full: (1xi32) <- ()
        full_9 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x142x28x28xf16) <- ([-1x24x28x28xf16, -1x118x28x28xf16], 1xi32)
        concat_9 = paddle._C_ops.concat(combine_9, full_9)

        # pd_op.conv2d: (-1x40x28x28xf16) <- (-1x142x28x28xf16, 40x142x3x3xf16)
        conv2d_20 = paddle._C_ops.conv2d(concat_9, parameter_100, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x28x28xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x28x28xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_20, parameter_101, parameter_102, parameter_103, parameter_104, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x40x28x28xf16) <- (-1x40x28x28xf16)
        relu6_20 = paddle._C_ops.relu6(batch_norm__120)

        # pd_op.conv2d: (-1x24x28x28xf16) <- (-1x40x28x28xf16, 24x40x3x3xf16)
        conv2d_21 = paddle._C_ops.conv2d(relu6_20, parameter_105, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x24x28x28xf16, 24xf32, 24xf32, xf32, xf32, None) <- (-1x24x28x28xf16, 24xf32, 24xf32, 24xf32, 24xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_21, parameter_106, parameter_107, parameter_108, parameter_109, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x24x28x28xf16) <- (-1x24x28x28xf16)
        relu6_21 = paddle._C_ops.relu6(batch_norm__126)

        # builtin.combine: ([-1x24x28x28xf16, -1x40x28x28xf16, -1x118x28x28xf16]) <- (-1x24x28x28xf16, -1x40x28x28xf16, -1x118x28x28xf16)
        combine_10 = [relu6_21, relu6_20, relu6_18]

        # pd_op.full: (1xi32) <- ()
        full_10 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x182x28x28xf16) <- ([-1x24x28x28xf16, -1x40x28x28xf16, -1x118x28x28xf16], 1xi32)
        concat_10 = paddle._C_ops.concat(combine_10, full_10)

        # pd_op.conv2d: (-1x70x28x28xf16) <- (-1x182x28x28xf16, 70x182x3x3xf16)
        conv2d_22 = paddle._C_ops.conv2d(concat_10, parameter_110, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x70x28x28xf16, 70xf32, 70xf32, xf32, xf32, None) <- (-1x70x28x28xf16, 70xf32, 70xf32, 70xf32, 70xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_22, parameter_111, parameter_112, parameter_113, parameter_114, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x70x28x28xf16) <- (-1x70x28x28xf16)
        relu6_22 = paddle._C_ops.relu6(batch_norm__132)

        # pd_op.conv2d: (-1x24x28x28xf16) <- (-1x70x28x28xf16, 24x70x3x3xf16)
        conv2d_23 = paddle._C_ops.conv2d(relu6_22, parameter_115, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x24x28x28xf16, 24xf32, 24xf32, xf32, xf32, None) <- (-1x24x28x28xf16, 24xf32, 24xf32, 24xf32, 24xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_23, parameter_116, parameter_117, parameter_118, parameter_119, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x24x28x28xf16) <- (-1x24x28x28xf16)
        relu6_23 = paddle._C_ops.relu6(batch_norm__138)

        # builtin.combine: ([-1x24x28x28xf16, -1x70x28x28xf16]) <- (-1x24x28x28xf16, -1x70x28x28xf16)
        combine_11 = [relu6_23, relu6_22]

        # pd_op.full: (1xi32) <- ()
        full_11 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x94x28x28xf16) <- ([-1x24x28x28xf16, -1x70x28x28xf16], 1xi32)
        concat_11 = paddle._C_ops.concat(combine_11, full_11)

        # pd_op.conv2d: (-1x40x28x28xf16) <- (-1x94x28x28xf16, 40x94x3x3xf16)
        conv2d_24 = paddle._C_ops.conv2d(concat_11, parameter_120, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x28x28xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x28x28xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_24, parameter_121, parameter_122, parameter_123, parameter_124, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x40x28x28xf16) <- (-1x40x28x28xf16)
        relu6_24 = paddle._C_ops.relu6(batch_norm__144)

        # pd_op.conv2d: (-1x24x28x28xf16) <- (-1x40x28x28xf16, 24x40x3x3xf16)
        conv2d_25 = paddle._C_ops.conv2d(relu6_24, parameter_125, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x24x28x28xf16, 24xf32, 24xf32, xf32, xf32, None) <- (-1x24x28x28xf16, 24xf32, 24xf32, 24xf32, 24xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_25, parameter_126, parameter_127, parameter_128, parameter_129, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x24x28x28xf16) <- (-1x24x28x28xf16)
        relu6_25 = paddle._C_ops.relu6(batch_norm__150)

        # builtin.combine: ([-1x24x28x28xf16, -1x40x28x28xf16, -1x70x28x28xf16, -1x118x28x28xf16, -1x192x28x28xf16]) <- (-1x24x28x28xf16, -1x40x28x28xf16, -1x70x28x28xf16, -1x118x28x28xf16, -1x192x28x28xf16)
        combine_12 = [relu6_25, relu6_24, relu6_22, relu6_18, pool2d_1]

        # pd_op.full: (1xi32) <- ()
        full_12 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x444x28x28xf16) <- ([-1x24x28x28xf16, -1x40x28x28xf16, -1x70x28x28xf16, -1x118x28x28xf16, -1x192x28x28xf16], 1xi32)
        concat_12 = paddle._C_ops.concat(combine_12, full_12)

        # pd_op.conv2d: (-1x200x28x28xf16) <- (-1x444x28x28xf16, 200x444x3x3xf16)
        conv2d_26 = paddle._C_ops.conv2d(concat_12, parameter_130, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x200x28x28xf16, 200xf32, 200xf32, xf32, xf32, None) <- (-1x200x28x28xf16, 200xf32, 200xf32, 200xf32, 200xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_26, parameter_131, parameter_132, parameter_133, parameter_134, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x200x28x28xf16) <- (-1x200x28x28xf16)
        relu6_26 = paddle._C_ops.relu6(batch_norm__156)

        # builtin.combine: ([-1x24x28x28xf16, -1x24x28x28xf16, -1x24x28x28xf16, -1x24x28x28xf16, -1x24x28x28xf16, -1x24x28x28xf16, -1x24x28x28xf16, -1x24x28x28xf16, -1x200x28x28xf16]) <- (-1x24x28x28xf16, -1x24x28x28xf16, -1x24x28x28xf16, -1x24x28x28xf16, -1x24x28x28xf16, -1x24x28x28xf16, -1x24x28x28xf16, -1x24x28x28xf16, -1x200x28x28xf16)
        combine_13 = [relu6_11, relu6_13, relu6_15, relu6_17, relu6_19, relu6_21, relu6_23, relu6_25, relu6_26]

        # pd_op.full: (1xi32) <- ()
        full_13 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x392x28x28xf16) <- ([-1x24x28x28xf16, -1x24x28x28xf16, -1x24x28x28xf16, -1x24x28x28xf16, -1x24x28x28xf16, -1x24x28x28xf16, -1x24x28x28xf16, -1x24x28x28xf16, -1x200x28x28xf16], 1xi32)
        concat_13 = paddle._C_ops.concat(combine_13, full_13)

        # pd_op.conv2d: (-1x256x28x28xf16) <- (-1x392x28x28xf16, 256x392x1x1xf16)
        conv2d_27 = paddle._C_ops.conv2d(concat_13, parameter_135, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x28x28xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x28x28xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_27, parameter_136, parameter_137, parameter_138, parameter_139, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x256x28x28xf16) <- (-1x256x28x28xf16)
        relu6_27 = paddle._C_ops.relu6(batch_norm__162)

        # pd_op.conv2d: (-1x28x28x28xf16) <- (-1x256x28x28xf16, 28x256x3x3xf16)
        conv2d_28 = paddle._C_ops.conv2d(relu6_27, parameter_140, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x28x28x28xf16, 28xf32, 28xf32, xf32, xf32, None) <- (-1x28x28x28xf16, 28xf32, 28xf32, 28xf32, 28xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_28, parameter_141, parameter_142, parameter_143, parameter_144, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x28x28x28xf16) <- (-1x28x28x28xf16)
        relu6_28 = paddle._C_ops.relu6(batch_norm__168)

        # builtin.combine: ([-1x28x28x28xf16, -1x256x28x28xf16]) <- (-1x28x28x28xf16, -1x256x28x28xf16)
        combine_14 = [relu6_28, relu6_27]

        # pd_op.full: (1xi32) <- ()
        full_14 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x284x28x28xf16) <- ([-1x28x28x28xf16, -1x256x28x28xf16], 1xi32)
        concat_14 = paddle._C_ops.concat(combine_14, full_14)

        # pd_op.conv2d: (-1x48x28x28xf16) <- (-1x284x28x28xf16, 48x284x3x3xf16)
        conv2d_29 = paddle._C_ops.conv2d(concat_14, parameter_145, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x28x28xf16, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x28x28xf16, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_29, parameter_146, parameter_147, parameter_148, parameter_149, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x48x28x28xf16) <- (-1x48x28x28xf16)
        relu6_29 = paddle._C_ops.relu6(batch_norm__174)

        # pd_op.conv2d: (-1x28x28x28xf16) <- (-1x48x28x28xf16, 28x48x3x3xf16)
        conv2d_30 = paddle._C_ops.conv2d(relu6_29, parameter_150, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x28x28x28xf16, 28xf32, 28xf32, xf32, xf32, None) <- (-1x28x28x28xf16, 28xf32, 28xf32, 28xf32, 28xf32)
        batch_norm__180, batch_norm__181, batch_norm__182, batch_norm__183, batch_norm__184, batch_norm__185 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_30, parameter_151, parameter_152, parameter_153, parameter_154, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x28x28x28xf16) <- (-1x28x28x28xf16)
        relu6_30 = paddle._C_ops.relu6(batch_norm__180)

        # builtin.combine: ([-1x28x28x28xf16, -1x48x28x28xf16, -1x256x28x28xf16]) <- (-1x28x28x28xf16, -1x48x28x28xf16, -1x256x28x28xf16)
        combine_15 = [relu6_30, relu6_29, relu6_27]

        # pd_op.full: (1xi32) <- ()
        full_15 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x332x28x28xf16) <- ([-1x28x28x28xf16, -1x48x28x28xf16, -1x256x28x28xf16], 1xi32)
        concat_15 = paddle._C_ops.concat(combine_15, full_15)

        # pd_op.conv2d: (-1x80x28x28xf16) <- (-1x332x28x28xf16, 80x332x3x3xf16)
        conv2d_31 = paddle._C_ops.conv2d(concat_15, parameter_155, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x28x28xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x28x28xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__186, batch_norm__187, batch_norm__188, batch_norm__189, batch_norm__190, batch_norm__191 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_31, parameter_156, parameter_157, parameter_158, parameter_159, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x80x28x28xf16) <- (-1x80x28x28xf16)
        relu6_31 = paddle._C_ops.relu6(batch_norm__186)

        # pd_op.conv2d: (-1x28x28x28xf16) <- (-1x80x28x28xf16, 28x80x3x3xf16)
        conv2d_32 = paddle._C_ops.conv2d(relu6_31, parameter_160, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x28x28x28xf16, 28xf32, 28xf32, xf32, xf32, None) <- (-1x28x28x28xf16, 28xf32, 28xf32, 28xf32, 28xf32)
        batch_norm__192, batch_norm__193, batch_norm__194, batch_norm__195, batch_norm__196, batch_norm__197 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_32, parameter_161, parameter_162, parameter_163, parameter_164, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x28x28x28xf16) <- (-1x28x28x28xf16)
        relu6_32 = paddle._C_ops.relu6(batch_norm__192)

        # builtin.combine: ([-1x28x28x28xf16, -1x80x28x28xf16]) <- (-1x28x28x28xf16, -1x80x28x28xf16)
        combine_16 = [relu6_32, relu6_31]

        # pd_op.full: (1xi32) <- ()
        full_16 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x108x28x28xf16) <- ([-1x28x28x28xf16, -1x80x28x28xf16], 1xi32)
        concat_16 = paddle._C_ops.concat(combine_16, full_16)

        # pd_op.conv2d: (-1x48x28x28xf16) <- (-1x108x28x28xf16, 48x108x3x3xf16)
        conv2d_33 = paddle._C_ops.conv2d(concat_16, parameter_165, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x28x28xf16, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x28x28xf16, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__198, batch_norm__199, batch_norm__200, batch_norm__201, batch_norm__202, batch_norm__203 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_33, parameter_166, parameter_167, parameter_168, parameter_169, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x48x28x28xf16) <- (-1x48x28x28xf16)
        relu6_33 = paddle._C_ops.relu6(batch_norm__198)

        # pd_op.conv2d: (-1x28x28x28xf16) <- (-1x48x28x28xf16, 28x48x3x3xf16)
        conv2d_34 = paddle._C_ops.conv2d(relu6_33, parameter_170, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x28x28x28xf16, 28xf32, 28xf32, xf32, xf32, None) <- (-1x28x28x28xf16, 28xf32, 28xf32, 28xf32, 28xf32)
        batch_norm__204, batch_norm__205, batch_norm__206, batch_norm__207, batch_norm__208, batch_norm__209 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_34, parameter_171, parameter_172, parameter_173, parameter_174, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x28x28x28xf16) <- (-1x28x28x28xf16)
        relu6_34 = paddle._C_ops.relu6(batch_norm__204)

        # builtin.combine: ([-1x28x28x28xf16, -1x48x28x28xf16, -1x80x28x28xf16, -1x256x28x28xf16]) <- (-1x28x28x28xf16, -1x48x28x28xf16, -1x80x28x28xf16, -1x256x28x28xf16)
        combine_17 = [relu6_34, relu6_33, relu6_31, relu6_27]

        # pd_op.full: (1xi32) <- ()
        full_17 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x412x28x28xf16) <- ([-1x28x28x28xf16, -1x48x28x28xf16, -1x80x28x28xf16, -1x256x28x28xf16], 1xi32)
        concat_17 = paddle._C_ops.concat(combine_17, full_17)

        # pd_op.conv2d: (-1x138x28x28xf16) <- (-1x412x28x28xf16, 138x412x3x3xf16)
        conv2d_35 = paddle._C_ops.conv2d(concat_17, parameter_175, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x138x28x28xf16, 138xf32, 138xf32, xf32, xf32, None) <- (-1x138x28x28xf16, 138xf32, 138xf32, 138xf32, 138xf32)
        batch_norm__210, batch_norm__211, batch_norm__212, batch_norm__213, batch_norm__214, batch_norm__215 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_35, parameter_176, parameter_177, parameter_178, parameter_179, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x138x28x28xf16) <- (-1x138x28x28xf16)
        relu6_35 = paddle._C_ops.relu6(batch_norm__210)

        # pd_op.conv2d: (-1x28x28x28xf16) <- (-1x138x28x28xf16, 28x138x3x3xf16)
        conv2d_36 = paddle._C_ops.conv2d(relu6_35, parameter_180, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x28x28x28xf16, 28xf32, 28xf32, xf32, xf32, None) <- (-1x28x28x28xf16, 28xf32, 28xf32, 28xf32, 28xf32)
        batch_norm__216, batch_norm__217, batch_norm__218, batch_norm__219, batch_norm__220, batch_norm__221 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_36, parameter_181, parameter_182, parameter_183, parameter_184, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x28x28x28xf16) <- (-1x28x28x28xf16)
        relu6_36 = paddle._C_ops.relu6(batch_norm__216)

        # builtin.combine: ([-1x28x28x28xf16, -1x138x28x28xf16]) <- (-1x28x28x28xf16, -1x138x28x28xf16)
        combine_18 = [relu6_36, relu6_35]

        # pd_op.full: (1xi32) <- ()
        full_18 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x166x28x28xf16) <- ([-1x28x28x28xf16, -1x138x28x28xf16], 1xi32)
        concat_18 = paddle._C_ops.concat(combine_18, full_18)

        # pd_op.conv2d: (-1x48x28x28xf16) <- (-1x166x28x28xf16, 48x166x3x3xf16)
        conv2d_37 = paddle._C_ops.conv2d(concat_18, parameter_185, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x28x28xf16, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x28x28xf16, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__222, batch_norm__223, batch_norm__224, batch_norm__225, batch_norm__226, batch_norm__227 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_37, parameter_186, parameter_187, parameter_188, parameter_189, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x48x28x28xf16) <- (-1x48x28x28xf16)
        relu6_37 = paddle._C_ops.relu6(batch_norm__222)

        # pd_op.conv2d: (-1x28x28x28xf16) <- (-1x48x28x28xf16, 28x48x3x3xf16)
        conv2d_38 = paddle._C_ops.conv2d(relu6_37, parameter_190, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x28x28x28xf16, 28xf32, 28xf32, xf32, xf32, None) <- (-1x28x28x28xf16, 28xf32, 28xf32, 28xf32, 28xf32)
        batch_norm__228, batch_norm__229, batch_norm__230, batch_norm__231, batch_norm__232, batch_norm__233 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_38, parameter_191, parameter_192, parameter_193, parameter_194, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x28x28x28xf16) <- (-1x28x28x28xf16)
        relu6_38 = paddle._C_ops.relu6(batch_norm__228)

        # builtin.combine: ([-1x28x28x28xf16, -1x48x28x28xf16, -1x138x28x28xf16]) <- (-1x28x28x28xf16, -1x48x28x28xf16, -1x138x28x28xf16)
        combine_19 = [relu6_38, relu6_37, relu6_35]

        # pd_op.full: (1xi32) <- ()
        full_19 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x214x28x28xf16) <- ([-1x28x28x28xf16, -1x48x28x28xf16, -1x138x28x28xf16], 1xi32)
        concat_19 = paddle._C_ops.concat(combine_19, full_19)

        # pd_op.conv2d: (-1x80x28x28xf16) <- (-1x214x28x28xf16, 80x214x3x3xf16)
        conv2d_39 = paddle._C_ops.conv2d(concat_19, parameter_195, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x28x28xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x28x28xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__234, batch_norm__235, batch_norm__236, batch_norm__237, batch_norm__238, batch_norm__239 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_39, parameter_196, parameter_197, parameter_198, parameter_199, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x80x28x28xf16) <- (-1x80x28x28xf16)
        relu6_39 = paddle._C_ops.relu6(batch_norm__234)

        # pd_op.conv2d: (-1x28x28x28xf16) <- (-1x80x28x28xf16, 28x80x3x3xf16)
        conv2d_40 = paddle._C_ops.conv2d(relu6_39, parameter_200, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x28x28x28xf16, 28xf32, 28xf32, xf32, xf32, None) <- (-1x28x28x28xf16, 28xf32, 28xf32, 28xf32, 28xf32)
        batch_norm__240, batch_norm__241, batch_norm__242, batch_norm__243, batch_norm__244, batch_norm__245 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_40, parameter_201, parameter_202, parameter_203, parameter_204, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x28x28x28xf16) <- (-1x28x28x28xf16)
        relu6_40 = paddle._C_ops.relu6(batch_norm__240)

        # builtin.combine: ([-1x28x28x28xf16, -1x80x28x28xf16]) <- (-1x28x28x28xf16, -1x80x28x28xf16)
        combine_20 = [relu6_40, relu6_39]

        # pd_op.full: (1xi32) <- ()
        full_20 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x108x28x28xf16) <- ([-1x28x28x28xf16, -1x80x28x28xf16], 1xi32)
        concat_20 = paddle._C_ops.concat(combine_20, full_20)

        # pd_op.conv2d: (-1x48x28x28xf16) <- (-1x108x28x28xf16, 48x108x3x3xf16)
        conv2d_41 = paddle._C_ops.conv2d(concat_20, parameter_205, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x28x28xf16, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x28x28xf16, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__246, batch_norm__247, batch_norm__248, batch_norm__249, batch_norm__250, batch_norm__251 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_41, parameter_206, parameter_207, parameter_208, parameter_209, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x48x28x28xf16) <- (-1x48x28x28xf16)
        relu6_41 = paddle._C_ops.relu6(batch_norm__246)

        # pd_op.conv2d: (-1x28x28x28xf16) <- (-1x48x28x28xf16, 28x48x3x3xf16)
        conv2d_42 = paddle._C_ops.conv2d(relu6_41, parameter_210, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x28x28x28xf16, 28xf32, 28xf32, xf32, xf32, None) <- (-1x28x28x28xf16, 28xf32, 28xf32, 28xf32, 28xf32)
        batch_norm__252, batch_norm__253, batch_norm__254, batch_norm__255, batch_norm__256, batch_norm__257 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_42, parameter_211, parameter_212, parameter_213, parameter_214, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x28x28x28xf16) <- (-1x28x28x28xf16)
        relu6_42 = paddle._C_ops.relu6(batch_norm__252)

        # builtin.combine: ([-1x28x28x28xf16, -1x48x28x28xf16, -1x80x28x28xf16, -1x138x28x28xf16, -1x256x28x28xf16]) <- (-1x28x28x28xf16, -1x48x28x28xf16, -1x80x28x28xf16, -1x138x28x28xf16, -1x256x28x28xf16)
        combine_21 = [relu6_42, relu6_41, relu6_39, relu6_35, relu6_27]

        # pd_op.full: (1xi32) <- ()
        full_21 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x550x28x28xf16) <- ([-1x28x28x28xf16, -1x48x28x28xf16, -1x80x28x28xf16, -1x138x28x28xf16, -1x256x28x28xf16], 1xi32)
        concat_21 = paddle._C_ops.concat(combine_21, full_21)

        # pd_op.conv2d: (-1x234x28x28xf16) <- (-1x550x28x28xf16, 234x550x3x3xf16)
        conv2d_43 = paddle._C_ops.conv2d(concat_21, parameter_215, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x234x28x28xf16, 234xf32, 234xf32, xf32, xf32, None) <- (-1x234x28x28xf16, 234xf32, 234xf32, 234xf32, 234xf32)
        batch_norm__258, batch_norm__259, batch_norm__260, batch_norm__261, batch_norm__262, batch_norm__263 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_43, parameter_216, parameter_217, parameter_218, parameter_219, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x234x28x28xf16) <- (-1x234x28x28xf16)
        relu6_43 = paddle._C_ops.relu6(batch_norm__258)

        # builtin.combine: ([-1x28x28x28xf16, -1x28x28x28xf16, -1x28x28x28xf16, -1x28x28x28xf16, -1x28x28x28xf16, -1x28x28x28xf16, -1x28x28x28xf16, -1x28x28x28xf16, -1x234x28x28xf16]) <- (-1x28x28x28xf16, -1x28x28x28xf16, -1x28x28x28xf16, -1x28x28x28xf16, -1x28x28x28xf16, -1x28x28x28xf16, -1x28x28x28xf16, -1x28x28x28xf16, -1x234x28x28xf16)
        combine_22 = [relu6_28, relu6_30, relu6_32, relu6_34, relu6_36, relu6_38, relu6_40, relu6_42, relu6_43]

        # pd_op.full: (1xi32) <- ()
        full_22 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x458x28x28xf16) <- ([-1x28x28x28xf16, -1x28x28x28xf16, -1x28x28x28xf16, -1x28x28x28xf16, -1x28x28x28xf16, -1x28x28x28xf16, -1x28x28x28xf16, -1x28x28x28xf16, -1x234x28x28xf16], 1xi32)
        concat_22 = paddle._C_ops.concat(combine_22, full_22)

        # pd_op.conv2d: (-1x320x28x28xf16) <- (-1x458x28x28xf16, 320x458x1x1xf16)
        conv2d_44 = paddle._C_ops.conv2d(concat_22, parameter_220, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x320x28x28xf16, 320xf32, 320xf32, xf32, xf32, None) <- (-1x320x28x28xf16, 320xf32, 320xf32, 320xf32, 320xf32)
        batch_norm__264, batch_norm__265, batch_norm__266, batch_norm__267, batch_norm__268, batch_norm__269 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_44, parameter_221, parameter_222, parameter_223, parameter_224, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x320x28x28xf16) <- (-1x320x28x28xf16)
        relu6_44 = paddle._C_ops.relu6(batch_norm__264)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [2, 2]

        # pd_op.pool2d: (-1x320x14x14xf16) <- (-1x320x28x28xf16, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(relu6_44, full_int_array_2, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x36x14x14xf16) <- (-1x320x14x14xf16, 36x320x3x3xf16)
        conv2d_45 = paddle._C_ops.conv2d(pool2d_2, parameter_225, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x14x14xf16, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x14x14xf16, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__270, batch_norm__271, batch_norm__272, batch_norm__273, batch_norm__274, batch_norm__275 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_45, parameter_226, parameter_227, parameter_228, parameter_229, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x36x14x14xf16) <- (-1x36x14x14xf16)
        relu6_45 = paddle._C_ops.relu6(batch_norm__270)

        # builtin.combine: ([-1x36x14x14xf16, -1x320x14x14xf16]) <- (-1x36x14x14xf16, -1x320x14x14xf16)
        combine_23 = [relu6_45, pool2d_2]

        # pd_op.full: (1xi32) <- ()
        full_23 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x356x14x14xf16) <- ([-1x36x14x14xf16, -1x320x14x14xf16], 1xi32)
        concat_23 = paddle._C_ops.concat(combine_23, full_23)

        # pd_op.conv2d: (-1x62x14x14xf16) <- (-1x356x14x14xf16, 62x356x3x3xf16)
        conv2d_46 = paddle._C_ops.conv2d(concat_23, parameter_230, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x62x14x14xf16, 62xf32, 62xf32, xf32, xf32, None) <- (-1x62x14x14xf16, 62xf32, 62xf32, 62xf32, 62xf32)
        batch_norm__276, batch_norm__277, batch_norm__278, batch_norm__279, batch_norm__280, batch_norm__281 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_46, parameter_231, parameter_232, parameter_233, parameter_234, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x62x14x14xf16) <- (-1x62x14x14xf16)
        relu6_46 = paddle._C_ops.relu6(batch_norm__276)

        # pd_op.conv2d: (-1x36x14x14xf16) <- (-1x62x14x14xf16, 36x62x3x3xf16)
        conv2d_47 = paddle._C_ops.conv2d(relu6_46, parameter_235, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x14x14xf16, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x14x14xf16, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__282, batch_norm__283, batch_norm__284, batch_norm__285, batch_norm__286, batch_norm__287 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_47, parameter_236, parameter_237, parameter_238, parameter_239, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x36x14x14xf16) <- (-1x36x14x14xf16)
        relu6_47 = paddle._C_ops.relu6(batch_norm__282)

        # builtin.combine: ([-1x36x14x14xf16, -1x62x14x14xf16, -1x320x14x14xf16]) <- (-1x36x14x14xf16, -1x62x14x14xf16, -1x320x14x14xf16)
        combine_24 = [relu6_47, relu6_46, pool2d_2]

        # pd_op.full: (1xi32) <- ()
        full_24 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x418x14x14xf16) <- ([-1x36x14x14xf16, -1x62x14x14xf16, -1x320x14x14xf16], 1xi32)
        concat_24 = paddle._C_ops.concat(combine_24, full_24)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x418x14x14xf16, 104x418x3x3xf16)
        conv2d_48 = paddle._C_ops.conv2d(concat_24, parameter_240, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, xf32, xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__288, batch_norm__289, batch_norm__290, batch_norm__291, batch_norm__292, batch_norm__293 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_48, parameter_241, parameter_242, parameter_243, parameter_244, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu6_48 = paddle._C_ops.relu6(batch_norm__288)

        # pd_op.conv2d: (-1x36x14x14xf16) <- (-1x104x14x14xf16, 36x104x3x3xf16)
        conv2d_49 = paddle._C_ops.conv2d(relu6_48, parameter_245, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x14x14xf16, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x14x14xf16, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__294, batch_norm__295, batch_norm__296, batch_norm__297, batch_norm__298, batch_norm__299 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_49, parameter_246, parameter_247, parameter_248, parameter_249, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x36x14x14xf16) <- (-1x36x14x14xf16)
        relu6_49 = paddle._C_ops.relu6(batch_norm__294)

        # builtin.combine: ([-1x36x14x14xf16, -1x104x14x14xf16]) <- (-1x36x14x14xf16, -1x104x14x14xf16)
        combine_25 = [relu6_49, relu6_48]

        # pd_op.full: (1xi32) <- ()
        full_25 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x140x14x14xf16) <- ([-1x36x14x14xf16, -1x104x14x14xf16], 1xi32)
        concat_25 = paddle._C_ops.concat(combine_25, full_25)

        # pd_op.conv2d: (-1x62x14x14xf16) <- (-1x140x14x14xf16, 62x140x3x3xf16)
        conv2d_50 = paddle._C_ops.conv2d(concat_25, parameter_250, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x62x14x14xf16, 62xf32, 62xf32, xf32, xf32, None) <- (-1x62x14x14xf16, 62xf32, 62xf32, 62xf32, 62xf32)
        batch_norm__300, batch_norm__301, batch_norm__302, batch_norm__303, batch_norm__304, batch_norm__305 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_50, parameter_251, parameter_252, parameter_253, parameter_254, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x62x14x14xf16) <- (-1x62x14x14xf16)
        relu6_50 = paddle._C_ops.relu6(batch_norm__300)

        # pd_op.conv2d: (-1x36x14x14xf16) <- (-1x62x14x14xf16, 36x62x3x3xf16)
        conv2d_51 = paddle._C_ops.conv2d(relu6_50, parameter_255, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x14x14xf16, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x14x14xf16, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__306, batch_norm__307, batch_norm__308, batch_norm__309, batch_norm__310, batch_norm__311 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_51, parameter_256, parameter_257, parameter_258, parameter_259, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x36x14x14xf16) <- (-1x36x14x14xf16)
        relu6_51 = paddle._C_ops.relu6(batch_norm__306)

        # builtin.combine: ([-1x36x14x14xf16, -1x62x14x14xf16, -1x104x14x14xf16, -1x320x14x14xf16]) <- (-1x36x14x14xf16, -1x62x14x14xf16, -1x104x14x14xf16, -1x320x14x14xf16)
        combine_26 = [relu6_51, relu6_50, relu6_48, pool2d_2]

        # pd_op.full: (1xi32) <- ()
        full_26 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x522x14x14xf16) <- ([-1x36x14x14xf16, -1x62x14x14xf16, -1x104x14x14xf16, -1x320x14x14xf16], 1xi32)
        concat_26 = paddle._C_ops.concat(combine_26, full_26)

        # pd_op.conv2d: (-1x176x14x14xf16) <- (-1x522x14x14xf16, 176x522x3x3xf16)
        conv2d_52 = paddle._C_ops.conv2d(concat_26, parameter_260, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x176x14x14xf16, 176xf32, 176xf32, xf32, xf32, None) <- (-1x176x14x14xf16, 176xf32, 176xf32, 176xf32, 176xf32)
        batch_norm__312, batch_norm__313, batch_norm__314, batch_norm__315, batch_norm__316, batch_norm__317 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_52, parameter_261, parameter_262, parameter_263, parameter_264, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x176x14x14xf16) <- (-1x176x14x14xf16)
        relu6_52 = paddle._C_ops.relu6(batch_norm__312)

        # pd_op.conv2d: (-1x36x14x14xf16) <- (-1x176x14x14xf16, 36x176x3x3xf16)
        conv2d_53 = paddle._C_ops.conv2d(relu6_52, parameter_265, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x14x14xf16, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x14x14xf16, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__318, batch_norm__319, batch_norm__320, batch_norm__321, batch_norm__322, batch_norm__323 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_53, parameter_266, parameter_267, parameter_268, parameter_269, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x36x14x14xf16) <- (-1x36x14x14xf16)
        relu6_53 = paddle._C_ops.relu6(batch_norm__318)

        # builtin.combine: ([-1x36x14x14xf16, -1x176x14x14xf16]) <- (-1x36x14x14xf16, -1x176x14x14xf16)
        combine_27 = [relu6_53, relu6_52]

        # pd_op.full: (1xi32) <- ()
        full_27 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x212x14x14xf16) <- ([-1x36x14x14xf16, -1x176x14x14xf16], 1xi32)
        concat_27 = paddle._C_ops.concat(combine_27, full_27)

        # pd_op.conv2d: (-1x62x14x14xf16) <- (-1x212x14x14xf16, 62x212x3x3xf16)
        conv2d_54 = paddle._C_ops.conv2d(concat_27, parameter_270, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x62x14x14xf16, 62xf32, 62xf32, xf32, xf32, None) <- (-1x62x14x14xf16, 62xf32, 62xf32, 62xf32, 62xf32)
        batch_norm__324, batch_norm__325, batch_norm__326, batch_norm__327, batch_norm__328, batch_norm__329 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_54, parameter_271, parameter_272, parameter_273, parameter_274, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x62x14x14xf16) <- (-1x62x14x14xf16)
        relu6_54 = paddle._C_ops.relu6(batch_norm__324)

        # pd_op.conv2d: (-1x36x14x14xf16) <- (-1x62x14x14xf16, 36x62x3x3xf16)
        conv2d_55 = paddle._C_ops.conv2d(relu6_54, parameter_275, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x14x14xf16, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x14x14xf16, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__330, batch_norm__331, batch_norm__332, batch_norm__333, batch_norm__334, batch_norm__335 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_55, parameter_276, parameter_277, parameter_278, parameter_279, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x36x14x14xf16) <- (-1x36x14x14xf16)
        relu6_55 = paddle._C_ops.relu6(batch_norm__330)

        # builtin.combine: ([-1x36x14x14xf16, -1x62x14x14xf16, -1x176x14x14xf16]) <- (-1x36x14x14xf16, -1x62x14x14xf16, -1x176x14x14xf16)
        combine_28 = [relu6_55, relu6_54, relu6_52]

        # pd_op.full: (1xi32) <- ()
        full_28 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x274x14x14xf16) <- ([-1x36x14x14xf16, -1x62x14x14xf16, -1x176x14x14xf16], 1xi32)
        concat_28 = paddle._C_ops.concat(combine_28, full_28)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x274x14x14xf16, 104x274x3x3xf16)
        conv2d_56 = paddle._C_ops.conv2d(concat_28, parameter_280, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, xf32, xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__336, batch_norm__337, batch_norm__338, batch_norm__339, batch_norm__340, batch_norm__341 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_56, parameter_281, parameter_282, parameter_283, parameter_284, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu6_56 = paddle._C_ops.relu6(batch_norm__336)

        # pd_op.conv2d: (-1x36x14x14xf16) <- (-1x104x14x14xf16, 36x104x3x3xf16)
        conv2d_57 = paddle._C_ops.conv2d(relu6_56, parameter_285, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x14x14xf16, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x14x14xf16, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__342, batch_norm__343, batch_norm__344, batch_norm__345, batch_norm__346, batch_norm__347 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_57, parameter_286, parameter_287, parameter_288, parameter_289, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x36x14x14xf16) <- (-1x36x14x14xf16)
        relu6_57 = paddle._C_ops.relu6(batch_norm__342)

        # builtin.combine: ([-1x36x14x14xf16, -1x104x14x14xf16]) <- (-1x36x14x14xf16, -1x104x14x14xf16)
        combine_29 = [relu6_57, relu6_56]

        # pd_op.full: (1xi32) <- ()
        full_29 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x140x14x14xf16) <- ([-1x36x14x14xf16, -1x104x14x14xf16], 1xi32)
        concat_29 = paddle._C_ops.concat(combine_29, full_29)

        # pd_op.conv2d: (-1x62x14x14xf16) <- (-1x140x14x14xf16, 62x140x3x3xf16)
        conv2d_58 = paddle._C_ops.conv2d(concat_29, parameter_290, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x62x14x14xf16, 62xf32, 62xf32, xf32, xf32, None) <- (-1x62x14x14xf16, 62xf32, 62xf32, 62xf32, 62xf32)
        batch_norm__348, batch_norm__349, batch_norm__350, batch_norm__351, batch_norm__352, batch_norm__353 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_58, parameter_291, parameter_292, parameter_293, parameter_294, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x62x14x14xf16) <- (-1x62x14x14xf16)
        relu6_58 = paddle._C_ops.relu6(batch_norm__348)

        # pd_op.conv2d: (-1x36x14x14xf16) <- (-1x62x14x14xf16, 36x62x3x3xf16)
        conv2d_59 = paddle._C_ops.conv2d(relu6_58, parameter_295, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x14x14xf16, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x14x14xf16, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__354, batch_norm__355, batch_norm__356, batch_norm__357, batch_norm__358, batch_norm__359 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_59, parameter_296, parameter_297, parameter_298, parameter_299, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x36x14x14xf16) <- (-1x36x14x14xf16)
        relu6_59 = paddle._C_ops.relu6(batch_norm__354)

        # builtin.combine: ([-1x36x14x14xf16, -1x62x14x14xf16, -1x104x14x14xf16, -1x176x14x14xf16, -1x320x14x14xf16]) <- (-1x36x14x14xf16, -1x62x14x14xf16, -1x104x14x14xf16, -1x176x14x14xf16, -1x320x14x14xf16)
        combine_30 = [relu6_59, relu6_58, relu6_56, relu6_52, pool2d_2]

        # pd_op.full: (1xi32) <- ()
        full_30 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x698x14x14xf16) <- ([-1x36x14x14xf16, -1x62x14x14xf16, -1x104x14x14xf16, -1x176x14x14xf16, -1x320x14x14xf16], 1xi32)
        concat_30 = paddle._C_ops.concat(combine_30, full_30)

        # pd_op.conv2d: (-1x300x14x14xf16) <- (-1x698x14x14xf16, 300x698x3x3xf16)
        conv2d_60 = paddle._C_ops.conv2d(concat_30, parameter_300, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x300x14x14xf16, 300xf32, 300xf32, xf32, xf32, None) <- (-1x300x14x14xf16, 300xf32, 300xf32, 300xf32, 300xf32)
        batch_norm__360, batch_norm__361, batch_norm__362, batch_norm__363, batch_norm__364, batch_norm__365 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_60, parameter_301, parameter_302, parameter_303, parameter_304, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x300x14x14xf16) <- (-1x300x14x14xf16)
        relu6_60 = paddle._C_ops.relu6(batch_norm__360)

        # builtin.combine: ([-1x36x14x14xf16, -1x36x14x14xf16, -1x36x14x14xf16, -1x36x14x14xf16, -1x36x14x14xf16, -1x36x14x14xf16, -1x36x14x14xf16, -1x36x14x14xf16, -1x300x14x14xf16]) <- (-1x36x14x14xf16, -1x36x14x14xf16, -1x36x14x14xf16, -1x36x14x14xf16, -1x36x14x14xf16, -1x36x14x14xf16, -1x36x14x14xf16, -1x36x14x14xf16, -1x300x14x14xf16)
        combine_31 = [relu6_45, relu6_47, relu6_49, relu6_51, relu6_53, relu6_55, relu6_57, relu6_59, relu6_60]

        # pd_op.full: (1xi32) <- ()
        full_31 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x588x14x14xf16) <- ([-1x36x14x14xf16, -1x36x14x14xf16, -1x36x14x14xf16, -1x36x14x14xf16, -1x36x14x14xf16, -1x36x14x14xf16, -1x36x14x14xf16, -1x36x14x14xf16, -1x300x14x14xf16], 1xi32)
        concat_31 = paddle._C_ops.concat(combine_31, full_31)

        # pd_op.conv2d: (-1x480x14x14xf16) <- (-1x588x14x14xf16, 480x588x1x1xf16)
        conv2d_61 = paddle._C_ops.conv2d(concat_31, parameter_305, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x480x14x14xf16, 480xf32, 480xf32, xf32, xf32, None) <- (-1x480x14x14xf16, 480xf32, 480xf32, 480xf32, 480xf32)
        batch_norm__366, batch_norm__367, batch_norm__368, batch_norm__369, batch_norm__370, batch_norm__371 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_61, parameter_306, parameter_307, parameter_308, parameter_309, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x480x14x14xf16) <- (-1x480x14x14xf16)
        relu6_61 = paddle._C_ops.relu6(batch_norm__366)

        # pd_op.conv2d: (-1x48x14x14xf16) <- (-1x480x14x14xf16, 48x480x3x3xf16)
        conv2d_62 = paddle._C_ops.conv2d(relu6_61, parameter_310, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x14x14xf16, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x14x14xf16, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__372, batch_norm__373, batch_norm__374, batch_norm__375, batch_norm__376, batch_norm__377 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_62, parameter_311, parameter_312, parameter_313, parameter_314, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x48x14x14xf16) <- (-1x48x14x14xf16)
        relu6_62 = paddle._C_ops.relu6(batch_norm__372)

        # builtin.combine: ([-1x48x14x14xf16, -1x480x14x14xf16]) <- (-1x48x14x14xf16, -1x480x14x14xf16)
        combine_32 = [relu6_62, relu6_61]

        # pd_op.full: (1xi32) <- ()
        full_32 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x528x14x14xf16) <- ([-1x48x14x14xf16, -1x480x14x14xf16], 1xi32)
        concat_32 = paddle._C_ops.concat(combine_32, full_32)

        # pd_op.conv2d: (-1x82x14x14xf16) <- (-1x528x14x14xf16, 82x528x3x3xf16)
        conv2d_63 = paddle._C_ops.conv2d(concat_32, parameter_315, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x82x14x14xf16, 82xf32, 82xf32, xf32, xf32, None) <- (-1x82x14x14xf16, 82xf32, 82xf32, 82xf32, 82xf32)
        batch_norm__378, batch_norm__379, batch_norm__380, batch_norm__381, batch_norm__382, batch_norm__383 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_63, parameter_316, parameter_317, parameter_318, parameter_319, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x82x14x14xf16) <- (-1x82x14x14xf16)
        relu6_63 = paddle._C_ops.relu6(batch_norm__378)

        # pd_op.conv2d: (-1x48x14x14xf16) <- (-1x82x14x14xf16, 48x82x3x3xf16)
        conv2d_64 = paddle._C_ops.conv2d(relu6_63, parameter_320, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x14x14xf16, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x14x14xf16, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__384, batch_norm__385, batch_norm__386, batch_norm__387, batch_norm__388, batch_norm__389 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_64, parameter_321, parameter_322, parameter_323, parameter_324, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x48x14x14xf16) <- (-1x48x14x14xf16)
        relu6_64 = paddle._C_ops.relu6(batch_norm__384)

        # builtin.combine: ([-1x48x14x14xf16, -1x82x14x14xf16, -1x480x14x14xf16]) <- (-1x48x14x14xf16, -1x82x14x14xf16, -1x480x14x14xf16)
        combine_33 = [relu6_64, relu6_63, relu6_61]

        # pd_op.full: (1xi32) <- ()
        full_33 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x610x14x14xf16) <- ([-1x48x14x14xf16, -1x82x14x14xf16, -1x480x14x14xf16], 1xi32)
        concat_33 = paddle._C_ops.concat(combine_33, full_33)

        # pd_op.conv2d: (-1x138x14x14xf16) <- (-1x610x14x14xf16, 138x610x3x3xf16)
        conv2d_65 = paddle._C_ops.conv2d(concat_33, parameter_325, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x138x14x14xf16, 138xf32, 138xf32, xf32, xf32, None) <- (-1x138x14x14xf16, 138xf32, 138xf32, 138xf32, 138xf32)
        batch_norm__390, batch_norm__391, batch_norm__392, batch_norm__393, batch_norm__394, batch_norm__395 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_65, parameter_326, parameter_327, parameter_328, parameter_329, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x138x14x14xf16) <- (-1x138x14x14xf16)
        relu6_65 = paddle._C_ops.relu6(batch_norm__390)

        # pd_op.conv2d: (-1x48x14x14xf16) <- (-1x138x14x14xf16, 48x138x3x3xf16)
        conv2d_66 = paddle._C_ops.conv2d(relu6_65, parameter_330, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x14x14xf16, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x14x14xf16, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__396, batch_norm__397, batch_norm__398, batch_norm__399, batch_norm__400, batch_norm__401 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_66, parameter_331, parameter_332, parameter_333, parameter_334, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x48x14x14xf16) <- (-1x48x14x14xf16)
        relu6_66 = paddle._C_ops.relu6(batch_norm__396)

        # builtin.combine: ([-1x48x14x14xf16, -1x138x14x14xf16]) <- (-1x48x14x14xf16, -1x138x14x14xf16)
        combine_34 = [relu6_66, relu6_65]

        # pd_op.full: (1xi32) <- ()
        full_34 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x186x14x14xf16) <- ([-1x48x14x14xf16, -1x138x14x14xf16], 1xi32)
        concat_34 = paddle._C_ops.concat(combine_34, full_34)

        # pd_op.conv2d: (-1x82x14x14xf16) <- (-1x186x14x14xf16, 82x186x3x3xf16)
        conv2d_67 = paddle._C_ops.conv2d(concat_34, parameter_335, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x82x14x14xf16, 82xf32, 82xf32, xf32, xf32, None) <- (-1x82x14x14xf16, 82xf32, 82xf32, 82xf32, 82xf32)
        batch_norm__402, batch_norm__403, batch_norm__404, batch_norm__405, batch_norm__406, batch_norm__407 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_67, parameter_336, parameter_337, parameter_338, parameter_339, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x82x14x14xf16) <- (-1x82x14x14xf16)
        relu6_67 = paddle._C_ops.relu6(batch_norm__402)

        # pd_op.conv2d: (-1x48x14x14xf16) <- (-1x82x14x14xf16, 48x82x3x3xf16)
        conv2d_68 = paddle._C_ops.conv2d(relu6_67, parameter_340, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x14x14xf16, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x14x14xf16, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__408, batch_norm__409, batch_norm__410, batch_norm__411, batch_norm__412, batch_norm__413 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_68, parameter_341, parameter_342, parameter_343, parameter_344, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x48x14x14xf16) <- (-1x48x14x14xf16)
        relu6_68 = paddle._C_ops.relu6(batch_norm__408)

        # builtin.combine: ([-1x48x14x14xf16, -1x82x14x14xf16, -1x138x14x14xf16, -1x480x14x14xf16]) <- (-1x48x14x14xf16, -1x82x14x14xf16, -1x138x14x14xf16, -1x480x14x14xf16)
        combine_35 = [relu6_68, relu6_67, relu6_65, relu6_61]

        # pd_op.full: (1xi32) <- ()
        full_35 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x748x14x14xf16) <- ([-1x48x14x14xf16, -1x82x14x14xf16, -1x138x14x14xf16, -1x480x14x14xf16], 1xi32)
        concat_35 = paddle._C_ops.concat(combine_35, full_35)

        # pd_op.conv2d: (-1x236x14x14xf16) <- (-1x748x14x14xf16, 236x748x3x3xf16)
        conv2d_69 = paddle._C_ops.conv2d(concat_35, parameter_345, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x236x14x14xf16, 236xf32, 236xf32, xf32, xf32, None) <- (-1x236x14x14xf16, 236xf32, 236xf32, 236xf32, 236xf32)
        batch_norm__414, batch_norm__415, batch_norm__416, batch_norm__417, batch_norm__418, batch_norm__419 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_69, parameter_346, parameter_347, parameter_348, parameter_349, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x236x14x14xf16) <- (-1x236x14x14xf16)
        relu6_69 = paddle._C_ops.relu6(batch_norm__414)

        # pd_op.conv2d: (-1x48x14x14xf16) <- (-1x236x14x14xf16, 48x236x3x3xf16)
        conv2d_70 = paddle._C_ops.conv2d(relu6_69, parameter_350, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x14x14xf16, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x14x14xf16, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__420, batch_norm__421, batch_norm__422, batch_norm__423, batch_norm__424, batch_norm__425 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_70, parameter_351, parameter_352, parameter_353, parameter_354, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x48x14x14xf16) <- (-1x48x14x14xf16)
        relu6_70 = paddle._C_ops.relu6(batch_norm__420)

        # builtin.combine: ([-1x48x14x14xf16, -1x236x14x14xf16]) <- (-1x48x14x14xf16, -1x236x14x14xf16)
        combine_36 = [relu6_70, relu6_69]

        # pd_op.full: (1xi32) <- ()
        full_36 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x284x14x14xf16) <- ([-1x48x14x14xf16, -1x236x14x14xf16], 1xi32)
        concat_36 = paddle._C_ops.concat(combine_36, full_36)

        # pd_op.conv2d: (-1x82x14x14xf16) <- (-1x284x14x14xf16, 82x284x3x3xf16)
        conv2d_71 = paddle._C_ops.conv2d(concat_36, parameter_355, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x82x14x14xf16, 82xf32, 82xf32, xf32, xf32, None) <- (-1x82x14x14xf16, 82xf32, 82xf32, 82xf32, 82xf32)
        batch_norm__426, batch_norm__427, batch_norm__428, batch_norm__429, batch_norm__430, batch_norm__431 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_71, parameter_356, parameter_357, parameter_358, parameter_359, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x82x14x14xf16) <- (-1x82x14x14xf16)
        relu6_71 = paddle._C_ops.relu6(batch_norm__426)

        # pd_op.conv2d: (-1x48x14x14xf16) <- (-1x82x14x14xf16, 48x82x3x3xf16)
        conv2d_72 = paddle._C_ops.conv2d(relu6_71, parameter_360, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x14x14xf16, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x14x14xf16, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__432, batch_norm__433, batch_norm__434, batch_norm__435, batch_norm__436, batch_norm__437 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_72, parameter_361, parameter_362, parameter_363, parameter_364, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x48x14x14xf16) <- (-1x48x14x14xf16)
        relu6_72 = paddle._C_ops.relu6(batch_norm__432)

        # builtin.combine: ([-1x48x14x14xf16, -1x82x14x14xf16, -1x236x14x14xf16]) <- (-1x48x14x14xf16, -1x82x14x14xf16, -1x236x14x14xf16)
        combine_37 = [relu6_72, relu6_71, relu6_69]

        # pd_op.full: (1xi32) <- ()
        full_37 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x366x14x14xf16) <- ([-1x48x14x14xf16, -1x82x14x14xf16, -1x236x14x14xf16], 1xi32)
        concat_37 = paddle._C_ops.concat(combine_37, full_37)

        # pd_op.conv2d: (-1x138x14x14xf16) <- (-1x366x14x14xf16, 138x366x3x3xf16)
        conv2d_73 = paddle._C_ops.conv2d(concat_37, parameter_365, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x138x14x14xf16, 138xf32, 138xf32, xf32, xf32, None) <- (-1x138x14x14xf16, 138xf32, 138xf32, 138xf32, 138xf32)
        batch_norm__438, batch_norm__439, batch_norm__440, batch_norm__441, batch_norm__442, batch_norm__443 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_73, parameter_366, parameter_367, parameter_368, parameter_369, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x138x14x14xf16) <- (-1x138x14x14xf16)
        relu6_73 = paddle._C_ops.relu6(batch_norm__438)

        # pd_op.conv2d: (-1x48x14x14xf16) <- (-1x138x14x14xf16, 48x138x3x3xf16)
        conv2d_74 = paddle._C_ops.conv2d(relu6_73, parameter_370, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x14x14xf16, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x14x14xf16, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__444, batch_norm__445, batch_norm__446, batch_norm__447, batch_norm__448, batch_norm__449 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_74, parameter_371, parameter_372, parameter_373, parameter_374, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x48x14x14xf16) <- (-1x48x14x14xf16)
        relu6_74 = paddle._C_ops.relu6(batch_norm__444)

        # builtin.combine: ([-1x48x14x14xf16, -1x138x14x14xf16]) <- (-1x48x14x14xf16, -1x138x14x14xf16)
        combine_38 = [relu6_74, relu6_73]

        # pd_op.full: (1xi32) <- ()
        full_38 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x186x14x14xf16) <- ([-1x48x14x14xf16, -1x138x14x14xf16], 1xi32)
        concat_38 = paddle._C_ops.concat(combine_38, full_38)

        # pd_op.conv2d: (-1x82x14x14xf16) <- (-1x186x14x14xf16, 82x186x3x3xf16)
        conv2d_75 = paddle._C_ops.conv2d(concat_38, parameter_375, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x82x14x14xf16, 82xf32, 82xf32, xf32, xf32, None) <- (-1x82x14x14xf16, 82xf32, 82xf32, 82xf32, 82xf32)
        batch_norm__450, batch_norm__451, batch_norm__452, batch_norm__453, batch_norm__454, batch_norm__455 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_75, parameter_376, parameter_377, parameter_378, parameter_379, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x82x14x14xf16) <- (-1x82x14x14xf16)
        relu6_75 = paddle._C_ops.relu6(batch_norm__450)

        # pd_op.conv2d: (-1x48x14x14xf16) <- (-1x82x14x14xf16, 48x82x3x3xf16)
        conv2d_76 = paddle._C_ops.conv2d(relu6_75, parameter_380, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x48x14x14xf16, 48xf32, 48xf32, xf32, xf32, None) <- (-1x48x14x14xf16, 48xf32, 48xf32, 48xf32, 48xf32)
        batch_norm__456, batch_norm__457, batch_norm__458, batch_norm__459, batch_norm__460, batch_norm__461 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_76, parameter_381, parameter_382, parameter_383, parameter_384, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x48x14x14xf16) <- (-1x48x14x14xf16)
        relu6_76 = paddle._C_ops.relu6(batch_norm__456)

        # builtin.combine: ([-1x48x14x14xf16, -1x82x14x14xf16, -1x138x14x14xf16, -1x236x14x14xf16, -1x480x14x14xf16]) <- (-1x48x14x14xf16, -1x82x14x14xf16, -1x138x14x14xf16, -1x236x14x14xf16, -1x480x14x14xf16)
        combine_39 = [relu6_76, relu6_75, relu6_73, relu6_69, relu6_61]

        # pd_op.full: (1xi32) <- ()
        full_39 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x984x14x14xf16) <- ([-1x48x14x14xf16, -1x82x14x14xf16, -1x138x14x14xf16, -1x236x14x14xf16, -1x480x14x14xf16], 1xi32)
        concat_39 = paddle._C_ops.concat(combine_39, full_39)

        # pd_op.conv2d: (-1x400x14x14xf16) <- (-1x984x14x14xf16, 400x984x3x3xf16)
        conv2d_77 = paddle._C_ops.conv2d(concat_39, parameter_385, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x400x14x14xf16, 400xf32, 400xf32, xf32, xf32, None) <- (-1x400x14x14xf16, 400xf32, 400xf32, 400xf32, 400xf32)
        batch_norm__462, batch_norm__463, batch_norm__464, batch_norm__465, batch_norm__466, batch_norm__467 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_77, parameter_386, parameter_387, parameter_388, parameter_389, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x400x14x14xf16) <- (-1x400x14x14xf16)
        relu6_77 = paddle._C_ops.relu6(batch_norm__462)

        # builtin.combine: ([-1x48x14x14xf16, -1x48x14x14xf16, -1x48x14x14xf16, -1x48x14x14xf16, -1x48x14x14xf16, -1x48x14x14xf16, -1x48x14x14xf16, -1x48x14x14xf16, -1x400x14x14xf16]) <- (-1x48x14x14xf16, -1x48x14x14xf16, -1x48x14x14xf16, -1x48x14x14xf16, -1x48x14x14xf16, -1x48x14x14xf16, -1x48x14x14xf16, -1x48x14x14xf16, -1x400x14x14xf16)
        combine_40 = [relu6_62, relu6_64, relu6_66, relu6_68, relu6_70, relu6_72, relu6_74, relu6_76, relu6_77]

        # pd_op.full: (1xi32) <- ()
        full_40 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x784x14x14xf16) <- ([-1x48x14x14xf16, -1x48x14x14xf16, -1x48x14x14xf16, -1x48x14x14xf16, -1x48x14x14xf16, -1x48x14x14xf16, -1x48x14x14xf16, -1x48x14x14xf16, -1x400x14x14xf16], 1xi32)
        concat_40 = paddle._C_ops.concat(combine_40, full_40)

        # pd_op.conv2d: (-1x720x14x14xf16) <- (-1x784x14x14xf16, 720x784x1x1xf16)
        conv2d_78 = paddle._C_ops.conv2d(concat_40, parameter_390, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x720x14x14xf16, 720xf32, 720xf32, xf32, xf32, None) <- (-1x720x14x14xf16, 720xf32, 720xf32, 720xf32, 720xf32)
        batch_norm__468, batch_norm__469, batch_norm__470, batch_norm__471, batch_norm__472, batch_norm__473 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_78, parameter_391, parameter_392, parameter_393, parameter_394, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x720x14x14xf16) <- (-1x720x14x14xf16)
        relu6_78 = paddle._C_ops.relu6(batch_norm__468)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [2, 2]

        # pd_op.pool2d: (-1x720x7x7xf16) <- (-1x720x14x14xf16, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(relu6_78, full_int_array_3, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x256x7x7xf16) <- (-1x720x7x7xf16, 256x720x3x3xf16)
        conv2d_79 = paddle._C_ops.conv2d(pool2d_3, parameter_395, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x7x7xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x7x7xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__474, batch_norm__475, batch_norm__476, batch_norm__477, batch_norm__478, batch_norm__479 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_79, parameter_396, parameter_397, parameter_398, parameter_399, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x256x7x7xf16) <- (-1x256x7x7xf16)
        relu6_79 = paddle._C_ops.relu6(batch_norm__474)

        # builtin.combine: ([-1x256x7x7xf16, -1x720x7x7xf16]) <- (-1x256x7x7xf16, -1x720x7x7xf16)
        combine_41 = [relu6_79, pool2d_3]

        # pd_op.full: (1xi32) <- ()
        full_41 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x976x7x7xf16) <- ([-1x256x7x7xf16, -1x720x7x7xf16], 1xi32)
        concat_41 = paddle._C_ops.concat(combine_41, full_41)

        # pd_op.conv2d: (-1x436x7x7xf16) <- (-1x976x7x7xf16, 436x976x3x3xf16)
        conv2d_80 = paddle._C_ops.conv2d(concat_41, parameter_400, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x436x7x7xf16, 436xf32, 436xf32, xf32, xf32, None) <- (-1x436x7x7xf16, 436xf32, 436xf32, 436xf32, 436xf32)
        batch_norm__480, batch_norm__481, batch_norm__482, batch_norm__483, batch_norm__484, batch_norm__485 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_80, parameter_401, parameter_402, parameter_403, parameter_404, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x436x7x7xf16) <- (-1x436x7x7xf16)
        relu6_80 = paddle._C_ops.relu6(batch_norm__480)

        # pd_op.conv2d: (-1x256x7x7xf16) <- (-1x436x7x7xf16, 256x436x3x3xf16)
        conv2d_81 = paddle._C_ops.conv2d(relu6_80, parameter_405, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x7x7xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x7x7xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__486, batch_norm__487, batch_norm__488, batch_norm__489, batch_norm__490, batch_norm__491 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_81, parameter_406, parameter_407, parameter_408, parameter_409, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x256x7x7xf16) <- (-1x256x7x7xf16)
        relu6_81 = paddle._C_ops.relu6(batch_norm__486)

        # builtin.combine: ([-1x256x7x7xf16, -1x436x7x7xf16, -1x720x7x7xf16]) <- (-1x256x7x7xf16, -1x436x7x7xf16, -1x720x7x7xf16)
        combine_42 = [relu6_81, relu6_80, pool2d_3]

        # pd_op.full: (1xi32) <- ()
        full_42 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1412x7x7xf16) <- ([-1x256x7x7xf16, -1x436x7x7xf16, -1x720x7x7xf16], 1xi32)
        concat_42 = paddle._C_ops.concat(combine_42, full_42)

        # pd_op.conv2d: (-1x740x7x7xf16) <- (-1x1412x7x7xf16, 740x1412x3x3xf16)
        conv2d_82 = paddle._C_ops.conv2d(concat_42, parameter_410, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x740x7x7xf16, 740xf32, 740xf32, xf32, xf32, None) <- (-1x740x7x7xf16, 740xf32, 740xf32, 740xf32, 740xf32)
        batch_norm__492, batch_norm__493, batch_norm__494, batch_norm__495, batch_norm__496, batch_norm__497 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_82, parameter_411, parameter_412, parameter_413, parameter_414, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x740x7x7xf16) <- (-1x740x7x7xf16)
        relu6_82 = paddle._C_ops.relu6(batch_norm__492)

        # builtin.combine: ([-1x256x7x7xf16, -1x256x7x7xf16, -1x740x7x7xf16]) <- (-1x256x7x7xf16, -1x256x7x7xf16, -1x740x7x7xf16)
        combine_43 = [relu6_79, relu6_81, relu6_82]

        # pd_op.full: (1xi32) <- ()
        full_43 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1252x7x7xf16) <- ([-1x256x7x7xf16, -1x256x7x7xf16, -1x740x7x7xf16], 1xi32)
        concat_43 = paddle._C_ops.concat(combine_43, full_43)

        # pd_op.full: (1xf32) <- ()
        full_44 = paddle._C_ops.full([1], float('0.1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.dropout: (-1x1252x7x7xf16, None) <- (-1x1252x7x7xf16, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(paddle._C_ops.dropout(concat_43, None, full_44, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x1280x7x7xf16) <- (-1x1252x7x7xf16, 1280x1252x1x1xf16)
        conv2d_83 = paddle._C_ops.conv2d(dropout_0, parameter_415, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1280x7x7xf16, 1280xf32, 1280xf32, xf32, xf32, None) <- (-1x1280x7x7xf16, 1280xf32, 1280xf32, 1280xf32, 1280xf32)
        batch_norm__498, batch_norm__499, batch_norm__500, batch_norm__501, batch_norm__502, batch_norm__503 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_83, parameter_416, parameter_417, parameter_418, parameter_419, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu6: (-1x1280x7x7xf16) <- (-1x1280x7x7xf16)
        relu6_83 = paddle._C_ops.relu6(batch_norm__498)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [1, 1]

        # pd_op.pool2d: (-1x1280x1x1xf16) <- (-1x1280x7x7xf16, 2xi64)
        pool2d_4 = paddle._C_ops.pool2d(relu6_83, full_int_array_4, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.flatten_: (-1x1280xf16, None) <- (-1x1280x1x1xf16)
        flatten__0, flatten__1 = (lambda x, f: f(x))(paddle._C_ops.flatten_(pool2d_4, 1, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xf32) <- ()
        full_45 = paddle._C_ops.full([1], float('0.2'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.dropout: (-1x1280xf16, None) <- (-1x1280xf16, None, 1xf32)
        dropout_2, dropout_3 = (lambda x, f: f(x))(paddle._C_ops.dropout(flatten__0, None, full_45, True, 'upscale_in_train', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x1000xf16) <- (-1x1280xf16, 1280x1000xf16)
        matmul_0 = paddle._C_ops.matmul(dropout_2, parameter_420, False, False)

        # pd_op.add_: (-1x1000xf16) <- (-1x1000xf16, 1000xf16)
        add__0 = paddle._C_ops.add_(matmul_0, parameter_421)

        # pd_op.softmax_: (-1x1000xf16) <- (-1x1000xf16)
        softmax__0 = paddle._C_ops.softmax_(add__0, -1)

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

    def forward(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_114, parameter_111, parameter_113, parameter_112, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_154, parameter_151, parameter_153, parameter_152, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_169, parameter_166, parameter_168, parameter_167, parameter_170, parameter_174, parameter_171, parameter_173, parameter_172, parameter_175, parameter_179, parameter_176, parameter_178, parameter_177, parameter_180, parameter_184, parameter_181, parameter_183, parameter_182, parameter_185, parameter_189, parameter_186, parameter_188, parameter_187, parameter_190, parameter_194, parameter_191, parameter_193, parameter_192, parameter_195, parameter_199, parameter_196, parameter_198, parameter_197, parameter_200, parameter_204, parameter_201, parameter_203, parameter_202, parameter_205, parameter_209, parameter_206, parameter_208, parameter_207, parameter_210, parameter_214, parameter_211, parameter_213, parameter_212, parameter_215, parameter_219, parameter_216, parameter_218, parameter_217, parameter_220, parameter_224, parameter_221, parameter_223, parameter_222, parameter_225, parameter_229, parameter_226, parameter_228, parameter_227, parameter_230, parameter_234, parameter_231, parameter_233, parameter_232, parameter_235, parameter_239, parameter_236, parameter_238, parameter_237, parameter_240, parameter_244, parameter_241, parameter_243, parameter_242, parameter_245, parameter_249, parameter_246, parameter_248, parameter_247, parameter_250, parameter_254, parameter_251, parameter_253, parameter_252, parameter_255, parameter_259, parameter_256, parameter_258, parameter_257, parameter_260, parameter_264, parameter_261, parameter_263, parameter_262, parameter_265, parameter_269, parameter_266, parameter_268, parameter_267, parameter_270, parameter_274, parameter_271, parameter_273, parameter_272, parameter_275, parameter_279, parameter_276, parameter_278, parameter_277, parameter_280, parameter_284, parameter_281, parameter_283, parameter_282, parameter_285, parameter_289, parameter_286, parameter_288, parameter_287, parameter_290, parameter_294, parameter_291, parameter_293, parameter_292, parameter_295, parameter_299, parameter_296, parameter_298, parameter_297, parameter_300, parameter_304, parameter_301, parameter_303, parameter_302, parameter_305, parameter_309, parameter_306, parameter_308, parameter_307, parameter_310, parameter_314, parameter_311, parameter_313, parameter_312, parameter_315, parameter_319, parameter_316, parameter_318, parameter_317, parameter_320, parameter_324, parameter_321, parameter_323, parameter_322, parameter_325, parameter_329, parameter_326, parameter_328, parameter_327, parameter_330, parameter_334, parameter_331, parameter_333, parameter_332, parameter_335, parameter_339, parameter_336, parameter_338, parameter_337, parameter_340, parameter_344, parameter_341, parameter_343, parameter_342, parameter_345, parameter_349, parameter_346, parameter_348, parameter_347, parameter_350, parameter_354, parameter_351, parameter_353, parameter_352, parameter_355, parameter_359, parameter_356, parameter_358, parameter_357, parameter_360, parameter_364, parameter_361, parameter_363, parameter_362, parameter_365, parameter_369, parameter_366, parameter_368, parameter_367, parameter_370, parameter_374, parameter_371, parameter_373, parameter_372, parameter_375, parameter_379, parameter_376, parameter_378, parameter_377, parameter_380, parameter_384, parameter_381, parameter_383, parameter_382, parameter_385, parameter_389, parameter_386, parameter_388, parameter_387, parameter_390, parameter_394, parameter_391, parameter_393, parameter_392, parameter_395, parameter_399, parameter_396, parameter_398, parameter_397, parameter_400, parameter_404, parameter_401, parameter_403, parameter_402, parameter_405, parameter_409, parameter_406, parameter_408, parameter_407, parameter_410, parameter_414, parameter_411, parameter_413, parameter_412, parameter_415, parameter_419, parameter_416, parameter_418, parameter_417, parameter_420, parameter_421, feed_0):
        return self.builtin_module_833_0_0(parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_114, parameter_111, parameter_113, parameter_112, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_154, parameter_151, parameter_153, parameter_152, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_169, parameter_166, parameter_168, parameter_167, parameter_170, parameter_174, parameter_171, parameter_173, parameter_172, parameter_175, parameter_179, parameter_176, parameter_178, parameter_177, parameter_180, parameter_184, parameter_181, parameter_183, parameter_182, parameter_185, parameter_189, parameter_186, parameter_188, parameter_187, parameter_190, parameter_194, parameter_191, parameter_193, parameter_192, parameter_195, parameter_199, parameter_196, parameter_198, parameter_197, parameter_200, parameter_204, parameter_201, parameter_203, parameter_202, parameter_205, parameter_209, parameter_206, parameter_208, parameter_207, parameter_210, parameter_214, parameter_211, parameter_213, parameter_212, parameter_215, parameter_219, parameter_216, parameter_218, parameter_217, parameter_220, parameter_224, parameter_221, parameter_223, parameter_222, parameter_225, parameter_229, parameter_226, parameter_228, parameter_227, parameter_230, parameter_234, parameter_231, parameter_233, parameter_232, parameter_235, parameter_239, parameter_236, parameter_238, parameter_237, parameter_240, parameter_244, parameter_241, parameter_243, parameter_242, parameter_245, parameter_249, parameter_246, parameter_248, parameter_247, parameter_250, parameter_254, parameter_251, parameter_253, parameter_252, parameter_255, parameter_259, parameter_256, parameter_258, parameter_257, parameter_260, parameter_264, parameter_261, parameter_263, parameter_262, parameter_265, parameter_269, parameter_266, parameter_268, parameter_267, parameter_270, parameter_274, parameter_271, parameter_273, parameter_272, parameter_275, parameter_279, parameter_276, parameter_278, parameter_277, parameter_280, parameter_284, parameter_281, parameter_283, parameter_282, parameter_285, parameter_289, parameter_286, parameter_288, parameter_287, parameter_290, parameter_294, parameter_291, parameter_293, parameter_292, parameter_295, parameter_299, parameter_296, parameter_298, parameter_297, parameter_300, parameter_304, parameter_301, parameter_303, parameter_302, parameter_305, parameter_309, parameter_306, parameter_308, parameter_307, parameter_310, parameter_314, parameter_311, parameter_313, parameter_312, parameter_315, parameter_319, parameter_316, parameter_318, parameter_317, parameter_320, parameter_324, parameter_321, parameter_323, parameter_322, parameter_325, parameter_329, parameter_326, parameter_328, parameter_327, parameter_330, parameter_334, parameter_331, parameter_333, parameter_332, parameter_335, parameter_339, parameter_336, parameter_338, parameter_337, parameter_340, parameter_344, parameter_341, parameter_343, parameter_342, parameter_345, parameter_349, parameter_346, parameter_348, parameter_347, parameter_350, parameter_354, parameter_351, parameter_353, parameter_352, parameter_355, parameter_359, parameter_356, parameter_358, parameter_357, parameter_360, parameter_364, parameter_361, parameter_363, parameter_362, parameter_365, parameter_369, parameter_366, parameter_368, parameter_367, parameter_370, parameter_374, parameter_371, parameter_373, parameter_372, parameter_375, parameter_379, parameter_376, parameter_378, parameter_377, parameter_380, parameter_384, parameter_381, parameter_383, parameter_382, parameter_385, parameter_389, parameter_386, parameter_388, parameter_387, parameter_390, parameter_394, parameter_391, parameter_393, parameter_392, parameter_395, parameter_399, parameter_396, parameter_398, parameter_397, parameter_400, parameter_404, parameter_401, parameter_403, parameter_402, parameter_405, parameter_409, parameter_406, parameter_408, parameter_407, parameter_410, parameter_414, parameter_411, parameter_413, parameter_412, parameter_415, parameter_419, parameter_416, parameter_418, parameter_417, parameter_420, parameter_421, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_833_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([48, 3, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_4
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([96, 48, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_9
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([24, 96, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_14
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([40, 120, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_19
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([24, 40, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_24
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([70, 160, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_29
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([24, 70, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_34
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([40, 94, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_39
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([24, 40, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_44
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([118, 230, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_49
            paddle.uniform([118], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([118], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([118], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([118], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([192, 214, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_54
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([24, 192, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_59
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([40, 216, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_64
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([24, 40, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_69
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([70, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_74
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([24, 70, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_79
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([40, 94, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_84
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([24, 40, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_89
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([118, 326, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_94
            paddle.uniform([118], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([118], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([118], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([118], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([24, 118, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_99
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([40, 142, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_104
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([24, 40, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_109
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([70, 182, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_114
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([70], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([24, 70, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_119
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([40, 94, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_124
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([24, 40, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_129
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([200, 444, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_134
            paddle.uniform([200], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([200], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([200], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([200], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([256, 392, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_139
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([28, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_144
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([48, 284, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_149
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([28, 48, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_154
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([80, 332, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_159
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([28, 80, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_164
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([48, 108, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_169
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([28, 48, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_174
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([138, 412, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_179
            paddle.uniform([138], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([138], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([138], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([138], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([28, 138, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_184
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([48, 166, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_189
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([28, 48, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_194
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([80, 214, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_199
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([28, 80, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_204
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([48, 108, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_209
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([28, 48, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_214
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_213
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_215
            paddle.uniform([234, 550, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_219
            paddle.uniform([234], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([234], dtype='float32', min=0, max=0.5),
            # parameter_218
            paddle.uniform([234], dtype='float32', min=0, max=0.5),
            # parameter_217
            paddle.uniform([234], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([320, 458, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_224
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_221
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_223
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_225
            paddle.uniform([36, 320, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_229
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_227
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([62, 356, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_234
            paddle.uniform([62], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([62], dtype='float32', min=0, max=0.5),
            # parameter_233
            paddle.uniform([62], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([62], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([36, 62, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_239
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_238
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([104, 418, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_244
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_241
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_243
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_245
            paddle.uniform([36, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_249
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_246
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_248
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_250
            paddle.uniform([62, 140, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_254
            paddle.uniform([62], dtype='float32', min=0, max=0.5),
            # parameter_251
            paddle.uniform([62], dtype='float32', min=0, max=0.5),
            # parameter_253
            paddle.uniform([62], dtype='float32', min=0, max=0.5),
            # parameter_252
            paddle.uniform([62], dtype='float32', min=0, max=0.5),
            # parameter_255
            paddle.uniform([36, 62, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_259
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_256
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_258
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_257
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([176, 522, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_264
            paddle.uniform([176], dtype='float32', min=0, max=0.5),
            # parameter_261
            paddle.uniform([176], dtype='float32', min=0, max=0.5),
            # parameter_263
            paddle.uniform([176], dtype='float32', min=0, max=0.5),
            # parameter_262
            paddle.uniform([176], dtype='float32', min=0, max=0.5),
            # parameter_265
            paddle.uniform([36, 176, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_269
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_266
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_268
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_267
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_270
            paddle.uniform([62, 212, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_274
            paddle.uniform([62], dtype='float32', min=0, max=0.5),
            # parameter_271
            paddle.uniform([62], dtype='float32', min=0, max=0.5),
            # parameter_273
            paddle.uniform([62], dtype='float32', min=0, max=0.5),
            # parameter_272
            paddle.uniform([62], dtype='float32', min=0, max=0.5),
            # parameter_275
            paddle.uniform([36, 62, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_279
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_276
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_278
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_277
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_280
            paddle.uniform([104, 274, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_284
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_281
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_283
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_282
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_285
            paddle.uniform([36, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_289
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_286
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_288
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_287
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_290
            paddle.uniform([62, 140, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_294
            paddle.uniform([62], dtype='float32', min=0, max=0.5),
            # parameter_291
            paddle.uniform([62], dtype='float32', min=0, max=0.5),
            # parameter_293
            paddle.uniform([62], dtype='float32', min=0, max=0.5),
            # parameter_292
            paddle.uniform([62], dtype='float32', min=0, max=0.5),
            # parameter_295
            paddle.uniform([36, 62, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_299
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_296
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_298
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_297
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_300
            paddle.uniform([300, 698, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_304
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            # parameter_301
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            # parameter_303
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            # parameter_302
            paddle.uniform([300], dtype='float32', min=0, max=0.5),
            # parameter_305
            paddle.uniform([480, 588, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_309
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_306
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_308
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_307
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_310
            paddle.uniform([48, 480, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_314
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_311
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_313
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_312
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_315
            paddle.uniform([82, 528, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_319
            paddle.uniform([82], dtype='float32', min=0, max=0.5),
            # parameter_316
            paddle.uniform([82], dtype='float32', min=0, max=0.5),
            # parameter_318
            paddle.uniform([82], dtype='float32', min=0, max=0.5),
            # parameter_317
            paddle.uniform([82], dtype='float32', min=0, max=0.5),
            # parameter_320
            paddle.uniform([48, 82, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_324
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_321
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_323
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_322
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_325
            paddle.uniform([138, 610, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_329
            paddle.uniform([138], dtype='float32', min=0, max=0.5),
            # parameter_326
            paddle.uniform([138], dtype='float32', min=0, max=0.5),
            # parameter_328
            paddle.uniform([138], dtype='float32', min=0, max=0.5),
            # parameter_327
            paddle.uniform([138], dtype='float32', min=0, max=0.5),
            # parameter_330
            paddle.uniform([48, 138, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_334
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_331
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_333
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_332
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_335
            paddle.uniform([82, 186, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_339
            paddle.uniform([82], dtype='float32', min=0, max=0.5),
            # parameter_336
            paddle.uniform([82], dtype='float32', min=0, max=0.5),
            # parameter_338
            paddle.uniform([82], dtype='float32', min=0, max=0.5),
            # parameter_337
            paddle.uniform([82], dtype='float32', min=0, max=0.5),
            # parameter_340
            paddle.uniform([48, 82, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_344
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_341
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_343
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_342
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_345
            paddle.uniform([236, 748, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_349
            paddle.uniform([236], dtype='float32', min=0, max=0.5),
            # parameter_346
            paddle.uniform([236], dtype='float32', min=0, max=0.5),
            # parameter_348
            paddle.uniform([236], dtype='float32', min=0, max=0.5),
            # parameter_347
            paddle.uniform([236], dtype='float32', min=0, max=0.5),
            # parameter_350
            paddle.uniform([48, 236, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_354
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_351
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_353
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_352
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_355
            paddle.uniform([82, 284, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_359
            paddle.uniform([82], dtype='float32', min=0, max=0.5),
            # parameter_356
            paddle.uniform([82], dtype='float32', min=0, max=0.5),
            # parameter_358
            paddle.uniform([82], dtype='float32', min=0, max=0.5),
            # parameter_357
            paddle.uniform([82], dtype='float32', min=0, max=0.5),
            # parameter_360
            paddle.uniform([48, 82, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_364
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_361
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_363
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_362
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_365
            paddle.uniform([138, 366, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_369
            paddle.uniform([138], dtype='float32', min=0, max=0.5),
            # parameter_366
            paddle.uniform([138], dtype='float32', min=0, max=0.5),
            # parameter_368
            paddle.uniform([138], dtype='float32', min=0, max=0.5),
            # parameter_367
            paddle.uniform([138], dtype='float32', min=0, max=0.5),
            # parameter_370
            paddle.uniform([48, 138, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_374
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_371
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_373
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_372
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_375
            paddle.uniform([82, 186, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_379
            paddle.uniform([82], dtype='float32', min=0, max=0.5),
            # parameter_376
            paddle.uniform([82], dtype='float32', min=0, max=0.5),
            # parameter_378
            paddle.uniform([82], dtype='float32', min=0, max=0.5),
            # parameter_377
            paddle.uniform([82], dtype='float32', min=0, max=0.5),
            # parameter_380
            paddle.uniform([48, 82, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_384
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_381
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_383
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_382
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            # parameter_385
            paddle.uniform([400, 984, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_389
            paddle.uniform([400], dtype='float32', min=0, max=0.5),
            # parameter_386
            paddle.uniform([400], dtype='float32', min=0, max=0.5),
            # parameter_388
            paddle.uniform([400], dtype='float32', min=0, max=0.5),
            # parameter_387
            paddle.uniform([400], dtype='float32', min=0, max=0.5),
            # parameter_390
            paddle.uniform([720, 784, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_394
            paddle.uniform([720], dtype='float32', min=0, max=0.5),
            # parameter_391
            paddle.uniform([720], dtype='float32', min=0, max=0.5),
            # parameter_393
            paddle.uniform([720], dtype='float32', min=0, max=0.5),
            # parameter_392
            paddle.uniform([720], dtype='float32', min=0, max=0.5),
            # parameter_395
            paddle.uniform([256, 720, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_399
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_396
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_398
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_397
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_400
            paddle.uniform([436, 976, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_404
            paddle.uniform([436], dtype='float32', min=0, max=0.5),
            # parameter_401
            paddle.uniform([436], dtype='float32', min=0, max=0.5),
            # parameter_403
            paddle.uniform([436], dtype='float32', min=0, max=0.5),
            # parameter_402
            paddle.uniform([436], dtype='float32', min=0, max=0.5),
            # parameter_405
            paddle.uniform([256, 436, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_409
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_406
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_408
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_407
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_410
            paddle.uniform([740, 1412, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_414
            paddle.uniform([740], dtype='float32', min=0, max=0.5),
            # parameter_411
            paddle.uniform([740], dtype='float32', min=0, max=0.5),
            # parameter_413
            paddle.uniform([740], dtype='float32', min=0, max=0.5),
            # parameter_412
            paddle.uniform([740], dtype='float32', min=0, max=0.5),
            # parameter_415
            paddle.uniform([1280, 1252, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_419
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_416
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_418
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_417
            paddle.uniform([1280], dtype='float32', min=0, max=0.5),
            # parameter_420
            paddle.uniform([1280, 1000], dtype='float16', min=0, max=0.5),
            # parameter_421
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
            paddle.static.InputSpec(shape=[48, 3, 3, 3], dtype='float16'),
            # parameter_4
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[96, 48, 3, 3], dtype='float16'),
            # parameter_9
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[24, 96, 3, 3], dtype='float16'),
            # parameter_14
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[40, 120, 3, 3], dtype='float16'),
            # parameter_19
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[24, 40, 3, 3], dtype='float16'),
            # parameter_24
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[70, 160, 3, 3], dtype='float16'),
            # parameter_29
            paddle.static.InputSpec(shape=[70], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[70], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[70], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[70], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[24, 70, 3, 3], dtype='float16'),
            # parameter_34
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[40, 94, 3, 3], dtype='float16'),
            # parameter_39
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[24, 40, 3, 3], dtype='float16'),
            # parameter_44
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[118, 230, 3, 3], dtype='float16'),
            # parameter_49
            paddle.static.InputSpec(shape=[118], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[118], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[118], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[118], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[192, 214, 1, 1], dtype='float16'),
            # parameter_54
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[24, 192, 3, 3], dtype='float16'),
            # parameter_59
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[40, 216, 3, 3], dtype='float16'),
            # parameter_64
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[24, 40, 3, 3], dtype='float16'),
            # parameter_69
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[70, 256, 3, 3], dtype='float16'),
            # parameter_74
            paddle.static.InputSpec(shape=[70], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[70], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[70], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[70], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[24, 70, 3, 3], dtype='float16'),
            # parameter_79
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[40, 94, 3, 3], dtype='float16'),
            # parameter_84
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[24, 40, 3, 3], dtype='float16'),
            # parameter_89
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[118, 326, 3, 3], dtype='float16'),
            # parameter_94
            paddle.static.InputSpec(shape=[118], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[118], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[118], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[118], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[24, 118, 3, 3], dtype='float16'),
            # parameter_99
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[40, 142, 3, 3], dtype='float16'),
            # parameter_104
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[24, 40, 3, 3], dtype='float16'),
            # parameter_109
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[70, 182, 3, 3], dtype='float16'),
            # parameter_114
            paddle.static.InputSpec(shape=[70], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[70], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[70], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[70], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[24, 70, 3, 3], dtype='float16'),
            # parameter_119
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[40, 94, 3, 3], dtype='float16'),
            # parameter_124
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[24, 40, 3, 3], dtype='float16'),
            # parameter_129
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[200, 444, 3, 3], dtype='float16'),
            # parameter_134
            paddle.static.InputSpec(shape=[200], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[200], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[200], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[200], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[256, 392, 1, 1], dtype='float16'),
            # parameter_139
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[28, 256, 3, 3], dtype='float16'),
            # parameter_144
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[48, 284, 3, 3], dtype='float16'),
            # parameter_149
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[28, 48, 3, 3], dtype='float16'),
            # parameter_154
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[80, 332, 3, 3], dtype='float16'),
            # parameter_159
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[28, 80, 3, 3], dtype='float16'),
            # parameter_164
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[48, 108, 3, 3], dtype='float16'),
            # parameter_169
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[28, 48, 3, 3], dtype='float16'),
            # parameter_174
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[138, 412, 3, 3], dtype='float16'),
            # parameter_179
            paddle.static.InputSpec(shape=[138], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[138], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[138], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[138], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[28, 138, 3, 3], dtype='float16'),
            # parameter_184
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[48, 166, 3, 3], dtype='float16'),
            # parameter_189
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[28, 48, 3, 3], dtype='float16'),
            # parameter_194
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[80, 214, 3, 3], dtype='float16'),
            # parameter_199
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[28, 80, 3, 3], dtype='float16'),
            # parameter_204
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[48, 108, 3, 3], dtype='float16'),
            # parameter_209
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[28, 48, 3, 3], dtype='float16'),
            # parameter_214
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_213
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_215
            paddle.static.InputSpec(shape=[234, 550, 3, 3], dtype='float16'),
            # parameter_219
            paddle.static.InputSpec(shape=[234], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[234], dtype='float32'),
            # parameter_218
            paddle.static.InputSpec(shape=[234], dtype='float32'),
            # parameter_217
            paddle.static.InputSpec(shape=[234], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[320, 458, 1, 1], dtype='float16'),
            # parameter_224
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_221
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_223
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_225
            paddle.static.InputSpec(shape=[36, 320, 3, 3], dtype='float16'),
            # parameter_229
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_227
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[62, 356, 3, 3], dtype='float16'),
            # parameter_234
            paddle.static.InputSpec(shape=[62], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[62], dtype='float32'),
            # parameter_233
            paddle.static.InputSpec(shape=[62], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[62], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[36, 62, 3, 3], dtype='float16'),
            # parameter_239
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_238
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[104, 418, 3, 3], dtype='float16'),
            # parameter_244
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_241
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_243
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_245
            paddle.static.InputSpec(shape=[36, 104, 3, 3], dtype='float16'),
            # parameter_249
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_246
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_248
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_250
            paddle.static.InputSpec(shape=[62, 140, 3, 3], dtype='float16'),
            # parameter_254
            paddle.static.InputSpec(shape=[62], dtype='float32'),
            # parameter_251
            paddle.static.InputSpec(shape=[62], dtype='float32'),
            # parameter_253
            paddle.static.InputSpec(shape=[62], dtype='float32'),
            # parameter_252
            paddle.static.InputSpec(shape=[62], dtype='float32'),
            # parameter_255
            paddle.static.InputSpec(shape=[36, 62, 3, 3], dtype='float16'),
            # parameter_259
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_256
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_258
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_257
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[176, 522, 3, 3], dtype='float16'),
            # parameter_264
            paddle.static.InputSpec(shape=[176], dtype='float32'),
            # parameter_261
            paddle.static.InputSpec(shape=[176], dtype='float32'),
            # parameter_263
            paddle.static.InputSpec(shape=[176], dtype='float32'),
            # parameter_262
            paddle.static.InputSpec(shape=[176], dtype='float32'),
            # parameter_265
            paddle.static.InputSpec(shape=[36, 176, 3, 3], dtype='float16'),
            # parameter_269
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_266
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_268
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_267
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_270
            paddle.static.InputSpec(shape=[62, 212, 3, 3], dtype='float16'),
            # parameter_274
            paddle.static.InputSpec(shape=[62], dtype='float32'),
            # parameter_271
            paddle.static.InputSpec(shape=[62], dtype='float32'),
            # parameter_273
            paddle.static.InputSpec(shape=[62], dtype='float32'),
            # parameter_272
            paddle.static.InputSpec(shape=[62], dtype='float32'),
            # parameter_275
            paddle.static.InputSpec(shape=[36, 62, 3, 3], dtype='float16'),
            # parameter_279
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_276
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_278
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_277
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_280
            paddle.static.InputSpec(shape=[104, 274, 3, 3], dtype='float16'),
            # parameter_284
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_281
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_283
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_282
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_285
            paddle.static.InputSpec(shape=[36, 104, 3, 3], dtype='float16'),
            # parameter_289
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_286
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_288
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_287
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_290
            paddle.static.InputSpec(shape=[62, 140, 3, 3], dtype='float16'),
            # parameter_294
            paddle.static.InputSpec(shape=[62], dtype='float32'),
            # parameter_291
            paddle.static.InputSpec(shape=[62], dtype='float32'),
            # parameter_293
            paddle.static.InputSpec(shape=[62], dtype='float32'),
            # parameter_292
            paddle.static.InputSpec(shape=[62], dtype='float32'),
            # parameter_295
            paddle.static.InputSpec(shape=[36, 62, 3, 3], dtype='float16'),
            # parameter_299
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_296
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_298
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_297
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_300
            paddle.static.InputSpec(shape=[300, 698, 3, 3], dtype='float16'),
            # parameter_304
            paddle.static.InputSpec(shape=[300], dtype='float32'),
            # parameter_301
            paddle.static.InputSpec(shape=[300], dtype='float32'),
            # parameter_303
            paddle.static.InputSpec(shape=[300], dtype='float32'),
            # parameter_302
            paddle.static.InputSpec(shape=[300], dtype='float32'),
            # parameter_305
            paddle.static.InputSpec(shape=[480, 588, 1, 1], dtype='float16'),
            # parameter_309
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_306
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_308
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_307
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_310
            paddle.static.InputSpec(shape=[48, 480, 3, 3], dtype='float16'),
            # parameter_314
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_311
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_313
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_312
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_315
            paddle.static.InputSpec(shape=[82, 528, 3, 3], dtype='float16'),
            # parameter_319
            paddle.static.InputSpec(shape=[82], dtype='float32'),
            # parameter_316
            paddle.static.InputSpec(shape=[82], dtype='float32'),
            # parameter_318
            paddle.static.InputSpec(shape=[82], dtype='float32'),
            # parameter_317
            paddle.static.InputSpec(shape=[82], dtype='float32'),
            # parameter_320
            paddle.static.InputSpec(shape=[48, 82, 3, 3], dtype='float16'),
            # parameter_324
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_321
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_323
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_322
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_325
            paddle.static.InputSpec(shape=[138, 610, 3, 3], dtype='float16'),
            # parameter_329
            paddle.static.InputSpec(shape=[138], dtype='float32'),
            # parameter_326
            paddle.static.InputSpec(shape=[138], dtype='float32'),
            # parameter_328
            paddle.static.InputSpec(shape=[138], dtype='float32'),
            # parameter_327
            paddle.static.InputSpec(shape=[138], dtype='float32'),
            # parameter_330
            paddle.static.InputSpec(shape=[48, 138, 3, 3], dtype='float16'),
            # parameter_334
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_331
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_333
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_332
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_335
            paddle.static.InputSpec(shape=[82, 186, 3, 3], dtype='float16'),
            # parameter_339
            paddle.static.InputSpec(shape=[82], dtype='float32'),
            # parameter_336
            paddle.static.InputSpec(shape=[82], dtype='float32'),
            # parameter_338
            paddle.static.InputSpec(shape=[82], dtype='float32'),
            # parameter_337
            paddle.static.InputSpec(shape=[82], dtype='float32'),
            # parameter_340
            paddle.static.InputSpec(shape=[48, 82, 3, 3], dtype='float16'),
            # parameter_344
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_341
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_343
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_342
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_345
            paddle.static.InputSpec(shape=[236, 748, 3, 3], dtype='float16'),
            # parameter_349
            paddle.static.InputSpec(shape=[236], dtype='float32'),
            # parameter_346
            paddle.static.InputSpec(shape=[236], dtype='float32'),
            # parameter_348
            paddle.static.InputSpec(shape=[236], dtype='float32'),
            # parameter_347
            paddle.static.InputSpec(shape=[236], dtype='float32'),
            # parameter_350
            paddle.static.InputSpec(shape=[48, 236, 3, 3], dtype='float16'),
            # parameter_354
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_351
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_353
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_352
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_355
            paddle.static.InputSpec(shape=[82, 284, 3, 3], dtype='float16'),
            # parameter_359
            paddle.static.InputSpec(shape=[82], dtype='float32'),
            # parameter_356
            paddle.static.InputSpec(shape=[82], dtype='float32'),
            # parameter_358
            paddle.static.InputSpec(shape=[82], dtype='float32'),
            # parameter_357
            paddle.static.InputSpec(shape=[82], dtype='float32'),
            # parameter_360
            paddle.static.InputSpec(shape=[48, 82, 3, 3], dtype='float16'),
            # parameter_364
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_361
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_363
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_362
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_365
            paddle.static.InputSpec(shape=[138, 366, 3, 3], dtype='float16'),
            # parameter_369
            paddle.static.InputSpec(shape=[138], dtype='float32'),
            # parameter_366
            paddle.static.InputSpec(shape=[138], dtype='float32'),
            # parameter_368
            paddle.static.InputSpec(shape=[138], dtype='float32'),
            # parameter_367
            paddle.static.InputSpec(shape=[138], dtype='float32'),
            # parameter_370
            paddle.static.InputSpec(shape=[48, 138, 3, 3], dtype='float16'),
            # parameter_374
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_371
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_373
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_372
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_375
            paddle.static.InputSpec(shape=[82, 186, 3, 3], dtype='float16'),
            # parameter_379
            paddle.static.InputSpec(shape=[82], dtype='float32'),
            # parameter_376
            paddle.static.InputSpec(shape=[82], dtype='float32'),
            # parameter_378
            paddle.static.InputSpec(shape=[82], dtype='float32'),
            # parameter_377
            paddle.static.InputSpec(shape=[82], dtype='float32'),
            # parameter_380
            paddle.static.InputSpec(shape=[48, 82, 3, 3], dtype='float16'),
            # parameter_384
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_381
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_383
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_382
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            # parameter_385
            paddle.static.InputSpec(shape=[400, 984, 3, 3], dtype='float16'),
            # parameter_389
            paddle.static.InputSpec(shape=[400], dtype='float32'),
            # parameter_386
            paddle.static.InputSpec(shape=[400], dtype='float32'),
            # parameter_388
            paddle.static.InputSpec(shape=[400], dtype='float32'),
            # parameter_387
            paddle.static.InputSpec(shape=[400], dtype='float32'),
            # parameter_390
            paddle.static.InputSpec(shape=[720, 784, 1, 1], dtype='float16'),
            # parameter_394
            paddle.static.InputSpec(shape=[720], dtype='float32'),
            # parameter_391
            paddle.static.InputSpec(shape=[720], dtype='float32'),
            # parameter_393
            paddle.static.InputSpec(shape=[720], dtype='float32'),
            # parameter_392
            paddle.static.InputSpec(shape=[720], dtype='float32'),
            # parameter_395
            paddle.static.InputSpec(shape=[256, 720, 3, 3], dtype='float16'),
            # parameter_399
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_396
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_398
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_397
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_400
            paddle.static.InputSpec(shape=[436, 976, 3, 3], dtype='float16'),
            # parameter_404
            paddle.static.InputSpec(shape=[436], dtype='float32'),
            # parameter_401
            paddle.static.InputSpec(shape=[436], dtype='float32'),
            # parameter_403
            paddle.static.InputSpec(shape=[436], dtype='float32'),
            # parameter_402
            paddle.static.InputSpec(shape=[436], dtype='float32'),
            # parameter_405
            paddle.static.InputSpec(shape=[256, 436, 3, 3], dtype='float16'),
            # parameter_409
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_406
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_408
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_407
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_410
            paddle.static.InputSpec(shape=[740, 1412, 3, 3], dtype='float16'),
            # parameter_414
            paddle.static.InputSpec(shape=[740], dtype='float32'),
            # parameter_411
            paddle.static.InputSpec(shape=[740], dtype='float32'),
            # parameter_413
            paddle.static.InputSpec(shape=[740], dtype='float32'),
            # parameter_412
            paddle.static.InputSpec(shape=[740], dtype='float32'),
            # parameter_415
            paddle.static.InputSpec(shape=[1280, 1252, 1, 1], dtype='float16'),
            # parameter_419
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_416
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_418
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_417
            paddle.static.InputSpec(shape=[1280], dtype='float32'),
            # parameter_420
            paddle.static.InputSpec(shape=[1280, 1000], dtype='float16'),
            # parameter_421
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