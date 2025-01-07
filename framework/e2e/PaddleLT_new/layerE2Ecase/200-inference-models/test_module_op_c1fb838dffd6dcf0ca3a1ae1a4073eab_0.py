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
    return [341][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_882_0_0(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_6, parameter_10, parameter_7, parameter_9, parameter_8, parameter_11, parameter_12, parameter_16, parameter_13, parameter_15, parameter_14, parameter_17, parameter_18, parameter_22, parameter_19, parameter_21, parameter_20, parameter_23, parameter_24, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_31, parameter_35, parameter_32, parameter_34, parameter_33, parameter_36, parameter_37, parameter_41, parameter_38, parameter_40, parameter_39, parameter_42, parameter_43, parameter_44, parameter_48, parameter_45, parameter_47, parameter_46, parameter_49, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_56, parameter_60, parameter_57, parameter_59, parameter_58, parameter_61, parameter_62, parameter_63, parameter_67, parameter_64, parameter_66, parameter_65, parameter_68, parameter_69, parameter_73, parameter_70, parameter_72, parameter_71, parameter_74, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_81, parameter_82, parameter_86, parameter_83, parameter_85, parameter_84, parameter_87, parameter_88, parameter_92, parameter_89, parameter_91, parameter_90, parameter_93, parameter_94, parameter_98, parameter_95, parameter_97, parameter_96, parameter_99, parameter_100, parameter_101, parameter_105, parameter_102, parameter_104, parameter_103, parameter_106, parameter_107, parameter_111, parameter_108, parameter_110, parameter_109, parameter_112, parameter_113, parameter_117, parameter_114, parameter_116, parameter_115, parameter_118, parameter_119, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_126, parameter_130, parameter_127, parameter_129, parameter_128, parameter_131, parameter_132, parameter_136, parameter_133, parameter_135, parameter_134, parameter_137, parameter_138, parameter_139, parameter_143, parameter_140, parameter_142, parameter_141, parameter_144, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_151, parameter_155, parameter_152, parameter_154, parameter_153, parameter_156, parameter_157, parameter_158, parameter_162, parameter_159, parameter_161, parameter_160, parameter_163, parameter_164, parameter_168, parameter_165, parameter_167, parameter_166, parameter_169, parameter_170, parameter_174, parameter_171, parameter_173, parameter_172, parameter_175, parameter_176, parameter_180, parameter_177, parameter_179, parameter_178, parameter_181, parameter_182, parameter_183, parameter_187, parameter_184, parameter_186, parameter_185, parameter_188, parameter_189, parameter_193, parameter_190, parameter_192, parameter_191, parameter_194, parameter_195, parameter_199, parameter_196, parameter_198, parameter_197, parameter_200, parameter_201, parameter_202, parameter_206, parameter_203, parameter_205, parameter_204, parameter_207, parameter_208, parameter_212, parameter_209, parameter_211, parameter_210, parameter_213, parameter_214, parameter_218, parameter_215, parameter_217, parameter_216, parameter_219, parameter_220, parameter_221, parameter_225, parameter_222, parameter_224, parameter_223, parameter_226, parameter_227, parameter_231, parameter_228, parameter_230, parameter_229, parameter_232, parameter_233, parameter_237, parameter_234, parameter_236, parameter_235, parameter_238, parameter_239, parameter_240, parameter_244, parameter_241, parameter_243, parameter_242, parameter_245, parameter_246, parameter_250, parameter_247, parameter_249, parameter_248, parameter_251, parameter_252, parameter_256, parameter_253, parameter_255, parameter_254, parameter_257, parameter_258, parameter_262, parameter_259, parameter_261, parameter_260, parameter_263, parameter_264, parameter_265, parameter_269, parameter_266, parameter_268, parameter_267, parameter_270, parameter_271, parameter_275, parameter_272, parameter_274, parameter_273, parameter_276, parameter_277, parameter_281, parameter_278, parameter_280, parameter_279, parameter_282, parameter_283, parameter_284, parameter_288, parameter_285, parameter_287, parameter_286, parameter_289, parameter_290, parameter_294, parameter_291, parameter_293, parameter_292, parameter_295, parameter_296, parameter_300, parameter_297, parameter_299, parameter_298, parameter_301, parameter_302, parameter_303, parameter_307, parameter_304, parameter_306, parameter_305, parameter_308, parameter_309, parameter_313, parameter_310, parameter_312, parameter_311, parameter_314, parameter_315, parameter_319, parameter_316, parameter_318, parameter_317, parameter_320, parameter_321, parameter_322, parameter_326, parameter_323, parameter_325, parameter_324, parameter_327, parameter_328, parameter_332, parameter_329, parameter_331, parameter_330, parameter_333, parameter_334, parameter_338, parameter_335, parameter_337, parameter_336, parameter_339, parameter_340, parameter_344, parameter_341, parameter_343, parameter_342, parameter_345, parameter_346, parameter_347, parameter_351, parameter_348, parameter_350, parameter_349, parameter_352, parameter_353, parameter_357, parameter_354, parameter_356, parameter_355, parameter_358, parameter_359, parameter_363, parameter_360, parameter_362, parameter_361, parameter_364, parameter_365, parameter_366, parameter_370, parameter_367, parameter_369, parameter_368, parameter_371, parameter_372, parameter_376, parameter_373, parameter_375, parameter_374, parameter_377, parameter_378, parameter_382, parameter_379, parameter_381, parameter_380, parameter_383, parameter_384, parameter_385, parameter_389, parameter_386, parameter_388, parameter_387, parameter_390, parameter_391, parameter_395, parameter_392, parameter_394, parameter_393, parameter_396, parameter_397, parameter_401, parameter_398, parameter_400, parameter_399, parameter_402, parameter_403, parameter_404, parameter_408, parameter_405, parameter_407, parameter_406, parameter_409, parameter_410, parameter_414, parameter_411, parameter_413, parameter_412, parameter_415, parameter_416, parameter_420, parameter_417, parameter_419, parameter_418, parameter_421, parameter_422, parameter_426, parameter_423, parameter_425, parameter_424, parameter_427, parameter_428, parameter_429, parameter_433, parameter_430, parameter_432, parameter_431, parameter_434, parameter_435, parameter_439, parameter_436, parameter_438, parameter_437, parameter_440, parameter_441, parameter_445, parameter_442, parameter_444, parameter_443, parameter_446, parameter_447, parameter_448, parameter_452, parameter_449, parameter_451, parameter_450, parameter_453, parameter_457, parameter_454, parameter_456, parameter_455, parameter_458, parameter_462, parameter_459, parameter_461, parameter_460, parameter_463, parameter_467, parameter_464, parameter_466, parameter_465, parameter_468, parameter_472, parameter_469, parameter_471, parameter_470, parameter_473, parameter_477, parameter_474, parameter_476, parameter_475, parameter_478, parameter_482, parameter_479, parameter_481, parameter_480, parameter_483, parameter_487, parameter_484, parameter_486, parameter_485, parameter_488, parameter_492, parameter_489, parameter_491, parameter_490, parameter_493, parameter_497, parameter_494, parameter_496, parameter_495, parameter_498, parameter_502, parameter_499, parameter_501, parameter_500, parameter_503, parameter_507, parameter_504, parameter_506, parameter_505, parameter_508, parameter_512, parameter_509, parameter_511, parameter_510, parameter_513, parameter_517, parameter_514, parameter_516, parameter_515, parameter_518, parameter_522, parameter_519, parameter_521, parameter_520, parameter_523, parameter_527, parameter_524, parameter_526, parameter_525, parameter_528, parameter_532, parameter_529, parameter_531, parameter_530, parameter_533, feed_0):

        # pd_op.conv2d: (1x13x512x1024xf32) <- (1x3x1024x2048xf32, 13x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(feed_0, parameter_0, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [3, 3]

        # pd_op.pool2d: (1x3x512x1024xf32) <- (1x3x1024x2048xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(feed_0, full_int_array_0, [2, 2], [1, 1], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # builtin.combine: ([1x13x512x1024xf32, 1x3x512x1024xf32]) <- (1x13x512x1024xf32, 1x3x512x1024xf32)
        combine_0 = [conv2d_0, pool2d_0]

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x16x512x1024xf32) <- ([1x13x512x1024xf32, 1x3x512x1024xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        # pd_op.batch_norm_: (1x16x512x1024xf32, 16xf32, 16xf32, xf32, xf32, None) <- (1x16x512x1024xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_0, parameter_1, parameter_2, parameter_3, parameter_4, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x16x512x1024xf32) <- (1x16x512x1024xf32, 1xf32)
        prelu_0 = paddle._C_ops.prelu(batch_norm__0, parameter_5, 'NCHW', 'all')

        # pd_op.max_pool2d_with_index: (1x16x256x512xf32, 1x16x256x512xi32) <- (1x16x512x1024xf32)
        max_pool2d_with_index_0, max_pool2d_with_index_1 = (lambda x, f: f(x))(paddle._C_ops.max_pool2d_with_index(prelu_0, [2, 2], [2, 2], [0, 0], False, False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x4x256x512xf32) <- (1x16x512x1024xf32, 4x16x2x2xf32)
        conv2d_1 = paddle._C_ops.conv2d(prelu_0, parameter_6, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x4x256x512xf32, 4xf32, 4xf32, xf32, xf32, None) <- (1x4x256x512xf32, 4xf32, 4xf32, 4xf32, 4xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_1, parameter_7, parameter_8, parameter_9, parameter_10, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x4x256x512xf32) <- (1x4x256x512xf32, 1xf32)
        prelu_1 = paddle._C_ops.prelu(batch_norm__6, parameter_11, 'NCHW', 'all')

        # pd_op.conv2d: (1x4x256x512xf32) <- (1x4x256x512xf32, 4x4x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(prelu_1, parameter_12, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x4x256x512xf32, 4xf32, 4xf32, xf32, xf32, None) <- (1x4x256x512xf32, 4xf32, 4xf32, 4xf32, 4xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_2, parameter_13, parameter_14, parameter_15, parameter_16, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x4x256x512xf32) <- (1x4x256x512xf32, 1xf32)
        prelu_2 = paddle._C_ops.prelu(batch_norm__12, parameter_17, 'NCHW', 'all')

        # pd_op.conv2d: (1x64x256x512xf32) <- (1x4x256x512xf32, 64x4x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(prelu_2, parameter_18, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x64x256x512xf32, 64xf32, 64xf32, xf32, xf32, None) <- (1x64x256x512xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_3, parameter_19, parameter_20, parameter_21, parameter_22, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x64x256x512xf32) <- (1x64x256x512xf32, 1xf32)
        prelu_3 = paddle._C_ops.prelu(batch_norm__18, parameter_23, 'NCHW', 'all')

        # pd_op.full: (1x48x256x512xf32) <- ()
        full_1 = paddle._C_ops.full([1, 48, 256, 512], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # builtin.combine: ([1x16x256x512xf32, 1x48x256x512xf32]) <- (1x16x256x512xf32, 1x48x256x512xf32)
        combine_1 = [max_pool2d_with_index_0, full_1]

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x64x256x512xf32) <- ([1x16x256x512xf32, 1x48x256x512xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_2)

        # pd_op.add_: (1x64x256x512xf32) <- (1x64x256x512xf32, 1x64x256x512xf32)
        add__0 = paddle._C_ops.add_(concat_1, prelu_3)

        # pd_op.prelu: (1x64x256x512xf32) <- (1x64x256x512xf32, 1xf32)
        prelu_4 = paddle._C_ops.prelu(add__0, parameter_24, 'NCHW', 'all')

        # pd_op.conv2d: (1x16x256x512xf32) <- (1x64x256x512xf32, 16x64x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(prelu_4, parameter_25, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x16x256x512xf32, 16xf32, 16xf32, xf32, xf32, None) <- (1x16x256x512xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_4, parameter_26, parameter_27, parameter_28, parameter_29, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x16x256x512xf32) <- (1x16x256x512xf32, 1xf32)
        prelu_5 = paddle._C_ops.prelu(batch_norm__24, parameter_30, 'NCHW', 'all')

        # pd_op.conv2d: (1x16x256x512xf32) <- (1x16x256x512xf32, 16x16x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(prelu_5, parameter_31, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x16x256x512xf32, 16xf32, 16xf32, xf32, xf32, None) <- (1x16x256x512xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_5, parameter_32, parameter_33, parameter_34, parameter_35, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x16x256x512xf32) <- (1x16x256x512xf32, 1xf32)
        prelu_6 = paddle._C_ops.prelu(batch_norm__30, parameter_36, 'NCHW', 'all')

        # pd_op.conv2d: (1x64x256x512xf32) <- (1x16x256x512xf32, 64x16x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(prelu_6, parameter_37, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x64x256x512xf32, 64xf32, 64xf32, xf32, xf32, None) <- (1x64x256x512xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_6, parameter_38, parameter_39, parameter_40, parameter_41, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x64x256x512xf32) <- (1x64x256x512xf32, 1xf32)
        prelu_7 = paddle._C_ops.prelu(batch_norm__36, parameter_42, 'NCHW', 'all')

        # pd_op.add_: (1x64x256x512xf32) <- (1x64x256x512xf32, 1x64x256x512xf32)
        add__1 = paddle._C_ops.add_(prelu_4, prelu_7)

        # pd_op.prelu: (1x64x256x512xf32) <- (1x64x256x512xf32, 1xf32)
        prelu_8 = paddle._C_ops.prelu(add__1, parameter_43, 'NCHW', 'all')

        # pd_op.conv2d: (1x16x256x512xf32) <- (1x64x256x512xf32, 16x64x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(prelu_8, parameter_44, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x16x256x512xf32, 16xf32, 16xf32, xf32, xf32, None) <- (1x16x256x512xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_7, parameter_45, parameter_46, parameter_47, parameter_48, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x16x256x512xf32) <- (1x16x256x512xf32, 1xf32)
        prelu_9 = paddle._C_ops.prelu(batch_norm__42, parameter_49, 'NCHW', 'all')

        # pd_op.conv2d: (1x16x256x512xf32) <- (1x16x256x512xf32, 16x16x3x3xf32)
        conv2d_8 = paddle._C_ops.conv2d(prelu_9, parameter_50, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x16x256x512xf32, 16xf32, 16xf32, xf32, xf32, None) <- (1x16x256x512xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_8, parameter_51, parameter_52, parameter_53, parameter_54, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x16x256x512xf32) <- (1x16x256x512xf32, 1xf32)
        prelu_10 = paddle._C_ops.prelu(batch_norm__48, parameter_55, 'NCHW', 'all')

        # pd_op.conv2d: (1x64x256x512xf32) <- (1x16x256x512xf32, 64x16x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(prelu_10, parameter_56, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x64x256x512xf32, 64xf32, 64xf32, xf32, xf32, None) <- (1x64x256x512xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_9, parameter_57, parameter_58, parameter_59, parameter_60, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x64x256x512xf32) <- (1x64x256x512xf32, 1xf32)
        prelu_11 = paddle._C_ops.prelu(batch_norm__54, parameter_61, 'NCHW', 'all')

        # pd_op.add_: (1x64x256x512xf32) <- (1x64x256x512xf32, 1x64x256x512xf32)
        add__2 = paddle._C_ops.add_(prelu_8, prelu_11)

        # pd_op.prelu: (1x64x256x512xf32) <- (1x64x256x512xf32, 1xf32)
        prelu_12 = paddle._C_ops.prelu(add__2, parameter_62, 'NCHW', 'all')

        # pd_op.conv2d: (1x16x256x512xf32) <- (1x64x256x512xf32, 16x64x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(prelu_12, parameter_63, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x16x256x512xf32, 16xf32, 16xf32, xf32, xf32, None) <- (1x16x256x512xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_10, parameter_64, parameter_65, parameter_66, parameter_67, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x16x256x512xf32) <- (1x16x256x512xf32, 1xf32)
        prelu_13 = paddle._C_ops.prelu(batch_norm__60, parameter_68, 'NCHW', 'all')

        # pd_op.conv2d: (1x16x256x512xf32) <- (1x16x256x512xf32, 16x16x3x3xf32)
        conv2d_11 = paddle._C_ops.conv2d(prelu_13, parameter_69, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x16x256x512xf32, 16xf32, 16xf32, xf32, xf32, None) <- (1x16x256x512xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_11, parameter_70, parameter_71, parameter_72, parameter_73, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x16x256x512xf32) <- (1x16x256x512xf32, 1xf32)
        prelu_14 = paddle._C_ops.prelu(batch_norm__66, parameter_74, 'NCHW', 'all')

        # pd_op.conv2d: (1x64x256x512xf32) <- (1x16x256x512xf32, 64x16x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(prelu_14, parameter_75, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x64x256x512xf32, 64xf32, 64xf32, xf32, xf32, None) <- (1x64x256x512xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_12, parameter_76, parameter_77, parameter_78, parameter_79, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x64x256x512xf32) <- (1x64x256x512xf32, 1xf32)
        prelu_15 = paddle._C_ops.prelu(batch_norm__72, parameter_80, 'NCHW', 'all')

        # pd_op.add_: (1x64x256x512xf32) <- (1x64x256x512xf32, 1x64x256x512xf32)
        add__3 = paddle._C_ops.add_(prelu_12, prelu_15)

        # pd_op.prelu: (1x64x256x512xf32) <- (1x64x256x512xf32, 1xf32)
        prelu_16 = paddle._C_ops.prelu(add__3, parameter_81, 'NCHW', 'all')

        # pd_op.conv2d: (1x16x256x512xf32) <- (1x64x256x512xf32, 16x64x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(prelu_16, parameter_82, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x16x256x512xf32, 16xf32, 16xf32, xf32, xf32, None) <- (1x16x256x512xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_13, parameter_83, parameter_84, parameter_85, parameter_86, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x16x256x512xf32) <- (1x16x256x512xf32, 1xf32)
        prelu_17 = paddle._C_ops.prelu(batch_norm__78, parameter_87, 'NCHW', 'all')

        # pd_op.conv2d: (1x16x256x512xf32) <- (1x16x256x512xf32, 16x16x3x3xf32)
        conv2d_14 = paddle._C_ops.conv2d(prelu_17, parameter_88, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x16x256x512xf32, 16xf32, 16xf32, xf32, xf32, None) <- (1x16x256x512xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_14, parameter_89, parameter_90, parameter_91, parameter_92, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x16x256x512xf32) <- (1x16x256x512xf32, 1xf32)
        prelu_18 = paddle._C_ops.prelu(batch_norm__84, parameter_93, 'NCHW', 'all')

        # pd_op.conv2d: (1x64x256x512xf32) <- (1x16x256x512xf32, 64x16x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(prelu_18, parameter_94, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x64x256x512xf32, 64xf32, 64xf32, xf32, xf32, None) <- (1x64x256x512xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_15, parameter_95, parameter_96, parameter_97, parameter_98, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x64x256x512xf32) <- (1x64x256x512xf32, 1xf32)
        prelu_19 = paddle._C_ops.prelu(batch_norm__90, parameter_99, 'NCHW', 'all')

        # pd_op.add_: (1x64x256x512xf32) <- (1x64x256x512xf32, 1x64x256x512xf32)
        add__4 = paddle._C_ops.add_(prelu_16, prelu_19)

        # pd_op.prelu: (1x64x256x512xf32) <- (1x64x256x512xf32, 1xf32)
        prelu_20 = paddle._C_ops.prelu(add__4, parameter_100, 'NCHW', 'all')

        # pd_op.max_pool2d_with_index: (1x64x128x256xf32, 1x64x128x256xi32) <- (1x64x256x512xf32)
        max_pool2d_with_index_2, max_pool2d_with_index_3 = (lambda x, f: f(x))(paddle._C_ops.max_pool2d_with_index(prelu_20, [2, 2], [2, 2], [0, 0], False, False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (1x16x128x256xf32) <- (1x64x256x512xf32, 16x64x2x2xf32)
        conv2d_16 = paddle._C_ops.conv2d(prelu_20, parameter_101, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x16x128x256xf32, 16xf32, 16xf32, xf32, xf32, None) <- (1x16x128x256xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_16, parameter_102, parameter_103, parameter_104, parameter_105, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x16x128x256xf32) <- (1x16x128x256xf32, 1xf32)
        prelu_21 = paddle._C_ops.prelu(batch_norm__96, parameter_106, 'NCHW', 'all')

        # pd_op.conv2d: (1x16x128x256xf32) <- (1x16x128x256xf32, 16x16x3x3xf32)
        conv2d_17 = paddle._C_ops.conv2d(prelu_21, parameter_107, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x16x128x256xf32, 16xf32, 16xf32, xf32, xf32, None) <- (1x16x128x256xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_17, parameter_108, parameter_109, parameter_110, parameter_111, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x16x128x256xf32) <- (1x16x128x256xf32, 1xf32)
        prelu_22 = paddle._C_ops.prelu(batch_norm__102, parameter_112, 'NCHW', 'all')

        # pd_op.conv2d: (1x128x128x256xf32) <- (1x16x128x256xf32, 128x16x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(prelu_22, parameter_113, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x128x128x256xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x128x256xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_18, parameter_114, parameter_115, parameter_116, parameter_117, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x128x128x256xf32) <- (1x128x128x256xf32, 1xf32)
        prelu_23 = paddle._C_ops.prelu(batch_norm__108, parameter_118, 'NCHW', 'all')

        # pd_op.full: (1x64x128x256xf32) <- ()
        full_3 = paddle._C_ops.full([1, 64, 128, 256], float('0'), paddle.float32, paddle.framework._current_expected_place())

        # builtin.combine: ([1x64x128x256xf32, 1x64x128x256xf32]) <- (1x64x128x256xf32, 1x64x128x256xf32)
        combine_2 = [max_pool2d_with_index_2, full_3]

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (1x128x128x256xf32) <- ([1x64x128x256xf32, 1x64x128x256xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_4)

        # pd_op.add_: (1x128x128x256xf32) <- (1x128x128x256xf32, 1x128x128x256xf32)
        add__5 = paddle._C_ops.add_(concat_2, prelu_23)

        # pd_op.prelu: (1x128x128x256xf32) <- (1x128x128x256xf32, 1xf32)
        prelu_24 = paddle._C_ops.prelu(add__5, parameter_119, 'NCHW', 'all')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x128x128x256xf32, 32x128x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(prelu_24, parameter_120, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_19, parameter_121, parameter_122, parameter_123, parameter_124, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x32x128x256xf32) <- (1x32x128x256xf32, 1xf32)
        prelu_25 = paddle._C_ops.prelu(batch_norm__114, parameter_125, 'NCHW', 'all')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x32x128x256xf32, 32x32x3x3xf32)
        conv2d_20 = paddle._C_ops.conv2d(prelu_25, parameter_126, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_20, parameter_127, parameter_128, parameter_129, parameter_130, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x32x128x256xf32) <- (1x32x128x256xf32, 1xf32)
        prelu_26 = paddle._C_ops.prelu(batch_norm__120, parameter_131, 'NCHW', 'all')

        # pd_op.conv2d: (1x128x128x256xf32) <- (1x32x128x256xf32, 128x32x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(prelu_26, parameter_132, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x128x128x256xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x128x256xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_21, parameter_133, parameter_134, parameter_135, parameter_136, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x128x128x256xf32) <- (1x128x128x256xf32, 1xf32)
        prelu_27 = paddle._C_ops.prelu(batch_norm__126, parameter_137, 'NCHW', 'all')

        # pd_op.add_: (1x128x128x256xf32) <- (1x128x128x256xf32, 1x128x128x256xf32)
        add__6 = paddle._C_ops.add_(prelu_24, prelu_27)

        # pd_op.prelu: (1x128x128x256xf32) <- (1x128x128x256xf32, 1xf32)
        prelu_28 = paddle._C_ops.prelu(add__6, parameter_138, 'NCHW', 'all')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x128x128x256xf32, 32x128x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(prelu_28, parameter_139, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_22, parameter_140, parameter_141, parameter_142, parameter_143, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x32x128x256xf32) <- (1x32x128x256xf32, 1xf32)
        prelu_29 = paddle._C_ops.prelu(batch_norm__132, parameter_144, 'NCHW', 'all')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x32x128x256xf32, 32x32x3x3xf32)
        conv2d_23 = paddle._C_ops.conv2d(prelu_29, parameter_145, [1, 1], [2, 2], 'EXPLICIT', [2, 2], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_23, parameter_146, parameter_147, parameter_148, parameter_149, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x32x128x256xf32) <- (1x32x128x256xf32, 1xf32)
        prelu_30 = paddle._C_ops.prelu(batch_norm__138, parameter_150, 'NCHW', 'all')

        # pd_op.conv2d: (1x128x128x256xf32) <- (1x32x128x256xf32, 128x32x1x1xf32)
        conv2d_24 = paddle._C_ops.conv2d(prelu_30, parameter_151, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x128x128x256xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x128x256xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_24, parameter_152, parameter_153, parameter_154, parameter_155, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x128x128x256xf32) <- (1x128x128x256xf32, 1xf32)
        prelu_31 = paddle._C_ops.prelu(batch_norm__144, parameter_156, 'NCHW', 'all')

        # pd_op.add_: (1x128x128x256xf32) <- (1x128x128x256xf32, 1x128x128x256xf32)
        add__7 = paddle._C_ops.add_(prelu_28, prelu_31)

        # pd_op.prelu: (1x128x128x256xf32) <- (1x128x128x256xf32, 1xf32)
        prelu_32 = paddle._C_ops.prelu(add__7, parameter_157, 'NCHW', 'all')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x128x128x256xf32, 32x128x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(prelu_32, parameter_158, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_25, parameter_159, parameter_160, parameter_161, parameter_162, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x32x128x256xf32) <- (1x32x128x256xf32, 1xf32)
        prelu_33 = paddle._C_ops.prelu(batch_norm__150, parameter_163, 'NCHW', 'all')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x32x128x256xf32, 32x32x5x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(prelu_33, parameter_164, [1, 1], [2, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_26, parameter_165, parameter_166, parameter_167, parameter_168, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x32x128x256xf32) <- (1x32x128x256xf32, 1xf32)
        prelu_34 = paddle._C_ops.prelu(batch_norm__156, parameter_169, 'NCHW', 'all')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x32x128x256xf32, 32x32x1x5xf32)
        conv2d_27 = paddle._C_ops.conv2d(prelu_34, parameter_170, [1, 1], [0, 2], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_27, parameter_171, parameter_172, parameter_173, parameter_174, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x32x128x256xf32) <- (1x32x128x256xf32, 1xf32)
        prelu_35 = paddle._C_ops.prelu(batch_norm__162, parameter_175, 'NCHW', 'all')

        # pd_op.conv2d: (1x128x128x256xf32) <- (1x32x128x256xf32, 128x32x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(prelu_35, parameter_176, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x128x128x256xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x128x256xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_28, parameter_177, parameter_178, parameter_179, parameter_180, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x128x128x256xf32) <- (1x128x128x256xf32, 1xf32)
        prelu_36 = paddle._C_ops.prelu(batch_norm__168, parameter_181, 'NCHW', 'all')

        # pd_op.add_: (1x128x128x256xf32) <- (1x128x128x256xf32, 1x128x128x256xf32)
        add__8 = paddle._C_ops.add_(prelu_32, prelu_36)

        # pd_op.prelu: (1x128x128x256xf32) <- (1x128x128x256xf32, 1xf32)
        prelu_37 = paddle._C_ops.prelu(add__8, parameter_182, 'NCHW', 'all')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x128x128x256xf32, 32x128x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(prelu_37, parameter_183, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_29, parameter_184, parameter_185, parameter_186, parameter_187, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x32x128x256xf32) <- (1x32x128x256xf32, 1xf32)
        prelu_38 = paddle._C_ops.prelu(batch_norm__174, parameter_188, 'NCHW', 'all')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x32x128x256xf32, 32x32x3x3xf32)
        conv2d_30 = paddle._C_ops.conv2d(prelu_38, parameter_189, [1, 1], [4, 4], 'EXPLICIT', [4, 4], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__180, batch_norm__181, batch_norm__182, batch_norm__183, batch_norm__184, batch_norm__185 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_30, parameter_190, parameter_191, parameter_192, parameter_193, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x32x128x256xf32) <- (1x32x128x256xf32, 1xf32)
        prelu_39 = paddle._C_ops.prelu(batch_norm__180, parameter_194, 'NCHW', 'all')

        # pd_op.conv2d: (1x128x128x256xf32) <- (1x32x128x256xf32, 128x32x1x1xf32)
        conv2d_31 = paddle._C_ops.conv2d(prelu_39, parameter_195, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x128x128x256xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x128x256xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__186, batch_norm__187, batch_norm__188, batch_norm__189, batch_norm__190, batch_norm__191 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_31, parameter_196, parameter_197, parameter_198, parameter_199, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x128x128x256xf32) <- (1x128x128x256xf32, 1xf32)
        prelu_40 = paddle._C_ops.prelu(batch_norm__186, parameter_200, 'NCHW', 'all')

        # pd_op.add_: (1x128x128x256xf32) <- (1x128x128x256xf32, 1x128x128x256xf32)
        add__9 = paddle._C_ops.add_(prelu_37, prelu_40)

        # pd_op.prelu: (1x128x128x256xf32) <- (1x128x128x256xf32, 1xf32)
        prelu_41 = paddle._C_ops.prelu(add__9, parameter_201, 'NCHW', 'all')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x128x128x256xf32, 32x128x1x1xf32)
        conv2d_32 = paddle._C_ops.conv2d(prelu_41, parameter_202, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__192, batch_norm__193, batch_norm__194, batch_norm__195, batch_norm__196, batch_norm__197 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_32, parameter_203, parameter_204, parameter_205, parameter_206, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x32x128x256xf32) <- (1x32x128x256xf32, 1xf32)
        prelu_42 = paddle._C_ops.prelu(batch_norm__192, parameter_207, 'NCHW', 'all')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x32x128x256xf32, 32x32x3x3xf32)
        conv2d_33 = paddle._C_ops.conv2d(prelu_42, parameter_208, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__198, batch_norm__199, batch_norm__200, batch_norm__201, batch_norm__202, batch_norm__203 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_33, parameter_209, parameter_210, parameter_211, parameter_212, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x32x128x256xf32) <- (1x32x128x256xf32, 1xf32)
        prelu_43 = paddle._C_ops.prelu(batch_norm__198, parameter_213, 'NCHW', 'all')

        # pd_op.conv2d: (1x128x128x256xf32) <- (1x32x128x256xf32, 128x32x1x1xf32)
        conv2d_34 = paddle._C_ops.conv2d(prelu_43, parameter_214, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x128x128x256xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x128x256xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__204, batch_norm__205, batch_norm__206, batch_norm__207, batch_norm__208, batch_norm__209 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_34, parameter_215, parameter_216, parameter_217, parameter_218, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x128x128x256xf32) <- (1x128x128x256xf32, 1xf32)
        prelu_44 = paddle._C_ops.prelu(batch_norm__204, parameter_219, 'NCHW', 'all')

        # pd_op.add_: (1x128x128x256xf32) <- (1x128x128x256xf32, 1x128x128x256xf32)
        add__10 = paddle._C_ops.add_(prelu_41, prelu_44)

        # pd_op.prelu: (1x128x128x256xf32) <- (1x128x128x256xf32, 1xf32)
        prelu_45 = paddle._C_ops.prelu(add__10, parameter_220, 'NCHW', 'all')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x128x128x256xf32, 32x128x1x1xf32)
        conv2d_35 = paddle._C_ops.conv2d(prelu_45, parameter_221, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__210, batch_norm__211, batch_norm__212, batch_norm__213, batch_norm__214, batch_norm__215 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_35, parameter_222, parameter_223, parameter_224, parameter_225, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x32x128x256xf32) <- (1x32x128x256xf32, 1xf32)
        prelu_46 = paddle._C_ops.prelu(batch_norm__210, parameter_226, 'NCHW', 'all')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x32x128x256xf32, 32x32x3x3xf32)
        conv2d_36 = paddle._C_ops.conv2d(prelu_46, parameter_227, [1, 1], [8, 8], 'EXPLICIT', [8, 8], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__216, batch_norm__217, batch_norm__218, batch_norm__219, batch_norm__220, batch_norm__221 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_36, parameter_228, parameter_229, parameter_230, parameter_231, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x32x128x256xf32) <- (1x32x128x256xf32, 1xf32)
        prelu_47 = paddle._C_ops.prelu(batch_norm__216, parameter_232, 'NCHW', 'all')

        # pd_op.conv2d: (1x128x128x256xf32) <- (1x32x128x256xf32, 128x32x1x1xf32)
        conv2d_37 = paddle._C_ops.conv2d(prelu_47, parameter_233, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x128x128x256xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x128x256xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__222, batch_norm__223, batch_norm__224, batch_norm__225, batch_norm__226, batch_norm__227 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_37, parameter_234, parameter_235, parameter_236, parameter_237, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x128x128x256xf32) <- (1x128x128x256xf32, 1xf32)
        prelu_48 = paddle._C_ops.prelu(batch_norm__222, parameter_238, 'NCHW', 'all')

        # pd_op.add_: (1x128x128x256xf32) <- (1x128x128x256xf32, 1x128x128x256xf32)
        add__11 = paddle._C_ops.add_(prelu_45, prelu_48)

        # pd_op.prelu: (1x128x128x256xf32) <- (1x128x128x256xf32, 1xf32)
        prelu_49 = paddle._C_ops.prelu(add__11, parameter_239, 'NCHW', 'all')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x128x128x256xf32, 32x128x1x1xf32)
        conv2d_38 = paddle._C_ops.conv2d(prelu_49, parameter_240, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__228, batch_norm__229, batch_norm__230, batch_norm__231, batch_norm__232, batch_norm__233 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_38, parameter_241, parameter_242, parameter_243, parameter_244, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x32x128x256xf32) <- (1x32x128x256xf32, 1xf32)
        prelu_50 = paddle._C_ops.prelu(batch_norm__228, parameter_245, 'NCHW', 'all')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x32x128x256xf32, 32x32x5x1xf32)
        conv2d_39 = paddle._C_ops.conv2d(prelu_50, parameter_246, [1, 1], [2, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__234, batch_norm__235, batch_norm__236, batch_norm__237, batch_norm__238, batch_norm__239 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_39, parameter_247, parameter_248, parameter_249, parameter_250, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x32x128x256xf32) <- (1x32x128x256xf32, 1xf32)
        prelu_51 = paddle._C_ops.prelu(batch_norm__234, parameter_251, 'NCHW', 'all')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x32x128x256xf32, 32x32x1x5xf32)
        conv2d_40 = paddle._C_ops.conv2d(prelu_51, parameter_252, [1, 1], [0, 2], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__240, batch_norm__241, batch_norm__242, batch_norm__243, batch_norm__244, batch_norm__245 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_40, parameter_253, parameter_254, parameter_255, parameter_256, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x32x128x256xf32) <- (1x32x128x256xf32, 1xf32)
        prelu_52 = paddle._C_ops.prelu(batch_norm__240, parameter_257, 'NCHW', 'all')

        # pd_op.conv2d: (1x128x128x256xf32) <- (1x32x128x256xf32, 128x32x1x1xf32)
        conv2d_41 = paddle._C_ops.conv2d(prelu_52, parameter_258, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x128x128x256xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x128x256xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__246, batch_norm__247, batch_norm__248, batch_norm__249, batch_norm__250, batch_norm__251 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_41, parameter_259, parameter_260, parameter_261, parameter_262, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x128x128x256xf32) <- (1x128x128x256xf32, 1xf32)
        prelu_53 = paddle._C_ops.prelu(batch_norm__246, parameter_263, 'NCHW', 'all')

        # pd_op.add_: (1x128x128x256xf32) <- (1x128x128x256xf32, 1x128x128x256xf32)
        add__12 = paddle._C_ops.add_(prelu_49, prelu_53)

        # pd_op.prelu: (1x128x128x256xf32) <- (1x128x128x256xf32, 1xf32)
        prelu_54 = paddle._C_ops.prelu(add__12, parameter_264, 'NCHW', 'all')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x128x128x256xf32, 32x128x1x1xf32)
        conv2d_42 = paddle._C_ops.conv2d(prelu_54, parameter_265, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__252, batch_norm__253, batch_norm__254, batch_norm__255, batch_norm__256, batch_norm__257 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_42, parameter_266, parameter_267, parameter_268, parameter_269, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x32x128x256xf32) <- (1x32x128x256xf32, 1xf32)
        prelu_55 = paddle._C_ops.prelu(batch_norm__252, parameter_270, 'NCHW', 'all')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x32x128x256xf32, 32x32x3x3xf32)
        conv2d_43 = paddle._C_ops.conv2d(prelu_55, parameter_271, [1, 1], [16, 16], 'EXPLICIT', [16, 16], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__258, batch_norm__259, batch_norm__260, batch_norm__261, batch_norm__262, batch_norm__263 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_43, parameter_272, parameter_273, parameter_274, parameter_275, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x32x128x256xf32) <- (1x32x128x256xf32, 1xf32)
        prelu_56 = paddle._C_ops.prelu(batch_norm__258, parameter_276, 'NCHW', 'all')

        # pd_op.conv2d: (1x128x128x256xf32) <- (1x32x128x256xf32, 128x32x1x1xf32)
        conv2d_44 = paddle._C_ops.conv2d(prelu_56, parameter_277, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x128x128x256xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x128x256xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__264, batch_norm__265, batch_norm__266, batch_norm__267, batch_norm__268, batch_norm__269 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_44, parameter_278, parameter_279, parameter_280, parameter_281, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x128x128x256xf32) <- (1x128x128x256xf32, 1xf32)
        prelu_57 = paddle._C_ops.prelu(batch_norm__264, parameter_282, 'NCHW', 'all')

        # pd_op.add_: (1x128x128x256xf32) <- (1x128x128x256xf32, 1x128x128x256xf32)
        add__13 = paddle._C_ops.add_(prelu_54, prelu_57)

        # pd_op.prelu: (1x128x128x256xf32) <- (1x128x128x256xf32, 1xf32)
        prelu_58 = paddle._C_ops.prelu(add__13, parameter_283, 'NCHW', 'all')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x128x128x256xf32, 32x128x1x1xf32)
        conv2d_45 = paddle._C_ops.conv2d(prelu_58, parameter_284, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__270, batch_norm__271, batch_norm__272, batch_norm__273, batch_norm__274, batch_norm__275 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_45, parameter_285, parameter_286, parameter_287, parameter_288, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x32x128x256xf32) <- (1x32x128x256xf32, 1xf32)
        prelu_59 = paddle._C_ops.prelu(batch_norm__270, parameter_289, 'NCHW', 'all')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x32x128x256xf32, 32x32x3x3xf32)
        conv2d_46 = paddle._C_ops.conv2d(prelu_59, parameter_290, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__276, batch_norm__277, batch_norm__278, batch_norm__279, batch_norm__280, batch_norm__281 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_46, parameter_291, parameter_292, parameter_293, parameter_294, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x32x128x256xf32) <- (1x32x128x256xf32, 1xf32)
        prelu_60 = paddle._C_ops.prelu(batch_norm__276, parameter_295, 'NCHW', 'all')

        # pd_op.conv2d: (1x128x128x256xf32) <- (1x32x128x256xf32, 128x32x1x1xf32)
        conv2d_47 = paddle._C_ops.conv2d(prelu_60, parameter_296, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x128x128x256xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x128x256xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__282, batch_norm__283, batch_norm__284, batch_norm__285, batch_norm__286, batch_norm__287 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_47, parameter_297, parameter_298, parameter_299, parameter_300, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x128x128x256xf32) <- (1x128x128x256xf32, 1xf32)
        prelu_61 = paddle._C_ops.prelu(batch_norm__282, parameter_301, 'NCHW', 'all')

        # pd_op.add_: (1x128x128x256xf32) <- (1x128x128x256xf32, 1x128x128x256xf32)
        add__14 = paddle._C_ops.add_(prelu_58, prelu_61)

        # pd_op.prelu: (1x128x128x256xf32) <- (1x128x128x256xf32, 1xf32)
        prelu_62 = paddle._C_ops.prelu(add__14, parameter_302, 'NCHW', 'all')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x128x128x256xf32, 32x128x1x1xf32)
        conv2d_48 = paddle._C_ops.conv2d(prelu_62, parameter_303, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__288, batch_norm__289, batch_norm__290, batch_norm__291, batch_norm__292, batch_norm__293 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_48, parameter_304, parameter_305, parameter_306, parameter_307, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x32x128x256xf32) <- (1x32x128x256xf32, 1xf32)
        prelu_63 = paddle._C_ops.prelu(batch_norm__288, parameter_308, 'NCHW', 'all')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x32x128x256xf32, 32x32x3x3xf32)
        conv2d_49 = paddle._C_ops.conv2d(prelu_63, parameter_309, [1, 1], [2, 2], 'EXPLICIT', [2, 2], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__294, batch_norm__295, batch_norm__296, batch_norm__297, batch_norm__298, batch_norm__299 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_49, parameter_310, parameter_311, parameter_312, parameter_313, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x32x128x256xf32) <- (1x32x128x256xf32, 1xf32)
        prelu_64 = paddle._C_ops.prelu(batch_norm__294, parameter_314, 'NCHW', 'all')

        # pd_op.conv2d: (1x128x128x256xf32) <- (1x32x128x256xf32, 128x32x1x1xf32)
        conv2d_50 = paddle._C_ops.conv2d(prelu_64, parameter_315, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x128x128x256xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x128x256xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__300, batch_norm__301, batch_norm__302, batch_norm__303, batch_norm__304, batch_norm__305 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_50, parameter_316, parameter_317, parameter_318, parameter_319, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x128x128x256xf32) <- (1x128x128x256xf32, 1xf32)
        prelu_65 = paddle._C_ops.prelu(batch_norm__300, parameter_320, 'NCHW', 'all')

        # pd_op.add_: (1x128x128x256xf32) <- (1x128x128x256xf32, 1x128x128x256xf32)
        add__15 = paddle._C_ops.add_(prelu_62, prelu_65)

        # pd_op.prelu: (1x128x128x256xf32) <- (1x128x128x256xf32, 1xf32)
        prelu_66 = paddle._C_ops.prelu(add__15, parameter_321, 'NCHW', 'all')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x128x128x256xf32, 32x128x1x1xf32)
        conv2d_51 = paddle._C_ops.conv2d(prelu_66, parameter_322, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__306, batch_norm__307, batch_norm__308, batch_norm__309, batch_norm__310, batch_norm__311 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_51, parameter_323, parameter_324, parameter_325, parameter_326, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x32x128x256xf32) <- (1x32x128x256xf32, 1xf32)
        prelu_67 = paddle._C_ops.prelu(batch_norm__306, parameter_327, 'NCHW', 'all')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x32x128x256xf32, 32x32x5x1xf32)
        conv2d_52 = paddle._C_ops.conv2d(prelu_67, parameter_328, [1, 1], [2, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__312, batch_norm__313, batch_norm__314, batch_norm__315, batch_norm__316, batch_norm__317 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_52, parameter_329, parameter_330, parameter_331, parameter_332, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x32x128x256xf32) <- (1x32x128x256xf32, 1xf32)
        prelu_68 = paddle._C_ops.prelu(batch_norm__312, parameter_333, 'NCHW', 'all')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x32x128x256xf32, 32x32x1x5xf32)
        conv2d_53 = paddle._C_ops.conv2d(prelu_68, parameter_334, [1, 1], [0, 2], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__318, batch_norm__319, batch_norm__320, batch_norm__321, batch_norm__322, batch_norm__323 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_53, parameter_335, parameter_336, parameter_337, parameter_338, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x32x128x256xf32) <- (1x32x128x256xf32, 1xf32)
        prelu_69 = paddle._C_ops.prelu(batch_norm__318, parameter_339, 'NCHW', 'all')

        # pd_op.conv2d: (1x128x128x256xf32) <- (1x32x128x256xf32, 128x32x1x1xf32)
        conv2d_54 = paddle._C_ops.conv2d(prelu_69, parameter_340, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x128x128x256xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x128x256xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__324, batch_norm__325, batch_norm__326, batch_norm__327, batch_norm__328, batch_norm__329 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_54, parameter_341, parameter_342, parameter_343, parameter_344, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x128x128x256xf32) <- (1x128x128x256xf32, 1xf32)
        prelu_70 = paddle._C_ops.prelu(batch_norm__324, parameter_345, 'NCHW', 'all')

        # pd_op.add_: (1x128x128x256xf32) <- (1x128x128x256xf32, 1x128x128x256xf32)
        add__16 = paddle._C_ops.add_(prelu_66, prelu_70)

        # pd_op.prelu: (1x128x128x256xf32) <- (1x128x128x256xf32, 1xf32)
        prelu_71 = paddle._C_ops.prelu(add__16, parameter_346, 'NCHW', 'all')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x128x128x256xf32, 32x128x1x1xf32)
        conv2d_55 = paddle._C_ops.conv2d(prelu_71, parameter_347, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__330, batch_norm__331, batch_norm__332, batch_norm__333, batch_norm__334, batch_norm__335 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_55, parameter_348, parameter_349, parameter_350, parameter_351, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x32x128x256xf32) <- (1x32x128x256xf32, 1xf32)
        prelu_72 = paddle._C_ops.prelu(batch_norm__330, parameter_352, 'NCHW', 'all')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x32x128x256xf32, 32x32x3x3xf32)
        conv2d_56 = paddle._C_ops.conv2d(prelu_72, parameter_353, [1, 1], [4, 4], 'EXPLICIT', [4, 4], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__336, batch_norm__337, batch_norm__338, batch_norm__339, batch_norm__340, batch_norm__341 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_56, parameter_354, parameter_355, parameter_356, parameter_357, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x32x128x256xf32) <- (1x32x128x256xf32, 1xf32)
        prelu_73 = paddle._C_ops.prelu(batch_norm__336, parameter_358, 'NCHW', 'all')

        # pd_op.conv2d: (1x128x128x256xf32) <- (1x32x128x256xf32, 128x32x1x1xf32)
        conv2d_57 = paddle._C_ops.conv2d(prelu_73, parameter_359, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x128x128x256xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x128x256xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__342, batch_norm__343, batch_norm__344, batch_norm__345, batch_norm__346, batch_norm__347 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_57, parameter_360, parameter_361, parameter_362, parameter_363, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x128x128x256xf32) <- (1x128x128x256xf32, 1xf32)
        prelu_74 = paddle._C_ops.prelu(batch_norm__342, parameter_364, 'NCHW', 'all')

        # pd_op.add_: (1x128x128x256xf32) <- (1x128x128x256xf32, 1x128x128x256xf32)
        add__17 = paddle._C_ops.add_(prelu_71, prelu_74)

        # pd_op.prelu: (1x128x128x256xf32) <- (1x128x128x256xf32, 1xf32)
        prelu_75 = paddle._C_ops.prelu(add__17, parameter_365, 'NCHW', 'all')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x128x128x256xf32, 32x128x1x1xf32)
        conv2d_58 = paddle._C_ops.conv2d(prelu_75, parameter_366, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__348, batch_norm__349, batch_norm__350, batch_norm__351, batch_norm__352, batch_norm__353 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_58, parameter_367, parameter_368, parameter_369, parameter_370, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x32x128x256xf32) <- (1x32x128x256xf32, 1xf32)
        prelu_76 = paddle._C_ops.prelu(batch_norm__348, parameter_371, 'NCHW', 'all')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x32x128x256xf32, 32x32x3x3xf32)
        conv2d_59 = paddle._C_ops.conv2d(prelu_76, parameter_372, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__354, batch_norm__355, batch_norm__356, batch_norm__357, batch_norm__358, batch_norm__359 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_59, parameter_373, parameter_374, parameter_375, parameter_376, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x32x128x256xf32) <- (1x32x128x256xf32, 1xf32)
        prelu_77 = paddle._C_ops.prelu(batch_norm__354, parameter_377, 'NCHW', 'all')

        # pd_op.conv2d: (1x128x128x256xf32) <- (1x32x128x256xf32, 128x32x1x1xf32)
        conv2d_60 = paddle._C_ops.conv2d(prelu_77, parameter_378, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x128x128x256xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x128x256xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__360, batch_norm__361, batch_norm__362, batch_norm__363, batch_norm__364, batch_norm__365 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_60, parameter_379, parameter_380, parameter_381, parameter_382, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x128x128x256xf32) <- (1x128x128x256xf32, 1xf32)
        prelu_78 = paddle._C_ops.prelu(batch_norm__360, parameter_383, 'NCHW', 'all')

        # pd_op.add_: (1x128x128x256xf32) <- (1x128x128x256xf32, 1x128x128x256xf32)
        add__18 = paddle._C_ops.add_(prelu_75, prelu_78)

        # pd_op.prelu: (1x128x128x256xf32) <- (1x128x128x256xf32, 1xf32)
        prelu_79 = paddle._C_ops.prelu(add__18, parameter_384, 'NCHW', 'all')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x128x128x256xf32, 32x128x1x1xf32)
        conv2d_61 = paddle._C_ops.conv2d(prelu_79, parameter_385, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__366, batch_norm__367, batch_norm__368, batch_norm__369, batch_norm__370, batch_norm__371 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_61, parameter_386, parameter_387, parameter_388, parameter_389, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x32x128x256xf32) <- (1x32x128x256xf32, 1xf32)
        prelu_80 = paddle._C_ops.prelu(batch_norm__366, parameter_390, 'NCHW', 'all')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x32x128x256xf32, 32x32x3x3xf32)
        conv2d_62 = paddle._C_ops.conv2d(prelu_80, parameter_391, [1, 1], [8, 8], 'EXPLICIT', [8, 8], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__372, batch_norm__373, batch_norm__374, batch_norm__375, batch_norm__376, batch_norm__377 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_62, parameter_392, parameter_393, parameter_394, parameter_395, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x32x128x256xf32) <- (1x32x128x256xf32, 1xf32)
        prelu_81 = paddle._C_ops.prelu(batch_norm__372, parameter_396, 'NCHW', 'all')

        # pd_op.conv2d: (1x128x128x256xf32) <- (1x32x128x256xf32, 128x32x1x1xf32)
        conv2d_63 = paddle._C_ops.conv2d(prelu_81, parameter_397, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x128x128x256xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x128x256xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__378, batch_norm__379, batch_norm__380, batch_norm__381, batch_norm__382, batch_norm__383 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_63, parameter_398, parameter_399, parameter_400, parameter_401, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x128x128x256xf32) <- (1x128x128x256xf32, 1xf32)
        prelu_82 = paddle._C_ops.prelu(batch_norm__378, parameter_402, 'NCHW', 'all')

        # pd_op.add_: (1x128x128x256xf32) <- (1x128x128x256xf32, 1x128x128x256xf32)
        add__19 = paddle._C_ops.add_(prelu_79, prelu_82)

        # pd_op.prelu: (1x128x128x256xf32) <- (1x128x128x256xf32, 1xf32)
        prelu_83 = paddle._C_ops.prelu(add__19, parameter_403, 'NCHW', 'all')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x128x128x256xf32, 32x128x1x1xf32)
        conv2d_64 = paddle._C_ops.conv2d(prelu_83, parameter_404, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__384, batch_norm__385, batch_norm__386, batch_norm__387, batch_norm__388, batch_norm__389 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_64, parameter_405, parameter_406, parameter_407, parameter_408, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x32x128x256xf32) <- (1x32x128x256xf32, 1xf32)
        prelu_84 = paddle._C_ops.prelu(batch_norm__384, parameter_409, 'NCHW', 'all')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x32x128x256xf32, 32x32x5x1xf32)
        conv2d_65 = paddle._C_ops.conv2d(prelu_84, parameter_410, [1, 1], [2, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__390, batch_norm__391, batch_norm__392, batch_norm__393, batch_norm__394, batch_norm__395 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_65, parameter_411, parameter_412, parameter_413, parameter_414, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x32x128x256xf32) <- (1x32x128x256xf32, 1xf32)
        prelu_85 = paddle._C_ops.prelu(batch_norm__390, parameter_415, 'NCHW', 'all')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x32x128x256xf32, 32x32x1x5xf32)
        conv2d_66 = paddle._C_ops.conv2d(prelu_85, parameter_416, [1, 1], [0, 2], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__396, batch_norm__397, batch_norm__398, batch_norm__399, batch_norm__400, batch_norm__401 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_66, parameter_417, parameter_418, parameter_419, parameter_420, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x32x128x256xf32) <- (1x32x128x256xf32, 1xf32)
        prelu_86 = paddle._C_ops.prelu(batch_norm__396, parameter_421, 'NCHW', 'all')

        # pd_op.conv2d: (1x128x128x256xf32) <- (1x32x128x256xf32, 128x32x1x1xf32)
        conv2d_67 = paddle._C_ops.conv2d(prelu_86, parameter_422, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x128x128x256xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x128x256xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__402, batch_norm__403, batch_norm__404, batch_norm__405, batch_norm__406, batch_norm__407 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_67, parameter_423, parameter_424, parameter_425, parameter_426, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x128x128x256xf32) <- (1x128x128x256xf32, 1xf32)
        prelu_87 = paddle._C_ops.prelu(batch_norm__402, parameter_427, 'NCHW', 'all')

        # pd_op.add_: (1x128x128x256xf32) <- (1x128x128x256xf32, 1x128x128x256xf32)
        add__20 = paddle._C_ops.add_(prelu_83, prelu_87)

        # pd_op.prelu: (1x128x128x256xf32) <- (1x128x128x256xf32, 1xf32)
        prelu_88 = paddle._C_ops.prelu(add__20, parameter_428, 'NCHW', 'all')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x128x128x256xf32, 32x128x1x1xf32)
        conv2d_68 = paddle._C_ops.conv2d(prelu_88, parameter_429, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__408, batch_norm__409, batch_norm__410, batch_norm__411, batch_norm__412, batch_norm__413 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_68, parameter_430, parameter_431, parameter_432, parameter_433, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x32x128x256xf32) <- (1x32x128x256xf32, 1xf32)
        prelu_89 = paddle._C_ops.prelu(batch_norm__408, parameter_434, 'NCHW', 'all')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x32x128x256xf32, 32x32x3x3xf32)
        conv2d_69 = paddle._C_ops.conv2d(prelu_89, parameter_435, [1, 1], [16, 16], 'EXPLICIT', [16, 16], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__414, batch_norm__415, batch_norm__416, batch_norm__417, batch_norm__418, batch_norm__419 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_69, parameter_436, parameter_437, parameter_438, parameter_439, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x32x128x256xf32) <- (1x32x128x256xf32, 1xf32)
        prelu_90 = paddle._C_ops.prelu(batch_norm__414, parameter_440, 'NCHW', 'all')

        # pd_op.conv2d: (1x128x128x256xf32) <- (1x32x128x256xf32, 128x32x1x1xf32)
        conv2d_70 = paddle._C_ops.conv2d(prelu_90, parameter_441, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x128x128x256xf32, 128xf32, 128xf32, xf32, xf32, None) <- (1x128x128x256xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__420, batch_norm__421, batch_norm__422, batch_norm__423, batch_norm__424, batch_norm__425 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_70, parameter_442, parameter_443, parameter_444, parameter_445, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.prelu: (1x128x128x256xf32) <- (1x128x128x256xf32, 1xf32)
        prelu_91 = paddle._C_ops.prelu(batch_norm__420, parameter_446, 'NCHW', 'all')

        # pd_op.add_: (1x128x128x256xf32) <- (1x128x128x256xf32, 1x128x128x256xf32)
        add__21 = paddle._C_ops.add_(prelu_88, prelu_91)

        # pd_op.prelu: (1x128x128x256xf32) <- (1x128x128x256xf32, 1xf32)
        prelu_92 = paddle._C_ops.prelu(add__21, parameter_447, 'NCHW', 'all')

        # pd_op.conv2d: (1x64x128x256xf32) <- (1x128x128x256xf32, 64x128x1x1xf32)
        conv2d_71 = paddle._C_ops.conv2d(prelu_92, parameter_448, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x64x128x256xf32, 64xf32, 64xf32, xf32, xf32, None) <- (1x64x128x256xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__426, batch_norm__427, batch_norm__428, batch_norm__429, batch_norm__430, batch_norm__431 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_71, parameter_449, parameter_450, parameter_451, parameter_452, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [256, 512]

        # pd_op.unpool: (1x64x256x512xf32) <- (1x64x128x256xf32, 1x64x128x256xi32, 2xi64)
        unpool_0 = paddle._C_ops.unpool(batch_norm__426, max_pool2d_with_index_3, [2, 2], [2, 2], [0, 0], full_int_array_1, 'NCHW')

        # pd_op.conv2d: (1x32x128x256xf32) <- (1x128x128x256xf32, 32x128x1x1xf32)
        conv2d_72 = paddle._C_ops.conv2d(prelu_92, parameter_453, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x32x128x256xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x128x256xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__432, batch_norm__433, batch_norm__434, batch_norm__435, batch_norm__436, batch_norm__437 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_72, parameter_454, parameter_455, parameter_456, parameter_457, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x32x128x256xf32) <- (1x32x128x256xf32)
        relu__0 = paddle._C_ops.relu_(batch_norm__432)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [256, 512]

        # pd_op.conv2d_transpose: (1x32x256x512xf32) <- (1x32x128x256xf32, 32x32x2x2xf32, 2xi64)
        conv2d_transpose_0 = paddle._C_ops.conv2d_transpose(relu__0, parameter_458, [2, 2], [0, 0], [], full_int_array_2, 'EXPLICIT', 1, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (1x32x256x512xf32, 32xf32, 32xf32, xf32, xf32, None) <- (1x32x256x512xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__438, batch_norm__439, batch_norm__440, batch_norm__441, batch_norm__442, batch_norm__443 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_transpose_0, parameter_459, parameter_460, parameter_461, parameter_462, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x32x256x512xf32) <- (1x32x256x512xf32)
        relu__1 = paddle._C_ops.relu_(batch_norm__438)

        # pd_op.conv2d: (1x64x256x512xf32) <- (1x32x256x512xf32, 64x32x1x1xf32)
        conv2d_73 = paddle._C_ops.conv2d(relu__1, parameter_463, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x64x256x512xf32, 64xf32, 64xf32, xf32, xf32, None) <- (1x64x256x512xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__444, batch_norm__445, batch_norm__446, batch_norm__447, batch_norm__448, batch_norm__449 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_73, parameter_464, parameter_465, parameter_466, parameter_467, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x64x256x512xf32) <- (1x64x256x512xf32, 1x64x256x512xf32)
        add__22 = paddle._C_ops.add_(unpool_0, batch_norm__444)

        # pd_op.relu_: (1x64x256x512xf32) <- (1x64x256x512xf32)
        relu__2 = paddle._C_ops.relu_(add__22)

        # pd_op.conv2d: (1x16x256x512xf32) <- (1x64x256x512xf32, 16x64x1x1xf32)
        conv2d_74 = paddle._C_ops.conv2d(relu__2, parameter_468, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x16x256x512xf32, 16xf32, 16xf32, xf32, xf32, None) <- (1x16x256x512xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__450, batch_norm__451, batch_norm__452, batch_norm__453, batch_norm__454, batch_norm__455 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_74, parameter_469, parameter_470, parameter_471, parameter_472, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x16x256x512xf32) <- (1x16x256x512xf32)
        relu__3 = paddle._C_ops.relu_(batch_norm__450)

        # pd_op.conv2d: (1x16x256x512xf32) <- (1x16x256x512xf32, 16x16x3x3xf32)
        conv2d_75 = paddle._C_ops.conv2d(relu__3, parameter_473, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x16x256x512xf32, 16xf32, 16xf32, xf32, xf32, None) <- (1x16x256x512xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__456, batch_norm__457, batch_norm__458, batch_norm__459, batch_norm__460, batch_norm__461 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_75, parameter_474, parameter_475, parameter_476, parameter_477, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x16x256x512xf32) <- (1x16x256x512xf32)
        relu__4 = paddle._C_ops.relu_(batch_norm__456)

        # pd_op.conv2d: (1x64x256x512xf32) <- (1x16x256x512xf32, 64x16x1x1xf32)
        conv2d_76 = paddle._C_ops.conv2d(relu__4, parameter_478, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x64x256x512xf32, 64xf32, 64xf32, xf32, xf32, None) <- (1x64x256x512xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__462, batch_norm__463, batch_norm__464, batch_norm__465, batch_norm__466, batch_norm__467 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_76, parameter_479, parameter_480, parameter_481, parameter_482, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x64x256x512xf32) <- (1x64x256x512xf32)
        relu__5 = paddle._C_ops.relu_(batch_norm__462)

        # pd_op.add_: (1x64x256x512xf32) <- (1x64x256x512xf32, 1x64x256x512xf32)
        add__23 = paddle._C_ops.add_(relu__2, relu__5)

        # pd_op.relu_: (1x64x256x512xf32) <- (1x64x256x512xf32)
        relu__6 = paddle._C_ops.relu_(add__23)

        # pd_op.conv2d: (1x16x256x512xf32) <- (1x64x256x512xf32, 16x64x1x1xf32)
        conv2d_77 = paddle._C_ops.conv2d(relu__6, parameter_483, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x16x256x512xf32, 16xf32, 16xf32, xf32, xf32, None) <- (1x16x256x512xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__468, batch_norm__469, batch_norm__470, batch_norm__471, batch_norm__472, batch_norm__473 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_77, parameter_484, parameter_485, parameter_486, parameter_487, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x16x256x512xf32) <- (1x16x256x512xf32)
        relu__7 = paddle._C_ops.relu_(batch_norm__468)

        # pd_op.conv2d: (1x16x256x512xf32) <- (1x16x256x512xf32, 16x16x3x3xf32)
        conv2d_78 = paddle._C_ops.conv2d(relu__7, parameter_488, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x16x256x512xf32, 16xf32, 16xf32, xf32, xf32, None) <- (1x16x256x512xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__474, batch_norm__475, batch_norm__476, batch_norm__477, batch_norm__478, batch_norm__479 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_78, parameter_489, parameter_490, parameter_491, parameter_492, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x16x256x512xf32) <- (1x16x256x512xf32)
        relu__8 = paddle._C_ops.relu_(batch_norm__474)

        # pd_op.conv2d: (1x64x256x512xf32) <- (1x16x256x512xf32, 64x16x1x1xf32)
        conv2d_79 = paddle._C_ops.conv2d(relu__8, parameter_493, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x64x256x512xf32, 64xf32, 64xf32, xf32, xf32, None) <- (1x64x256x512xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__480, batch_norm__481, batch_norm__482, batch_norm__483, batch_norm__484, batch_norm__485 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_79, parameter_494, parameter_495, parameter_496, parameter_497, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x64x256x512xf32) <- (1x64x256x512xf32)
        relu__9 = paddle._C_ops.relu_(batch_norm__480)

        # pd_op.add_: (1x64x256x512xf32) <- (1x64x256x512xf32, 1x64x256x512xf32)
        add__24 = paddle._C_ops.add_(relu__6, relu__9)

        # pd_op.relu_: (1x64x256x512xf32) <- (1x64x256x512xf32)
        relu__10 = paddle._C_ops.relu_(add__24)

        # pd_op.conv2d: (1x16x256x512xf32) <- (1x64x256x512xf32, 16x64x1x1xf32)
        conv2d_80 = paddle._C_ops.conv2d(relu__10, parameter_498, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x16x256x512xf32, 16xf32, 16xf32, xf32, xf32, None) <- (1x16x256x512xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__486, batch_norm__487, batch_norm__488, batch_norm__489, batch_norm__490, batch_norm__491 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_80, parameter_499, parameter_500, parameter_501, parameter_502, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [512, 1024]

        # pd_op.unpool: (1x16x512x1024xf32) <- (1x16x256x512xf32, 1x16x256x512xi32, 2xi64)
        unpool_1 = paddle._C_ops.unpool(batch_norm__486, max_pool2d_with_index_1, [2, 2], [2, 2], [0, 0], full_int_array_3, 'NCHW')

        # pd_op.conv2d: (1x16x256x512xf32) <- (1x64x256x512xf32, 16x64x1x1xf32)
        conv2d_81 = paddle._C_ops.conv2d(relu__10, parameter_503, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x16x256x512xf32, 16xf32, 16xf32, xf32, xf32, None) <- (1x16x256x512xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__492, batch_norm__493, batch_norm__494, batch_norm__495, batch_norm__496, batch_norm__497 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_81, parameter_504, parameter_505, parameter_506, parameter_507, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x16x256x512xf32) <- (1x16x256x512xf32)
        relu__11 = paddle._C_ops.relu_(batch_norm__492)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [512, 1024]

        # pd_op.conv2d_transpose: (1x16x512x1024xf32) <- (1x16x256x512xf32, 16x16x2x2xf32, 2xi64)
        conv2d_transpose_1 = paddle._C_ops.conv2d_transpose(relu__11, parameter_508, [2, 2], [0, 0], [], full_int_array_4, 'EXPLICIT', 1, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (1x16x512x1024xf32, 16xf32, 16xf32, xf32, xf32, None) <- (1x16x512x1024xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__498, batch_norm__499, batch_norm__500, batch_norm__501, batch_norm__502, batch_norm__503 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_transpose_1, parameter_509, parameter_510, parameter_511, parameter_512, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x16x512x1024xf32) <- (1x16x512x1024xf32)
        relu__12 = paddle._C_ops.relu_(batch_norm__498)

        # pd_op.conv2d: (1x16x512x1024xf32) <- (1x16x512x1024xf32, 16x16x1x1xf32)
        conv2d_82 = paddle._C_ops.conv2d(relu__12, parameter_513, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x16x512x1024xf32, 16xf32, 16xf32, xf32, xf32, None) <- (1x16x512x1024xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__504, batch_norm__505, batch_norm__506, batch_norm__507, batch_norm__508, batch_norm__509 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_82, parameter_514, parameter_515, parameter_516, parameter_517, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (1x16x512x1024xf32) <- (1x16x512x1024xf32, 1x16x512x1024xf32)
        add__25 = paddle._C_ops.add_(unpool_1, batch_norm__504)

        # pd_op.relu_: (1x16x512x1024xf32) <- (1x16x512x1024xf32)
        relu__13 = paddle._C_ops.relu_(add__25)

        # pd_op.conv2d: (1x4x512x1024xf32) <- (1x16x512x1024xf32, 4x16x1x1xf32)
        conv2d_83 = paddle._C_ops.conv2d(relu__13, parameter_518, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x4x512x1024xf32, 4xf32, 4xf32, xf32, xf32, None) <- (1x4x512x1024xf32, 4xf32, 4xf32, 4xf32, 4xf32)
        batch_norm__510, batch_norm__511, batch_norm__512, batch_norm__513, batch_norm__514, batch_norm__515 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_83, parameter_519, parameter_520, parameter_521, parameter_522, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x4x512x1024xf32) <- (1x4x512x1024xf32)
        relu__14 = paddle._C_ops.relu_(batch_norm__510)

        # pd_op.conv2d: (1x4x512x1024xf32) <- (1x4x512x1024xf32, 4x4x3x3xf32)
        conv2d_84 = paddle._C_ops.conv2d(relu__14, parameter_523, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x4x512x1024xf32, 4xf32, 4xf32, xf32, xf32, None) <- (1x4x512x1024xf32, 4xf32, 4xf32, 4xf32, 4xf32)
        batch_norm__516, batch_norm__517, batch_norm__518, batch_norm__519, batch_norm__520, batch_norm__521 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_84, parameter_524, parameter_525, parameter_526, parameter_527, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x4x512x1024xf32) <- (1x4x512x1024xf32)
        relu__15 = paddle._C_ops.relu_(batch_norm__516)

        # pd_op.conv2d: (1x16x512x1024xf32) <- (1x4x512x1024xf32, 16x4x1x1xf32)
        conv2d_85 = paddle._C_ops.conv2d(relu__15, parameter_528, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (1x16x512x1024xf32, 16xf32, 16xf32, xf32, xf32, None) <- (1x16x512x1024xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__522, batch_norm__523, batch_norm__524, batch_norm__525, batch_norm__526, batch_norm__527 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_85, parameter_529, parameter_530, parameter_531, parameter_532, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (1x16x512x1024xf32) <- (1x16x512x1024xf32)
        relu__16 = paddle._C_ops.relu_(batch_norm__522)

        # pd_op.add_: (1x16x512x1024xf32) <- (1x16x512x1024xf32, 1x16x512x1024xf32)
        add__26 = paddle._C_ops.add_(relu__13, relu__16)

        # pd_op.relu_: (1x16x512x1024xf32) <- (1x16x512x1024xf32)
        relu__17 = paddle._C_ops.relu_(add__26)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [1024, 2048]

        # pd_op.conv2d_transpose: (1x19x1024x2048xf32) <- (1x16x512x1024xf32, 16x19x3x3xf32, 2xi64)
        conv2d_transpose_2 = paddle._C_ops.conv2d_transpose(relu__17, parameter_533, [2, 2], [1, 1], [], full_int_array_5, 'EXPLICIT', 1, [1, 1], 'NCHW')

        # pd_op.full: (1xi64) <- ()
        full_5 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (1x1024x2048xi32) <- (1x19x1024x2048xf32, 1xi64)
        argmax_0 = paddle._C_ops.argmax(conv2d_transpose_2, full_5, False, False, paddle.int32)

        # pd_op.full: (1xf32) <- ()
        full_6 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (1x1024x2048xi32) <- (1x1024x2048xi32, 1xf32)
        scale__0 = paddle._C_ops.scale_(argmax_0, full_6, float('0'), True)
        return scale__0



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

    def forward(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_6, parameter_10, parameter_7, parameter_9, parameter_8, parameter_11, parameter_12, parameter_16, parameter_13, parameter_15, parameter_14, parameter_17, parameter_18, parameter_22, parameter_19, parameter_21, parameter_20, parameter_23, parameter_24, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_31, parameter_35, parameter_32, parameter_34, parameter_33, parameter_36, parameter_37, parameter_41, parameter_38, parameter_40, parameter_39, parameter_42, parameter_43, parameter_44, parameter_48, parameter_45, parameter_47, parameter_46, parameter_49, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_56, parameter_60, parameter_57, parameter_59, parameter_58, parameter_61, parameter_62, parameter_63, parameter_67, parameter_64, parameter_66, parameter_65, parameter_68, parameter_69, parameter_73, parameter_70, parameter_72, parameter_71, parameter_74, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_81, parameter_82, parameter_86, parameter_83, parameter_85, parameter_84, parameter_87, parameter_88, parameter_92, parameter_89, parameter_91, parameter_90, parameter_93, parameter_94, parameter_98, parameter_95, parameter_97, parameter_96, parameter_99, parameter_100, parameter_101, parameter_105, parameter_102, parameter_104, parameter_103, parameter_106, parameter_107, parameter_111, parameter_108, parameter_110, parameter_109, parameter_112, parameter_113, parameter_117, parameter_114, parameter_116, parameter_115, parameter_118, parameter_119, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_126, parameter_130, parameter_127, parameter_129, parameter_128, parameter_131, parameter_132, parameter_136, parameter_133, parameter_135, parameter_134, parameter_137, parameter_138, parameter_139, parameter_143, parameter_140, parameter_142, parameter_141, parameter_144, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_151, parameter_155, parameter_152, parameter_154, parameter_153, parameter_156, parameter_157, parameter_158, parameter_162, parameter_159, parameter_161, parameter_160, parameter_163, parameter_164, parameter_168, parameter_165, parameter_167, parameter_166, parameter_169, parameter_170, parameter_174, parameter_171, parameter_173, parameter_172, parameter_175, parameter_176, parameter_180, parameter_177, parameter_179, parameter_178, parameter_181, parameter_182, parameter_183, parameter_187, parameter_184, parameter_186, parameter_185, parameter_188, parameter_189, parameter_193, parameter_190, parameter_192, parameter_191, parameter_194, parameter_195, parameter_199, parameter_196, parameter_198, parameter_197, parameter_200, parameter_201, parameter_202, parameter_206, parameter_203, parameter_205, parameter_204, parameter_207, parameter_208, parameter_212, parameter_209, parameter_211, parameter_210, parameter_213, parameter_214, parameter_218, parameter_215, parameter_217, parameter_216, parameter_219, parameter_220, parameter_221, parameter_225, parameter_222, parameter_224, parameter_223, parameter_226, parameter_227, parameter_231, parameter_228, parameter_230, parameter_229, parameter_232, parameter_233, parameter_237, parameter_234, parameter_236, parameter_235, parameter_238, parameter_239, parameter_240, parameter_244, parameter_241, parameter_243, parameter_242, parameter_245, parameter_246, parameter_250, parameter_247, parameter_249, parameter_248, parameter_251, parameter_252, parameter_256, parameter_253, parameter_255, parameter_254, parameter_257, parameter_258, parameter_262, parameter_259, parameter_261, parameter_260, parameter_263, parameter_264, parameter_265, parameter_269, parameter_266, parameter_268, parameter_267, parameter_270, parameter_271, parameter_275, parameter_272, parameter_274, parameter_273, parameter_276, parameter_277, parameter_281, parameter_278, parameter_280, parameter_279, parameter_282, parameter_283, parameter_284, parameter_288, parameter_285, parameter_287, parameter_286, parameter_289, parameter_290, parameter_294, parameter_291, parameter_293, parameter_292, parameter_295, parameter_296, parameter_300, parameter_297, parameter_299, parameter_298, parameter_301, parameter_302, parameter_303, parameter_307, parameter_304, parameter_306, parameter_305, parameter_308, parameter_309, parameter_313, parameter_310, parameter_312, parameter_311, parameter_314, parameter_315, parameter_319, parameter_316, parameter_318, parameter_317, parameter_320, parameter_321, parameter_322, parameter_326, parameter_323, parameter_325, parameter_324, parameter_327, parameter_328, parameter_332, parameter_329, parameter_331, parameter_330, parameter_333, parameter_334, parameter_338, parameter_335, parameter_337, parameter_336, parameter_339, parameter_340, parameter_344, parameter_341, parameter_343, parameter_342, parameter_345, parameter_346, parameter_347, parameter_351, parameter_348, parameter_350, parameter_349, parameter_352, parameter_353, parameter_357, parameter_354, parameter_356, parameter_355, parameter_358, parameter_359, parameter_363, parameter_360, parameter_362, parameter_361, parameter_364, parameter_365, parameter_366, parameter_370, parameter_367, parameter_369, parameter_368, parameter_371, parameter_372, parameter_376, parameter_373, parameter_375, parameter_374, parameter_377, parameter_378, parameter_382, parameter_379, parameter_381, parameter_380, parameter_383, parameter_384, parameter_385, parameter_389, parameter_386, parameter_388, parameter_387, parameter_390, parameter_391, parameter_395, parameter_392, parameter_394, parameter_393, parameter_396, parameter_397, parameter_401, parameter_398, parameter_400, parameter_399, parameter_402, parameter_403, parameter_404, parameter_408, parameter_405, parameter_407, parameter_406, parameter_409, parameter_410, parameter_414, parameter_411, parameter_413, parameter_412, parameter_415, parameter_416, parameter_420, parameter_417, parameter_419, parameter_418, parameter_421, parameter_422, parameter_426, parameter_423, parameter_425, parameter_424, parameter_427, parameter_428, parameter_429, parameter_433, parameter_430, parameter_432, parameter_431, parameter_434, parameter_435, parameter_439, parameter_436, parameter_438, parameter_437, parameter_440, parameter_441, parameter_445, parameter_442, parameter_444, parameter_443, parameter_446, parameter_447, parameter_448, parameter_452, parameter_449, parameter_451, parameter_450, parameter_453, parameter_457, parameter_454, parameter_456, parameter_455, parameter_458, parameter_462, parameter_459, parameter_461, parameter_460, parameter_463, parameter_467, parameter_464, parameter_466, parameter_465, parameter_468, parameter_472, parameter_469, parameter_471, parameter_470, parameter_473, parameter_477, parameter_474, parameter_476, parameter_475, parameter_478, parameter_482, parameter_479, parameter_481, parameter_480, parameter_483, parameter_487, parameter_484, parameter_486, parameter_485, parameter_488, parameter_492, parameter_489, parameter_491, parameter_490, parameter_493, parameter_497, parameter_494, parameter_496, parameter_495, parameter_498, parameter_502, parameter_499, parameter_501, parameter_500, parameter_503, parameter_507, parameter_504, parameter_506, parameter_505, parameter_508, parameter_512, parameter_509, parameter_511, parameter_510, parameter_513, parameter_517, parameter_514, parameter_516, parameter_515, parameter_518, parameter_522, parameter_519, parameter_521, parameter_520, parameter_523, parameter_527, parameter_524, parameter_526, parameter_525, parameter_528, parameter_532, parameter_529, parameter_531, parameter_530, parameter_533, feed_0):
        return self.builtin_module_882_0_0(parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_6, parameter_10, parameter_7, parameter_9, parameter_8, parameter_11, parameter_12, parameter_16, parameter_13, parameter_15, parameter_14, parameter_17, parameter_18, parameter_22, parameter_19, parameter_21, parameter_20, parameter_23, parameter_24, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_31, parameter_35, parameter_32, parameter_34, parameter_33, parameter_36, parameter_37, parameter_41, parameter_38, parameter_40, parameter_39, parameter_42, parameter_43, parameter_44, parameter_48, parameter_45, parameter_47, parameter_46, parameter_49, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_56, parameter_60, parameter_57, parameter_59, parameter_58, parameter_61, parameter_62, parameter_63, parameter_67, parameter_64, parameter_66, parameter_65, parameter_68, parameter_69, parameter_73, parameter_70, parameter_72, parameter_71, parameter_74, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_81, parameter_82, parameter_86, parameter_83, parameter_85, parameter_84, parameter_87, parameter_88, parameter_92, parameter_89, parameter_91, parameter_90, parameter_93, parameter_94, parameter_98, parameter_95, parameter_97, parameter_96, parameter_99, parameter_100, parameter_101, parameter_105, parameter_102, parameter_104, parameter_103, parameter_106, parameter_107, parameter_111, parameter_108, parameter_110, parameter_109, parameter_112, parameter_113, parameter_117, parameter_114, parameter_116, parameter_115, parameter_118, parameter_119, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_126, parameter_130, parameter_127, parameter_129, parameter_128, parameter_131, parameter_132, parameter_136, parameter_133, parameter_135, parameter_134, parameter_137, parameter_138, parameter_139, parameter_143, parameter_140, parameter_142, parameter_141, parameter_144, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_151, parameter_155, parameter_152, parameter_154, parameter_153, parameter_156, parameter_157, parameter_158, parameter_162, parameter_159, parameter_161, parameter_160, parameter_163, parameter_164, parameter_168, parameter_165, parameter_167, parameter_166, parameter_169, parameter_170, parameter_174, parameter_171, parameter_173, parameter_172, parameter_175, parameter_176, parameter_180, parameter_177, parameter_179, parameter_178, parameter_181, parameter_182, parameter_183, parameter_187, parameter_184, parameter_186, parameter_185, parameter_188, parameter_189, parameter_193, parameter_190, parameter_192, parameter_191, parameter_194, parameter_195, parameter_199, parameter_196, parameter_198, parameter_197, parameter_200, parameter_201, parameter_202, parameter_206, parameter_203, parameter_205, parameter_204, parameter_207, parameter_208, parameter_212, parameter_209, parameter_211, parameter_210, parameter_213, parameter_214, parameter_218, parameter_215, parameter_217, parameter_216, parameter_219, parameter_220, parameter_221, parameter_225, parameter_222, parameter_224, parameter_223, parameter_226, parameter_227, parameter_231, parameter_228, parameter_230, parameter_229, parameter_232, parameter_233, parameter_237, parameter_234, parameter_236, parameter_235, parameter_238, parameter_239, parameter_240, parameter_244, parameter_241, parameter_243, parameter_242, parameter_245, parameter_246, parameter_250, parameter_247, parameter_249, parameter_248, parameter_251, parameter_252, parameter_256, parameter_253, parameter_255, parameter_254, parameter_257, parameter_258, parameter_262, parameter_259, parameter_261, parameter_260, parameter_263, parameter_264, parameter_265, parameter_269, parameter_266, parameter_268, parameter_267, parameter_270, parameter_271, parameter_275, parameter_272, parameter_274, parameter_273, parameter_276, parameter_277, parameter_281, parameter_278, parameter_280, parameter_279, parameter_282, parameter_283, parameter_284, parameter_288, parameter_285, parameter_287, parameter_286, parameter_289, parameter_290, parameter_294, parameter_291, parameter_293, parameter_292, parameter_295, parameter_296, parameter_300, parameter_297, parameter_299, parameter_298, parameter_301, parameter_302, parameter_303, parameter_307, parameter_304, parameter_306, parameter_305, parameter_308, parameter_309, parameter_313, parameter_310, parameter_312, parameter_311, parameter_314, parameter_315, parameter_319, parameter_316, parameter_318, parameter_317, parameter_320, parameter_321, parameter_322, parameter_326, parameter_323, parameter_325, parameter_324, parameter_327, parameter_328, parameter_332, parameter_329, parameter_331, parameter_330, parameter_333, parameter_334, parameter_338, parameter_335, parameter_337, parameter_336, parameter_339, parameter_340, parameter_344, parameter_341, parameter_343, parameter_342, parameter_345, parameter_346, parameter_347, parameter_351, parameter_348, parameter_350, parameter_349, parameter_352, parameter_353, parameter_357, parameter_354, parameter_356, parameter_355, parameter_358, parameter_359, parameter_363, parameter_360, parameter_362, parameter_361, parameter_364, parameter_365, parameter_366, parameter_370, parameter_367, parameter_369, parameter_368, parameter_371, parameter_372, parameter_376, parameter_373, parameter_375, parameter_374, parameter_377, parameter_378, parameter_382, parameter_379, parameter_381, parameter_380, parameter_383, parameter_384, parameter_385, parameter_389, parameter_386, parameter_388, parameter_387, parameter_390, parameter_391, parameter_395, parameter_392, parameter_394, parameter_393, parameter_396, parameter_397, parameter_401, parameter_398, parameter_400, parameter_399, parameter_402, parameter_403, parameter_404, parameter_408, parameter_405, parameter_407, parameter_406, parameter_409, parameter_410, parameter_414, parameter_411, parameter_413, parameter_412, parameter_415, parameter_416, parameter_420, parameter_417, parameter_419, parameter_418, parameter_421, parameter_422, parameter_426, parameter_423, parameter_425, parameter_424, parameter_427, parameter_428, parameter_429, parameter_433, parameter_430, parameter_432, parameter_431, parameter_434, parameter_435, parameter_439, parameter_436, parameter_438, parameter_437, parameter_440, parameter_441, parameter_445, parameter_442, parameter_444, parameter_443, parameter_446, parameter_447, parameter_448, parameter_452, parameter_449, parameter_451, parameter_450, parameter_453, parameter_457, parameter_454, parameter_456, parameter_455, parameter_458, parameter_462, parameter_459, parameter_461, parameter_460, parameter_463, parameter_467, parameter_464, parameter_466, parameter_465, parameter_468, parameter_472, parameter_469, parameter_471, parameter_470, parameter_473, parameter_477, parameter_474, parameter_476, parameter_475, parameter_478, parameter_482, parameter_479, parameter_481, parameter_480, parameter_483, parameter_487, parameter_484, parameter_486, parameter_485, parameter_488, parameter_492, parameter_489, parameter_491, parameter_490, parameter_493, parameter_497, parameter_494, parameter_496, parameter_495, parameter_498, parameter_502, parameter_499, parameter_501, parameter_500, parameter_503, parameter_507, parameter_504, parameter_506, parameter_505, parameter_508, parameter_512, parameter_509, parameter_511, parameter_510, parameter_513, parameter_517, parameter_514, parameter_516, parameter_515, parameter_518, parameter_522, parameter_519, parameter_521, parameter_520, parameter_523, parameter_527, parameter_524, parameter_526, parameter_525, parameter_528, parameter_532, parameter_529, parameter_531, parameter_530, parameter_533, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_882_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([13, 3, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([4, 16, 2, 2], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([4], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([4], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([4], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([4], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([4, 4, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([4], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([4], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([4], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([4], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([64, 4, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([16, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([16, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([16, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([16, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([16, 64, 2, 2], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([16, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([128, 16, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_139
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_149
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_154
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([32, 32, 5, 1], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([32, 32, 1, 5], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_189
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_209
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_213
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_214
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_218
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_215
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_217
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_219
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_221
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_225
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_224
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_223
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_227
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_229
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_233
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_234
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_238
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_239
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_244
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_241
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_243
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_245
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_246
            paddle.uniform([32, 32, 5, 1], dtype='float32', min=0, max=0.5),
            # parameter_250
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_249
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_248
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_251
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_252
            paddle.uniform([32, 32, 1, 5], dtype='float32', min=0, max=0.5),
            # parameter_256
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_253
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_255
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_254
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_257
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_258
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_262
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_259
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_261
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_263
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_264
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_265
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_269
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_266
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_268
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_267
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_270
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_271
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_275
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_272
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_274
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_273
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_276
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_277
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_281
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_278
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_280
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_279
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_282
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_283
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_284
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_288
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_285
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_287
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_286
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_289
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_290
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_294
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_291
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_293
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_292
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_295
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_296
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_300
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_297
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_299
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_298
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_301
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_302
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_303
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_307
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_304
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_306
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_305
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_308
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_309
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_313
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_310
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_312
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_311
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_314
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_315
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_319
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_316
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_318
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_317
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_320
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_321
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_322
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_326
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_323
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_325
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_324
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_327
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_328
            paddle.uniform([32, 32, 5, 1], dtype='float32', min=0, max=0.5),
            # parameter_332
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_329
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_331
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_330
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_333
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_334
            paddle.uniform([32, 32, 1, 5], dtype='float32', min=0, max=0.5),
            # parameter_338
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_335
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_337
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_336
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_339
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_340
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_344
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_341
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_343
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_342
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_345
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_346
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_347
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_351
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_348
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_350
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_349
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_352
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_353
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_357
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_354
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_356
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_355
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_358
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_359
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_363
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_360
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_362
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_361
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_364
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_365
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_366
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_370
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_367
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_369
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_368
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_371
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_372
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_376
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_373
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_375
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_374
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_377
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_378
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_382
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_379
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_381
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_380
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_383
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_384
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_385
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_389
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_386
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_388
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_387
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_390
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_391
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_395
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_392
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_394
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_393
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_396
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_397
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_401
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_398
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_400
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_399
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_402
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_403
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_404
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_408
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_405
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_407
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_406
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_409
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_410
            paddle.uniform([32, 32, 5, 1], dtype='float32', min=0, max=0.5),
            # parameter_414
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_411
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_413
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_412
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_415
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_416
            paddle.uniform([32, 32, 1, 5], dtype='float32', min=0, max=0.5),
            # parameter_420
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_417
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_419
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_418
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_421
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_422
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_426
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_423
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_425
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_424
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_427
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_428
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_429
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_433
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_430
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_432
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_431
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_434
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_435
            paddle.uniform([32, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_439
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_436
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_438
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_437
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_440
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_441
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_445
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_442
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_444
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_443
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_446
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_447
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_448
            paddle.uniform([64, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_452
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_449
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_451
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_450
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_453
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_457
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_454
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_456
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_455
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_458
            paddle.uniform([32, 32, 2, 2], dtype='float32', min=0, max=0.5),
            # parameter_462
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_459
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_461
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_460
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_463
            paddle.uniform([64, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_467
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_464
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_466
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_465
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_468
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_472
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_469
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_471
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_470
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_473
            paddle.uniform([16, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_477
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_474
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_476
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_475
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_478
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_482
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_479
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_481
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_480
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_483
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_487
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_484
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_486
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_485
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_488
            paddle.uniform([16, 16, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_492
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_489
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_491
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_490
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_493
            paddle.uniform([64, 16, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_497
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_494
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_496
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_495
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_498
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_502
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_499
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_501
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_500
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_503
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_507
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_504
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_506
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_505
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_508
            paddle.uniform([16, 16, 2, 2], dtype='float32', min=0, max=0.5),
            # parameter_512
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_509
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_511
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_510
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_513
            paddle.uniform([16, 16, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_517
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_514
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_516
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_515
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_518
            paddle.uniform([4, 16, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_522
            paddle.uniform([4], dtype='float32', min=0, max=0.5),
            # parameter_519
            paddle.uniform([4], dtype='float32', min=0, max=0.5),
            # parameter_521
            paddle.uniform([4], dtype='float32', min=0, max=0.5),
            # parameter_520
            paddle.uniform([4], dtype='float32', min=0, max=0.5),
            # parameter_523
            paddle.uniform([4, 4, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_527
            paddle.uniform([4], dtype='float32', min=0, max=0.5),
            # parameter_524
            paddle.uniform([4], dtype='float32', min=0, max=0.5),
            # parameter_526
            paddle.uniform([4], dtype='float32', min=0, max=0.5),
            # parameter_525
            paddle.uniform([4], dtype='float32', min=0, max=0.5),
            # parameter_528
            paddle.uniform([16, 4, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_532
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_529
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_531
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_530
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_533
            paddle.uniform([16, 19, 3, 3], dtype='float32', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 3, 1024, 2048], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_0
            paddle.static.InputSpec(shape=[13, 3, 3, 3], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[4, 16, 2, 2], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[4], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[4], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[4], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[4], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[4, 4, 3, 3], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[4], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[4], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[4], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[4], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[64, 4, 1, 1], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[16, 16, 3, 3], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[64, 16, 1, 1], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[16, 16, 3, 3], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[64, 16, 1, 1], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[16, 16, 3, 3], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[64, 16, 1, 1], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[16, 16, 3, 3], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[64, 16, 1, 1], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[16, 64, 2, 2], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[16, 16, 3, 3], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[128, 16, 1, 1], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[128, 32, 1, 1], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_139
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_149
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[128, 32, 1, 1], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_154
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[32, 32, 5, 1], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[32, 32, 1, 5], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[128, 32, 1, 1], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_189
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[128, 32, 1, 1], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_209
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_213
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_214
            paddle.static.InputSpec(shape=[128, 32, 1, 1], dtype='float32'),
            # parameter_218
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_215
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_217
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_219
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_221
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float32'),
            # parameter_225
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_224
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_223
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_227
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_229
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_233
            paddle.static.InputSpec(shape=[128, 32, 1, 1], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_234
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_238
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_239
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float32'),
            # parameter_244
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_241
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_243
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_245
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_246
            paddle.static.InputSpec(shape=[32, 32, 5, 1], dtype='float32'),
            # parameter_250
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_249
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_248
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_251
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_252
            paddle.static.InputSpec(shape=[32, 32, 1, 5], dtype='float32'),
            # parameter_256
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_253
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_255
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_254
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_257
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_258
            paddle.static.InputSpec(shape=[128, 32, 1, 1], dtype='float32'),
            # parameter_262
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_259
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_261
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_263
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_264
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_265
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float32'),
            # parameter_269
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_266
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_268
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_267
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_270
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_271
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_275
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_272
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_274
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_273
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_276
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_277
            paddle.static.InputSpec(shape=[128, 32, 1, 1], dtype='float32'),
            # parameter_281
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_278
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_280
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_279
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_282
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_283
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_284
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float32'),
            # parameter_288
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_285
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_287
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_286
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_289
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_290
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_294
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_291
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_293
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_292
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_295
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_296
            paddle.static.InputSpec(shape=[128, 32, 1, 1], dtype='float32'),
            # parameter_300
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_297
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_299
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_298
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_301
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_302
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_303
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float32'),
            # parameter_307
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_304
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_306
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_305
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_308
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_309
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_313
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_310
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_312
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_311
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_314
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_315
            paddle.static.InputSpec(shape=[128, 32, 1, 1], dtype='float32'),
            # parameter_319
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_316
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_318
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_317
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_320
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_321
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_322
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float32'),
            # parameter_326
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_323
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_325
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_324
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_327
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_328
            paddle.static.InputSpec(shape=[32, 32, 5, 1], dtype='float32'),
            # parameter_332
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_329
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_331
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_330
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_333
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_334
            paddle.static.InputSpec(shape=[32, 32, 1, 5], dtype='float32'),
            # parameter_338
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_335
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_337
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_336
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_339
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_340
            paddle.static.InputSpec(shape=[128, 32, 1, 1], dtype='float32'),
            # parameter_344
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_341
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_343
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_342
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_345
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_346
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_347
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float32'),
            # parameter_351
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_348
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_350
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_349
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_352
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_353
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_357
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_354
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_356
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_355
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_358
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_359
            paddle.static.InputSpec(shape=[128, 32, 1, 1], dtype='float32'),
            # parameter_363
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_360
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_362
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_361
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_364
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_365
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_366
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float32'),
            # parameter_370
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_367
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_369
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_368
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_371
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_372
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_376
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_373
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_375
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_374
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_377
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_378
            paddle.static.InputSpec(shape=[128, 32, 1, 1], dtype='float32'),
            # parameter_382
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_379
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_381
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_380
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_383
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_384
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_385
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float32'),
            # parameter_389
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_386
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_388
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_387
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_390
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_391
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_395
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_392
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_394
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_393
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_396
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_397
            paddle.static.InputSpec(shape=[128, 32, 1, 1], dtype='float32'),
            # parameter_401
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_398
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_400
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_399
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_402
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_403
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_404
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float32'),
            # parameter_408
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_405
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_407
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_406
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_409
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_410
            paddle.static.InputSpec(shape=[32, 32, 5, 1], dtype='float32'),
            # parameter_414
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_411
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_413
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_412
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_415
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_416
            paddle.static.InputSpec(shape=[32, 32, 1, 5], dtype='float32'),
            # parameter_420
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_417
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_419
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_418
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_421
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_422
            paddle.static.InputSpec(shape=[128, 32, 1, 1], dtype='float32'),
            # parameter_426
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_423
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_425
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_424
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_427
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_428
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_429
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float32'),
            # parameter_433
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_430
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_432
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_431
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_434
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_435
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float32'),
            # parameter_439
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_436
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_438
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_437
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_440
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_441
            paddle.static.InputSpec(shape=[128, 32, 1, 1], dtype='float32'),
            # parameter_445
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_442
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_444
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_443
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_446
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_447
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_448
            paddle.static.InputSpec(shape=[64, 128, 1, 1], dtype='float32'),
            # parameter_452
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_449
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_451
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_450
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_453
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float32'),
            # parameter_457
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_454
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_456
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_455
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_458
            paddle.static.InputSpec(shape=[32, 32, 2, 2], dtype='float32'),
            # parameter_462
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_459
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_461
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_460
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_463
            paddle.static.InputSpec(shape=[64, 32, 1, 1], dtype='float32'),
            # parameter_467
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_464
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_466
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_465
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_468
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float32'),
            # parameter_472
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_469
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_471
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_470
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_473
            paddle.static.InputSpec(shape=[16, 16, 3, 3], dtype='float32'),
            # parameter_477
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_474
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_476
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_475
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_478
            paddle.static.InputSpec(shape=[64, 16, 1, 1], dtype='float32'),
            # parameter_482
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_479
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_481
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_480
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_483
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float32'),
            # parameter_487
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_484
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_486
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_485
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_488
            paddle.static.InputSpec(shape=[16, 16, 3, 3], dtype='float32'),
            # parameter_492
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_489
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_491
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_490
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_493
            paddle.static.InputSpec(shape=[64, 16, 1, 1], dtype='float32'),
            # parameter_497
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_494
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_496
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_495
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_498
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float32'),
            # parameter_502
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_499
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_501
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_500
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_503
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float32'),
            # parameter_507
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_504
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_506
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_505
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_508
            paddle.static.InputSpec(shape=[16, 16, 2, 2], dtype='float32'),
            # parameter_512
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_509
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_511
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_510
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_513
            paddle.static.InputSpec(shape=[16, 16, 1, 1], dtype='float32'),
            # parameter_517
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_514
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_516
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_515
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_518
            paddle.static.InputSpec(shape=[4, 16, 1, 1], dtype='float32'),
            # parameter_522
            paddle.static.InputSpec(shape=[4], dtype='float32'),
            # parameter_519
            paddle.static.InputSpec(shape=[4], dtype='float32'),
            # parameter_521
            paddle.static.InputSpec(shape=[4], dtype='float32'),
            # parameter_520
            paddle.static.InputSpec(shape=[4], dtype='float32'),
            # parameter_523
            paddle.static.InputSpec(shape=[4, 4, 3, 3], dtype='float32'),
            # parameter_527
            paddle.static.InputSpec(shape=[4], dtype='float32'),
            # parameter_524
            paddle.static.InputSpec(shape=[4], dtype='float32'),
            # parameter_526
            paddle.static.InputSpec(shape=[4], dtype='float32'),
            # parameter_525
            paddle.static.InputSpec(shape=[4], dtype='float32'),
            # parameter_528
            paddle.static.InputSpec(shape=[16, 4, 1, 1], dtype='float32'),
            # parameter_532
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_529
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_531
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_530
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_533
            paddle.static.InputSpec(shape=[16, 19, 3, 3], dtype='float32'),
            # feed_0
            paddle.static.InputSpec(shape=[1, 3, 1024, 2048], dtype='float32'),
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