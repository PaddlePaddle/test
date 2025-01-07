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
    return [633][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_1174_0_0(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_114, parameter_111, parameter_113, parameter_112, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_154, parameter_151, parameter_153, parameter_152, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_169, parameter_166, parameter_168, parameter_167, parameter_170, parameter_174, parameter_171, parameter_173, parameter_172, parameter_175, parameter_179, parameter_176, parameter_178, parameter_177, parameter_180, parameter_184, parameter_181, parameter_183, parameter_182, parameter_185, parameter_189, parameter_186, parameter_188, parameter_187, parameter_190, parameter_194, parameter_191, parameter_193, parameter_192, parameter_195, parameter_199, parameter_196, parameter_198, parameter_197, parameter_200, parameter_204, parameter_201, parameter_203, parameter_202, parameter_205, parameter_209, parameter_206, parameter_208, parameter_207, parameter_210, parameter_214, parameter_211, parameter_213, parameter_212, parameter_215, parameter_219, parameter_216, parameter_218, parameter_217, parameter_220, parameter_224, parameter_221, parameter_223, parameter_222, parameter_225, parameter_229, parameter_226, parameter_228, parameter_227, parameter_230, parameter_231, parameter_232, parameter_236, parameter_233, parameter_235, parameter_234, parameter_237, parameter_241, parameter_238, parameter_240, parameter_239, parameter_242, parameter_246, parameter_243, parameter_245, parameter_244, parameter_247, parameter_251, parameter_248, parameter_250, parameter_249, parameter_252, parameter_253, parameter_254, parameter_258, parameter_255, parameter_257, parameter_256, parameter_259, parameter_263, parameter_260, parameter_262, parameter_261, parameter_264, parameter_268, parameter_265, parameter_267, parameter_266, parameter_269, parameter_270, parameter_271, parameter_275, parameter_272, parameter_274, parameter_273, parameter_276, parameter_280, parameter_277, parameter_279, parameter_278, parameter_281, parameter_285, parameter_282, parameter_284, parameter_283, parameter_286, parameter_290, parameter_287, parameter_289, parameter_288, parameter_291, parameter_295, parameter_292, parameter_294, parameter_293, parameter_296, parameter_300, parameter_297, parameter_299, parameter_298, parameter_301, parameter_305, parameter_302, parameter_304, parameter_303, parameter_306, parameter_310, parameter_307, parameter_309, parameter_308, parameter_311, parameter_315, parameter_312, parameter_314, parameter_313, parameter_316, parameter_320, parameter_317, parameter_319, parameter_318, parameter_321, parameter_325, parameter_322, parameter_324, parameter_323, parameter_326, parameter_330, parameter_327, parameter_329, parameter_328, parameter_331, parameter_335, parameter_332, parameter_334, parameter_333, parameter_336, parameter_340, parameter_337, parameter_339, parameter_338, parameter_341, parameter_345, parameter_342, parameter_344, parameter_343, parameter_346, parameter_350, parameter_347, parameter_349, parameter_348, parameter_351, parameter_355, parameter_352, parameter_354, parameter_353, parameter_356, parameter_360, parameter_357, parameter_359, parameter_358, parameter_361, parameter_365, parameter_362, parameter_364, parameter_363, parameter_366, parameter_370, parameter_367, parameter_369, parameter_368, parameter_371, parameter_375, parameter_372, parameter_374, parameter_373, parameter_376, parameter_380, parameter_377, parameter_379, parameter_378, parameter_381, parameter_385, parameter_382, parameter_384, parameter_383, parameter_386, parameter_390, parameter_387, parameter_389, parameter_388, parameter_391, parameter_395, parameter_392, parameter_394, parameter_393, parameter_396, parameter_400, parameter_397, parameter_399, parameter_398, parameter_401, parameter_405, parameter_402, parameter_404, parameter_403, parameter_406, parameter_410, parameter_407, parameter_409, parameter_408, parameter_411, parameter_415, parameter_412, parameter_414, parameter_413, parameter_416, parameter_420, parameter_417, parameter_419, parameter_418, parameter_421, parameter_425, parameter_422, parameter_424, parameter_423, parameter_426, parameter_430, parameter_427, parameter_429, parameter_428, parameter_431, parameter_435, parameter_432, parameter_434, parameter_433, parameter_436, parameter_440, parameter_437, parameter_439, parameter_438, parameter_441, parameter_445, parameter_442, parameter_444, parameter_443, parameter_446, parameter_450, parameter_447, parameter_449, parameter_448, parameter_451, parameter_455, parameter_452, parameter_454, parameter_453, parameter_456, parameter_460, parameter_457, parameter_459, parameter_458, parameter_461, parameter_465, parameter_462, parameter_464, parameter_463, parameter_466, parameter_470, parameter_467, parameter_469, parameter_468, parameter_471, parameter_475, parameter_472, parameter_474, parameter_473, parameter_476, parameter_480, parameter_477, parameter_479, parameter_478, parameter_481, parameter_485, parameter_482, parameter_484, parameter_483, parameter_486, parameter_490, parameter_487, parameter_489, parameter_488, parameter_491, parameter_495, parameter_492, parameter_494, parameter_493, parameter_496, parameter_500, parameter_497, parameter_499, parameter_498, parameter_501, parameter_505, parameter_502, parameter_504, parameter_503, parameter_506, parameter_510, parameter_507, parameter_509, parameter_508, parameter_511, parameter_515, parameter_512, parameter_514, parameter_513, parameter_516, parameter_520, parameter_517, parameter_519, parameter_518, parameter_521, parameter_525, parameter_522, parameter_524, parameter_523, parameter_526, parameter_527, parameter_528, parameter_529, parameter_530, parameter_531, feed_2, feed_0, feed_1):

        # pd_op.cast: (-1x3x640x640xf16) <- (-1x3x640x640xf32)
        cast_0 = paddle._C_ops.cast(feed_0, paddle.float16)

        # pd_op.conv2d: (-1x32x320x320xf16) <- (-1x3x640x640xf16, 32x3x3x3xf16)
        conv2d_0 = paddle._C_ops.conv2d(cast_0, parameter_0, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x320x320xf16, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x320x320xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_0, parameter_1, parameter_2, parameter_3, parameter_4, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x32x320x320xf16) <- (-1x32x320x320xf16)
        relu__0 = paddle._C_ops.relu_(batch_norm__0)

        # pd_op.conv2d: (-1x32x320x320xf16) <- (-1x32x320x320xf16, 32x32x3x3xf16)
        conv2d_1 = paddle._C_ops.conv2d(relu__0, parameter_5, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x320x320xf16, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x320x320xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_1, parameter_6, parameter_7, parameter_8, parameter_9, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x32x320x320xf16) <- (-1x32x320x320xf16)
        relu__1 = paddle._C_ops.relu_(batch_norm__6)

        # pd_op.conv2d: (-1x64x320x320xf16) <- (-1x32x320x320xf16, 64x32x3x3xf16)
        conv2d_2 = paddle._C_ops.conv2d(relu__1, parameter_10, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x320x320xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x320x320xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_2, parameter_11, parameter_12, parameter_13, parameter_14, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x320x320xf16) <- (-1x64x320x320xf16)
        relu__2 = paddle._C_ops.relu_(batch_norm__12)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [3, 3]

        # pd_op.pool2d: (-1x64x160x160xf16) <- (-1x64x320x320xf16, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(relu__2, full_int_array_0, [2, 2], [1, 1], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x64x160x160xf16) <- (-1x64x160x160xf16, 64x64x1x1xf16)
        conv2d_3 = paddle._C_ops.conv2d(pool2d_0, parameter_15, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x160x160xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x160x160xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_3, parameter_16, parameter_17, parameter_18, parameter_19, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x160x160xf16) <- (-1x64x160x160xf16)
        relu__3 = paddle._C_ops.relu_(batch_norm__18)

        # pd_op.conv2d: (-1x64x160x160xf16) <- (-1x64x160x160xf16, 64x64x3x3xf16)
        conv2d_4 = paddle._C_ops.conv2d(relu__3, parameter_20, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x160x160xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x160x160xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_4, parameter_21, parameter_22, parameter_23, parameter_24, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x160x160xf16) <- (-1x64x160x160xf16)
        relu__4 = paddle._C_ops.relu_(batch_norm__24)

        # pd_op.conv2d: (-1x256x160x160xf16) <- (-1x64x160x160xf16, 256x64x1x1xf16)
        conv2d_5 = paddle._C_ops.conv2d(relu__4, parameter_25, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x160x160xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x160x160xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_5, parameter_26, parameter_27, parameter_28, parameter_29, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x256x160x160xf16) <- (-1x64x160x160xf16, 256x64x1x1xf16)
        conv2d_6 = paddle._C_ops.conv2d(pool2d_0, parameter_30, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x160x160xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x160x160xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_6, parameter_31, parameter_32, parameter_33, parameter_34, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x256x160x160xf16) <- (-1x256x160x160xf16, -1x256x160x160xf16)
        add__0 = paddle._C_ops.add_(batch_norm__30, batch_norm__36)

        # pd_op.relu_: (-1x256x160x160xf16) <- (-1x256x160x160xf16)
        relu__5 = paddle._C_ops.relu_(add__0)

        # pd_op.conv2d: (-1x64x160x160xf16) <- (-1x256x160x160xf16, 64x256x1x1xf16)
        conv2d_7 = paddle._C_ops.conv2d(relu__5, parameter_35, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x160x160xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x160x160xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_7, parameter_36, parameter_37, parameter_38, parameter_39, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x160x160xf16) <- (-1x64x160x160xf16)
        relu__6 = paddle._C_ops.relu_(batch_norm__42)

        # pd_op.conv2d: (-1x64x160x160xf16) <- (-1x64x160x160xf16, 64x64x3x3xf16)
        conv2d_8 = paddle._C_ops.conv2d(relu__6, parameter_40, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x160x160xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x160x160xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_8, parameter_41, parameter_42, parameter_43, parameter_44, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x160x160xf16) <- (-1x64x160x160xf16)
        relu__7 = paddle._C_ops.relu_(batch_norm__48)

        # pd_op.conv2d: (-1x256x160x160xf16) <- (-1x64x160x160xf16, 256x64x1x1xf16)
        conv2d_9 = paddle._C_ops.conv2d(relu__7, parameter_45, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x160x160xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x160x160xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_9, parameter_46, parameter_47, parameter_48, parameter_49, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x256x160x160xf16) <- (-1x256x160x160xf16, -1x256x160x160xf16)
        add__1 = paddle._C_ops.add_(batch_norm__54, relu__5)

        # pd_op.relu_: (-1x256x160x160xf16) <- (-1x256x160x160xf16)
        relu__8 = paddle._C_ops.relu_(add__1)

        # pd_op.conv2d: (-1x64x160x160xf16) <- (-1x256x160x160xf16, 64x256x1x1xf16)
        conv2d_10 = paddle._C_ops.conv2d(relu__8, parameter_50, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x160x160xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x160x160xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_10, parameter_51, parameter_52, parameter_53, parameter_54, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x160x160xf16) <- (-1x64x160x160xf16)
        relu__9 = paddle._C_ops.relu_(batch_norm__60)

        # pd_op.conv2d: (-1x64x160x160xf16) <- (-1x64x160x160xf16, 64x64x3x3xf16)
        conv2d_11 = paddle._C_ops.conv2d(relu__9, parameter_55, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x160x160xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x160x160xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_11, parameter_56, parameter_57, parameter_58, parameter_59, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x160x160xf16) <- (-1x64x160x160xf16)
        relu__10 = paddle._C_ops.relu_(batch_norm__66)

        # pd_op.conv2d: (-1x256x160x160xf16) <- (-1x64x160x160xf16, 256x64x1x1xf16)
        conv2d_12 = paddle._C_ops.conv2d(relu__10, parameter_60, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x160x160xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x160x160xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_12, parameter_61, parameter_62, parameter_63, parameter_64, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x256x160x160xf16) <- (-1x256x160x160xf16, -1x256x160x160xf16)
        add__2 = paddle._C_ops.add_(batch_norm__72, relu__8)

        # pd_op.relu_: (-1x256x160x160xf16) <- (-1x256x160x160xf16)
        relu__11 = paddle._C_ops.relu_(add__2)

        # pd_op.conv2d: (-1x128x160x160xf16) <- (-1x256x160x160xf16, 128x256x1x1xf16)
        conv2d_13 = paddle._C_ops.conv2d(relu__11, parameter_65, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x160x160xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x160x160xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_13, parameter_66, parameter_67, parameter_68, parameter_69, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x160x160xf16) <- (-1x128x160x160xf16)
        relu__12 = paddle._C_ops.relu_(batch_norm__78)

        # pd_op.conv2d: (-1x128x80x80xf16) <- (-1x128x160x160xf16, 128x128x3x3xf16)
        conv2d_14 = paddle._C_ops.conv2d(relu__12, parameter_70, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x80x80xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x80x80xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_14, parameter_71, parameter_72, parameter_73, parameter_74, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x80x80xf16) <- (-1x128x80x80xf16)
        relu__13 = paddle._C_ops.relu_(batch_norm__84)

        # pd_op.conv2d: (-1x512x80x80xf16) <- (-1x128x80x80xf16, 512x128x1x1xf16)
        conv2d_15 = paddle._C_ops.conv2d(relu__13, parameter_75, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x80x80xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x80x80xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_15, parameter_76, parameter_77, parameter_78, parameter_79, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [2, 2]

        # pd_op.pool2d: (-1x256x80x80xf16) <- (-1x256x160x160xf16, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(relu__11, full_int_array_1, [2, 2], [0, 0], True, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x512x80x80xf16) <- (-1x256x80x80xf16, 512x256x1x1xf16)
        conv2d_16 = paddle._C_ops.conv2d(pool2d_1, parameter_80, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x80x80xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x80x80xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_16, parameter_81, parameter_82, parameter_83, parameter_84, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x80x80xf16) <- (-1x512x80x80xf16, -1x512x80x80xf16)
        add__3 = paddle._C_ops.add_(batch_norm__90, batch_norm__96)

        # pd_op.relu_: (-1x512x80x80xf16) <- (-1x512x80x80xf16)
        relu__14 = paddle._C_ops.relu_(add__3)

        # pd_op.conv2d: (-1x128x80x80xf16) <- (-1x512x80x80xf16, 128x512x1x1xf16)
        conv2d_17 = paddle._C_ops.conv2d(relu__14, parameter_85, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x80x80xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x80x80xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_17, parameter_86, parameter_87, parameter_88, parameter_89, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x80x80xf16) <- (-1x128x80x80xf16)
        relu__15 = paddle._C_ops.relu_(batch_norm__102)

        # pd_op.conv2d: (-1x128x80x80xf16) <- (-1x128x80x80xf16, 128x128x3x3xf16)
        conv2d_18 = paddle._C_ops.conv2d(relu__15, parameter_90, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x80x80xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x80x80xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_18, parameter_91, parameter_92, parameter_93, parameter_94, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x80x80xf16) <- (-1x128x80x80xf16)
        relu__16 = paddle._C_ops.relu_(batch_norm__108)

        # pd_op.conv2d: (-1x512x80x80xf16) <- (-1x128x80x80xf16, 512x128x1x1xf16)
        conv2d_19 = paddle._C_ops.conv2d(relu__16, parameter_95, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x80x80xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x80x80xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_19, parameter_96, parameter_97, parameter_98, parameter_99, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x80x80xf16) <- (-1x512x80x80xf16, -1x512x80x80xf16)
        add__4 = paddle._C_ops.add_(batch_norm__114, relu__14)

        # pd_op.relu_: (-1x512x80x80xf16) <- (-1x512x80x80xf16)
        relu__17 = paddle._C_ops.relu_(add__4)

        # pd_op.conv2d: (-1x128x80x80xf16) <- (-1x512x80x80xf16, 128x512x1x1xf16)
        conv2d_20 = paddle._C_ops.conv2d(relu__17, parameter_100, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x80x80xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x80x80xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_20, parameter_101, parameter_102, parameter_103, parameter_104, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x80x80xf16) <- (-1x128x80x80xf16)
        relu__18 = paddle._C_ops.relu_(batch_norm__120)

        # pd_op.conv2d: (-1x128x80x80xf16) <- (-1x128x80x80xf16, 128x128x3x3xf16)
        conv2d_21 = paddle._C_ops.conv2d(relu__18, parameter_105, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x80x80xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x80x80xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_21, parameter_106, parameter_107, parameter_108, parameter_109, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x80x80xf16) <- (-1x128x80x80xf16)
        relu__19 = paddle._C_ops.relu_(batch_norm__126)

        # pd_op.conv2d: (-1x512x80x80xf16) <- (-1x128x80x80xf16, 512x128x1x1xf16)
        conv2d_22 = paddle._C_ops.conv2d(relu__19, parameter_110, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x80x80xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x80x80xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_22, parameter_111, parameter_112, parameter_113, parameter_114, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x80x80xf16) <- (-1x512x80x80xf16, -1x512x80x80xf16)
        add__5 = paddle._C_ops.add_(batch_norm__132, relu__17)

        # pd_op.relu_: (-1x512x80x80xf16) <- (-1x512x80x80xf16)
        relu__20 = paddle._C_ops.relu_(add__5)

        # pd_op.conv2d: (-1x128x80x80xf16) <- (-1x512x80x80xf16, 128x512x1x1xf16)
        conv2d_23 = paddle._C_ops.conv2d(relu__20, parameter_115, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x80x80xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x80x80xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_23, parameter_116, parameter_117, parameter_118, parameter_119, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x80x80xf16) <- (-1x128x80x80xf16)
        relu__21 = paddle._C_ops.relu_(batch_norm__138)

        # pd_op.conv2d: (-1x128x80x80xf16) <- (-1x128x80x80xf16, 128x128x3x3xf16)
        conv2d_24 = paddle._C_ops.conv2d(relu__21, parameter_120, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x80x80xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x80x80xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_24, parameter_121, parameter_122, parameter_123, parameter_124, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x80x80xf16) <- (-1x128x80x80xf16)
        relu__22 = paddle._C_ops.relu_(batch_norm__144)

        # pd_op.conv2d: (-1x512x80x80xf16) <- (-1x128x80x80xf16, 512x128x1x1xf16)
        conv2d_25 = paddle._C_ops.conv2d(relu__22, parameter_125, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x80x80xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x80x80xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_25, parameter_126, parameter_127, parameter_128, parameter_129, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x80x80xf16) <- (-1x512x80x80xf16, -1x512x80x80xf16)
        add__6 = paddle._C_ops.add_(batch_norm__150, relu__20)

        # pd_op.relu_: (-1x512x80x80xf16) <- (-1x512x80x80xf16)
        relu__23 = paddle._C_ops.relu_(add__6)

        # pd_op.conv2d: (-1x256x80x80xf16) <- (-1x512x80x80xf16, 256x512x1x1xf16)
        conv2d_26 = paddle._C_ops.conv2d(relu__23, parameter_130, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x80x80xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x80x80xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_26, parameter_131, parameter_132, parameter_133, parameter_134, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x80x80xf16) <- (-1x256x80x80xf16)
        relu__24 = paddle._C_ops.relu_(batch_norm__156)

        # pd_op.conv2d: (-1x256x40x40xf16) <- (-1x256x80x80xf16, 256x256x3x3xf16)
        conv2d_27 = paddle._C_ops.conv2d(relu__24, parameter_135, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x40x40xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x40x40xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_27, parameter_136, parameter_137, parameter_138, parameter_139, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x40x40xf16) <- (-1x256x40x40xf16)
        relu__25 = paddle._C_ops.relu_(batch_norm__162)

        # pd_op.conv2d: (-1x1024x40x40xf16) <- (-1x256x40x40xf16, 1024x256x1x1xf16)
        conv2d_28 = paddle._C_ops.conv2d(relu__25, parameter_140, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x40x40xf16, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x40x40xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_28, parameter_141, parameter_142, parameter_143, parameter_144, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [2, 2]

        # pd_op.pool2d: (-1x512x40x40xf16) <- (-1x512x80x80xf16, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(relu__23, full_int_array_2, [2, 2], [0, 0], True, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x1024x40x40xf16) <- (-1x512x40x40xf16, 1024x512x1x1xf16)
        conv2d_29 = paddle._C_ops.conv2d(pool2d_2, parameter_145, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x40x40xf16, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x40x40xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_29, parameter_146, parameter_147, parameter_148, parameter_149, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x40x40xf16) <- (-1x1024x40x40xf16, -1x1024x40x40xf16)
        add__7 = paddle._C_ops.add_(batch_norm__168, batch_norm__174)

        # pd_op.relu_: (-1x1024x40x40xf16) <- (-1x1024x40x40xf16)
        relu__26 = paddle._C_ops.relu_(add__7)

        # pd_op.conv2d: (-1x256x40x40xf16) <- (-1x1024x40x40xf16, 256x1024x1x1xf16)
        conv2d_30 = paddle._C_ops.conv2d(relu__26, parameter_150, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x40x40xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x40x40xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__180, batch_norm__181, batch_norm__182, batch_norm__183, batch_norm__184, batch_norm__185 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_30, parameter_151, parameter_152, parameter_153, parameter_154, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x40x40xf16) <- (-1x256x40x40xf16)
        relu__27 = paddle._C_ops.relu_(batch_norm__180)

        # pd_op.conv2d: (-1x256x40x40xf16) <- (-1x256x40x40xf16, 256x256x3x3xf16)
        conv2d_31 = paddle._C_ops.conv2d(relu__27, parameter_155, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x40x40xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x40x40xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__186, batch_norm__187, batch_norm__188, batch_norm__189, batch_norm__190, batch_norm__191 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_31, parameter_156, parameter_157, parameter_158, parameter_159, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x40x40xf16) <- (-1x256x40x40xf16)
        relu__28 = paddle._C_ops.relu_(batch_norm__186)

        # pd_op.conv2d: (-1x1024x40x40xf16) <- (-1x256x40x40xf16, 1024x256x1x1xf16)
        conv2d_32 = paddle._C_ops.conv2d(relu__28, parameter_160, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x40x40xf16, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x40x40xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__192, batch_norm__193, batch_norm__194, batch_norm__195, batch_norm__196, batch_norm__197 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_32, parameter_161, parameter_162, parameter_163, parameter_164, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x40x40xf16) <- (-1x1024x40x40xf16, -1x1024x40x40xf16)
        add__8 = paddle._C_ops.add_(batch_norm__192, relu__26)

        # pd_op.relu_: (-1x1024x40x40xf16) <- (-1x1024x40x40xf16)
        relu__29 = paddle._C_ops.relu_(add__8)

        # pd_op.conv2d: (-1x256x40x40xf16) <- (-1x1024x40x40xf16, 256x1024x1x1xf16)
        conv2d_33 = paddle._C_ops.conv2d(relu__29, parameter_165, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x40x40xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x40x40xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__198, batch_norm__199, batch_norm__200, batch_norm__201, batch_norm__202, batch_norm__203 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_33, parameter_166, parameter_167, parameter_168, parameter_169, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x40x40xf16) <- (-1x256x40x40xf16)
        relu__30 = paddle._C_ops.relu_(batch_norm__198)

        # pd_op.conv2d: (-1x256x40x40xf16) <- (-1x256x40x40xf16, 256x256x3x3xf16)
        conv2d_34 = paddle._C_ops.conv2d(relu__30, parameter_170, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x40x40xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x40x40xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__204, batch_norm__205, batch_norm__206, batch_norm__207, batch_norm__208, batch_norm__209 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_34, parameter_171, parameter_172, parameter_173, parameter_174, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x40x40xf16) <- (-1x256x40x40xf16)
        relu__31 = paddle._C_ops.relu_(batch_norm__204)

        # pd_op.conv2d: (-1x1024x40x40xf16) <- (-1x256x40x40xf16, 1024x256x1x1xf16)
        conv2d_35 = paddle._C_ops.conv2d(relu__31, parameter_175, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x40x40xf16, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x40x40xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__210, batch_norm__211, batch_norm__212, batch_norm__213, batch_norm__214, batch_norm__215 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_35, parameter_176, parameter_177, parameter_178, parameter_179, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x40x40xf16) <- (-1x1024x40x40xf16, -1x1024x40x40xf16)
        add__9 = paddle._C_ops.add_(batch_norm__210, relu__29)

        # pd_op.relu_: (-1x1024x40x40xf16) <- (-1x1024x40x40xf16)
        relu__32 = paddle._C_ops.relu_(add__9)

        # pd_op.conv2d: (-1x256x40x40xf16) <- (-1x1024x40x40xf16, 256x1024x1x1xf16)
        conv2d_36 = paddle._C_ops.conv2d(relu__32, parameter_180, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x40x40xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x40x40xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__216, batch_norm__217, batch_norm__218, batch_norm__219, batch_norm__220, batch_norm__221 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_36, parameter_181, parameter_182, parameter_183, parameter_184, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x40x40xf16) <- (-1x256x40x40xf16)
        relu__33 = paddle._C_ops.relu_(batch_norm__216)

        # pd_op.conv2d: (-1x256x40x40xf16) <- (-1x256x40x40xf16, 256x256x3x3xf16)
        conv2d_37 = paddle._C_ops.conv2d(relu__33, parameter_185, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x40x40xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x40x40xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__222, batch_norm__223, batch_norm__224, batch_norm__225, batch_norm__226, batch_norm__227 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_37, parameter_186, parameter_187, parameter_188, parameter_189, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x40x40xf16) <- (-1x256x40x40xf16)
        relu__34 = paddle._C_ops.relu_(batch_norm__222)

        # pd_op.conv2d: (-1x1024x40x40xf16) <- (-1x256x40x40xf16, 1024x256x1x1xf16)
        conv2d_38 = paddle._C_ops.conv2d(relu__34, parameter_190, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x40x40xf16, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x40x40xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__228, batch_norm__229, batch_norm__230, batch_norm__231, batch_norm__232, batch_norm__233 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_38, parameter_191, parameter_192, parameter_193, parameter_194, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x40x40xf16) <- (-1x1024x40x40xf16, -1x1024x40x40xf16)
        add__10 = paddle._C_ops.add_(batch_norm__228, relu__32)

        # pd_op.relu_: (-1x1024x40x40xf16) <- (-1x1024x40x40xf16)
        relu__35 = paddle._C_ops.relu_(add__10)

        # pd_op.conv2d: (-1x256x40x40xf16) <- (-1x1024x40x40xf16, 256x1024x1x1xf16)
        conv2d_39 = paddle._C_ops.conv2d(relu__35, parameter_195, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x40x40xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x40x40xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__234, batch_norm__235, batch_norm__236, batch_norm__237, batch_norm__238, batch_norm__239 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_39, parameter_196, parameter_197, parameter_198, parameter_199, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x40x40xf16) <- (-1x256x40x40xf16)
        relu__36 = paddle._C_ops.relu_(batch_norm__234)

        # pd_op.conv2d: (-1x256x40x40xf16) <- (-1x256x40x40xf16, 256x256x3x3xf16)
        conv2d_40 = paddle._C_ops.conv2d(relu__36, parameter_200, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x40x40xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x40x40xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__240, batch_norm__241, batch_norm__242, batch_norm__243, batch_norm__244, batch_norm__245 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_40, parameter_201, parameter_202, parameter_203, parameter_204, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x40x40xf16) <- (-1x256x40x40xf16)
        relu__37 = paddle._C_ops.relu_(batch_norm__240)

        # pd_op.conv2d: (-1x1024x40x40xf16) <- (-1x256x40x40xf16, 1024x256x1x1xf16)
        conv2d_41 = paddle._C_ops.conv2d(relu__37, parameter_205, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x40x40xf16, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x40x40xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__246, batch_norm__247, batch_norm__248, batch_norm__249, batch_norm__250, batch_norm__251 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_41, parameter_206, parameter_207, parameter_208, parameter_209, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x40x40xf16) <- (-1x1024x40x40xf16, -1x1024x40x40xf16)
        add__11 = paddle._C_ops.add_(batch_norm__246, relu__35)

        # pd_op.relu_: (-1x1024x40x40xf16) <- (-1x1024x40x40xf16)
        relu__38 = paddle._C_ops.relu_(add__11)

        # pd_op.conv2d: (-1x256x40x40xf16) <- (-1x1024x40x40xf16, 256x1024x1x1xf16)
        conv2d_42 = paddle._C_ops.conv2d(relu__38, parameter_210, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x40x40xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x40x40xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__252, batch_norm__253, batch_norm__254, batch_norm__255, batch_norm__256, batch_norm__257 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_42, parameter_211, parameter_212, parameter_213, parameter_214, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x40x40xf16) <- (-1x256x40x40xf16)
        relu__39 = paddle._C_ops.relu_(batch_norm__252)

        # pd_op.conv2d: (-1x256x40x40xf16) <- (-1x256x40x40xf16, 256x256x3x3xf16)
        conv2d_43 = paddle._C_ops.conv2d(relu__39, parameter_215, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x40x40xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x40x40xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__258, batch_norm__259, batch_norm__260, batch_norm__261, batch_norm__262, batch_norm__263 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_43, parameter_216, parameter_217, parameter_218, parameter_219, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x40x40xf16) <- (-1x256x40x40xf16)
        relu__40 = paddle._C_ops.relu_(batch_norm__258)

        # pd_op.conv2d: (-1x1024x40x40xf16) <- (-1x256x40x40xf16, 1024x256x1x1xf16)
        conv2d_44 = paddle._C_ops.conv2d(relu__40, parameter_220, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x40x40xf16, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x40x40xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__264, batch_norm__265, batch_norm__266, batch_norm__267, batch_norm__268, batch_norm__269 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_44, parameter_221, parameter_222, parameter_223, parameter_224, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x40x40xf16) <- (-1x1024x40x40xf16, -1x1024x40x40xf16)
        add__12 = paddle._C_ops.add_(batch_norm__264, relu__38)

        # pd_op.relu_: (-1x1024x40x40xf16) <- (-1x1024x40x40xf16)
        relu__41 = paddle._C_ops.relu_(add__12)

        # pd_op.conv2d: (-1x512x40x40xf16) <- (-1x1024x40x40xf16, 512x1024x1x1xf16)
        conv2d_45 = paddle._C_ops.conv2d(relu__41, parameter_225, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x40x40xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x40x40xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__270, batch_norm__271, batch_norm__272, batch_norm__273, batch_norm__274, batch_norm__275 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_45, parameter_226, parameter_227, parameter_228, parameter_229, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x40x40xf16) <- (-1x512x40x40xf16)
        relu__42 = paddle._C_ops.relu_(batch_norm__270)

        # pd_op.conv2d: (-1x27x20x20xf16) <- (-1x512x40x40xf16, 27x512x3x3xf16)
        conv2d_46 = paddle._C_ops.conv2d(relu__42, parameter_230, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [1, 27, 1, 1]

        # pd_op.reshape: (1x27x1x1xf16, 0x27xf16) <- (27xf16, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_231, full_int_array_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x27x20x20xf16) <- (-1x27x20x20xf16, 1x27x1x1xf16)
        add__13 = paddle._C_ops.add_(conv2d_46, reshape_0)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [18, 9]

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x18x20x20xf16, -1x9x20x20xf16]) <- (-1x27x20x20xf16, 2xi64, 1xi32)
        split_0 = paddle._C_ops.split(add__13, full_int_array_4, full_0)

        # builtin.slice: (-1x9x20x20xf16) <- ([-1x18x20x20xf16, -1x9x20x20xf16])
        slice_0 = split_0[1]

        # pd_op.sigmoid_: (-1x9x20x20xf16) <- (-1x9x20x20xf16)
        sigmoid__0 = paddle._C_ops.sigmoid_(slice_0)

        # pd_op.cast: (-1x512x40x40xf32) <- (-1x512x40x40xf16)
        cast_1 = paddle._C_ops.cast(relu__42, paddle.float32)

        # pd_op.cast: (-1x9x20x20xf32) <- (-1x9x20x20xf16)
        cast_2 = paddle._C_ops.cast(sigmoid__0, paddle.float32)

        # builtin.slice: (-1x18x20x20xf16) <- ([-1x18x20x20xf16, -1x9x20x20xf16])
        slice_1 = split_0[0]

        # pd_op.cast: (-1x18x20x20xf32) <- (-1x18x20x20xf16)
        cast_3 = paddle._C_ops.cast(slice_1, paddle.float32)

        # pd_op.deformable_conv: (-1x512x20x20xf32) <- (-1x512x40x40xf32, -1x18x20x20xf32, 512x512x3x3xf32, -1x9x20x20xf32)
        deformable_conv_0 = paddle._C_ops.deformable_conv(cast_1, cast_3, parameter_232, cast_2, [2, 2], [1, 1], [1, 1], 1, 1, 1)

        # pd_op.cast: (-1x512x20x20xf16) <- (-1x512x20x20xf32)
        cast_4 = paddle._C_ops.cast(deformable_conv_0, paddle.float16)

        # pd_op.batch_norm_: (-1x512x20x20xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x20x20xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__276, batch_norm__277, batch_norm__278, batch_norm__279, batch_norm__280, batch_norm__281 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(cast_4, parameter_233, parameter_234, parameter_235, parameter_236, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x20x20xf16) <- (-1x512x20x20xf16)
        relu__43 = paddle._C_ops.relu_(batch_norm__276)

        # pd_op.conv2d: (-1x2048x20x20xf16) <- (-1x512x20x20xf16, 2048x512x1x1xf16)
        conv2d_47 = paddle._C_ops.conv2d(relu__43, parameter_237, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x2048x20x20xf16, 2048xf32, 2048xf32, xf32, xf32, None) <- (-1x2048x20x20xf16, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        batch_norm__282, batch_norm__283, batch_norm__284, batch_norm__285, batch_norm__286, batch_norm__287 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_47, parameter_238, parameter_239, parameter_240, parameter_241, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [2, 2]

        # pd_op.pool2d: (-1x1024x20x20xf16) <- (-1x1024x40x40xf16, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(relu__41, full_int_array_5, [2, 2], [0, 0], True, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x2048x20x20xf16) <- (-1x1024x20x20xf16, 2048x1024x1x1xf16)
        conv2d_48 = paddle._C_ops.conv2d(pool2d_3, parameter_242, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x2048x20x20xf16, 2048xf32, 2048xf32, xf32, xf32, None) <- (-1x2048x20x20xf16, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        batch_norm__288, batch_norm__289, batch_norm__290, batch_norm__291, batch_norm__292, batch_norm__293 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_48, parameter_243, parameter_244, parameter_245, parameter_246, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x2048x20x20xf16) <- (-1x2048x20x20xf16, -1x2048x20x20xf16)
        add__14 = paddle._C_ops.add_(batch_norm__282, batch_norm__288)

        # pd_op.relu_: (-1x2048x20x20xf16) <- (-1x2048x20x20xf16)
        relu__44 = paddle._C_ops.relu_(add__14)

        # pd_op.conv2d: (-1x512x20x20xf16) <- (-1x2048x20x20xf16, 512x2048x1x1xf16)
        conv2d_49 = paddle._C_ops.conv2d(relu__44, parameter_247, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x20x20xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x20x20xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__294, batch_norm__295, batch_norm__296, batch_norm__297, batch_norm__298, batch_norm__299 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_49, parameter_248, parameter_249, parameter_250, parameter_251, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x20x20xf16) <- (-1x512x20x20xf16)
        relu__45 = paddle._C_ops.relu_(batch_norm__294)

        # pd_op.conv2d: (-1x27x20x20xf16) <- (-1x512x20x20xf16, 27x512x3x3xf16)
        conv2d_50 = paddle._C_ops.conv2d(relu__45, parameter_252, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_6 = [1, 27, 1, 1]

        # pd_op.reshape: (1x27x1x1xf16, 0x27xf16) <- (27xf16, 4xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_253, full_int_array_6), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x27x20x20xf16) <- (-1x27x20x20xf16, 1x27x1x1xf16)
        add__15 = paddle._C_ops.add_(conv2d_50, reshape_2)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_7 = [18, 9]

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x18x20x20xf16, -1x9x20x20xf16]) <- (-1x27x20x20xf16, 2xi64, 1xi32)
        split_1 = paddle._C_ops.split(add__15, full_int_array_7, full_1)

        # builtin.slice: (-1x9x20x20xf16) <- ([-1x18x20x20xf16, -1x9x20x20xf16])
        slice_2 = split_1[1]

        # pd_op.sigmoid_: (-1x9x20x20xf16) <- (-1x9x20x20xf16)
        sigmoid__1 = paddle._C_ops.sigmoid_(slice_2)

        # pd_op.cast: (-1x512x20x20xf32) <- (-1x512x20x20xf16)
        cast_5 = paddle._C_ops.cast(relu__45, paddle.float32)

        # pd_op.cast: (-1x9x20x20xf32) <- (-1x9x20x20xf16)
        cast_6 = paddle._C_ops.cast(sigmoid__1, paddle.float32)

        # builtin.slice: (-1x18x20x20xf16) <- ([-1x18x20x20xf16, -1x9x20x20xf16])
        slice_3 = split_1[0]

        # pd_op.cast: (-1x18x20x20xf32) <- (-1x18x20x20xf16)
        cast_7 = paddle._C_ops.cast(slice_3, paddle.float32)

        # pd_op.deformable_conv: (-1x512x20x20xf32) <- (-1x512x20x20xf32, -1x18x20x20xf32, 512x512x3x3xf32, -1x9x20x20xf32)
        deformable_conv_1 = paddle._C_ops.deformable_conv(cast_5, cast_7, parameter_254, cast_6, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        # pd_op.cast: (-1x512x20x20xf16) <- (-1x512x20x20xf32)
        cast_8 = paddle._C_ops.cast(deformable_conv_1, paddle.float16)

        # pd_op.batch_norm_: (-1x512x20x20xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x20x20xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__300, batch_norm__301, batch_norm__302, batch_norm__303, batch_norm__304, batch_norm__305 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(cast_8, parameter_255, parameter_256, parameter_257, parameter_258, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x20x20xf16) <- (-1x512x20x20xf16)
        relu__46 = paddle._C_ops.relu_(batch_norm__300)

        # pd_op.conv2d: (-1x2048x20x20xf16) <- (-1x512x20x20xf16, 2048x512x1x1xf16)
        conv2d_51 = paddle._C_ops.conv2d(relu__46, parameter_259, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x2048x20x20xf16, 2048xf32, 2048xf32, xf32, xf32, None) <- (-1x2048x20x20xf16, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        batch_norm__306, batch_norm__307, batch_norm__308, batch_norm__309, batch_norm__310, batch_norm__311 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_51, parameter_260, parameter_261, parameter_262, parameter_263, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x2048x20x20xf16) <- (-1x2048x20x20xf16, -1x2048x20x20xf16)
        add__16 = paddle._C_ops.add_(batch_norm__306, relu__44)

        # pd_op.relu_: (-1x2048x20x20xf16) <- (-1x2048x20x20xf16)
        relu__47 = paddle._C_ops.relu_(add__16)

        # pd_op.conv2d: (-1x512x20x20xf16) <- (-1x2048x20x20xf16, 512x2048x1x1xf16)
        conv2d_52 = paddle._C_ops.conv2d(relu__47, parameter_264, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x20x20xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x20x20xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__312, batch_norm__313, batch_norm__314, batch_norm__315, batch_norm__316, batch_norm__317 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_52, parameter_265, parameter_266, parameter_267, parameter_268, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x20x20xf16) <- (-1x512x20x20xf16)
        relu__48 = paddle._C_ops.relu_(batch_norm__312)

        # pd_op.conv2d: (-1x27x20x20xf16) <- (-1x512x20x20xf16, 27x512x3x3xf16)
        conv2d_53 = paddle._C_ops.conv2d(relu__48, parameter_269, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_8 = [1, 27, 1, 1]

        # pd_op.reshape: (1x27x1x1xf16, 0x27xf16) <- (27xf16, 4xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_270, full_int_array_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x27x20x20xf16) <- (-1x27x20x20xf16, 1x27x1x1xf16)
        add__17 = paddle._C_ops.add_(conv2d_53, reshape_4)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_9 = [18, 9]

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x18x20x20xf16, -1x9x20x20xf16]) <- (-1x27x20x20xf16, 2xi64, 1xi32)
        split_2 = paddle._C_ops.split(add__17, full_int_array_9, full_2)

        # builtin.slice: (-1x9x20x20xf16) <- ([-1x18x20x20xf16, -1x9x20x20xf16])
        slice_4 = split_2[1]

        # pd_op.sigmoid_: (-1x9x20x20xf16) <- (-1x9x20x20xf16)
        sigmoid__2 = paddle._C_ops.sigmoid_(slice_4)

        # pd_op.cast: (-1x512x20x20xf32) <- (-1x512x20x20xf16)
        cast_9 = paddle._C_ops.cast(relu__48, paddle.float32)

        # pd_op.cast: (-1x9x20x20xf32) <- (-1x9x20x20xf16)
        cast_10 = paddle._C_ops.cast(sigmoid__2, paddle.float32)

        # builtin.slice: (-1x18x20x20xf16) <- ([-1x18x20x20xf16, -1x9x20x20xf16])
        slice_5 = split_2[0]

        # pd_op.cast: (-1x18x20x20xf32) <- (-1x18x20x20xf16)
        cast_11 = paddle._C_ops.cast(slice_5, paddle.float32)

        # pd_op.deformable_conv: (-1x512x20x20xf32) <- (-1x512x20x20xf32, -1x18x20x20xf32, 512x512x3x3xf32, -1x9x20x20xf32)
        deformable_conv_2 = paddle._C_ops.deformable_conv(cast_9, cast_11, parameter_271, cast_10, [1, 1], [1, 1], [1, 1], 1, 1, 1)

        # pd_op.cast: (-1x512x20x20xf16) <- (-1x512x20x20xf32)
        cast_12 = paddle._C_ops.cast(deformable_conv_2, paddle.float16)

        # pd_op.batch_norm_: (-1x512x20x20xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x20x20xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__318, batch_norm__319, batch_norm__320, batch_norm__321, batch_norm__322, batch_norm__323 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(cast_12, parameter_272, parameter_273, parameter_274, parameter_275, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x20x20xf16) <- (-1x512x20x20xf16)
        relu__49 = paddle._C_ops.relu_(batch_norm__318)

        # pd_op.conv2d: (-1x2048x20x20xf16) <- (-1x512x20x20xf16, 2048x512x1x1xf16)
        conv2d_54 = paddle._C_ops.conv2d(relu__49, parameter_276, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x2048x20x20xf16, 2048xf32, 2048xf32, xf32, xf32, None) <- (-1x2048x20x20xf16, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        batch_norm__324, batch_norm__325, batch_norm__326, batch_norm__327, batch_norm__328, batch_norm__329 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_54, parameter_277, parameter_278, parameter_279, parameter_280, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x2048x20x20xf16) <- (-1x2048x20x20xf16, -1x2048x20x20xf16)
        add__18 = paddle._C_ops.add_(batch_norm__324, relu__47)

        # pd_op.relu_: (-1x2048x20x20xf16) <- (-1x2048x20x20xf16)
        relu__50 = paddle._C_ops.relu_(add__18)

        # pd_op.conv2d: (-1x512x20x20xf16) <- (-1x2048x20x20xf16, 512x2048x1x1xf16)
        conv2d_55 = paddle._C_ops.conv2d(relu__50, parameter_281, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x20x20xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x20x20xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__330, batch_norm__331, batch_norm__332, batch_norm__333, batch_norm__334, batch_norm__335 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_55, parameter_282, parameter_283, parameter_284, parameter_285, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x512x20x20xf16) <- (-1x512x20x20xf16)
        mish_0 = paddle._C_ops.mish(batch_norm__330, float('20'))

        # pd_op.conv2d: (-1x512x20x20xf16) <- (-1x2048x20x20xf16, 512x2048x1x1xf16)
        conv2d_56 = paddle._C_ops.conv2d(relu__50, parameter_286, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x20x20xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x20x20xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__336, batch_norm__337, batch_norm__338, batch_norm__339, batch_norm__340, batch_norm__341 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_56, parameter_287, parameter_288, parameter_289, parameter_290, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x512x20x20xf16) <- (-1x512x20x20xf16)
        mish_1 = paddle._C_ops.mish(batch_norm__336, float('20'))

        # pd_op.conv2d: (-1x512x20x20xf16) <- (-1x512x20x20xf16, 512x512x1x1xf16)
        conv2d_57 = paddle._C_ops.conv2d(mish_0, parameter_291, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x20x20xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x20x20xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__342, batch_norm__343, batch_norm__344, batch_norm__345, batch_norm__346, batch_norm__347 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_57, parameter_292, parameter_293, parameter_294, parameter_295, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x512x20x20xf16) <- (-1x512x20x20xf16)
        mish_2 = paddle._C_ops.mish(batch_norm__342, float('20'))

        # pd_op.conv2d: (-1x512x20x20xf16) <- (-1x512x20x20xf16, 512x512x3x3xf16)
        conv2d_58 = paddle._C_ops.conv2d(mish_2, parameter_296, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x20x20xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x20x20xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__348, batch_norm__349, batch_norm__350, batch_norm__351, batch_norm__352, batch_norm__353 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_58, parameter_297, parameter_298, parameter_299, parameter_300, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x512x20x20xf16) <- (-1x512x20x20xf16)
        mish_3 = paddle._C_ops.mish(batch_norm__348, float('20'))

        # pd_op.conv2d: (-1x512x20x20xf16) <- (-1x512x20x20xf16, 512x512x1x1xf16)
        conv2d_59 = paddle._C_ops.conv2d(mish_3, parameter_301, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x20x20xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x20x20xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__354, batch_norm__355, batch_norm__356, batch_norm__357, batch_norm__358, batch_norm__359 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_59, parameter_302, parameter_303, parameter_304, parameter_305, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x512x20x20xf16) <- (-1x512x20x20xf16)
        mish_4 = paddle._C_ops.mish(batch_norm__354, float('20'))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_10 = [5, 5]

        # pd_op.pool2d: (-1x512x20x20xf16) <- (-1x512x20x20xf16, 2xi64)
        pool2d_4 = paddle._C_ops.pool2d(mish_4, full_int_array_10, [1, 1], [2, 2], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_11 = [9, 9]

        # pd_op.pool2d: (-1x512x20x20xf16) <- (-1x512x20x20xf16, 2xi64)
        pool2d_5 = paddle._C_ops.pool2d(mish_4, full_int_array_11, [1, 1], [4, 4], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_12 = [13, 13]

        # pd_op.pool2d: (-1x512x20x20xf16) <- (-1x512x20x20xf16, 2xi64)
        pool2d_6 = paddle._C_ops.pool2d(mish_4, full_int_array_12, [1, 1], [6, 6], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # builtin.combine: ([-1x512x20x20xf16, -1x512x20x20xf16, -1x512x20x20xf16, -1x512x20x20xf16]) <- (-1x512x20x20xf16, -1x512x20x20xf16, -1x512x20x20xf16, -1x512x20x20xf16)
        combine_0 = [mish_4, pool2d_4, pool2d_5, pool2d_6]

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x2048x20x20xf16) <- ([-1x512x20x20xf16, -1x512x20x20xf16, -1x512x20x20xf16, -1x512x20x20xf16], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_3)

        # pd_op.conv2d: (-1x512x20x20xf16) <- (-1x2048x20x20xf16, 512x2048x1x1xf16)
        conv2d_60 = paddle._C_ops.conv2d(concat_0, parameter_306, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x20x20xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x20x20xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__360, batch_norm__361, batch_norm__362, batch_norm__363, batch_norm__364, batch_norm__365 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_60, parameter_307, parameter_308, parameter_309, parameter_310, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x512x20x20xf16) <- (-1x512x20x20xf16)
        mish_5 = paddle._C_ops.mish(batch_norm__360, float('20'))

        # pd_op.conv2d: (-1x512x20x20xf16) <- (-1x512x20x20xf16, 512x512x1x1xf16)
        conv2d_61 = paddle._C_ops.conv2d(mish_5, parameter_311, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x20x20xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x20x20xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__366, batch_norm__367, batch_norm__368, batch_norm__369, batch_norm__370, batch_norm__371 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_61, parameter_312, parameter_313, parameter_314, parameter_315, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x512x20x20xf16) <- (-1x512x20x20xf16)
        mish_6 = paddle._C_ops.mish(batch_norm__366, float('20'))

        # pd_op.conv2d: (-1x512x20x20xf16) <- (-1x512x20x20xf16, 512x512x3x3xf16)
        conv2d_62 = paddle._C_ops.conv2d(mish_6, parameter_316, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x20x20xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x20x20xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__372, batch_norm__373, batch_norm__374, batch_norm__375, batch_norm__376, batch_norm__377 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_62, parameter_317, parameter_318, parameter_319, parameter_320, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x512x20x20xf16) <- (-1x512x20x20xf16)
        mish_7 = paddle._C_ops.mish(batch_norm__372, float('20'))

        # builtin.combine: ([-1x512x20x20xf16, -1x512x20x20xf16]) <- (-1x512x20x20xf16, -1x512x20x20xf16)
        combine_1 = [mish_7, mish_1]

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024x20x20xf16) <- ([-1x512x20x20xf16, -1x512x20x20xf16], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_4)

        # pd_op.conv2d: (-1x1024x20x20xf16) <- (-1x1024x20x20xf16, 1024x1024x1x1xf16)
        conv2d_63 = paddle._C_ops.conv2d(concat_1, parameter_321, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x20x20xf16, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x20x20xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__378, batch_norm__379, batch_norm__380, batch_norm__381, batch_norm__382, batch_norm__383 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_63, parameter_322, parameter_323, parameter_324, parameter_325, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x1024x20x20xf16) <- (-1x1024x20x20xf16)
        mish_8 = paddle._C_ops.mish(batch_norm__378, float('20'))

        # pd_op.conv2d: (-1x512x20x20xf16) <- (-1x1024x20x20xf16, 512x1024x1x1xf16)
        conv2d_64 = paddle._C_ops.conv2d(mish_8, parameter_326, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x20x20xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x20x20xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__384, batch_norm__385, batch_norm__386, batch_norm__387, batch_norm__388, batch_norm__389 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_64, parameter_327, parameter_328, parameter_329, parameter_330, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x512x20x20xf16) <- (-1x512x20x20xf16)
        mish_9 = paddle._C_ops.mish(batch_norm__384, float('20'))

        # pd_op.nearest_interp: (-1x512x40x40xf16) <- (-1x512x20x20xf16, None, None, None)
        nearest_interp_0 = paddle._C_ops.nearest_interp(mish_9, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 0)

        # builtin.combine: ([-1x512x40x40xf16, -1x1024x40x40xf16]) <- (-1x512x40x40xf16, -1x1024x40x40xf16)
        combine_2 = [nearest_interp_0, relu__41]

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1536x40x40xf16) <- ([-1x512x40x40xf16, -1x1024x40x40xf16], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_5)

        # pd_op.conv2d: (-1x256x40x40xf16) <- (-1x1536x40x40xf16, 256x1536x1x1xf16)
        conv2d_65 = paddle._C_ops.conv2d(concat_2, parameter_331, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x40x40xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x40x40xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__390, batch_norm__391, batch_norm__392, batch_norm__393, batch_norm__394, batch_norm__395 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_65, parameter_332, parameter_333, parameter_334, parameter_335, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x256x40x40xf16) <- (-1x256x40x40xf16)
        mish_10 = paddle._C_ops.mish(batch_norm__390, float('20'))

        # pd_op.conv2d: (-1x256x40x40xf16) <- (-1x1536x40x40xf16, 256x1536x1x1xf16)
        conv2d_66 = paddle._C_ops.conv2d(concat_2, parameter_336, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x40x40xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x40x40xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__396, batch_norm__397, batch_norm__398, batch_norm__399, batch_norm__400, batch_norm__401 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_66, parameter_337, parameter_338, parameter_339, parameter_340, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x256x40x40xf16) <- (-1x256x40x40xf16)
        mish_11 = paddle._C_ops.mish(batch_norm__396, float('20'))

        # pd_op.conv2d: (-1x256x40x40xf16) <- (-1x256x40x40xf16, 256x256x1x1xf16)
        conv2d_67 = paddle._C_ops.conv2d(mish_10, parameter_341, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x40x40xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x40x40xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__402, batch_norm__403, batch_norm__404, batch_norm__405, batch_norm__406, batch_norm__407 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_67, parameter_342, parameter_343, parameter_344, parameter_345, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x256x40x40xf16) <- (-1x256x40x40xf16)
        mish_12 = paddle._C_ops.mish(batch_norm__402, float('20'))

        # pd_op.conv2d: (-1x256x40x40xf16) <- (-1x256x40x40xf16, 256x256x3x3xf16)
        conv2d_68 = paddle._C_ops.conv2d(mish_12, parameter_346, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x40x40xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x40x40xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__408, batch_norm__409, batch_norm__410, batch_norm__411, batch_norm__412, batch_norm__413 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_68, parameter_347, parameter_348, parameter_349, parameter_350, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x256x40x40xf16) <- (-1x256x40x40xf16)
        mish_13 = paddle._C_ops.mish(batch_norm__408, float('20'))

        # pd_op.conv2d: (-1x256x40x40xf16) <- (-1x256x40x40xf16, 256x256x1x1xf16)
        conv2d_69 = paddle._C_ops.conv2d(mish_13, parameter_351, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x40x40xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x40x40xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__414, batch_norm__415, batch_norm__416, batch_norm__417, batch_norm__418, batch_norm__419 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_69, parameter_352, parameter_353, parameter_354, parameter_355, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x256x40x40xf16) <- (-1x256x40x40xf16)
        mish_14 = paddle._C_ops.mish(batch_norm__414, float('20'))

        # pd_op.conv2d: (-1x256x40x40xf16) <- (-1x256x40x40xf16, 256x256x3x3xf16)
        conv2d_70 = paddle._C_ops.conv2d(mish_14, parameter_356, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x40x40xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x40x40xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__420, batch_norm__421, batch_norm__422, batch_norm__423, batch_norm__424, batch_norm__425 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_70, parameter_357, parameter_358, parameter_359, parameter_360, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x256x40x40xf16) <- (-1x256x40x40xf16)
        mish_15 = paddle._C_ops.mish(batch_norm__420, float('20'))

        # pd_op.conv2d: (-1x256x40x40xf16) <- (-1x256x40x40xf16, 256x256x1x1xf16)
        conv2d_71 = paddle._C_ops.conv2d(mish_15, parameter_361, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x40x40xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x40x40xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__426, batch_norm__427, batch_norm__428, batch_norm__429, batch_norm__430, batch_norm__431 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_71, parameter_362, parameter_363, parameter_364, parameter_365, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x256x40x40xf16) <- (-1x256x40x40xf16)
        mish_16 = paddle._C_ops.mish(batch_norm__426, float('20'))

        # pd_op.conv2d: (-1x256x40x40xf16) <- (-1x256x40x40xf16, 256x256x3x3xf16)
        conv2d_72 = paddle._C_ops.conv2d(mish_16, parameter_366, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x40x40xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x40x40xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__432, batch_norm__433, batch_norm__434, batch_norm__435, batch_norm__436, batch_norm__437 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_72, parameter_367, parameter_368, parameter_369, parameter_370, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x256x40x40xf16) <- (-1x256x40x40xf16)
        mish_17 = paddle._C_ops.mish(batch_norm__432, float('20'))

        # builtin.combine: ([-1x256x40x40xf16, -1x256x40x40xf16]) <- (-1x256x40x40xf16, -1x256x40x40xf16)
        combine_3 = [mish_17, mish_11]

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x512x40x40xf16) <- ([-1x256x40x40xf16, -1x256x40x40xf16], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_3, full_6)

        # pd_op.conv2d: (-1x512x40x40xf16) <- (-1x512x40x40xf16, 512x512x1x1xf16)
        conv2d_73 = paddle._C_ops.conv2d(concat_3, parameter_371, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x40x40xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x40x40xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__438, batch_norm__439, batch_norm__440, batch_norm__441, batch_norm__442, batch_norm__443 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_73, parameter_372, parameter_373, parameter_374, parameter_375, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x512x40x40xf16) <- (-1x512x40x40xf16)
        mish_18 = paddle._C_ops.mish(batch_norm__438, float('20'))

        # pd_op.conv2d: (-1x256x40x40xf16) <- (-1x512x40x40xf16, 256x512x1x1xf16)
        conv2d_74 = paddle._C_ops.conv2d(mish_18, parameter_376, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x40x40xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x40x40xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__444, batch_norm__445, batch_norm__446, batch_norm__447, batch_norm__448, batch_norm__449 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_74, parameter_377, parameter_378, parameter_379, parameter_380, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x256x40x40xf16) <- (-1x256x40x40xf16)
        mish_19 = paddle._C_ops.mish(batch_norm__444, float('20'))

        # pd_op.nearest_interp: (-1x256x80x80xf16) <- (-1x256x40x40xf16, None, None, None)
        nearest_interp_1 = paddle._C_ops.nearest_interp(mish_19, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 0)

        # builtin.combine: ([-1x256x80x80xf16, -1x512x80x80xf16]) <- (-1x256x80x80xf16, -1x512x80x80xf16)
        combine_4 = [nearest_interp_1, relu__23]

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x768x80x80xf16) <- ([-1x256x80x80xf16, -1x512x80x80xf16], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_4, full_7)

        # pd_op.conv2d: (-1x128x80x80xf16) <- (-1x768x80x80xf16, 128x768x1x1xf16)
        conv2d_75 = paddle._C_ops.conv2d(concat_4, parameter_381, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x80x80xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x80x80xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__450, batch_norm__451, batch_norm__452, batch_norm__453, batch_norm__454, batch_norm__455 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_75, parameter_382, parameter_383, parameter_384, parameter_385, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x128x80x80xf16) <- (-1x128x80x80xf16)
        mish_20 = paddle._C_ops.mish(batch_norm__450, float('20'))

        # pd_op.conv2d: (-1x128x80x80xf16) <- (-1x768x80x80xf16, 128x768x1x1xf16)
        conv2d_76 = paddle._C_ops.conv2d(concat_4, parameter_386, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x80x80xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x80x80xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__456, batch_norm__457, batch_norm__458, batch_norm__459, batch_norm__460, batch_norm__461 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_76, parameter_387, parameter_388, parameter_389, parameter_390, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x128x80x80xf16) <- (-1x128x80x80xf16)
        mish_21 = paddle._C_ops.mish(batch_norm__456, float('20'))

        # pd_op.conv2d: (-1x128x80x80xf16) <- (-1x128x80x80xf16, 128x128x1x1xf16)
        conv2d_77 = paddle._C_ops.conv2d(mish_20, parameter_391, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x80x80xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x80x80xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__462, batch_norm__463, batch_norm__464, batch_norm__465, batch_norm__466, batch_norm__467 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_77, parameter_392, parameter_393, parameter_394, parameter_395, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x128x80x80xf16) <- (-1x128x80x80xf16)
        mish_22 = paddle._C_ops.mish(batch_norm__462, float('20'))

        # pd_op.conv2d: (-1x128x80x80xf16) <- (-1x128x80x80xf16, 128x128x3x3xf16)
        conv2d_78 = paddle._C_ops.conv2d(mish_22, parameter_396, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x80x80xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x80x80xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__468, batch_norm__469, batch_norm__470, batch_norm__471, batch_norm__472, batch_norm__473 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_78, parameter_397, parameter_398, parameter_399, parameter_400, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x128x80x80xf16) <- (-1x128x80x80xf16)
        mish_23 = paddle._C_ops.mish(batch_norm__468, float('20'))

        # pd_op.conv2d: (-1x128x80x80xf16) <- (-1x128x80x80xf16, 128x128x1x1xf16)
        conv2d_79 = paddle._C_ops.conv2d(mish_23, parameter_401, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x80x80xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x80x80xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__474, batch_norm__475, batch_norm__476, batch_norm__477, batch_norm__478, batch_norm__479 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_79, parameter_402, parameter_403, parameter_404, parameter_405, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x128x80x80xf16) <- (-1x128x80x80xf16)
        mish_24 = paddle._C_ops.mish(batch_norm__474, float('20'))

        # pd_op.conv2d: (-1x128x80x80xf16) <- (-1x128x80x80xf16, 128x128x3x3xf16)
        conv2d_80 = paddle._C_ops.conv2d(mish_24, parameter_406, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x80x80xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x80x80xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__480, batch_norm__481, batch_norm__482, batch_norm__483, batch_norm__484, batch_norm__485 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_80, parameter_407, parameter_408, parameter_409, parameter_410, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x128x80x80xf16) <- (-1x128x80x80xf16)
        mish_25 = paddle._C_ops.mish(batch_norm__480, float('20'))

        # pd_op.conv2d: (-1x128x80x80xf16) <- (-1x128x80x80xf16, 128x128x1x1xf16)
        conv2d_81 = paddle._C_ops.conv2d(mish_25, parameter_411, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x80x80xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x80x80xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__486, batch_norm__487, batch_norm__488, batch_norm__489, batch_norm__490, batch_norm__491 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_81, parameter_412, parameter_413, parameter_414, parameter_415, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x128x80x80xf16) <- (-1x128x80x80xf16)
        mish_26 = paddle._C_ops.mish(batch_norm__486, float('20'))

        # pd_op.conv2d: (-1x128x80x80xf16) <- (-1x128x80x80xf16, 128x128x3x3xf16)
        conv2d_82 = paddle._C_ops.conv2d(mish_26, parameter_416, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x80x80xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x80x80xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__492, batch_norm__493, batch_norm__494, batch_norm__495, batch_norm__496, batch_norm__497 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_82, parameter_417, parameter_418, parameter_419, parameter_420, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x128x80x80xf16) <- (-1x128x80x80xf16)
        mish_27 = paddle._C_ops.mish(batch_norm__492, float('20'))

        # builtin.combine: ([-1x128x80x80xf16, -1x128x80x80xf16]) <- (-1x128x80x80xf16, -1x128x80x80xf16)
        combine_5 = [mish_27, mish_21]

        # pd_op.full: (1xi32) <- ()
        full_8 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x256x80x80xf16) <- ([-1x128x80x80xf16, -1x128x80x80xf16], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_5, full_8)

        # pd_op.conv2d: (-1x256x80x80xf16) <- (-1x256x80x80xf16, 256x256x1x1xf16)
        conv2d_83 = paddle._C_ops.conv2d(concat_5, parameter_421, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x80x80xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x80x80xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__498, batch_norm__499, batch_norm__500, batch_norm__501, batch_norm__502, batch_norm__503 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_83, parameter_422, parameter_423, parameter_424, parameter_425, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x256x80x80xf16) <- (-1x256x80x80xf16)
        mish_28 = paddle._C_ops.mish(batch_norm__498, float('20'))

        # pd_op.conv2d: (-1x256x40x40xf16) <- (-1x256x80x80xf16, 256x256x3x3xf16)
        conv2d_84 = paddle._C_ops.conv2d(mish_28, parameter_426, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x40x40xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x40x40xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__504, batch_norm__505, batch_norm__506, batch_norm__507, batch_norm__508, batch_norm__509 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_84, parameter_427, parameter_428, parameter_429, parameter_430, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x256x40x40xf16) <- (-1x256x40x40xf16)
        mish_29 = paddle._C_ops.mish(batch_norm__504, float('20'))

        # builtin.combine: ([-1x256x40x40xf16, -1x512x40x40xf16]) <- (-1x256x40x40xf16, -1x512x40x40xf16)
        combine_6 = [mish_29, mish_18]

        # pd_op.full: (1xi32) <- ()
        full_9 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x768x40x40xf16) <- ([-1x256x40x40xf16, -1x512x40x40xf16], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_6, full_9)

        # pd_op.conv2d: (-1x256x40x40xf16) <- (-1x768x40x40xf16, 256x768x1x1xf16)
        conv2d_85 = paddle._C_ops.conv2d(concat_6, parameter_431, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x40x40xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x40x40xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__510, batch_norm__511, batch_norm__512, batch_norm__513, batch_norm__514, batch_norm__515 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_85, parameter_432, parameter_433, parameter_434, parameter_435, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x256x40x40xf16) <- (-1x256x40x40xf16)
        mish_30 = paddle._C_ops.mish(batch_norm__510, float('20'))

        # pd_op.conv2d: (-1x256x40x40xf16) <- (-1x768x40x40xf16, 256x768x1x1xf16)
        conv2d_86 = paddle._C_ops.conv2d(concat_6, parameter_436, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x40x40xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x40x40xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__516, batch_norm__517, batch_norm__518, batch_norm__519, batch_norm__520, batch_norm__521 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_86, parameter_437, parameter_438, parameter_439, parameter_440, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x256x40x40xf16) <- (-1x256x40x40xf16)
        mish_31 = paddle._C_ops.mish(batch_norm__516, float('20'))

        # pd_op.conv2d: (-1x256x40x40xf16) <- (-1x256x40x40xf16, 256x256x1x1xf16)
        conv2d_87 = paddle._C_ops.conv2d(mish_30, parameter_441, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x40x40xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x40x40xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__522, batch_norm__523, batch_norm__524, batch_norm__525, batch_norm__526, batch_norm__527 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_87, parameter_442, parameter_443, parameter_444, parameter_445, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x256x40x40xf16) <- (-1x256x40x40xf16)
        mish_32 = paddle._C_ops.mish(batch_norm__522, float('20'))

        # pd_op.conv2d: (-1x256x40x40xf16) <- (-1x256x40x40xf16, 256x256x3x3xf16)
        conv2d_88 = paddle._C_ops.conv2d(mish_32, parameter_446, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x40x40xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x40x40xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__528, batch_norm__529, batch_norm__530, batch_norm__531, batch_norm__532, batch_norm__533 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_88, parameter_447, parameter_448, parameter_449, parameter_450, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x256x40x40xf16) <- (-1x256x40x40xf16)
        mish_33 = paddle._C_ops.mish(batch_norm__528, float('20'))

        # pd_op.conv2d: (-1x256x40x40xf16) <- (-1x256x40x40xf16, 256x256x1x1xf16)
        conv2d_89 = paddle._C_ops.conv2d(mish_33, parameter_451, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x40x40xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x40x40xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__534, batch_norm__535, batch_norm__536, batch_norm__537, batch_norm__538, batch_norm__539 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_89, parameter_452, parameter_453, parameter_454, parameter_455, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x256x40x40xf16) <- (-1x256x40x40xf16)
        mish_34 = paddle._C_ops.mish(batch_norm__534, float('20'))

        # pd_op.conv2d: (-1x256x40x40xf16) <- (-1x256x40x40xf16, 256x256x3x3xf16)
        conv2d_90 = paddle._C_ops.conv2d(mish_34, parameter_456, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x40x40xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x40x40xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__540, batch_norm__541, batch_norm__542, batch_norm__543, batch_norm__544, batch_norm__545 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_90, parameter_457, parameter_458, parameter_459, parameter_460, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x256x40x40xf16) <- (-1x256x40x40xf16)
        mish_35 = paddle._C_ops.mish(batch_norm__540, float('20'))

        # pd_op.conv2d: (-1x256x40x40xf16) <- (-1x256x40x40xf16, 256x256x1x1xf16)
        conv2d_91 = paddle._C_ops.conv2d(mish_35, parameter_461, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x40x40xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x40x40xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__546, batch_norm__547, batch_norm__548, batch_norm__549, batch_norm__550, batch_norm__551 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_91, parameter_462, parameter_463, parameter_464, parameter_465, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x256x40x40xf16) <- (-1x256x40x40xf16)
        mish_36 = paddle._C_ops.mish(batch_norm__546, float('20'))

        # pd_op.conv2d: (-1x256x40x40xf16) <- (-1x256x40x40xf16, 256x256x3x3xf16)
        conv2d_92 = paddle._C_ops.conv2d(mish_36, parameter_466, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x40x40xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x40x40xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__552, batch_norm__553, batch_norm__554, batch_norm__555, batch_norm__556, batch_norm__557 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_92, parameter_467, parameter_468, parameter_469, parameter_470, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x256x40x40xf16) <- (-1x256x40x40xf16)
        mish_37 = paddle._C_ops.mish(batch_norm__552, float('20'))

        # builtin.combine: ([-1x256x40x40xf16, -1x256x40x40xf16]) <- (-1x256x40x40xf16, -1x256x40x40xf16)
        combine_7 = [mish_37, mish_31]

        # pd_op.full: (1xi32) <- ()
        full_10 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x512x40x40xf16) <- ([-1x256x40x40xf16, -1x256x40x40xf16], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_7, full_10)

        # pd_op.conv2d: (-1x512x40x40xf16) <- (-1x512x40x40xf16, 512x512x1x1xf16)
        conv2d_93 = paddle._C_ops.conv2d(concat_7, parameter_471, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x40x40xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x40x40xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__558, batch_norm__559, batch_norm__560, batch_norm__561, batch_norm__562, batch_norm__563 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_93, parameter_472, parameter_473, parameter_474, parameter_475, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x512x40x40xf16) <- (-1x512x40x40xf16)
        mish_38 = paddle._C_ops.mish(batch_norm__558, float('20'))

        # pd_op.conv2d: (-1x512x20x20xf16) <- (-1x512x40x40xf16, 512x512x3x3xf16)
        conv2d_94 = paddle._C_ops.conv2d(mish_38, parameter_476, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x20x20xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x20x20xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__564, batch_norm__565, batch_norm__566, batch_norm__567, batch_norm__568, batch_norm__569 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_94, parameter_477, parameter_478, parameter_479, parameter_480, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x512x20x20xf16) <- (-1x512x20x20xf16)
        mish_39 = paddle._C_ops.mish(batch_norm__564, float('20'))

        # builtin.combine: ([-1x512x20x20xf16, -1x1024x20x20xf16]) <- (-1x512x20x20xf16, -1x1024x20x20xf16)
        combine_8 = [mish_39, mish_8]

        # pd_op.full: (1xi32) <- ()
        full_11 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1536x20x20xf16) <- ([-1x512x20x20xf16, -1x1024x20x20xf16], 1xi32)
        concat_8 = paddle._C_ops.concat(combine_8, full_11)

        # pd_op.conv2d: (-1x512x20x20xf16) <- (-1x1536x20x20xf16, 512x1536x1x1xf16)
        conv2d_95 = paddle._C_ops.conv2d(concat_8, parameter_481, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x20x20xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x20x20xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__570, batch_norm__571, batch_norm__572, batch_norm__573, batch_norm__574, batch_norm__575 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_95, parameter_482, parameter_483, parameter_484, parameter_485, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x512x20x20xf16) <- (-1x512x20x20xf16)
        mish_40 = paddle._C_ops.mish(batch_norm__570, float('20'))

        # pd_op.conv2d: (-1x512x20x20xf16) <- (-1x1536x20x20xf16, 512x1536x1x1xf16)
        conv2d_96 = paddle._C_ops.conv2d(concat_8, parameter_486, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x20x20xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x20x20xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__576, batch_norm__577, batch_norm__578, batch_norm__579, batch_norm__580, batch_norm__581 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_96, parameter_487, parameter_488, parameter_489, parameter_490, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x512x20x20xf16) <- (-1x512x20x20xf16)
        mish_41 = paddle._C_ops.mish(batch_norm__576, float('20'))

        # pd_op.conv2d: (-1x512x20x20xf16) <- (-1x512x20x20xf16, 512x512x1x1xf16)
        conv2d_97 = paddle._C_ops.conv2d(mish_40, parameter_491, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x20x20xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x20x20xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__582, batch_norm__583, batch_norm__584, batch_norm__585, batch_norm__586, batch_norm__587 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_97, parameter_492, parameter_493, parameter_494, parameter_495, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x512x20x20xf16) <- (-1x512x20x20xf16)
        mish_42 = paddle._C_ops.mish(batch_norm__582, float('20'))

        # pd_op.conv2d: (-1x512x20x20xf16) <- (-1x512x20x20xf16, 512x512x3x3xf16)
        conv2d_98 = paddle._C_ops.conv2d(mish_42, parameter_496, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x20x20xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x20x20xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__588, batch_norm__589, batch_norm__590, batch_norm__591, batch_norm__592, batch_norm__593 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_98, parameter_497, parameter_498, parameter_499, parameter_500, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x512x20x20xf16) <- (-1x512x20x20xf16)
        mish_43 = paddle._C_ops.mish(batch_norm__588, float('20'))

        # pd_op.conv2d: (-1x512x20x20xf16) <- (-1x512x20x20xf16, 512x512x1x1xf16)
        conv2d_99 = paddle._C_ops.conv2d(mish_43, parameter_501, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x20x20xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x20x20xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__594, batch_norm__595, batch_norm__596, batch_norm__597, batch_norm__598, batch_norm__599 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_99, parameter_502, parameter_503, parameter_504, parameter_505, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x512x20x20xf16) <- (-1x512x20x20xf16)
        mish_44 = paddle._C_ops.mish(batch_norm__594, float('20'))

        # pd_op.conv2d: (-1x512x20x20xf16) <- (-1x512x20x20xf16, 512x512x3x3xf16)
        conv2d_100 = paddle._C_ops.conv2d(mish_44, parameter_506, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x20x20xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x20x20xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__600, batch_norm__601, batch_norm__602, batch_norm__603, batch_norm__604, batch_norm__605 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_100, parameter_507, parameter_508, parameter_509, parameter_510, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x512x20x20xf16) <- (-1x512x20x20xf16)
        mish_45 = paddle._C_ops.mish(batch_norm__600, float('20'))

        # pd_op.conv2d: (-1x512x20x20xf16) <- (-1x512x20x20xf16, 512x512x1x1xf16)
        conv2d_101 = paddle._C_ops.conv2d(mish_45, parameter_511, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x20x20xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x20x20xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__606, batch_norm__607, batch_norm__608, batch_norm__609, batch_norm__610, batch_norm__611 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_101, parameter_512, parameter_513, parameter_514, parameter_515, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x512x20x20xf16) <- (-1x512x20x20xf16)
        mish_46 = paddle._C_ops.mish(batch_norm__606, float('20'))

        # pd_op.conv2d: (-1x512x20x20xf16) <- (-1x512x20x20xf16, 512x512x3x3xf16)
        conv2d_102 = paddle._C_ops.conv2d(mish_46, parameter_516, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x20x20xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x20x20xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__612, batch_norm__613, batch_norm__614, batch_norm__615, batch_norm__616, batch_norm__617 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_102, parameter_517, parameter_518, parameter_519, parameter_520, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x512x20x20xf16) <- (-1x512x20x20xf16)
        mish_47 = paddle._C_ops.mish(batch_norm__612, float('20'))

        # builtin.combine: ([-1x512x20x20xf16, -1x512x20x20xf16]) <- (-1x512x20x20xf16, -1x512x20x20xf16)
        combine_9 = [mish_47, mish_41]

        # pd_op.full: (1xi32) <- ()
        full_12 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024x20x20xf16) <- ([-1x512x20x20xf16, -1x512x20x20xf16], 1xi32)
        concat_9 = paddle._C_ops.concat(combine_9, full_12)

        # pd_op.conv2d: (-1x1024x20x20xf16) <- (-1x1024x20x20xf16, 1024x1024x1x1xf16)
        conv2d_103 = paddle._C_ops.conv2d(concat_9, parameter_521, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x20x20xf16, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x20x20xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__618, batch_norm__619, batch_norm__620, batch_norm__621, batch_norm__622, batch_norm__623 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_103, parameter_522, parameter_523, parameter_524, parameter_525, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.mish: (-1x1024x20x20xf16) <- (-1x1024x20x20xf16)
        mish_48 = paddle._C_ops.mish(batch_norm__618, float('20'))

        # pd_op.conv2d: (-1x258x20x20xf16) <- (-1x1024x20x20xf16, 258x1024x1x1xf16)
        conv2d_104 = paddle._C_ops.conv2d(mish_48, parameter_526, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_13 = [1, 258, 1, 1]

        # pd_op.reshape: (1x258x1x1xf16, 0x258xf16) <- (258xf16, 4xi64)
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_527, full_int_array_13), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x258x20x20xf16) <- (-1x258x20x20xf16, 1x258x1x1xf16)
        add__19 = paddle._C_ops.add_(conv2d_104, reshape_6)

        # pd_op.conv2d: (-1x258x40x40xf16) <- (-1x512x40x40xf16, 258x512x1x1xf16)
        conv2d_105 = paddle._C_ops.conv2d(mish_38, parameter_528, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_14 = [1, 258, 1, 1]

        # pd_op.reshape: (1x258x1x1xf16, 0x258xf16) <- (258xf16, 4xi64)
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_529, full_int_array_14), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x258x40x40xf16) <- (-1x258x40x40xf16, 1x258x1x1xf16)
        add__20 = paddle._C_ops.add_(conv2d_105, reshape_8)

        # pd_op.conv2d: (-1x258x80x80xf16) <- (-1x256x80x80xf16, 258x256x1x1xf16)
        conv2d_106 = paddle._C_ops.conv2d(mish_28, parameter_530, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_15 = [1, 258, 1, 1]

        # pd_op.reshape: (1x258x1x1xf16, 0x258xf16) <- (258xf16, 4xi64)
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_531, full_int_array_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x258x80x80xf16) <- (-1x258x80x80xf16, 1x258x1x1xf16)
        add__21 = paddle._C_ops.add_(conv2d_106, reshape_10)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_16 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_17 = [3]

        # pd_op.slice: (-1x3x20x20xf16) <- (-1x258x20x20xf16, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(add__19, [1], full_int_array_16, full_int_array_17, [1], [])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_18 = [3]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_19 = [258]

        # pd_op.slice: (-1x255x20x20xf16) <- (-1x258x20x20xf16, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(add__19, [1], full_int_array_18, full_int_array_19, [1], [])

        # pd_op.shape: (4xi32) <- (-1x255x20x20xf16)
        shape_0 = paddle._C_ops.shape(paddle.cast(slice_7, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_20 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_21 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(shape_0, [0], full_int_array_20, full_int_array_21, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_13 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_14 = paddle._C_ops.full([1], float('85'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_15 = paddle._C_ops.full([1], float('400'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_10 = [slice_8, full_13, full_14, full_15]

        # pd_op.reshape_: (-1x3x85x400xf16, 0x-1x255x20x20xf16) <- (-1x255x20x20xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(slice_7, [x.reshape([]) for x in combine_10]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_16 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_17 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_18 = paddle._C_ops.full([1], float('400'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_11 = [slice_8, full_16, full_17, full_18]

        # pd_op.reshape_: (-1x3x1x400xf16, 0x-1x3x20x20xf16) <- (-1x3x20x20xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape_(slice_6, [x.reshape([]) for x in combine_11]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_22 = [4]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_23 = [5]

        # pd_op.slice: (-1x3x1x400xf16) <- (-1x3x85x400xf16, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(reshape__0, [2], full_int_array_22, full_int_array_23, [1], [])

        # pd_op.sigmoid_: (-1x3x1x400xf16) <- (-1x3x1x400xf16)
        sigmoid__3 = paddle._C_ops.sigmoid_(reshape__2)

        # pd_op.sigmoid_: (-1x3x1x400xf16) <- (-1x3x1x400xf16)
        sigmoid__4 = paddle._C_ops.sigmoid_(slice_9)

        # pd_op.full: (xf16) <- ()
        full_19 = paddle._C_ops.full([], float('0.5'), paddle.float16, paddle.framework._current_expected_place())

        # pd_op.elementwise_pow: (-1x3x1x400xf16) <- (-1x3x1x400xf16, xf16)
        elementwise_pow_0 = paddle._C_ops.elementwise_pow(sigmoid__4, full_19)

        # pd_op.full: (xf16) <- ()
        full_20 = paddle._C_ops.full([], float('0.5'), paddle.float16, paddle.framework._current_expected_place())

        # pd_op.elementwise_pow: (-1x3x1x400xf16) <- (-1x3x1x400xf16, xf16)
        elementwise_pow_1 = paddle._C_ops.elementwise_pow(sigmoid__3, full_20)

        # pd_op.multiply_: (-1x3x1x400xf16) <- (-1x3x1x400xf16, -1x3x1x400xf16)
        multiply__0 = paddle._C_ops.multiply_(elementwise_pow_0, elementwise_pow_1)

        # pd_op.full: (1xf32) <- ()
        full_21 = paddle._C_ops.full([1], float('1e-07'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_22 = paddle._C_ops.full([1], float('1e+07'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.clip_: (-1x3x1x400xf16) <- (-1x3x1x400xf16, 1xf32, 1xf32)
        clip__0 = paddle._C_ops.clip_(multiply__0, full_21, full_22)

        # pd_op.full_batch_size_like: (-1x3x1x400xf16) <- (-1x3x1x400xf16)
        full_batch_size_like_0 = paddle._C_ops.full_batch_size_like(clip__0, [-1, 3, 1, 400], paddle.float16, float('1'), 0, 0, paddle.framework._current_expected_place())

        # pd_op.divide_: (-1x3x1x400xf16) <- (-1x3x1x400xf16, -1x3x1x400xf16)
        divide__0 = paddle._C_ops.divide_(full_batch_size_like_0, clip__0)

        # pd_op.full: (1xf32) <- ()
        full_23 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x3x1x400xf16) <- (-1x3x1x400xf16, 1xf32)
        scale__0 = paddle._C_ops.scale_(divide__0, full_23, float('-1'), True)

        # pd_op.full: (1xf32) <- ()
        full_24 = paddle._C_ops.full([1], float('1e-07'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_25 = paddle._C_ops.full([1], float('1e+07'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.clip_: (-1x3x1x400xf16) <- (-1x3x1x400xf16, 1xf32, 1xf32)
        clip__1 = paddle._C_ops.clip_(scale__0, full_24, full_25)

        # pd_op.cast: (-1x3x1x400xf32) <- (-1x3x1x400xf16)
        cast_13 = paddle._C_ops.cast(clip__1, paddle.float32)

        # pd_op.log_: (-1x3x1x400xf32) <- (-1x3x1x400xf32)
        log__0 = paddle._C_ops.log_(cast_13)

        # pd_op.cast: (-1x3x1x400xf16) <- (-1x3x1x400xf32)
        cast_14 = paddle._C_ops.cast(log__0, paddle.float16)

        # pd_op.full: (1xf32) <- ()
        full_26 = paddle._C_ops.full([1], float('-1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x3x1x400xf16) <- (-1x3x1x400xf16, 1xf32)
        scale__1 = paddle._C_ops.scale_(cast_14, full_26, float('0'), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_24 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_25 = [4]

        # pd_op.slice: (-1x3x4x400xf16) <- (-1x3x85x400xf16, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(reshape__0, [2], full_int_array_24, full_int_array_25, [1], [])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_26 = [5]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_27 = [85]

        # pd_op.slice: (-1x3x80x400xf16) <- (-1x3x85x400xf16, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(reshape__0, [2], full_int_array_26, full_int_array_27, [1], [])

        # builtin.combine: ([-1x3x4x400xf16, -1x3x1x400xf16, -1x3x80x400xf16]) <- (-1x3x4x400xf16, -1x3x1x400xf16, -1x3x80x400xf16)
        combine_12 = [slice_10, scale__1, slice_11]

        # pd_op.full: (1xi32) <- ()
        full_27 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x3x85x400xf16) <- ([-1x3x4x400xf16, -1x3x1x400xf16, -1x3x80x400xf16], 1xi32)
        concat_10 = paddle._C_ops.concat(combine_12, full_27)

        # pd_op.full: (1xi32) <- ()
        full_28 = paddle._C_ops.full([1], float('255'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_29 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_30 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_13 = [slice_8, full_28, full_29, full_30]

        # pd_op.reshape_: (-1x255x20x20xf16, 0x-1x3x85x400xf16) <- (-1x3x85x400xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_10, [x.reshape([]) for x in combine_13]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_28 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_29 = [3]

        # pd_op.slice: (-1x3x40x40xf16) <- (-1x258x40x40xf16, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(add__20, [1], full_int_array_28, full_int_array_29, [1], [])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_30 = [3]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_31 = [258]

        # pd_op.slice: (-1x255x40x40xf16) <- (-1x258x40x40xf16, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(add__20, [1], full_int_array_30, full_int_array_31, [1], [])

        # pd_op.shape: (4xi32) <- (-1x255x40x40xf16)
        shape_1 = paddle._C_ops.shape(paddle.cast(slice_13, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_32 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_33 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(shape_1, [0], full_int_array_32, full_int_array_33, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_31 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_32 = paddle._C_ops.full([1], float('85'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_33 = paddle._C_ops.full([1], float('1600'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_14 = [slice_14, full_31, full_32, full_33]

        # pd_op.reshape_: (-1x3x85x1600xf16, 0x-1x255x40x40xf16) <- (-1x255x40x40xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape_(slice_13, [x.reshape([]) for x in combine_14]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_34 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_35 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_36 = paddle._C_ops.full([1], float('1600'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_15 = [slice_14, full_34, full_35, full_36]

        # pd_op.reshape_: (-1x3x1x1600xf16, 0x-1x3x40x40xf16) <- (-1x3x40x40xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape_(slice_12, [x.reshape([]) for x in combine_15]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_34 = [4]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_35 = [5]

        # pd_op.slice: (-1x3x1x1600xf16) <- (-1x3x85x1600xf16, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(reshape__6, [2], full_int_array_34, full_int_array_35, [1], [])

        # pd_op.sigmoid_: (-1x3x1x1600xf16) <- (-1x3x1x1600xf16)
        sigmoid__5 = paddle._C_ops.sigmoid_(reshape__8)

        # pd_op.sigmoid_: (-1x3x1x1600xf16) <- (-1x3x1x1600xf16)
        sigmoid__6 = paddle._C_ops.sigmoid_(slice_15)

        # pd_op.full: (xf16) <- ()
        full_37 = paddle._C_ops.full([], float('0.5'), paddle.float16, paddle.framework._current_expected_place())

        # pd_op.elementwise_pow: (-1x3x1x1600xf16) <- (-1x3x1x1600xf16, xf16)
        elementwise_pow_2 = paddle._C_ops.elementwise_pow(sigmoid__6, full_37)

        # pd_op.full: (xf16) <- ()
        full_38 = paddle._C_ops.full([], float('0.5'), paddle.float16, paddle.framework._current_expected_place())

        # pd_op.elementwise_pow: (-1x3x1x1600xf16) <- (-1x3x1x1600xf16, xf16)
        elementwise_pow_3 = paddle._C_ops.elementwise_pow(sigmoid__5, full_38)

        # pd_op.multiply_: (-1x3x1x1600xf16) <- (-1x3x1x1600xf16, -1x3x1x1600xf16)
        multiply__1 = paddle._C_ops.multiply_(elementwise_pow_2, elementwise_pow_3)

        # pd_op.full: (1xf32) <- ()
        full_39 = paddle._C_ops.full([1], float('1e-07'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_40 = paddle._C_ops.full([1], float('1e+07'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.clip_: (-1x3x1x1600xf16) <- (-1x3x1x1600xf16, 1xf32, 1xf32)
        clip__2 = paddle._C_ops.clip_(multiply__1, full_39, full_40)

        # pd_op.full_batch_size_like: (-1x3x1x1600xf16) <- (-1x3x1x1600xf16)
        full_batch_size_like_1 = paddle._C_ops.full_batch_size_like(clip__2, [-1, 3, 1, 1600], paddle.float16, float('1'), 0, 0, paddle.framework._current_expected_place())

        # pd_op.divide_: (-1x3x1x1600xf16) <- (-1x3x1x1600xf16, -1x3x1x1600xf16)
        divide__1 = paddle._C_ops.divide_(full_batch_size_like_1, clip__2)

        # pd_op.full: (1xf32) <- ()
        full_41 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x3x1x1600xf16) <- (-1x3x1x1600xf16, 1xf32)
        scale__2 = paddle._C_ops.scale_(divide__1, full_41, float('-1'), True)

        # pd_op.full: (1xf32) <- ()
        full_42 = paddle._C_ops.full([1], float('1e-07'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_43 = paddle._C_ops.full([1], float('1e+07'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.clip_: (-1x3x1x1600xf16) <- (-1x3x1x1600xf16, 1xf32, 1xf32)
        clip__3 = paddle._C_ops.clip_(scale__2, full_42, full_43)

        # pd_op.cast: (-1x3x1x1600xf32) <- (-1x3x1x1600xf16)
        cast_15 = paddle._C_ops.cast(clip__3, paddle.float32)

        # pd_op.log_: (-1x3x1x1600xf32) <- (-1x3x1x1600xf32)
        log__1 = paddle._C_ops.log_(cast_15)

        # pd_op.cast: (-1x3x1x1600xf16) <- (-1x3x1x1600xf32)
        cast_16 = paddle._C_ops.cast(log__1, paddle.float16)

        # pd_op.full: (1xf32) <- ()
        full_44 = paddle._C_ops.full([1], float('-1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x3x1x1600xf16) <- (-1x3x1x1600xf16, 1xf32)
        scale__3 = paddle._C_ops.scale_(cast_16, full_44, float('0'), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_36 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_37 = [4]

        # pd_op.slice: (-1x3x4x1600xf16) <- (-1x3x85x1600xf16, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(reshape__6, [2], full_int_array_36, full_int_array_37, [1], [])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_38 = [5]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_39 = [85]

        # pd_op.slice: (-1x3x80x1600xf16) <- (-1x3x85x1600xf16, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(reshape__6, [2], full_int_array_38, full_int_array_39, [1], [])

        # builtin.combine: ([-1x3x4x1600xf16, -1x3x1x1600xf16, -1x3x80x1600xf16]) <- (-1x3x4x1600xf16, -1x3x1x1600xf16, -1x3x80x1600xf16)
        combine_16 = [slice_16, scale__3, slice_17]

        # pd_op.full: (1xi32) <- ()
        full_45 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x3x85x1600xf16) <- ([-1x3x4x1600xf16, -1x3x1x1600xf16, -1x3x80x1600xf16], 1xi32)
        concat_11 = paddle._C_ops.concat(combine_16, full_45)

        # pd_op.full: (1xi32) <- ()
        full_46 = paddle._C_ops.full([1], float('255'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_47 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_48 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_17 = [slice_14, full_46, full_47, full_48]

        # pd_op.reshape_: (-1x255x40x40xf16, 0x-1x3x85x1600xf16) <- (-1x3x85x1600xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_11, [x.reshape([]) for x in combine_17]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_40 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_41 = [3]

        # pd_op.slice: (-1x3x80x80xf16) <- (-1x258x80x80xf16, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(add__21, [1], full_int_array_40, full_int_array_41, [1], [])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_42 = [3]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_43 = [258]

        # pd_op.slice: (-1x255x80x80xf16) <- (-1x258x80x80xf16, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(add__21, [1], full_int_array_42, full_int_array_43, [1], [])

        # pd_op.shape: (4xi32) <- (-1x255x80x80xf16)
        shape_2 = paddle._C_ops.shape(paddle.cast(slice_19, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_44 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_45 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(shape_2, [0], full_int_array_44, full_int_array_45, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_49 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_50 = paddle._C_ops.full([1], float('85'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_51 = paddle._C_ops.full([1], float('6400'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_18 = [slice_20, full_49, full_50, full_51]

        # pd_op.reshape_: (-1x3x85x6400xf16, 0x-1x255x80x80xf16) <- (-1x255x80x80xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape_(slice_19, [x.reshape([]) for x in combine_18]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_52 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_53 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_54 = paddle._C_ops.full([1], float('6400'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_19 = [slice_20, full_52, full_53, full_54]

        # pd_op.reshape_: (-1x3x1x6400xf16, 0x-1x3x80x80xf16) <- (-1x3x80x80xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape_(slice_18, [x.reshape([]) for x in combine_19]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_46 = [4]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_47 = [5]

        # pd_op.slice: (-1x3x1x6400xf16) <- (-1x3x85x6400xf16, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(reshape__12, [2], full_int_array_46, full_int_array_47, [1], [])

        # pd_op.sigmoid_: (-1x3x1x6400xf16) <- (-1x3x1x6400xf16)
        sigmoid__7 = paddle._C_ops.sigmoid_(reshape__14)

        # pd_op.sigmoid_: (-1x3x1x6400xf16) <- (-1x3x1x6400xf16)
        sigmoid__8 = paddle._C_ops.sigmoid_(slice_21)

        # pd_op.full: (xf16) <- ()
        full_55 = paddle._C_ops.full([], float('0.5'), paddle.float16, paddle.framework._current_expected_place())

        # pd_op.elementwise_pow: (-1x3x1x6400xf16) <- (-1x3x1x6400xf16, xf16)
        elementwise_pow_4 = paddle._C_ops.elementwise_pow(sigmoid__8, full_55)

        # pd_op.full: (xf16) <- ()
        full_56 = paddle._C_ops.full([], float('0.5'), paddle.float16, paddle.framework._current_expected_place())

        # pd_op.elementwise_pow: (-1x3x1x6400xf16) <- (-1x3x1x6400xf16, xf16)
        elementwise_pow_5 = paddle._C_ops.elementwise_pow(sigmoid__7, full_56)

        # pd_op.multiply_: (-1x3x1x6400xf16) <- (-1x3x1x6400xf16, -1x3x1x6400xf16)
        multiply__2 = paddle._C_ops.multiply_(elementwise_pow_4, elementwise_pow_5)

        # pd_op.full: (1xf32) <- ()
        full_57 = paddle._C_ops.full([1], float('1e-07'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_58 = paddle._C_ops.full([1], float('1e+07'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.clip_: (-1x3x1x6400xf16) <- (-1x3x1x6400xf16, 1xf32, 1xf32)
        clip__4 = paddle._C_ops.clip_(multiply__2, full_57, full_58)

        # pd_op.full_batch_size_like: (-1x3x1x6400xf16) <- (-1x3x1x6400xf16)
        full_batch_size_like_2 = paddle._C_ops.full_batch_size_like(clip__4, [-1, 3, 1, 6400], paddle.float16, float('1'), 0, 0, paddle.framework._current_expected_place())

        # pd_op.divide_: (-1x3x1x6400xf16) <- (-1x3x1x6400xf16, -1x3x1x6400xf16)
        divide__2 = paddle._C_ops.divide_(full_batch_size_like_2, clip__4)

        # pd_op.full: (1xf32) <- ()
        full_59 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x3x1x6400xf16) <- (-1x3x1x6400xf16, 1xf32)
        scale__4 = paddle._C_ops.scale_(divide__2, full_59, float('-1'), True)

        # pd_op.full: (1xf32) <- ()
        full_60 = paddle._C_ops.full([1], float('1e-07'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.full: (1xf32) <- ()
        full_61 = paddle._C_ops.full([1], float('1e+07'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.clip_: (-1x3x1x6400xf16) <- (-1x3x1x6400xf16, 1xf32, 1xf32)
        clip__5 = paddle._C_ops.clip_(scale__4, full_60, full_61)

        # pd_op.cast: (-1x3x1x6400xf32) <- (-1x3x1x6400xf16)
        cast_17 = paddle._C_ops.cast(clip__5, paddle.float32)

        # pd_op.log_: (-1x3x1x6400xf32) <- (-1x3x1x6400xf32)
        log__2 = paddle._C_ops.log_(cast_17)

        # pd_op.cast: (-1x3x1x6400xf16) <- (-1x3x1x6400xf32)
        cast_18 = paddle._C_ops.cast(log__2, paddle.float16)

        # pd_op.full: (1xf32) <- ()
        full_62 = paddle._C_ops.full([1], float('-1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x3x1x6400xf16) <- (-1x3x1x6400xf16, 1xf32)
        scale__5 = paddle._C_ops.scale_(cast_18, full_62, float('0'), True)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_48 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_49 = [4]

        # pd_op.slice: (-1x3x4x6400xf16) <- (-1x3x85x6400xf16, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(reshape__12, [2], full_int_array_48, full_int_array_49, [1], [])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_50 = [5]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_51 = [85]

        # pd_op.slice: (-1x3x80x6400xf16) <- (-1x3x85x6400xf16, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(reshape__12, [2], full_int_array_50, full_int_array_51, [1], [])

        # builtin.combine: ([-1x3x4x6400xf16, -1x3x1x6400xf16, -1x3x80x6400xf16]) <- (-1x3x4x6400xf16, -1x3x1x6400xf16, -1x3x80x6400xf16)
        combine_20 = [slice_22, scale__5, slice_23]

        # pd_op.full: (1xi32) <- ()
        full_63 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x3x85x6400xf16) <- ([-1x3x4x6400xf16, -1x3x1x6400xf16, -1x3x80x6400xf16], 1xi32)
        concat_12 = paddle._C_ops.concat(combine_20, full_63)

        # pd_op.full: (1xi32) <- ()
        full_64 = paddle._C_ops.full([1], float('255'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_65 = paddle._C_ops.full([1], float('80'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_66 = paddle._C_ops.full([1], float('80'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_21 = [slice_20, full_64, full_65, full_66]

        # pd_op.reshape_: (-1x255x80x80xf16, 0x-1x3x85x6400xf16) <- (-1x3x85x6400xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__16, reshape__17 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_12, [x.reshape([]) for x in combine_21]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.cast: (-1x2xf16) <- (-1x2xf32)
        cast_19 = paddle._C_ops.cast(feed_1, paddle.float16)

        # pd_op.cast: (-1x2xf16) <- (-1x2xf32)
        cast_20 = paddle._C_ops.cast(feed_2, paddle.float16)

        # pd_op.divide_: (-1x2xf16) <- (-1x2xf16, -1x2xf16)
        divide__3 = paddle._C_ops.divide_(cast_19, cast_20)

        # pd_op.cast: (-1x2xf32) <- (-1x2xf16)
        cast_21 = paddle._C_ops.cast(divide__3, paddle.float32)

        # pd_op.cast: (-1x2xi32) <- (-1x2xf32)
        cast_22 = paddle._C_ops.cast(cast_21, paddle.int32)

        # pd_op.cast: (-1x255x20x20xf32) <- (-1x255x20x20xf16)
        cast_23 = paddle._C_ops.cast(reshape__4, paddle.float32)

        # pd_op.yolo_box: (-1x1200x4xf32, -1x1200x80xf32) <- (-1x255x20x20xf32, -1x2xi32)
        yolo_box_0, yolo_box_1 = (lambda x, f: f(x))(paddle._C_ops.yolo_box(cast_23, cast_22, [116, 90, 156, 198, 373, 326], 80, float('0.01'), 32, True, float('1.05'), False, float('0.5')), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.cast: (-1x1200x80xf16) <- (-1x1200x80xf32)
        cast_24 = paddle._C_ops.cast(yolo_box_1, paddle.float16)

        # pd_op.transpose: (-1x80x1200xf16) <- (-1x1200x80xf16)
        transpose_0 = paddle._C_ops.transpose(cast_24, [0, 2, 1])

        # pd_op.cast: (-1x255x40x40xf32) <- (-1x255x40x40xf16)
        cast_25 = paddle._C_ops.cast(reshape__10, paddle.float32)

        # pd_op.yolo_box: (-1x4800x4xf32, -1x4800x80xf32) <- (-1x255x40x40xf32, -1x2xi32)
        yolo_box_2, yolo_box_3 = (lambda x, f: f(x))(paddle._C_ops.yolo_box(cast_25, cast_22, [30, 61, 62, 45, 59, 119], 80, float('0.01'), 16, True, float('1.05'), False, float('0.5')), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.cast: (-1x4800x80xf16) <- (-1x4800x80xf32)
        cast_26 = paddle._C_ops.cast(yolo_box_3, paddle.float16)

        # pd_op.transpose: (-1x80x4800xf16) <- (-1x4800x80xf16)
        transpose_1 = paddle._C_ops.transpose(cast_26, [0, 2, 1])

        # pd_op.cast: (-1x255x80x80xf32) <- (-1x255x80x80xf16)
        cast_27 = paddle._C_ops.cast(reshape__16, paddle.float32)

        # pd_op.yolo_box: (-1x19200x4xf32, -1x19200x80xf32) <- (-1x255x80x80xf32, -1x2xi32)
        yolo_box_4, yolo_box_5 = (lambda x, f: f(x))(paddle._C_ops.yolo_box(cast_27, cast_22, [10, 13, 16, 30, 33, 23], 80, float('0.01'), 8, True, float('1.05'), False, float('0.5')), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.cast: (-1x19200x80xf16) <- (-1x19200x80xf32)
        cast_28 = paddle._C_ops.cast(yolo_box_5, paddle.float16)

        # pd_op.transpose: (-1x80x19200xf16) <- (-1x19200x80xf16)
        transpose_2 = paddle._C_ops.transpose(cast_28, [0, 2, 1])

        # pd_op.cast: (-1x1200x4xf16) <- (-1x1200x4xf32)
        cast_29 = paddle._C_ops.cast(yolo_box_0, paddle.float16)

        # pd_op.cast: (-1x4800x4xf16) <- (-1x4800x4xf32)
        cast_30 = paddle._C_ops.cast(yolo_box_2, paddle.float16)

        # pd_op.cast: (-1x19200x4xf16) <- (-1x19200x4xf32)
        cast_31 = paddle._C_ops.cast(yolo_box_4, paddle.float16)

        # builtin.combine: ([-1x1200x4xf16, -1x4800x4xf16, -1x19200x4xf16]) <- (-1x1200x4xf16, -1x4800x4xf16, -1x19200x4xf16)
        combine_22 = [cast_29, cast_30, cast_31]

        # pd_op.full: (1xi32) <- ()
        full_67 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x25200x4xf16) <- ([-1x1200x4xf16, -1x4800x4xf16, -1x19200x4xf16], 1xi32)
        concat_13 = paddle._C_ops.concat(combine_22, full_67)

        # builtin.combine: ([-1x80x1200xf16, -1x80x4800xf16, -1x80x19200xf16]) <- (-1x80x1200xf16, -1x80x4800xf16, -1x80x19200xf16)
        combine_23 = [transpose_0, transpose_1, transpose_2]

        # pd_op.full: (1xi32) <- ()
        full_68 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x80x25200xf16) <- ([-1x80x1200xf16, -1x80x4800xf16, -1x80x19200xf16], 1xi32)
        concat_14 = paddle._C_ops.concat(combine_23, full_68)

        # pd_op.cast: (-1x25200x4xf32) <- (-1x25200x4xf16)
        cast_32 = paddle._C_ops.cast(concat_13, paddle.float32)

        # pd_op.cast: (-1x80x25200xf32) <- (-1x80x25200xf16)
        cast_33 = paddle._C_ops.cast(concat_14, paddle.float32)

        # pd_op.memcpy_d2h: (-1x25200x4xf32) <- (-1x25200x4xf32)
        memcpy_d2h_0 = paddle._C_ops.memcpy_d2h(cast_32, 0)

        # pd_op.memcpy_d2h: (-1x80x25200xf32) <- (-1x80x25200xf32)
        memcpy_d2h_1 = paddle._C_ops.memcpy_d2h(cast_33, 0)

        # pd_op.matrix_nms: (25200x6xf32, 25200x1xi32, -1xi32) <- (-1x25200x4xf32, -1x80x25200xf32)
        matrix_nms_0, matrix_nms_1, matrix_nms_2 = (lambda x, f: f(x))(paddle._C_ops.matrix_nms(memcpy_d2h_0, memcpy_d2h_1, float('0.01'), -1, 100, float('0.01'), False, float('2'), -1, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))
        return matrix_nms_0, matrix_nms_2



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

    def forward(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_114, parameter_111, parameter_113, parameter_112, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_154, parameter_151, parameter_153, parameter_152, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_169, parameter_166, parameter_168, parameter_167, parameter_170, parameter_174, parameter_171, parameter_173, parameter_172, parameter_175, parameter_179, parameter_176, parameter_178, parameter_177, parameter_180, parameter_184, parameter_181, parameter_183, parameter_182, parameter_185, parameter_189, parameter_186, parameter_188, parameter_187, parameter_190, parameter_194, parameter_191, parameter_193, parameter_192, parameter_195, parameter_199, parameter_196, parameter_198, parameter_197, parameter_200, parameter_204, parameter_201, parameter_203, parameter_202, parameter_205, parameter_209, parameter_206, parameter_208, parameter_207, parameter_210, parameter_214, parameter_211, parameter_213, parameter_212, parameter_215, parameter_219, parameter_216, parameter_218, parameter_217, parameter_220, parameter_224, parameter_221, parameter_223, parameter_222, parameter_225, parameter_229, parameter_226, parameter_228, parameter_227, parameter_230, parameter_231, parameter_232, parameter_236, parameter_233, parameter_235, parameter_234, parameter_237, parameter_241, parameter_238, parameter_240, parameter_239, parameter_242, parameter_246, parameter_243, parameter_245, parameter_244, parameter_247, parameter_251, parameter_248, parameter_250, parameter_249, parameter_252, parameter_253, parameter_254, parameter_258, parameter_255, parameter_257, parameter_256, parameter_259, parameter_263, parameter_260, parameter_262, parameter_261, parameter_264, parameter_268, parameter_265, parameter_267, parameter_266, parameter_269, parameter_270, parameter_271, parameter_275, parameter_272, parameter_274, parameter_273, parameter_276, parameter_280, parameter_277, parameter_279, parameter_278, parameter_281, parameter_285, parameter_282, parameter_284, parameter_283, parameter_286, parameter_290, parameter_287, parameter_289, parameter_288, parameter_291, parameter_295, parameter_292, parameter_294, parameter_293, parameter_296, parameter_300, parameter_297, parameter_299, parameter_298, parameter_301, parameter_305, parameter_302, parameter_304, parameter_303, parameter_306, parameter_310, parameter_307, parameter_309, parameter_308, parameter_311, parameter_315, parameter_312, parameter_314, parameter_313, parameter_316, parameter_320, parameter_317, parameter_319, parameter_318, parameter_321, parameter_325, parameter_322, parameter_324, parameter_323, parameter_326, parameter_330, parameter_327, parameter_329, parameter_328, parameter_331, parameter_335, parameter_332, parameter_334, parameter_333, parameter_336, parameter_340, parameter_337, parameter_339, parameter_338, parameter_341, parameter_345, parameter_342, parameter_344, parameter_343, parameter_346, parameter_350, parameter_347, parameter_349, parameter_348, parameter_351, parameter_355, parameter_352, parameter_354, parameter_353, parameter_356, parameter_360, parameter_357, parameter_359, parameter_358, parameter_361, parameter_365, parameter_362, parameter_364, parameter_363, parameter_366, parameter_370, parameter_367, parameter_369, parameter_368, parameter_371, parameter_375, parameter_372, parameter_374, parameter_373, parameter_376, parameter_380, parameter_377, parameter_379, parameter_378, parameter_381, parameter_385, parameter_382, parameter_384, parameter_383, parameter_386, parameter_390, parameter_387, parameter_389, parameter_388, parameter_391, parameter_395, parameter_392, parameter_394, parameter_393, parameter_396, parameter_400, parameter_397, parameter_399, parameter_398, parameter_401, parameter_405, parameter_402, parameter_404, parameter_403, parameter_406, parameter_410, parameter_407, parameter_409, parameter_408, parameter_411, parameter_415, parameter_412, parameter_414, parameter_413, parameter_416, parameter_420, parameter_417, parameter_419, parameter_418, parameter_421, parameter_425, parameter_422, parameter_424, parameter_423, parameter_426, parameter_430, parameter_427, parameter_429, parameter_428, parameter_431, parameter_435, parameter_432, parameter_434, parameter_433, parameter_436, parameter_440, parameter_437, parameter_439, parameter_438, parameter_441, parameter_445, parameter_442, parameter_444, parameter_443, parameter_446, parameter_450, parameter_447, parameter_449, parameter_448, parameter_451, parameter_455, parameter_452, parameter_454, parameter_453, parameter_456, parameter_460, parameter_457, parameter_459, parameter_458, parameter_461, parameter_465, parameter_462, parameter_464, parameter_463, parameter_466, parameter_470, parameter_467, parameter_469, parameter_468, parameter_471, parameter_475, parameter_472, parameter_474, parameter_473, parameter_476, parameter_480, parameter_477, parameter_479, parameter_478, parameter_481, parameter_485, parameter_482, parameter_484, parameter_483, parameter_486, parameter_490, parameter_487, parameter_489, parameter_488, parameter_491, parameter_495, parameter_492, parameter_494, parameter_493, parameter_496, parameter_500, parameter_497, parameter_499, parameter_498, parameter_501, parameter_505, parameter_502, parameter_504, parameter_503, parameter_506, parameter_510, parameter_507, parameter_509, parameter_508, parameter_511, parameter_515, parameter_512, parameter_514, parameter_513, parameter_516, parameter_520, parameter_517, parameter_519, parameter_518, parameter_521, parameter_525, parameter_522, parameter_524, parameter_523, parameter_526, parameter_527, parameter_528, parameter_529, parameter_530, parameter_531, feed_2, feed_0, feed_1):
        return self.builtin_module_1174_0_0(parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_114, parameter_111, parameter_113, parameter_112, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_154, parameter_151, parameter_153, parameter_152, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_169, parameter_166, parameter_168, parameter_167, parameter_170, parameter_174, parameter_171, parameter_173, parameter_172, parameter_175, parameter_179, parameter_176, parameter_178, parameter_177, parameter_180, parameter_184, parameter_181, parameter_183, parameter_182, parameter_185, parameter_189, parameter_186, parameter_188, parameter_187, parameter_190, parameter_194, parameter_191, parameter_193, parameter_192, parameter_195, parameter_199, parameter_196, parameter_198, parameter_197, parameter_200, parameter_204, parameter_201, parameter_203, parameter_202, parameter_205, parameter_209, parameter_206, parameter_208, parameter_207, parameter_210, parameter_214, parameter_211, parameter_213, parameter_212, parameter_215, parameter_219, parameter_216, parameter_218, parameter_217, parameter_220, parameter_224, parameter_221, parameter_223, parameter_222, parameter_225, parameter_229, parameter_226, parameter_228, parameter_227, parameter_230, parameter_231, parameter_232, parameter_236, parameter_233, parameter_235, parameter_234, parameter_237, parameter_241, parameter_238, parameter_240, parameter_239, parameter_242, parameter_246, parameter_243, parameter_245, parameter_244, parameter_247, parameter_251, parameter_248, parameter_250, parameter_249, parameter_252, parameter_253, parameter_254, parameter_258, parameter_255, parameter_257, parameter_256, parameter_259, parameter_263, parameter_260, parameter_262, parameter_261, parameter_264, parameter_268, parameter_265, parameter_267, parameter_266, parameter_269, parameter_270, parameter_271, parameter_275, parameter_272, parameter_274, parameter_273, parameter_276, parameter_280, parameter_277, parameter_279, parameter_278, parameter_281, parameter_285, parameter_282, parameter_284, parameter_283, parameter_286, parameter_290, parameter_287, parameter_289, parameter_288, parameter_291, parameter_295, parameter_292, parameter_294, parameter_293, parameter_296, parameter_300, parameter_297, parameter_299, parameter_298, parameter_301, parameter_305, parameter_302, parameter_304, parameter_303, parameter_306, parameter_310, parameter_307, parameter_309, parameter_308, parameter_311, parameter_315, parameter_312, parameter_314, parameter_313, parameter_316, parameter_320, parameter_317, parameter_319, parameter_318, parameter_321, parameter_325, parameter_322, parameter_324, parameter_323, parameter_326, parameter_330, parameter_327, parameter_329, parameter_328, parameter_331, parameter_335, parameter_332, parameter_334, parameter_333, parameter_336, parameter_340, parameter_337, parameter_339, parameter_338, parameter_341, parameter_345, parameter_342, parameter_344, parameter_343, parameter_346, parameter_350, parameter_347, parameter_349, parameter_348, parameter_351, parameter_355, parameter_352, parameter_354, parameter_353, parameter_356, parameter_360, parameter_357, parameter_359, parameter_358, parameter_361, parameter_365, parameter_362, parameter_364, parameter_363, parameter_366, parameter_370, parameter_367, parameter_369, parameter_368, parameter_371, parameter_375, parameter_372, parameter_374, parameter_373, parameter_376, parameter_380, parameter_377, parameter_379, parameter_378, parameter_381, parameter_385, parameter_382, parameter_384, parameter_383, parameter_386, parameter_390, parameter_387, parameter_389, parameter_388, parameter_391, parameter_395, parameter_392, parameter_394, parameter_393, parameter_396, parameter_400, parameter_397, parameter_399, parameter_398, parameter_401, parameter_405, parameter_402, parameter_404, parameter_403, parameter_406, parameter_410, parameter_407, parameter_409, parameter_408, parameter_411, parameter_415, parameter_412, parameter_414, parameter_413, parameter_416, parameter_420, parameter_417, parameter_419, parameter_418, parameter_421, parameter_425, parameter_422, parameter_424, parameter_423, parameter_426, parameter_430, parameter_427, parameter_429, parameter_428, parameter_431, parameter_435, parameter_432, parameter_434, parameter_433, parameter_436, parameter_440, parameter_437, parameter_439, parameter_438, parameter_441, parameter_445, parameter_442, parameter_444, parameter_443, parameter_446, parameter_450, parameter_447, parameter_449, parameter_448, parameter_451, parameter_455, parameter_452, parameter_454, parameter_453, parameter_456, parameter_460, parameter_457, parameter_459, parameter_458, parameter_461, parameter_465, parameter_462, parameter_464, parameter_463, parameter_466, parameter_470, parameter_467, parameter_469, parameter_468, parameter_471, parameter_475, parameter_472, parameter_474, parameter_473, parameter_476, parameter_480, parameter_477, parameter_479, parameter_478, parameter_481, parameter_485, parameter_482, parameter_484, parameter_483, parameter_486, parameter_490, parameter_487, parameter_489, parameter_488, parameter_491, parameter_495, parameter_492, parameter_494, parameter_493, parameter_496, parameter_500, parameter_497, parameter_499, parameter_498, parameter_501, parameter_505, parameter_502, parameter_504, parameter_503, parameter_506, parameter_510, parameter_507, parameter_509, parameter_508, parameter_511, parameter_515, parameter_512, parameter_514, parameter_513, parameter_516, parameter_520, parameter_517, parameter_519, parameter_518, parameter_521, parameter_525, parameter_522, parameter_524, parameter_523, parameter_526, parameter_527, parameter_528, parameter_529, parameter_530, parameter_531, feed_2, feed_0, feed_1)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_1174_0_0(CinnTestBase, unittest.TestCase):
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
            paddle.uniform([32, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_9
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([64, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_14
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_19
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([64, 64, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_24
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([256, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_29
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([256, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_34
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([64, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_39
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([64, 64, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_44
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([256, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_49
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([64, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_54
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([64, 64, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_59
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([256, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_64
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([128, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_69
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([128, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_74
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([512, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_79
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([512, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_84
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([128, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_89
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([128, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_94
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([512, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_99
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([128, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_104
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([128, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_109
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([512, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_114
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([128, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_119
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([128, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_124
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([512, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_129
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([256, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_134
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_139
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([1024, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_144
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([1024, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_149
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([256, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_154
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_159
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([1024, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_164
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([256, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_169
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_174
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([1024, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_179
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([256, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_184
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_189
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([1024, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_194
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([256, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_199
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_204
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([1024, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_209
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([256, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_214
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_213
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_215
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_219
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_218
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_217
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([1024, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_224
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_221
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_223
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_225
            paddle.uniform([512, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_229
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_227
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([27, 512, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_231
            paddle.uniform([27], dtype='float16', min=0, max=0.5),
            # parameter_232
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_233
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_234
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([2048, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_241
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_238
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_239
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([2048, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_246
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_243
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_245
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_244
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([512, 2048, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_251
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_248
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_250
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_249
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_252
            paddle.uniform([27, 512, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_253
            paddle.uniform([27], dtype='float16', min=0, max=0.5),
            # parameter_254
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_258
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_255
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_257
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_256
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_259
            paddle.uniform([2048, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_263
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_262
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_261
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_264
            paddle.uniform([512, 2048, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_268
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_265
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_267
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_266
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_269
            paddle.uniform([27, 512, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_270
            paddle.uniform([27], dtype='float16', min=0, max=0.5),
            # parameter_271
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_275
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_272
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_274
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_273
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_276
            paddle.uniform([2048, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_280
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_277
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_279
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_278
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_281
            paddle.uniform([512, 2048, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_285
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_282
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_284
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_283
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_286
            paddle.uniform([512, 2048, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_290
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_287
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_289
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_288
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_291
            paddle.uniform([512, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_295
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_292
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_294
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_293
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_296
            paddle.uniform([512, 512, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_300
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_297
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_299
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_298
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_301
            paddle.uniform([512, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_305
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_302
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_304
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_303
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_306
            paddle.uniform([512, 2048, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_310
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_307
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_309
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_308
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_311
            paddle.uniform([512, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_315
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_312
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_314
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_313
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_316
            paddle.uniform([512, 512, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_320
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_317
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_319
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_318
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_321
            paddle.uniform([1024, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_325
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_322
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_324
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_323
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_326
            paddle.uniform([512, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_330
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_327
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_329
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_328
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_331
            paddle.uniform([256, 1536, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_335
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_332
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_334
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_333
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_336
            paddle.uniform([256, 1536, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_340
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_337
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_339
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_338
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_341
            paddle.uniform([256, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_345
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_342
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_344
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_343
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_346
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_350
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_347
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_349
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_348
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_351
            paddle.uniform([256, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_355
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_352
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_354
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_353
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_356
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_360
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_357
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_359
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_358
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_361
            paddle.uniform([256, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_365
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_362
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_364
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_363
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_366
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_370
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_367
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_369
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_368
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_371
            paddle.uniform([512, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_375
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_372
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_374
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_373
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_376
            paddle.uniform([256, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_380
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_377
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_379
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_378
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_381
            paddle.uniform([128, 768, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_385
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_382
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_384
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_383
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_386
            paddle.uniform([128, 768, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_390
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_387
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_389
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_388
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_391
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_395
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_392
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_394
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_393
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_396
            paddle.uniform([128, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_400
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_397
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_399
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_398
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_401
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_405
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_402
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_404
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_403
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_406
            paddle.uniform([128, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_410
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_407
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_409
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_408
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_411
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_415
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_412
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_414
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_413
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_416
            paddle.uniform([128, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_420
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_417
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_419
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_418
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_421
            paddle.uniform([256, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_425
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_422
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_424
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_423
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_426
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_430
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_427
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_429
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_428
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_431
            paddle.uniform([256, 768, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_435
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_432
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_434
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_433
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_436
            paddle.uniform([256, 768, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_440
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_437
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_439
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_438
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_441
            paddle.uniform([256, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_445
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_442
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_444
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_443
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_446
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_450
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_447
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_449
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_448
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_451
            paddle.uniform([256, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_455
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_452
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_454
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_453
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_456
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_460
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_457
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_459
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_458
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_461
            paddle.uniform([256, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_465
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_462
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_464
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_463
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_466
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_470
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_467
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_469
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_468
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_471
            paddle.uniform([512, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_475
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_472
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_474
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_473
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_476
            paddle.uniform([512, 512, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_480
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_477
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_479
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_478
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_481
            paddle.uniform([512, 1536, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_485
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_482
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_484
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_483
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_486
            paddle.uniform([512, 1536, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_490
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_487
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_489
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_488
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_491
            paddle.uniform([512, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_495
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_492
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_494
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_493
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_496
            paddle.uniform([512, 512, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_500
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_497
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_499
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_498
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_501
            paddle.uniform([512, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_505
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_502
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_504
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_503
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_506
            paddle.uniform([512, 512, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_510
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_507
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_509
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_508
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_511
            paddle.uniform([512, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_515
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_512
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_514
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_513
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_516
            paddle.uniform([512, 512, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_520
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_517
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_519
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_518
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_521
            paddle.uniform([1024, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_525
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_522
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_524
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_523
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_526
            paddle.uniform([258, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_527
            paddle.uniform([258], dtype='float16', min=0, max=0.5),
            # parameter_528
            paddle.uniform([258, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_529
            paddle.uniform([258], dtype='float16', min=0, max=0.5),
            # parameter_530
            paddle.uniform([258, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_531
            paddle.uniform([258], dtype='float16', min=0, max=0.5),
            # feed_2
            paddle.to_tensor([1.0, 1.0], dtype='float32').reshape([1, 2]),
            # feed_0
            paddle.uniform([1, 3, 640, 640], dtype='float32', min=0, max=0.5),
            # feed_1
            paddle.to_tensor([640.0, 640.0], dtype='float32').reshape([1, 2]),
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
            paddle.static.InputSpec(shape=[32, 32, 3, 3], dtype='float16'),
            # parameter_9
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[64, 32, 3, 3], dtype='float16'),
            # parameter_14
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_19
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float16'),
            # parameter_24
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float16'),
            # parameter_29
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float16'),
            # parameter_34
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float16'),
            # parameter_39
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float16'),
            # parameter_44
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float16'),
            # parameter_49
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float16'),
            # parameter_54
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float16'),
            # parameter_59
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float16'),
            # parameter_64
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float16'),
            # parameter_69
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float16'),
            # parameter_74
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float16'),
            # parameter_79
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[512, 256, 1, 1], dtype='float16'),
            # parameter_84
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float16'),
            # parameter_89
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float16'),
            # parameter_94
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float16'),
            # parameter_99
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float16'),
            # parameter_104
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float16'),
            # parameter_109
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float16'),
            # parameter_114
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float16'),
            # parameter_119
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float16'),
            # parameter_124
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float16'),
            # parameter_129
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[256, 512, 1, 1], dtype='float16'),
            # parameter_134
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_139
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float16'),
            # parameter_144
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[1024, 512, 1, 1], dtype='float16'),
            # parameter_149
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float16'),
            # parameter_154
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_159
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float16'),
            # parameter_164
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float16'),
            # parameter_169
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_174
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float16'),
            # parameter_179
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float16'),
            # parameter_184
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_189
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float16'),
            # parameter_194
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float16'),
            # parameter_199
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_204
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float16'),
            # parameter_209
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float16'),
            # parameter_214
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_213
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_215
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_219
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_218
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_217
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float16'),
            # parameter_224
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_221
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_223
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_225
            paddle.static.InputSpec(shape=[512, 1024, 1, 1], dtype='float16'),
            # parameter_229
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_227
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[27, 512, 3, 3], dtype='float16'),
            # parameter_231
            paddle.static.InputSpec(shape=[27], dtype='float16'),
            # parameter_232
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_233
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_234
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[2048, 512, 1, 1], dtype='float16'),
            # parameter_241
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_238
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_239
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[2048, 1024, 1, 1], dtype='float16'),
            # parameter_246
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_243
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_245
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_244
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[512, 2048, 1, 1], dtype='float16'),
            # parameter_251
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_248
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_250
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_249
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_252
            paddle.static.InputSpec(shape=[27, 512, 3, 3], dtype='float16'),
            # parameter_253
            paddle.static.InputSpec(shape=[27], dtype='float16'),
            # parameter_254
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_258
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_255
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_257
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_256
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_259
            paddle.static.InputSpec(shape=[2048, 512, 1, 1], dtype='float16'),
            # parameter_263
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_262
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_261
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_264
            paddle.static.InputSpec(shape=[512, 2048, 1, 1], dtype='float16'),
            # parameter_268
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_265
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_267
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_266
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_269
            paddle.static.InputSpec(shape=[27, 512, 3, 3], dtype='float16'),
            # parameter_270
            paddle.static.InputSpec(shape=[27], dtype='float16'),
            # parameter_271
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            # parameter_275
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_272
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_274
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_273
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_276
            paddle.static.InputSpec(shape=[2048, 512, 1, 1], dtype='float16'),
            # parameter_280
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_277
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_279
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_278
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_281
            paddle.static.InputSpec(shape=[512, 2048, 1, 1], dtype='float16'),
            # parameter_285
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_282
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_284
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_283
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_286
            paddle.static.InputSpec(shape=[512, 2048, 1, 1], dtype='float16'),
            # parameter_290
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_287
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_289
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_288
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_291
            paddle.static.InputSpec(shape=[512, 512, 1, 1], dtype='float16'),
            # parameter_295
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_292
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_294
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_293
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_296
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float16'),
            # parameter_300
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_297
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_299
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_298
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_301
            paddle.static.InputSpec(shape=[512, 512, 1, 1], dtype='float16'),
            # parameter_305
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_302
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_304
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_303
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_306
            paddle.static.InputSpec(shape=[512, 2048, 1, 1], dtype='float16'),
            # parameter_310
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_307
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_309
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_308
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_311
            paddle.static.InputSpec(shape=[512, 512, 1, 1], dtype='float16'),
            # parameter_315
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_312
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_314
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_313
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_316
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float16'),
            # parameter_320
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_317
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_319
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_318
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_321
            paddle.static.InputSpec(shape=[1024, 1024, 1, 1], dtype='float16'),
            # parameter_325
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_322
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_324
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_323
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_326
            paddle.static.InputSpec(shape=[512, 1024, 1, 1], dtype='float16'),
            # parameter_330
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_327
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_329
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_328
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_331
            paddle.static.InputSpec(shape=[256, 1536, 1, 1], dtype='float16'),
            # parameter_335
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_332
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_334
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_333
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_336
            paddle.static.InputSpec(shape=[256, 1536, 1, 1], dtype='float16'),
            # parameter_340
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_337
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_339
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_338
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_341
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float16'),
            # parameter_345
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_342
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_344
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_343
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_346
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_350
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_347
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_349
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_348
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_351
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float16'),
            # parameter_355
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_352
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_354
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_353
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_356
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_360
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_357
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_359
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_358
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_361
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float16'),
            # parameter_365
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_362
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_364
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_363
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_366
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_370
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_367
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_369
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_368
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_371
            paddle.static.InputSpec(shape=[512, 512, 1, 1], dtype='float16'),
            # parameter_375
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_372
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_374
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_373
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_376
            paddle.static.InputSpec(shape=[256, 512, 1, 1], dtype='float16'),
            # parameter_380
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_377
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_379
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_378
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_381
            paddle.static.InputSpec(shape=[128, 768, 1, 1], dtype='float16'),
            # parameter_385
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_382
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_384
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_383
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_386
            paddle.static.InputSpec(shape=[128, 768, 1, 1], dtype='float16'),
            # parameter_390
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_387
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_389
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_388
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_391
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_395
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_392
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_394
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_393
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_396
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float16'),
            # parameter_400
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_397
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_399
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_398
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_401
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_405
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_402
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_404
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_403
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_406
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float16'),
            # parameter_410
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_407
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_409
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_408
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_411
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_415
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_412
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_414
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_413
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_416
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float16'),
            # parameter_420
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_417
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_419
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_418
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_421
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float16'),
            # parameter_425
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_422
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_424
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_423
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_426
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_430
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_427
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_429
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_428
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_431
            paddle.static.InputSpec(shape=[256, 768, 1, 1], dtype='float16'),
            # parameter_435
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_432
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_434
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_433
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_436
            paddle.static.InputSpec(shape=[256, 768, 1, 1], dtype='float16'),
            # parameter_440
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_437
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_439
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_438
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_441
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float16'),
            # parameter_445
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_442
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_444
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_443
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_446
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_450
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_447
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_449
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_448
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_451
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float16'),
            # parameter_455
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_452
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_454
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_453
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_456
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_460
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_457
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_459
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_458
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_461
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float16'),
            # parameter_465
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_462
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_464
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_463
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_466
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_470
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_467
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_469
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_468
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_471
            paddle.static.InputSpec(shape=[512, 512, 1, 1], dtype='float16'),
            # parameter_475
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_472
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_474
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_473
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_476
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float16'),
            # parameter_480
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_477
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_479
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_478
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_481
            paddle.static.InputSpec(shape=[512, 1536, 1, 1], dtype='float16'),
            # parameter_485
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_482
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_484
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_483
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_486
            paddle.static.InputSpec(shape=[512, 1536, 1, 1], dtype='float16'),
            # parameter_490
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_487
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_489
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_488
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_491
            paddle.static.InputSpec(shape=[512, 512, 1, 1], dtype='float16'),
            # parameter_495
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_492
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_494
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_493
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_496
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float16'),
            # parameter_500
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_497
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_499
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_498
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_501
            paddle.static.InputSpec(shape=[512, 512, 1, 1], dtype='float16'),
            # parameter_505
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_502
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_504
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_503
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_506
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float16'),
            # parameter_510
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_507
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_509
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_508
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_511
            paddle.static.InputSpec(shape=[512, 512, 1, 1], dtype='float16'),
            # parameter_515
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_512
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_514
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_513
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_516
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float16'),
            # parameter_520
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_517
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_519
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_518
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_521
            paddle.static.InputSpec(shape=[1024, 1024, 1, 1], dtype='float16'),
            # parameter_525
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_522
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_524
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_523
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_526
            paddle.static.InputSpec(shape=[258, 1024, 1, 1], dtype='float16'),
            # parameter_527
            paddle.static.InputSpec(shape=[258], dtype='float16'),
            # parameter_528
            paddle.static.InputSpec(shape=[258, 512, 1, 1], dtype='float16'),
            # parameter_529
            paddle.static.InputSpec(shape=[258], dtype='float16'),
            # parameter_530
            paddle.static.InputSpec(shape=[258, 256, 1, 1], dtype='float16'),
            # parameter_531
            paddle.static.InputSpec(shape=[258], dtype='float16'),
            # feed_2
            paddle.static.InputSpec(shape=[None, 2], dtype='float32'),
            # feed_0
            paddle.static.InputSpec(shape=[None, 3, 640, 640], dtype='float32'),
            # feed_1
            paddle.static.InputSpec(shape=[None, 2], dtype='float32'),
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