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
    return [962][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_1572_0_0(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_36, parameter_37, parameter_38, parameter_39, parameter_43, parameter_40, parameter_42, parameter_41, parameter_44, parameter_48, parameter_45, parameter_47, parameter_46, parameter_49, parameter_50, parameter_51, parameter_52, parameter_53, parameter_57, parameter_54, parameter_56, parameter_55, parameter_58, parameter_62, parameter_59, parameter_61, parameter_60, parameter_63, parameter_64, parameter_65, parameter_66, parameter_67, parameter_71, parameter_68, parameter_70, parameter_69, parameter_72, parameter_76, parameter_73, parameter_75, parameter_74, parameter_77, parameter_81, parameter_78, parameter_80, parameter_79, parameter_82, parameter_83, parameter_84, parameter_85, parameter_86, parameter_90, parameter_87, parameter_89, parameter_88, parameter_91, parameter_95, parameter_92, parameter_94, parameter_93, parameter_96, parameter_97, parameter_98, parameter_99, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_111, parameter_112, parameter_113, parameter_117, parameter_114, parameter_116, parameter_115, parameter_118, parameter_119, parameter_123, parameter_120, parameter_122, parameter_121, parameter_124, parameter_125, parameter_126, parameter_127, parameter_128, parameter_129, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_146, parameter_147, parameter_148, parameter_149, parameter_153, parameter_150, parameter_152, parameter_151, parameter_154, parameter_158, parameter_155, parameter_157, parameter_156, parameter_159, parameter_163, parameter_160, parameter_162, parameter_161, parameter_164, parameter_165, parameter_166, parameter_167, parameter_168, parameter_172, parameter_169, parameter_171, parameter_170, parameter_173, parameter_177, parameter_174, parameter_176, parameter_175, parameter_178, parameter_179, parameter_180, parameter_181, parameter_182, parameter_186, parameter_183, parameter_185, parameter_184, parameter_187, parameter_191, parameter_188, parameter_190, parameter_189, parameter_192, parameter_193, parameter_194, parameter_195, parameter_196, parameter_200, parameter_197, parameter_199, parameter_198, parameter_201, parameter_205, parameter_202, parameter_204, parameter_203, parameter_206, parameter_207, parameter_208, parameter_209, parameter_213, parameter_210, parameter_212, parameter_211, parameter_214, parameter_215, parameter_219, parameter_216, parameter_218, parameter_217, parameter_220, parameter_221, parameter_222, parameter_223, parameter_224, parameter_225, parameter_226, parameter_230, parameter_227, parameter_229, parameter_228, parameter_231, parameter_235, parameter_232, parameter_234, parameter_233, parameter_236, parameter_240, parameter_237, parameter_239, parameter_238, parameter_241, parameter_242, parameter_243, parameter_244, parameter_245, parameter_249, parameter_246, parameter_248, parameter_247, parameter_250, parameter_254, parameter_251, parameter_253, parameter_252, parameter_255, parameter_259, parameter_256, parameter_258, parameter_257, parameter_260, parameter_261, parameter_262, parameter_263, parameter_264, parameter_268, parameter_265, parameter_267, parameter_266, parameter_269, parameter_273, parameter_270, parameter_272, parameter_271, parameter_274, parameter_275, parameter_276, parameter_277, parameter_278, parameter_282, parameter_279, parameter_281, parameter_280, parameter_283, parameter_287, parameter_284, parameter_286, parameter_285, parameter_288, parameter_289, parameter_290, parameter_291, parameter_292, parameter_296, parameter_293, parameter_295, parameter_294, parameter_297, parameter_301, parameter_298, parameter_300, parameter_299, parameter_302, parameter_303, parameter_304, parameter_305, parameter_309, parameter_306, parameter_308, parameter_307, parameter_310, parameter_311, parameter_315, parameter_312, parameter_314, parameter_313, parameter_316, parameter_317, parameter_318, parameter_319, parameter_320, parameter_321, parameter_322, parameter_326, parameter_323, parameter_325, parameter_324, parameter_327, parameter_331, parameter_328, parameter_330, parameter_329, parameter_332, parameter_336, parameter_333, parameter_335, parameter_334, parameter_337, parameter_338, parameter_339, parameter_340, parameter_341, parameter_345, parameter_342, parameter_344, parameter_343, parameter_346, parameter_350, parameter_347, parameter_349, parameter_348, parameter_351, parameter_355, parameter_352, parameter_354, parameter_353, parameter_356, parameter_357, parameter_358, parameter_359, parameter_360, parameter_364, parameter_361, parameter_363, parameter_362, parameter_365, parameter_369, parameter_366, parameter_368, parameter_367, parameter_370, parameter_371, parameter_372, parameter_373, parameter_374, parameter_378, parameter_375, parameter_377, parameter_376, parameter_379, parameter_383, parameter_380, parameter_382, parameter_381, parameter_384, parameter_385, parameter_386, parameter_387, parameter_388, parameter_392, parameter_389, parameter_391, parameter_390, parameter_393, parameter_397, parameter_394, parameter_396, parameter_395, parameter_398, parameter_399, parameter_400, parameter_401, parameter_405, parameter_402, parameter_404, parameter_403, parameter_406, parameter_407, parameter_411, parameter_408, parameter_410, parameter_409, parameter_412, parameter_413, parameter_414, parameter_415, parameter_416, parameter_417, parameter_418, parameter_422, parameter_419, parameter_421, parameter_420, parameter_423, parameter_427, parameter_424, parameter_426, parameter_425, parameter_428, parameter_432, parameter_429, parameter_431, parameter_430, parameter_433, parameter_434, parameter_435, parameter_436, parameter_437, parameter_441, parameter_438, parameter_440, parameter_439, parameter_442, parameter_446, parameter_443, parameter_445, parameter_444, parameter_447, parameter_451, parameter_448, parameter_450, parameter_449, parameter_452, parameter_453, parameter_454, parameter_455, parameter_456, parameter_460, parameter_457, parameter_459, parameter_458, parameter_461, parameter_465, parameter_462, parameter_464, parameter_463, parameter_466, parameter_467, parameter_468, parameter_469, parameter_470, parameter_474, parameter_471, parameter_473, parameter_472, parameter_475, parameter_479, parameter_476, parameter_478, parameter_477, parameter_480, parameter_481, parameter_482, parameter_483, parameter_484, parameter_488, parameter_485, parameter_487, parameter_486, parameter_489, parameter_493, parameter_490, parameter_492, parameter_491, parameter_494, parameter_495, parameter_496, parameter_497, parameter_501, parameter_498, parameter_500, parameter_499, parameter_502, parameter_503, parameter_507, parameter_504, parameter_506, parameter_505, parameter_508, parameter_509, parameter_510, parameter_511, parameter_512, parameter_513, parameter_514, parameter_518, parameter_515, parameter_517, parameter_516, parameter_519, parameter_523, parameter_520, parameter_522, parameter_521, parameter_524, parameter_528, parameter_525, parameter_527, parameter_526, parameter_529, parameter_530, parameter_531, parameter_532, parameter_533, parameter_537, parameter_534, parameter_536, parameter_535, parameter_538, parameter_542, parameter_539, parameter_541, parameter_540, parameter_543, parameter_547, parameter_544, parameter_546, parameter_545, parameter_548, parameter_549, parameter_550, parameter_551, parameter_552, parameter_556, parameter_553, parameter_555, parameter_554, parameter_557, parameter_561, parameter_558, parameter_560, parameter_559, parameter_562, parameter_563, parameter_564, parameter_565, parameter_569, parameter_566, parameter_568, parameter_567, parameter_570, parameter_571, parameter_572, parameter_573, parameter_574, parameter_575, parameter_576, parameter_577, parameter_578, parameter_582, parameter_579, parameter_581, parameter_580, parameter_583, parameter_587, parameter_584, parameter_586, parameter_585, parameter_588, parameter_592, parameter_589, parameter_591, parameter_590, parameter_593, parameter_594, parameter_595, parameter_596, parameter_600, parameter_597, parameter_599, parameter_598, parameter_601, parameter_602, feed_0):

        # pd_op.cast: (-1x3x224x224xf16) <- (-1x3x224x224xf32)
        cast_0 = paddle._C_ops.cast(feed_0, paddle.float16)

        # pd_op.conv2d: (-1x64x112x112xf16) <- (-1x3x224x224xf16, 64x3x3x3xf16)
        conv2d_0 = paddle._C_ops.conv2d(cast_0, parameter_0, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x112x112xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x112x112xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_0, parameter_1, parameter_2, parameter_3, parameter_4, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x112x112xf16) <- (-1x64x112x112xf16)
        relu__0 = paddle._C_ops.relu_(batch_norm__0)

        # pd_op.conv2d: (-1x32x112x112xf16) <- (-1x64x112x112xf16, 32x64x3x3xf16)
        conv2d_1 = paddle._C_ops.conv2d(relu__0, parameter_5, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x112x112xf16, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x112x112xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_1, parameter_6, parameter_7, parameter_8, parameter_9, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x32x112x112xf16) <- (-1x32x112x112xf16)
        relu__1 = paddle._C_ops.relu_(batch_norm__6)

        # pd_op.conv2d: (-1x64x112x112xf16) <- (-1x32x112x112xf16, 64x32x3x3xf16)
        conv2d_2 = paddle._C_ops.conv2d(relu__1, parameter_10, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x112x112xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x112x112xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_2, parameter_11, parameter_12, parameter_13, parameter_14, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x112x112xf16) <- (-1x64x112x112xf16)
        relu__2 = paddle._C_ops.relu_(batch_norm__12)

        # pd_op.conv2d: (-1x64x56x56xf16) <- (-1x64x112x112xf16, 64x64x3x3xf16)
        conv2d_3 = paddle._C_ops.conv2d(relu__2, parameter_15, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x56x56xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x56x56xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_3, parameter_16, parameter_17, parameter_18, parameter_19, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x56x56xf16) <- (-1x64x56x56xf16)
        relu__3 = paddle._C_ops.relu_(batch_norm__18)

        # pd_op.conv2d: (-1x96x56x56xf16) <- (-1x64x56x56xf16, 96x64x1x1xf16)
        conv2d_4 = paddle._C_ops.conv2d(relu__3, parameter_20, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x96x56x56xf16, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x56x56xf16, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_4, parameter_21, parameter_22, parameter_23, parameter_24, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x96x56x56xf16) <- (-1x96x56x56xf16, 96x32x3x3xf16)
        conv2d_5 = paddle._C_ops.conv2d(batch_norm__24, parameter_25, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 3, 'NCHW')

        # pd_op.batch_norm_: (-1x96x56x56xf16, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x56x56xf16, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_5, parameter_26, parameter_27, parameter_28, parameter_29, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x96x56x56xf16) <- (-1x96x56x56xf16)
        relu__4 = paddle._C_ops.relu_(batch_norm__30)

        # pd_op.conv2d: (-1x96x56x56xf16) <- (-1x96x56x56xf16, 96x96x1x1xf16)
        conv2d_6 = paddle._C_ops.conv2d(relu__4, parameter_30, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x96x56x56xf16) <- (-1x96x56x56xf16, -1x96x56x56xf16)
        add__0 = paddle._C_ops.add_(batch_norm__24, conv2d_6)

        # pd_op.batch_norm_: (-1x96x56x56xf16, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x56x56xf16, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__0, parameter_31, parameter_32, parameter_33, parameter_34, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x288x56x56xf16) <- (-1x96x56x56xf16, 288x96x1x1xf16)
        conv2d_7 = paddle._C_ops.conv2d(batch_norm__36, parameter_35, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, 288, 1, 1]

        # pd_op.reshape: (1x288x1x1xf16, 0x288xf16) <- (288xf16, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_36, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x288x56x56xf16) <- (-1x288x56x56xf16, 1x288x1x1xf16)
        add__1 = paddle._C_ops.add_(conv2d_7, reshape_0)

        # pd_op.relu_: (-1x288x56x56xf16) <- (-1x288x56x56xf16)
        relu__5 = paddle._C_ops.relu_(add__1)

        # pd_op.conv2d: (-1x96x56x56xf16) <- (-1x288x56x56xf16, 96x288x1x1xf16)
        conv2d_8 = paddle._C_ops.conv2d(relu__5, parameter_37, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_1 = [1, 96, 1, 1]

        # pd_op.reshape: (1x96x1x1xf16, 0x96xf16) <- (96xf16, 4xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_38, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x96x56x56xf16) <- (-1x96x56x56xf16, 1x96x1x1xf16)
        add__2 = paddle._C_ops.add_(conv2d_8, reshape_2)

        # pd_op.add_: (-1x96x56x56xf16) <- (-1x96x56x56xf16, -1x96x56x56xf16)
        add__3 = paddle._C_ops.add_(add__0, add__2)

        # pd_op.conv2d: (-1x96x56x56xf16) <- (-1x96x56x56xf16, 96x32x3x3xf16)
        conv2d_9 = paddle._C_ops.conv2d(add__3, parameter_39, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 3, 'NCHW')

        # pd_op.batch_norm_: (-1x96x56x56xf16, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x56x56xf16, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_9, parameter_40, parameter_41, parameter_42, parameter_43, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x96x56x56xf16) <- (-1x96x56x56xf16)
        relu__6 = paddle._C_ops.relu_(batch_norm__42)

        # pd_op.conv2d: (-1x96x56x56xf16) <- (-1x96x56x56xf16, 96x96x1x1xf16)
        conv2d_10 = paddle._C_ops.conv2d(relu__6, parameter_44, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x96x56x56xf16) <- (-1x96x56x56xf16, -1x96x56x56xf16)
        add__4 = paddle._C_ops.add_(add__3, conv2d_10)

        # pd_op.batch_norm_: (-1x96x56x56xf16, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x56x56xf16, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__4, parameter_45, parameter_46, parameter_47, parameter_48, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x288x56x56xf16) <- (-1x96x56x56xf16, 288x96x1x1xf16)
        conv2d_11 = paddle._C_ops.conv2d(batch_norm__48, parameter_49, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_2 = [1, 288, 1, 1]

        # pd_op.reshape: (1x288x1x1xf16, 0x288xf16) <- (288xf16, 4xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_50, full_int_array_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x288x56x56xf16) <- (-1x288x56x56xf16, 1x288x1x1xf16)
        add__5 = paddle._C_ops.add_(conv2d_11, reshape_4)

        # pd_op.relu_: (-1x288x56x56xf16) <- (-1x288x56x56xf16)
        relu__7 = paddle._C_ops.relu_(add__5)

        # pd_op.conv2d: (-1x96x56x56xf16) <- (-1x288x56x56xf16, 96x288x1x1xf16)
        conv2d_12 = paddle._C_ops.conv2d(relu__7, parameter_51, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [1, 96, 1, 1]

        # pd_op.reshape: (1x96x1x1xf16, 0x96xf16) <- (96xf16, 4xi64)
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_52, full_int_array_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x96x56x56xf16) <- (-1x96x56x56xf16, 1x96x1x1xf16)
        add__6 = paddle._C_ops.add_(conv2d_12, reshape_6)

        # pd_op.add_: (-1x96x56x56xf16) <- (-1x96x56x56xf16, -1x96x56x56xf16)
        add__7 = paddle._C_ops.add_(add__4, add__6)

        # pd_op.conv2d: (-1x96x56x56xf16) <- (-1x96x56x56xf16, 96x32x3x3xf16)
        conv2d_13 = paddle._C_ops.conv2d(add__7, parameter_53, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 3, 'NCHW')

        # pd_op.batch_norm_: (-1x96x56x56xf16, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x56x56xf16, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_13, parameter_54, parameter_55, parameter_56, parameter_57, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x96x56x56xf16) <- (-1x96x56x56xf16)
        relu__8 = paddle._C_ops.relu_(batch_norm__54)

        # pd_op.conv2d: (-1x96x56x56xf16) <- (-1x96x56x56xf16, 96x96x1x1xf16)
        conv2d_14 = paddle._C_ops.conv2d(relu__8, parameter_58, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x96x56x56xf16) <- (-1x96x56x56xf16, -1x96x56x56xf16)
        add__8 = paddle._C_ops.add_(add__7, conv2d_14)

        # pd_op.batch_norm_: (-1x96x56x56xf16, 96xf32, 96xf32, xf32, xf32, None) <- (-1x96x56x56xf16, 96xf32, 96xf32, 96xf32, 96xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__8, parameter_59, parameter_60, parameter_61, parameter_62, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x288x56x56xf16) <- (-1x96x56x56xf16, 288x96x1x1xf16)
        conv2d_15 = paddle._C_ops.conv2d(batch_norm__60, parameter_63, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_4 = [1, 288, 1, 1]

        # pd_op.reshape: (1x288x1x1xf16, 0x288xf16) <- (288xf16, 4xi64)
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_64, full_int_array_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x288x56x56xf16) <- (-1x288x56x56xf16, 1x288x1x1xf16)
        add__9 = paddle._C_ops.add_(conv2d_15, reshape_8)

        # pd_op.relu_: (-1x288x56x56xf16) <- (-1x288x56x56xf16)
        relu__9 = paddle._C_ops.relu_(add__9)

        # pd_op.conv2d: (-1x96x56x56xf16) <- (-1x288x56x56xf16, 96x288x1x1xf16)
        conv2d_16 = paddle._C_ops.conv2d(relu__9, parameter_65, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_5 = [1, 96, 1, 1]

        # pd_op.reshape: (1x96x1x1xf16, 0x96xf16) <- (96xf16, 4xi64)
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_66, full_int_array_5), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x96x56x56xf16) <- (-1x96x56x56xf16, 1x96x1x1xf16)
        add__10 = paddle._C_ops.add_(conv2d_16, reshape_10)

        # pd_op.add_: (-1x96x56x56xf16) <- (-1x96x56x56xf16, -1x96x56x56xf16)
        add__11 = paddle._C_ops.add_(add__8, add__10)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_6 = [2, 2]

        # pd_op.pool2d: (-1x96x28x28xf16) <- (-1x96x56x56xf16, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(add__11, full_int_array_6, [2, 2], [0, 0], True, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x192x28x28xf16) <- (-1x96x28x28xf16, 192x96x1x1xf16)
        conv2d_17 = paddle._C_ops.conv2d(pool2d_0, parameter_67, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x192x28x28xf16, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x28x28xf16, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_17, parameter_68, parameter_69, parameter_70, parameter_71, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x192x28x28xf16) <- (-1x192x28x28xf16, 192x32x3x3xf16)
        conv2d_18 = paddle._C_ops.conv2d(batch_norm__66, parameter_72, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 6, 'NCHW')

        # pd_op.batch_norm_: (-1x192x28x28xf16, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x28x28xf16, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_18, parameter_73, parameter_74, parameter_75, parameter_76, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x192x28x28xf16) <- (-1x192x28x28xf16)
        relu__10 = paddle._C_ops.relu_(batch_norm__72)

        # pd_op.conv2d: (-1x192x28x28xf16) <- (-1x192x28x28xf16, 192x192x1x1xf16)
        conv2d_19 = paddle._C_ops.conv2d(relu__10, parameter_77, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x192x28x28xf16) <- (-1x192x28x28xf16, -1x192x28x28xf16)
        add__12 = paddle._C_ops.add_(batch_norm__66, conv2d_19)

        # pd_op.batch_norm_: (-1x192x28x28xf16, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x28x28xf16, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__12, parameter_78, parameter_79, parameter_80, parameter_81, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x576x28x28xf16) <- (-1x192x28x28xf16, 576x192x1x1xf16)
        conv2d_20 = paddle._C_ops.conv2d(batch_norm__78, parameter_82, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_7 = [1, 576, 1, 1]

        # pd_op.reshape: (1x576x1x1xf16, 0x576xf16) <- (576xf16, 4xi64)
        reshape_12, reshape_13 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_83, full_int_array_7), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x576x28x28xf16) <- (-1x576x28x28xf16, 1x576x1x1xf16)
        add__13 = paddle._C_ops.add_(conv2d_20, reshape_12)

        # pd_op.relu_: (-1x576x28x28xf16) <- (-1x576x28x28xf16)
        relu__11 = paddle._C_ops.relu_(add__13)

        # pd_op.conv2d: (-1x192x28x28xf16) <- (-1x576x28x28xf16, 192x576x1x1xf16)
        conv2d_21 = paddle._C_ops.conv2d(relu__11, parameter_84, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_8 = [1, 192, 1, 1]

        # pd_op.reshape: (1x192x1x1xf16, 0x192xf16) <- (192xf16, 4xi64)
        reshape_14, reshape_15 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_85, full_int_array_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x192x28x28xf16) <- (-1x192x28x28xf16, 1x192x1x1xf16)
        add__14 = paddle._C_ops.add_(conv2d_21, reshape_14)

        # pd_op.add_: (-1x192x28x28xf16) <- (-1x192x28x28xf16, -1x192x28x28xf16)
        add__15 = paddle._C_ops.add_(add__12, add__14)

        # pd_op.conv2d: (-1x192x28x28xf16) <- (-1x192x28x28xf16, 192x32x3x3xf16)
        conv2d_22 = paddle._C_ops.conv2d(add__15, parameter_86, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 6, 'NCHW')

        # pd_op.batch_norm_: (-1x192x28x28xf16, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x28x28xf16, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_22, parameter_87, parameter_88, parameter_89, parameter_90, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x192x28x28xf16) <- (-1x192x28x28xf16)
        relu__12 = paddle._C_ops.relu_(batch_norm__84)

        # pd_op.conv2d: (-1x192x28x28xf16) <- (-1x192x28x28xf16, 192x192x1x1xf16)
        conv2d_23 = paddle._C_ops.conv2d(relu__12, parameter_91, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x192x28x28xf16) <- (-1x192x28x28xf16, -1x192x28x28xf16)
        add__16 = paddle._C_ops.add_(add__15, conv2d_23)

        # pd_op.batch_norm_: (-1x192x28x28xf16, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x28x28xf16, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__16, parameter_92, parameter_93, parameter_94, parameter_95, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x576x28x28xf16) <- (-1x192x28x28xf16, 576x192x1x1xf16)
        conv2d_24 = paddle._C_ops.conv2d(batch_norm__90, parameter_96, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_9 = [1, 576, 1, 1]

        # pd_op.reshape: (1x576x1x1xf16, 0x576xf16) <- (576xf16, 4xi64)
        reshape_16, reshape_17 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_97, full_int_array_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x576x28x28xf16) <- (-1x576x28x28xf16, 1x576x1x1xf16)
        add__17 = paddle._C_ops.add_(conv2d_24, reshape_16)

        # pd_op.relu_: (-1x576x28x28xf16) <- (-1x576x28x28xf16)
        relu__13 = paddle._C_ops.relu_(add__17)

        # pd_op.conv2d: (-1x192x28x28xf16) <- (-1x576x28x28xf16, 192x576x1x1xf16)
        conv2d_25 = paddle._C_ops.conv2d(relu__13, parameter_98, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_10 = [1, 192, 1, 1]

        # pd_op.reshape: (1x192x1x1xf16, 0x192xf16) <- (192xf16, 4xi64)
        reshape_18, reshape_19 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_99, full_int_array_10), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x192x28x28xf16) <- (-1x192x28x28xf16, 1x192x1x1xf16)
        add__18 = paddle._C_ops.add_(conv2d_25, reshape_18)

        # pd_op.add_: (-1x192x28x28xf16) <- (-1x192x28x28xf16, -1x192x28x28xf16)
        add__19 = paddle._C_ops.add_(add__16, add__18)

        # pd_op.conv2d: (-1x192x28x28xf16) <- (-1x192x28x28xf16, 192x32x3x3xf16)
        conv2d_26 = paddle._C_ops.conv2d(add__19, parameter_100, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 6, 'NCHW')

        # pd_op.batch_norm_: (-1x192x28x28xf16, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x28x28xf16, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_26, parameter_101, parameter_102, parameter_103, parameter_104, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x192x28x28xf16) <- (-1x192x28x28xf16)
        relu__14 = paddle._C_ops.relu_(batch_norm__96)

        # pd_op.conv2d: (-1x192x28x28xf16) <- (-1x192x28x28xf16, 192x192x1x1xf16)
        conv2d_27 = paddle._C_ops.conv2d(relu__14, parameter_105, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x192x28x28xf16) <- (-1x192x28x28xf16, -1x192x28x28xf16)
        add__20 = paddle._C_ops.add_(add__19, conv2d_27)

        # pd_op.batch_norm_: (-1x192x28x28xf16, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x28x28xf16, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__20, parameter_106, parameter_107, parameter_108, parameter_109, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x576x28x28xf16) <- (-1x192x28x28xf16, 576x192x1x1xf16)
        conv2d_28 = paddle._C_ops.conv2d(batch_norm__102, parameter_110, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_11 = [1, 576, 1, 1]

        # pd_op.reshape: (1x576x1x1xf16, 0x576xf16) <- (576xf16, 4xi64)
        reshape_20, reshape_21 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_111, full_int_array_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x576x28x28xf16) <- (-1x576x28x28xf16, 1x576x1x1xf16)
        add__21 = paddle._C_ops.add_(conv2d_28, reshape_20)

        # pd_op.relu_: (-1x576x28x28xf16) <- (-1x576x28x28xf16)
        relu__15 = paddle._C_ops.relu_(add__21)

        # pd_op.conv2d: (-1x192x28x28xf16) <- (-1x576x28x28xf16, 192x576x1x1xf16)
        conv2d_29 = paddle._C_ops.conv2d(relu__15, parameter_112, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_12 = [1, 192, 1, 1]

        # pd_op.reshape: (1x192x1x1xf16, 0x192xf16) <- (192xf16, 4xi64)
        reshape_22, reshape_23 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_113, full_int_array_12), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x192x28x28xf16) <- (-1x192x28x28xf16, 1x192x1x1xf16)
        add__22 = paddle._C_ops.add_(conv2d_29, reshape_22)

        # pd_op.add_: (-1x192x28x28xf16) <- (-1x192x28x28xf16, -1x192x28x28xf16)
        add__23 = paddle._C_ops.add_(add__20, add__22)

        # pd_op.batch_norm_: (-1x192x28x28xf16, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x28x28xf16, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__23, parameter_114, parameter_115, parameter_116, parameter_117, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (4xi32) <- (-1x192x28x28xf16)
        shape_0 = paddle._C_ops.shape(paddle.cast(batch_norm__108, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_13 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_14 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], full_int_array_13, full_int_array_14, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('192'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('784'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_0 = [slice_0, full_0, full_1]

        # pd_op.reshape_: (-1x192x784xf16, 0x-1x192x28x28xf16) <- (-1x192x28x28xf16, [1xi32, 1xi32, 1xi32])
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__108, [x.reshape([]) for x in combine_0]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x784x192xf16) <- (-1x192x784xf16)
        transpose_0 = paddle._C_ops.transpose(reshape__0, [0, 2, 1])

        # pd_op.shape: (3xi32) <- (-1x784x192xf16)
        shape_1 = paddle._C_ops.shape(paddle.cast(transpose_0, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_15 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_16 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(shape_1, [0], full_int_array_15, full_int_array_16, [1], [0])

        # pd_op.matmul: (-1x784x192xf16) <- (-1x784x192xf16, 192x192xf16)
        matmul_0 = paddle._C_ops.matmul(transpose_0, parameter_118, False, False)

        # pd_op.add_: (-1x784x192xf16) <- (-1x784x192xf16, 192xf16)
        add__24 = paddle._C_ops.add_(matmul_0, parameter_119)

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], float('784'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_1 = [slice_1, full_2, full_3, full_4]

        # pd_op.reshape_: (-1x784x6x32xf16, 0x-1x784x192xf16) <- (-1x784x192xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__24, [x.reshape([]) for x in combine_1]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x6x784x32xf16) <- (-1x784x6x32xf16)
        transpose_1 = paddle._C_ops.transpose(reshape__2, [0, 2, 1, 3])

        # pd_op.transpose: (-1x192x784xf16) <- (-1x784x192xf16)
        transpose_2 = paddle._C_ops.transpose(transpose_0, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_17 = [2]

        # pd_op.unsqueeze_: (-1x192x1x784xf16, None) <- (-1x192x784xf16, 1xi64)
        unsqueeze__0, unsqueeze__1 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(transpose_2, full_int_array_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_18 = [1, 16]

        # pd_op.pool2d: (-1x192x1x49xf16) <- (-1x192x1x784xf16, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(unsqueeze__0, full_int_array_18, [1, 16], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_19 = [2]

        # pd_op.squeeze_: (-1x192x49xf16, None) <- (-1x192x1x49xf16, 1xi64)
        squeeze__0, squeeze__1 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_1, full_int_array_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x192x49xf16, 192xf32, 192xf32, xf32, xf32, None) <- (-1x192x49xf16, 192xf32, 192xf32, 192xf32, 192xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(squeeze__0, parameter_120, parameter_121, parameter_122, parameter_123, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.transpose: (-1x49x192xf16) <- (-1x192x49xf16)
        transpose_3 = paddle._C_ops.transpose(batch_norm__114, [0, 2, 1])

        # pd_op.matmul: (-1x49x192xf16) <- (-1x49x192xf16, 192x192xf16)
        matmul_1 = paddle._C_ops.matmul(transpose_3, parameter_124, False, False)

        # pd_op.add_: (-1x49x192xf16) <- (-1x49x192xf16, 192xf16)
        add__25 = paddle._C_ops.add_(matmul_1, parameter_125)

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_2 = [slice_1, full_5, full_6, full_7]

        # pd_op.reshape_: (-1x49x6x32xf16, 0x-1x49x192xf16) <- (-1x49x192xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__25, [x.reshape([]) for x in combine_2]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x6x32x49xf16) <- (-1x49x6x32xf16)
        transpose_4 = paddle._C_ops.transpose(reshape__4, [0, 2, 3, 1])

        # pd_op.matmul: (-1x49x192xf16) <- (-1x49x192xf16, 192x192xf16)
        matmul_2 = paddle._C_ops.matmul(transpose_3, parameter_126, False, False)

        # pd_op.add_: (-1x49x192xf16) <- (-1x49x192xf16, 192xf16)
        add__26 = paddle._C_ops.add_(matmul_2, parameter_127)

        # pd_op.full: (1xi32) <- ()
        full_8 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_9 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_10 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_3 = [slice_1, full_8, full_9, full_10]

        # pd_op.reshape_: (-1x49x6x32xf16, 0x-1x49x192xf16) <- (-1x49x192xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__26, [x.reshape([]) for x in combine_3]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x6x49x32xf16) <- (-1x49x6x32xf16)
        transpose_5 = paddle._C_ops.transpose(reshape__6, [0, 2, 1, 3])

        # pd_op.matmul: (-1x6x784x49xf16) <- (-1x6x784x32xf16, -1x6x32x49xf16)
        matmul_3 = paddle._C_ops.matmul(transpose_1, transpose_4, False, False)

        # pd_op.full: (1xf32) <- ()
        full_11 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x6x784x49xf16) <- (-1x6x784x49xf16, 1xf32)
        scale__0 = paddle._C_ops.scale_(matmul_3, full_11, float('0'), True)

        # pd_op.softmax_: (-1x6x784x49xf16) <- (-1x6x784x49xf16)
        softmax__0 = paddle._C_ops.softmax_(scale__0, -1)

        # pd_op.matmul: (-1x6x784x32xf16) <- (-1x6x784x49xf16, -1x6x49x32xf16)
        matmul_4 = paddle._C_ops.matmul(softmax__0, transpose_5, False, False)

        # pd_op.transpose: (-1x784x6x32xf16) <- (-1x6x784x32xf16)
        transpose_6 = paddle._C_ops.transpose(matmul_4, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_12 = paddle._C_ops.full([1], float('784'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_13 = paddle._C_ops.full([1], float('192'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_4 = [slice_1, full_12, full_13]

        # pd_op.reshape_: (-1x784x192xf16, 0x-1x784x6x32xf16) <- (-1x784x6x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_6, [x.reshape([]) for x in combine_4]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x784x192xf16) <- (-1x784x192xf16, 192x192xf16)
        matmul_5 = paddle._C_ops.matmul(reshape__8, parameter_128, False, False)

        # pd_op.add_: (-1x784x192xf16) <- (-1x784x192xf16, 192xf16)
        add__27 = paddle._C_ops.add_(matmul_5, parameter_129)

        # pd_op.shape: (3xi32) <- (-1x784x192xf16)
        shape_2 = paddle._C_ops.shape(paddle.cast(add__27, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_20 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_21 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(shape_2, [0], full_int_array_20, full_int_array_21, [1], [0])

        # pd_op.transpose: (-1x192x784xf16) <- (-1x784x192xf16)
        transpose_7 = paddle._C_ops.transpose(add__27, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_14 = paddle._C_ops.full([1], float('192'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_15 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_16 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_5 = [slice_2, full_14, full_15, full_16]

        # pd_op.reshape_: (-1x192x28x28xf16, 0x-1x192x784xf16) <- (-1x192x784xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_7, [x.reshape([]) for x in combine_5]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x192x28x28xf16) <- (-1x192x28x28xf16, -1x192x28x28xf16)
        add__28 = paddle._C_ops.add_(add__23, reshape__10)

        # pd_op.conv2d: (-1x64x28x28xf16) <- (-1x192x28x28xf16, 64x192x1x1xf16)
        conv2d_30 = paddle._C_ops.conv2d(add__28, parameter_130, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x28x28xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x28x28xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_30, parameter_131, parameter_132, parameter_133, parameter_134, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 64x32x3x3xf16)
        conv2d_31 = paddle._C_ops.conv2d(batch_norm__120, parameter_135, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x64x28x28xf16, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x28x28xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_31, parameter_136, parameter_137, parameter_138, parameter_139, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x28x28xf16) <- (-1x64x28x28xf16)
        relu__16 = paddle._C_ops.relu_(batch_norm__126)

        # pd_op.conv2d: (-1x64x28x28xf16) <- (-1x64x28x28xf16, 64x64x1x1xf16)
        conv2d_32 = paddle._C_ops.conv2d(relu__16, parameter_140, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x64x28x28xf16) <- (-1x64x28x28xf16, -1x64x28x28xf16)
        add__29 = paddle._C_ops.add_(batch_norm__120, conv2d_32)

        # builtin.combine: ([-1x192x28x28xf16, -1x64x28x28xf16]) <- (-1x192x28x28xf16, -1x64x28x28xf16)
        combine_6 = [add__28, add__29]

        # pd_op.full: (1xi32) <- ()
        full_17 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x256x28x28xf16) <- ([-1x192x28x28xf16, -1x64x28x28xf16], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_6, full_17)

        # pd_op.batch_norm_: (-1x256x28x28xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x28x28xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_0, parameter_141, parameter_142, parameter_143, parameter_144, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x512x28x28xf16) <- (-1x256x28x28xf16, 512x256x1x1xf16)
        conv2d_33 = paddle._C_ops.conv2d(batch_norm__132, parameter_145, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_22 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf16, 0x512xf16) <- (512xf16, 4xi64)
        reshape_24, reshape_25 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_146, full_int_array_22), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x28x28xf16) <- (-1x512x28x28xf16, 1x512x1x1xf16)
        add__30 = paddle._C_ops.add_(conv2d_33, reshape_24)

        # pd_op.relu_: (-1x512x28x28xf16) <- (-1x512x28x28xf16)
        relu__17 = paddle._C_ops.relu_(add__30)

        # pd_op.conv2d: (-1x256x28x28xf16) <- (-1x512x28x28xf16, 256x512x1x1xf16)
        conv2d_34 = paddle._C_ops.conv2d(relu__17, parameter_147, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_23 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf16, 0x256xf16) <- (256xf16, 4xi64)
        reshape_26, reshape_27 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_148, full_int_array_23), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x28x28xf16) <- (-1x256x28x28xf16, 1x256x1x1xf16)
        add__31 = paddle._C_ops.add_(conv2d_34, reshape_26)

        # pd_op.add_: (-1x256x28x28xf16) <- (-1x256x28x28xf16, -1x256x28x28xf16)
        add__32 = paddle._C_ops.add_(concat_0, add__31)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_24 = [2, 2]

        # pd_op.pool2d: (-1x256x14x14xf16) <- (-1x256x28x28xf16, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(add__32, full_int_array_24, [2, 2], [0, 0], True, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x256x14x14xf16, 384x256x1x1xf16)
        conv2d_35 = paddle._C_ops.conv2d(pool2d_2, parameter_149, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_35, parameter_150, parameter_151, parameter_152, parameter_153, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 384x32x3x3xf16)
        conv2d_36 = paddle._C_ops.conv2d(batch_norm__138, parameter_154, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 12, 'NCHW')

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_36, parameter_155, parameter_156, parameter_157, parameter_158, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x384x14x14xf16) <- (-1x384x14x14xf16)
        relu__18 = paddle._C_ops.relu_(batch_norm__144)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 384x384x1x1xf16)
        conv2d_37 = paddle._C_ops.conv2d(relu__18, parameter_159, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__33 = paddle._C_ops.add_(batch_norm__138, conv2d_37)

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__33, parameter_160, parameter_161, parameter_162, parameter_163, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x1152x14x14xf16) <- (-1x384x14x14xf16, 1152x384x1x1xf16)
        conv2d_38 = paddle._C_ops.conv2d(batch_norm__150, parameter_164, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_25 = [1, 1152, 1, 1]

        # pd_op.reshape: (1x1152x1x1xf16, 0x1152xf16) <- (1152xf16, 4xi64)
        reshape_28, reshape_29 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_165, full_int_array_25), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1152x14x14xf16) <- (-1x1152x14x14xf16, 1x1152x1x1xf16)
        add__34 = paddle._C_ops.add_(conv2d_38, reshape_28)

        # pd_op.relu_: (-1x1152x14x14xf16) <- (-1x1152x14x14xf16)
        relu__19 = paddle._C_ops.relu_(add__34)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x1152x14x14xf16, 384x1152x1x1xf16)
        conv2d_39 = paddle._C_ops.conv2d(relu__19, parameter_166, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_26 = [1, 384, 1, 1]

        # pd_op.reshape: (1x384x1x1xf16, 0x384xf16) <- (384xf16, 4xi64)
        reshape_30, reshape_31 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_167, full_int_array_26), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 1x384x1x1xf16)
        add__35 = paddle._C_ops.add_(conv2d_39, reshape_30)

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__36 = paddle._C_ops.add_(add__33, add__35)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 384x32x3x3xf16)
        conv2d_40 = paddle._C_ops.conv2d(add__36, parameter_168, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 12, 'NCHW')

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_40, parameter_169, parameter_170, parameter_171, parameter_172, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x384x14x14xf16) <- (-1x384x14x14xf16)
        relu__20 = paddle._C_ops.relu_(batch_norm__156)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 384x384x1x1xf16)
        conv2d_41 = paddle._C_ops.conv2d(relu__20, parameter_173, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__37 = paddle._C_ops.add_(add__36, conv2d_41)

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__37, parameter_174, parameter_175, parameter_176, parameter_177, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x1152x14x14xf16) <- (-1x384x14x14xf16, 1152x384x1x1xf16)
        conv2d_42 = paddle._C_ops.conv2d(batch_norm__162, parameter_178, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_27 = [1, 1152, 1, 1]

        # pd_op.reshape: (1x1152x1x1xf16, 0x1152xf16) <- (1152xf16, 4xi64)
        reshape_32, reshape_33 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_179, full_int_array_27), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1152x14x14xf16) <- (-1x1152x14x14xf16, 1x1152x1x1xf16)
        add__38 = paddle._C_ops.add_(conv2d_42, reshape_32)

        # pd_op.relu_: (-1x1152x14x14xf16) <- (-1x1152x14x14xf16)
        relu__21 = paddle._C_ops.relu_(add__38)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x1152x14x14xf16, 384x1152x1x1xf16)
        conv2d_43 = paddle._C_ops.conv2d(relu__21, parameter_180, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_28 = [1, 384, 1, 1]

        # pd_op.reshape: (1x384x1x1xf16, 0x384xf16) <- (384xf16, 4xi64)
        reshape_34, reshape_35 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_181, full_int_array_28), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 1x384x1x1xf16)
        add__39 = paddle._C_ops.add_(conv2d_43, reshape_34)

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__40 = paddle._C_ops.add_(add__37, add__39)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 384x32x3x3xf16)
        conv2d_44 = paddle._C_ops.conv2d(add__40, parameter_182, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 12, 'NCHW')

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_44, parameter_183, parameter_184, parameter_185, parameter_186, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x384x14x14xf16) <- (-1x384x14x14xf16)
        relu__22 = paddle._C_ops.relu_(batch_norm__168)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 384x384x1x1xf16)
        conv2d_45 = paddle._C_ops.conv2d(relu__22, parameter_187, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__41 = paddle._C_ops.add_(add__40, conv2d_45)

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__41, parameter_188, parameter_189, parameter_190, parameter_191, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x1152x14x14xf16) <- (-1x384x14x14xf16, 1152x384x1x1xf16)
        conv2d_46 = paddle._C_ops.conv2d(batch_norm__174, parameter_192, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_29 = [1, 1152, 1, 1]

        # pd_op.reshape: (1x1152x1x1xf16, 0x1152xf16) <- (1152xf16, 4xi64)
        reshape_36, reshape_37 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_193, full_int_array_29), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1152x14x14xf16) <- (-1x1152x14x14xf16, 1x1152x1x1xf16)
        add__42 = paddle._C_ops.add_(conv2d_46, reshape_36)

        # pd_op.relu_: (-1x1152x14x14xf16) <- (-1x1152x14x14xf16)
        relu__23 = paddle._C_ops.relu_(add__42)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x1152x14x14xf16, 384x1152x1x1xf16)
        conv2d_47 = paddle._C_ops.conv2d(relu__23, parameter_194, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_30 = [1, 384, 1, 1]

        # pd_op.reshape: (1x384x1x1xf16, 0x384xf16) <- (384xf16, 4xi64)
        reshape_38, reshape_39 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_195, full_int_array_30), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 1x384x1x1xf16)
        add__43 = paddle._C_ops.add_(conv2d_47, reshape_38)

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__44 = paddle._C_ops.add_(add__41, add__43)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 384x32x3x3xf16)
        conv2d_48 = paddle._C_ops.conv2d(add__44, parameter_196, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 12, 'NCHW')

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__180, batch_norm__181, batch_norm__182, batch_norm__183, batch_norm__184, batch_norm__185 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_48, parameter_197, parameter_198, parameter_199, parameter_200, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x384x14x14xf16) <- (-1x384x14x14xf16)
        relu__24 = paddle._C_ops.relu_(batch_norm__180)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 384x384x1x1xf16)
        conv2d_49 = paddle._C_ops.conv2d(relu__24, parameter_201, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__45 = paddle._C_ops.add_(add__44, conv2d_49)

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__186, batch_norm__187, batch_norm__188, batch_norm__189, batch_norm__190, batch_norm__191 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__45, parameter_202, parameter_203, parameter_204, parameter_205, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x1152x14x14xf16) <- (-1x384x14x14xf16, 1152x384x1x1xf16)
        conv2d_50 = paddle._C_ops.conv2d(batch_norm__186, parameter_206, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_31 = [1, 1152, 1, 1]

        # pd_op.reshape: (1x1152x1x1xf16, 0x1152xf16) <- (1152xf16, 4xi64)
        reshape_40, reshape_41 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_207, full_int_array_31), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1152x14x14xf16) <- (-1x1152x14x14xf16, 1x1152x1x1xf16)
        add__46 = paddle._C_ops.add_(conv2d_50, reshape_40)

        # pd_op.relu_: (-1x1152x14x14xf16) <- (-1x1152x14x14xf16)
        relu__25 = paddle._C_ops.relu_(add__46)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x1152x14x14xf16, 384x1152x1x1xf16)
        conv2d_51 = paddle._C_ops.conv2d(relu__25, parameter_208, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_32 = [1, 384, 1, 1]

        # pd_op.reshape: (1x384x1x1xf16, 0x384xf16) <- (384xf16, 4xi64)
        reshape_42, reshape_43 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_209, full_int_array_32), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 1x384x1x1xf16)
        add__47 = paddle._C_ops.add_(conv2d_51, reshape_42)

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__48 = paddle._C_ops.add_(add__45, add__47)

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__192, batch_norm__193, batch_norm__194, batch_norm__195, batch_norm__196, batch_norm__197 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__48, parameter_210, parameter_211, parameter_212, parameter_213, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (4xi32) <- (-1x384x14x14xf16)
        shape_3 = paddle._C_ops.shape(paddle.cast(batch_norm__192, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_33 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_34 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(shape_3, [0], full_int_array_33, full_int_array_34, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_18 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_19 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_7 = [slice_3, full_18, full_19]

        # pd_op.reshape_: (-1x384x196xf16, 0x-1x384x14x14xf16) <- (-1x384x14x14xf16, [1xi32, 1xi32, 1xi32])
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__192, [x.reshape([]) for x in combine_7]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x384xf16) <- (-1x384x196xf16)
        transpose_8 = paddle._C_ops.transpose(reshape__12, [0, 2, 1])

        # pd_op.shape: (3xi32) <- (-1x196x384xf16)
        shape_4 = paddle._C_ops.shape(paddle.cast(transpose_8, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_35 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_36 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(shape_4, [0], full_int_array_35, full_int_array_36, [1], [0])

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_6 = paddle._C_ops.matmul(transpose_8, parameter_214, False, False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__49 = paddle._C_ops.add_(matmul_6, parameter_215)

        # pd_op.full: (1xi32) <- ()
        full_20 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_21 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_22 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_8 = [slice_4, full_20, full_21, full_22]

        # pd_op.reshape_: (-1x196x12x32xf16, 0x-1x196x384xf16) <- (-1x196x384xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__49, [x.reshape([]) for x in combine_8]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x12x196x32xf16) <- (-1x196x12x32xf16)
        transpose_9 = paddle._C_ops.transpose(reshape__14, [0, 2, 1, 3])

        # pd_op.transpose: (-1x384x196xf16) <- (-1x196x384xf16)
        transpose_10 = paddle._C_ops.transpose(transpose_8, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_37 = [2]

        # pd_op.unsqueeze_: (-1x384x1x196xf16, None) <- (-1x384x196xf16, 1xi64)
        unsqueeze__2, unsqueeze__3 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(transpose_10, full_int_array_37), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_38 = [1, 4]

        # pd_op.pool2d: (-1x384x1x49xf16) <- (-1x384x1x196xf16, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(unsqueeze__2, full_int_array_38, [1, 4], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_39 = [2]

        # pd_op.squeeze_: (-1x384x49xf16, None) <- (-1x384x1x49xf16, 1xi64)
        squeeze__2, squeeze__3 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_3, full_int_array_39), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x384x49xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x49xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__198, batch_norm__199, batch_norm__200, batch_norm__201, batch_norm__202, batch_norm__203 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(squeeze__2, parameter_216, parameter_217, parameter_218, parameter_219, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.transpose: (-1x49x384xf16) <- (-1x384x49xf16)
        transpose_11 = paddle._C_ops.transpose(batch_norm__198, [0, 2, 1])

        # pd_op.matmul: (-1x49x384xf16) <- (-1x49x384xf16, 384x384xf16)
        matmul_7 = paddle._C_ops.matmul(transpose_11, parameter_220, False, False)

        # pd_op.add_: (-1x49x384xf16) <- (-1x49x384xf16, 384xf16)
        add__50 = paddle._C_ops.add_(matmul_7, parameter_221)

        # pd_op.full: (1xi32) <- ()
        full_23 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_24 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_25 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_9 = [slice_4, full_23, full_24, full_25]

        # pd_op.reshape_: (-1x49x12x32xf16, 0x-1x49x384xf16) <- (-1x49x384xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__16, reshape__17 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__50, [x.reshape([]) for x in combine_9]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x12x32x49xf16) <- (-1x49x12x32xf16)
        transpose_12 = paddle._C_ops.transpose(reshape__16, [0, 2, 3, 1])

        # pd_op.matmul: (-1x49x384xf16) <- (-1x49x384xf16, 384x384xf16)
        matmul_8 = paddle._C_ops.matmul(transpose_11, parameter_222, False, False)

        # pd_op.add_: (-1x49x384xf16) <- (-1x49x384xf16, 384xf16)
        add__51 = paddle._C_ops.add_(matmul_8, parameter_223)

        # pd_op.full: (1xi32) <- ()
        full_26 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_27 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_28 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_10 = [slice_4, full_26, full_27, full_28]

        # pd_op.reshape_: (-1x49x12x32xf16, 0x-1x49x384xf16) <- (-1x49x384xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__18, reshape__19 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__51, [x.reshape([]) for x in combine_10]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x12x49x32xf16) <- (-1x49x12x32xf16)
        transpose_13 = paddle._C_ops.transpose(reshape__18, [0, 2, 1, 3])

        # pd_op.matmul: (-1x12x196x49xf16) <- (-1x12x196x32xf16, -1x12x32x49xf16)
        matmul_9 = paddle._C_ops.matmul(transpose_9, transpose_12, False, False)

        # pd_op.full: (1xf32) <- ()
        full_29 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x12x196x49xf16) <- (-1x12x196x49xf16, 1xf32)
        scale__1 = paddle._C_ops.scale_(matmul_9, full_29, float('0'), True)

        # pd_op.softmax_: (-1x12x196x49xf16) <- (-1x12x196x49xf16)
        softmax__1 = paddle._C_ops.softmax_(scale__1, -1)

        # pd_op.matmul: (-1x12x196x32xf16) <- (-1x12x196x49xf16, -1x12x49x32xf16)
        matmul_10 = paddle._C_ops.matmul(softmax__1, transpose_13, False, False)

        # pd_op.transpose: (-1x196x12x32xf16) <- (-1x12x196x32xf16)
        transpose_14 = paddle._C_ops.transpose(matmul_10, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_30 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_31 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_11 = [slice_4, full_30, full_31]

        # pd_op.reshape_: (-1x196x384xf16, 0x-1x196x12x32xf16) <- (-1x196x12x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__20, reshape__21 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_14, [x.reshape([]) for x in combine_11]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_11 = paddle._C_ops.matmul(reshape__20, parameter_224, False, False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__52 = paddle._C_ops.add_(matmul_11, parameter_225)

        # pd_op.shape: (3xi32) <- (-1x196x384xf16)
        shape_5 = paddle._C_ops.shape(paddle.cast(add__52, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_40 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_41 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(shape_5, [0], full_int_array_40, full_int_array_41, [1], [0])

        # pd_op.transpose: (-1x384x196xf16) <- (-1x196x384xf16)
        transpose_15 = paddle._C_ops.transpose(add__52, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_32 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_33 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_34 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_12 = [slice_5, full_32, full_33, full_34]

        # pd_op.reshape_: (-1x384x14x14xf16, 0x-1x384x196xf16) <- (-1x384x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__22, reshape__23 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_15, [x.reshape([]) for x in combine_12]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__53 = paddle._C_ops.add_(add__48, reshape__22)

        # pd_op.conv2d: (-1x128x14x14xf16) <- (-1x384x14x14xf16, 128x384x1x1xf16)
        conv2d_52 = paddle._C_ops.conv2d(add__53, parameter_226, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x14x14xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x14x14xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__204, batch_norm__205, batch_norm__206, batch_norm__207, batch_norm__208, batch_norm__209 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_52, parameter_227, parameter_228, parameter_229, parameter_230, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x128x14x14xf16) <- (-1x128x14x14xf16, 128x32x3x3xf16)
        conv2d_53 = paddle._C_ops.conv2d(batch_norm__204, parameter_231, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 4, 'NCHW')

        # pd_op.batch_norm_: (-1x128x14x14xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x14x14xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__210, batch_norm__211, batch_norm__212, batch_norm__213, batch_norm__214, batch_norm__215 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_53, parameter_232, parameter_233, parameter_234, parameter_235, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x14x14xf16) <- (-1x128x14x14xf16)
        relu__26 = paddle._C_ops.relu_(batch_norm__210)

        # pd_op.conv2d: (-1x128x14x14xf16) <- (-1x128x14x14xf16, 128x128x1x1xf16)
        conv2d_54 = paddle._C_ops.conv2d(relu__26, parameter_236, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x14x14xf16) <- (-1x128x14x14xf16, -1x128x14x14xf16)
        add__54 = paddle._C_ops.add_(batch_norm__204, conv2d_54)

        # builtin.combine: ([-1x384x14x14xf16, -1x128x14x14xf16]) <- (-1x384x14x14xf16, -1x128x14x14xf16)
        combine_13 = [add__53, add__54]

        # pd_op.full: (1xi32) <- ()
        full_35 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x512x14x14xf16) <- ([-1x384x14x14xf16, -1x128x14x14xf16], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_13, full_35)

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__216, batch_norm__217, batch_norm__218, batch_norm__219, batch_norm__220, batch_norm__221 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_1, parameter_237, parameter_238, parameter_239, parameter_240, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x512x14x14xf16, 1024x512x1x1xf16)
        conv2d_55 = paddle._C_ops.conv2d(batch_norm__216, parameter_241, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_42 = [1, 1024, 1, 1]

        # pd_op.reshape: (1x1024x1x1xf16, 0x1024xf16) <- (1024xf16, 4xi64)
        reshape_44, reshape_45 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_242, full_int_array_42), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, 1x1024x1x1xf16)
        add__55 = paddle._C_ops.add_(conv2d_55, reshape_44)

        # pd_op.relu_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16)
        relu__27 = paddle._C_ops.relu_(add__55)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x1024x14x14xf16, 512x1024x1x1xf16)
        conv2d_56 = paddle._C_ops.conv2d(relu__27, parameter_243, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_43 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf16, 0x512xf16) <- (512xf16, 4xi64)
        reshape_46, reshape_47 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_244, full_int_array_43), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x14x14xf16) <- (-1x512x14x14xf16, 1x512x1x1xf16)
        add__56 = paddle._C_ops.add_(conv2d_56, reshape_46)

        # pd_op.add_: (-1x512x14x14xf16) <- (-1x512x14x14xf16, -1x512x14x14xf16)
        add__57 = paddle._C_ops.add_(concat_1, add__56)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x512x14x14xf16, 384x512x1x1xf16)
        conv2d_57 = paddle._C_ops.conv2d(add__57, parameter_245, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__222, batch_norm__223, batch_norm__224, batch_norm__225, batch_norm__226, batch_norm__227 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_57, parameter_246, parameter_247, parameter_248, parameter_249, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 384x32x3x3xf16)
        conv2d_58 = paddle._C_ops.conv2d(batch_norm__222, parameter_250, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 12, 'NCHW')

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__228, batch_norm__229, batch_norm__230, batch_norm__231, batch_norm__232, batch_norm__233 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_58, parameter_251, parameter_252, parameter_253, parameter_254, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x384x14x14xf16) <- (-1x384x14x14xf16)
        relu__28 = paddle._C_ops.relu_(batch_norm__228)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 384x384x1x1xf16)
        conv2d_59 = paddle._C_ops.conv2d(relu__28, parameter_255, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__58 = paddle._C_ops.add_(batch_norm__222, conv2d_59)

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__234, batch_norm__235, batch_norm__236, batch_norm__237, batch_norm__238, batch_norm__239 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__58, parameter_256, parameter_257, parameter_258, parameter_259, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x1152x14x14xf16) <- (-1x384x14x14xf16, 1152x384x1x1xf16)
        conv2d_60 = paddle._C_ops.conv2d(batch_norm__234, parameter_260, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_44 = [1, 1152, 1, 1]

        # pd_op.reshape: (1x1152x1x1xf16, 0x1152xf16) <- (1152xf16, 4xi64)
        reshape_48, reshape_49 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_261, full_int_array_44), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1152x14x14xf16) <- (-1x1152x14x14xf16, 1x1152x1x1xf16)
        add__59 = paddle._C_ops.add_(conv2d_60, reshape_48)

        # pd_op.relu_: (-1x1152x14x14xf16) <- (-1x1152x14x14xf16)
        relu__29 = paddle._C_ops.relu_(add__59)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x1152x14x14xf16, 384x1152x1x1xf16)
        conv2d_61 = paddle._C_ops.conv2d(relu__29, parameter_262, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_45 = [1, 384, 1, 1]

        # pd_op.reshape: (1x384x1x1xf16, 0x384xf16) <- (384xf16, 4xi64)
        reshape_50, reshape_51 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_263, full_int_array_45), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 1x384x1x1xf16)
        add__60 = paddle._C_ops.add_(conv2d_61, reshape_50)

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__61 = paddle._C_ops.add_(add__58, add__60)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 384x32x3x3xf16)
        conv2d_62 = paddle._C_ops.conv2d(add__61, parameter_264, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 12, 'NCHW')

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__240, batch_norm__241, batch_norm__242, batch_norm__243, batch_norm__244, batch_norm__245 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_62, parameter_265, parameter_266, parameter_267, parameter_268, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x384x14x14xf16) <- (-1x384x14x14xf16)
        relu__30 = paddle._C_ops.relu_(batch_norm__240)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 384x384x1x1xf16)
        conv2d_63 = paddle._C_ops.conv2d(relu__30, parameter_269, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__62 = paddle._C_ops.add_(add__61, conv2d_63)

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__246, batch_norm__247, batch_norm__248, batch_norm__249, batch_norm__250, batch_norm__251 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__62, parameter_270, parameter_271, parameter_272, parameter_273, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x1152x14x14xf16) <- (-1x384x14x14xf16, 1152x384x1x1xf16)
        conv2d_64 = paddle._C_ops.conv2d(batch_norm__246, parameter_274, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_46 = [1, 1152, 1, 1]

        # pd_op.reshape: (1x1152x1x1xf16, 0x1152xf16) <- (1152xf16, 4xi64)
        reshape_52, reshape_53 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_275, full_int_array_46), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1152x14x14xf16) <- (-1x1152x14x14xf16, 1x1152x1x1xf16)
        add__63 = paddle._C_ops.add_(conv2d_64, reshape_52)

        # pd_op.relu_: (-1x1152x14x14xf16) <- (-1x1152x14x14xf16)
        relu__31 = paddle._C_ops.relu_(add__63)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x1152x14x14xf16, 384x1152x1x1xf16)
        conv2d_65 = paddle._C_ops.conv2d(relu__31, parameter_276, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_47 = [1, 384, 1, 1]

        # pd_op.reshape: (1x384x1x1xf16, 0x384xf16) <- (384xf16, 4xi64)
        reshape_54, reshape_55 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_277, full_int_array_47), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 1x384x1x1xf16)
        add__64 = paddle._C_ops.add_(conv2d_65, reshape_54)

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__65 = paddle._C_ops.add_(add__62, add__64)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 384x32x3x3xf16)
        conv2d_66 = paddle._C_ops.conv2d(add__65, parameter_278, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 12, 'NCHW')

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__252, batch_norm__253, batch_norm__254, batch_norm__255, batch_norm__256, batch_norm__257 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_66, parameter_279, parameter_280, parameter_281, parameter_282, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x384x14x14xf16) <- (-1x384x14x14xf16)
        relu__32 = paddle._C_ops.relu_(batch_norm__252)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 384x384x1x1xf16)
        conv2d_67 = paddle._C_ops.conv2d(relu__32, parameter_283, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__66 = paddle._C_ops.add_(add__65, conv2d_67)

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__258, batch_norm__259, batch_norm__260, batch_norm__261, batch_norm__262, batch_norm__263 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__66, parameter_284, parameter_285, parameter_286, parameter_287, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x1152x14x14xf16) <- (-1x384x14x14xf16, 1152x384x1x1xf16)
        conv2d_68 = paddle._C_ops.conv2d(batch_norm__258, parameter_288, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_48 = [1, 1152, 1, 1]

        # pd_op.reshape: (1x1152x1x1xf16, 0x1152xf16) <- (1152xf16, 4xi64)
        reshape_56, reshape_57 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_289, full_int_array_48), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1152x14x14xf16) <- (-1x1152x14x14xf16, 1x1152x1x1xf16)
        add__67 = paddle._C_ops.add_(conv2d_68, reshape_56)

        # pd_op.relu_: (-1x1152x14x14xf16) <- (-1x1152x14x14xf16)
        relu__33 = paddle._C_ops.relu_(add__67)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x1152x14x14xf16, 384x1152x1x1xf16)
        conv2d_69 = paddle._C_ops.conv2d(relu__33, parameter_290, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_49 = [1, 384, 1, 1]

        # pd_op.reshape: (1x384x1x1xf16, 0x384xf16) <- (384xf16, 4xi64)
        reshape_58, reshape_59 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_291, full_int_array_49), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 1x384x1x1xf16)
        add__68 = paddle._C_ops.add_(conv2d_69, reshape_58)

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__69 = paddle._C_ops.add_(add__66, add__68)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 384x32x3x3xf16)
        conv2d_70 = paddle._C_ops.conv2d(add__69, parameter_292, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 12, 'NCHW')

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__264, batch_norm__265, batch_norm__266, batch_norm__267, batch_norm__268, batch_norm__269 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_70, parameter_293, parameter_294, parameter_295, parameter_296, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x384x14x14xf16) <- (-1x384x14x14xf16)
        relu__34 = paddle._C_ops.relu_(batch_norm__264)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 384x384x1x1xf16)
        conv2d_71 = paddle._C_ops.conv2d(relu__34, parameter_297, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__70 = paddle._C_ops.add_(add__69, conv2d_71)

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__270, batch_norm__271, batch_norm__272, batch_norm__273, batch_norm__274, batch_norm__275 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__70, parameter_298, parameter_299, parameter_300, parameter_301, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x1152x14x14xf16) <- (-1x384x14x14xf16, 1152x384x1x1xf16)
        conv2d_72 = paddle._C_ops.conv2d(batch_norm__270, parameter_302, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_50 = [1, 1152, 1, 1]

        # pd_op.reshape: (1x1152x1x1xf16, 0x1152xf16) <- (1152xf16, 4xi64)
        reshape_60, reshape_61 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_303, full_int_array_50), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1152x14x14xf16) <- (-1x1152x14x14xf16, 1x1152x1x1xf16)
        add__71 = paddle._C_ops.add_(conv2d_72, reshape_60)

        # pd_op.relu_: (-1x1152x14x14xf16) <- (-1x1152x14x14xf16)
        relu__35 = paddle._C_ops.relu_(add__71)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x1152x14x14xf16, 384x1152x1x1xf16)
        conv2d_73 = paddle._C_ops.conv2d(relu__35, parameter_304, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_51 = [1, 384, 1, 1]

        # pd_op.reshape: (1x384x1x1xf16, 0x384xf16) <- (384xf16, 4xi64)
        reshape_62, reshape_63 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_305, full_int_array_51), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 1x384x1x1xf16)
        add__72 = paddle._C_ops.add_(conv2d_73, reshape_62)

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__73 = paddle._C_ops.add_(add__70, add__72)

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__276, batch_norm__277, batch_norm__278, batch_norm__279, batch_norm__280, batch_norm__281 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__73, parameter_306, parameter_307, parameter_308, parameter_309, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (4xi32) <- (-1x384x14x14xf16)
        shape_6 = paddle._C_ops.shape(paddle.cast(batch_norm__276, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_52 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_53 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(shape_6, [0], full_int_array_52, full_int_array_53, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_36 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_37 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_14 = [slice_6, full_36, full_37]

        # pd_op.reshape_: (-1x384x196xf16, 0x-1x384x14x14xf16) <- (-1x384x14x14xf16, [1xi32, 1xi32, 1xi32])
        reshape__24, reshape__25 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__276, [x.reshape([]) for x in combine_14]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x384xf16) <- (-1x384x196xf16)
        transpose_16 = paddle._C_ops.transpose(reshape__24, [0, 2, 1])

        # pd_op.shape: (3xi32) <- (-1x196x384xf16)
        shape_7 = paddle._C_ops.shape(paddle.cast(transpose_16, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_54 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_55 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(shape_7, [0], full_int_array_54, full_int_array_55, [1], [0])

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_12 = paddle._C_ops.matmul(transpose_16, parameter_310, False, False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__74 = paddle._C_ops.add_(matmul_12, parameter_311)

        # pd_op.full: (1xi32) <- ()
        full_38 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_39 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_40 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_15 = [slice_7, full_38, full_39, full_40]

        # pd_op.reshape_: (-1x196x12x32xf16, 0x-1x196x384xf16) <- (-1x196x384xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__26, reshape__27 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__74, [x.reshape([]) for x in combine_15]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x12x196x32xf16) <- (-1x196x12x32xf16)
        transpose_17 = paddle._C_ops.transpose(reshape__26, [0, 2, 1, 3])

        # pd_op.transpose: (-1x384x196xf16) <- (-1x196x384xf16)
        transpose_18 = paddle._C_ops.transpose(transpose_16, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_56 = [2]

        # pd_op.unsqueeze_: (-1x384x1x196xf16, None) <- (-1x384x196xf16, 1xi64)
        unsqueeze__4, unsqueeze__5 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(transpose_18, full_int_array_56), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_57 = [1, 4]

        # pd_op.pool2d: (-1x384x1x49xf16) <- (-1x384x1x196xf16, 2xi64)
        pool2d_4 = paddle._C_ops.pool2d(unsqueeze__4, full_int_array_57, [1, 4], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_58 = [2]

        # pd_op.squeeze_: (-1x384x49xf16, None) <- (-1x384x1x49xf16, 1xi64)
        squeeze__4, squeeze__5 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_4, full_int_array_58), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x384x49xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x49xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__282, batch_norm__283, batch_norm__284, batch_norm__285, batch_norm__286, batch_norm__287 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(squeeze__4, parameter_312, parameter_313, parameter_314, parameter_315, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.transpose: (-1x49x384xf16) <- (-1x384x49xf16)
        transpose_19 = paddle._C_ops.transpose(batch_norm__282, [0, 2, 1])

        # pd_op.matmul: (-1x49x384xf16) <- (-1x49x384xf16, 384x384xf16)
        matmul_13 = paddle._C_ops.matmul(transpose_19, parameter_316, False, False)

        # pd_op.add_: (-1x49x384xf16) <- (-1x49x384xf16, 384xf16)
        add__75 = paddle._C_ops.add_(matmul_13, parameter_317)

        # pd_op.full: (1xi32) <- ()
        full_41 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_42 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_43 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_16 = [slice_7, full_41, full_42, full_43]

        # pd_op.reshape_: (-1x49x12x32xf16, 0x-1x49x384xf16) <- (-1x49x384xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__28, reshape__29 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__75, [x.reshape([]) for x in combine_16]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x12x32x49xf16) <- (-1x49x12x32xf16)
        transpose_20 = paddle._C_ops.transpose(reshape__28, [0, 2, 3, 1])

        # pd_op.matmul: (-1x49x384xf16) <- (-1x49x384xf16, 384x384xf16)
        matmul_14 = paddle._C_ops.matmul(transpose_19, parameter_318, False, False)

        # pd_op.add_: (-1x49x384xf16) <- (-1x49x384xf16, 384xf16)
        add__76 = paddle._C_ops.add_(matmul_14, parameter_319)

        # pd_op.full: (1xi32) <- ()
        full_44 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_45 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_46 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_17 = [slice_7, full_44, full_45, full_46]

        # pd_op.reshape_: (-1x49x12x32xf16, 0x-1x49x384xf16) <- (-1x49x384xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__30, reshape__31 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__76, [x.reshape([]) for x in combine_17]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x12x49x32xf16) <- (-1x49x12x32xf16)
        transpose_21 = paddle._C_ops.transpose(reshape__30, [0, 2, 1, 3])

        # pd_op.matmul: (-1x12x196x49xf16) <- (-1x12x196x32xf16, -1x12x32x49xf16)
        matmul_15 = paddle._C_ops.matmul(transpose_17, transpose_20, False, False)

        # pd_op.full: (1xf32) <- ()
        full_47 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x12x196x49xf16) <- (-1x12x196x49xf16, 1xf32)
        scale__2 = paddle._C_ops.scale_(matmul_15, full_47, float('0'), True)

        # pd_op.softmax_: (-1x12x196x49xf16) <- (-1x12x196x49xf16)
        softmax__2 = paddle._C_ops.softmax_(scale__2, -1)

        # pd_op.matmul: (-1x12x196x32xf16) <- (-1x12x196x49xf16, -1x12x49x32xf16)
        matmul_16 = paddle._C_ops.matmul(softmax__2, transpose_21, False, False)

        # pd_op.transpose: (-1x196x12x32xf16) <- (-1x12x196x32xf16)
        transpose_22 = paddle._C_ops.transpose(matmul_16, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_48 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_49 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_18 = [slice_7, full_48, full_49]

        # pd_op.reshape_: (-1x196x384xf16, 0x-1x196x12x32xf16) <- (-1x196x12x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__32, reshape__33 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_22, [x.reshape([]) for x in combine_18]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_17 = paddle._C_ops.matmul(reshape__32, parameter_320, False, False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__77 = paddle._C_ops.add_(matmul_17, parameter_321)

        # pd_op.shape: (3xi32) <- (-1x196x384xf16)
        shape_8 = paddle._C_ops.shape(paddle.cast(add__77, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_59 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_60 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(shape_8, [0], full_int_array_59, full_int_array_60, [1], [0])

        # pd_op.transpose: (-1x384x196xf16) <- (-1x196x384xf16)
        transpose_23 = paddle._C_ops.transpose(add__77, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_50 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_51 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_52 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_19 = [slice_8, full_50, full_51, full_52]

        # pd_op.reshape_: (-1x384x14x14xf16, 0x-1x384x196xf16) <- (-1x384x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__34, reshape__35 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_23, [x.reshape([]) for x in combine_19]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__78 = paddle._C_ops.add_(add__73, reshape__34)

        # pd_op.conv2d: (-1x128x14x14xf16) <- (-1x384x14x14xf16, 128x384x1x1xf16)
        conv2d_74 = paddle._C_ops.conv2d(add__78, parameter_322, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x14x14xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x14x14xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__288, batch_norm__289, batch_norm__290, batch_norm__291, batch_norm__292, batch_norm__293 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_74, parameter_323, parameter_324, parameter_325, parameter_326, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x128x14x14xf16) <- (-1x128x14x14xf16, 128x32x3x3xf16)
        conv2d_75 = paddle._C_ops.conv2d(batch_norm__288, parameter_327, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 4, 'NCHW')

        # pd_op.batch_norm_: (-1x128x14x14xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x14x14xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__294, batch_norm__295, batch_norm__296, batch_norm__297, batch_norm__298, batch_norm__299 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_75, parameter_328, parameter_329, parameter_330, parameter_331, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x14x14xf16) <- (-1x128x14x14xf16)
        relu__36 = paddle._C_ops.relu_(batch_norm__294)

        # pd_op.conv2d: (-1x128x14x14xf16) <- (-1x128x14x14xf16, 128x128x1x1xf16)
        conv2d_76 = paddle._C_ops.conv2d(relu__36, parameter_332, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x14x14xf16) <- (-1x128x14x14xf16, -1x128x14x14xf16)
        add__79 = paddle._C_ops.add_(batch_norm__288, conv2d_76)

        # builtin.combine: ([-1x384x14x14xf16, -1x128x14x14xf16]) <- (-1x384x14x14xf16, -1x128x14x14xf16)
        combine_20 = [add__78, add__79]

        # pd_op.full: (1xi32) <- ()
        full_53 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x512x14x14xf16) <- ([-1x384x14x14xf16, -1x128x14x14xf16], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_20, full_53)

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__300, batch_norm__301, batch_norm__302, batch_norm__303, batch_norm__304, batch_norm__305 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_2, parameter_333, parameter_334, parameter_335, parameter_336, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x512x14x14xf16, 1024x512x1x1xf16)
        conv2d_77 = paddle._C_ops.conv2d(batch_norm__300, parameter_337, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_61 = [1, 1024, 1, 1]

        # pd_op.reshape: (1x1024x1x1xf16, 0x1024xf16) <- (1024xf16, 4xi64)
        reshape_64, reshape_65 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_338, full_int_array_61), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, 1x1024x1x1xf16)
        add__80 = paddle._C_ops.add_(conv2d_77, reshape_64)

        # pd_op.relu_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16)
        relu__37 = paddle._C_ops.relu_(add__80)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x1024x14x14xf16, 512x1024x1x1xf16)
        conv2d_78 = paddle._C_ops.conv2d(relu__37, parameter_339, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_62 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf16, 0x512xf16) <- (512xf16, 4xi64)
        reshape_66, reshape_67 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_340, full_int_array_62), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x14x14xf16) <- (-1x512x14x14xf16, 1x512x1x1xf16)
        add__81 = paddle._C_ops.add_(conv2d_78, reshape_66)

        # pd_op.add_: (-1x512x14x14xf16) <- (-1x512x14x14xf16, -1x512x14x14xf16)
        add__82 = paddle._C_ops.add_(concat_2, add__81)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x512x14x14xf16, 384x512x1x1xf16)
        conv2d_79 = paddle._C_ops.conv2d(add__82, parameter_341, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__306, batch_norm__307, batch_norm__308, batch_norm__309, batch_norm__310, batch_norm__311 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_79, parameter_342, parameter_343, parameter_344, parameter_345, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 384x32x3x3xf16)
        conv2d_80 = paddle._C_ops.conv2d(batch_norm__306, parameter_346, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 12, 'NCHW')

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__312, batch_norm__313, batch_norm__314, batch_norm__315, batch_norm__316, batch_norm__317 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_80, parameter_347, parameter_348, parameter_349, parameter_350, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x384x14x14xf16) <- (-1x384x14x14xf16)
        relu__38 = paddle._C_ops.relu_(batch_norm__312)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 384x384x1x1xf16)
        conv2d_81 = paddle._C_ops.conv2d(relu__38, parameter_351, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__83 = paddle._C_ops.add_(batch_norm__306, conv2d_81)

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__318, batch_norm__319, batch_norm__320, batch_norm__321, batch_norm__322, batch_norm__323 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__83, parameter_352, parameter_353, parameter_354, parameter_355, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x1152x14x14xf16) <- (-1x384x14x14xf16, 1152x384x1x1xf16)
        conv2d_82 = paddle._C_ops.conv2d(batch_norm__318, parameter_356, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_63 = [1, 1152, 1, 1]

        # pd_op.reshape: (1x1152x1x1xf16, 0x1152xf16) <- (1152xf16, 4xi64)
        reshape_68, reshape_69 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_357, full_int_array_63), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1152x14x14xf16) <- (-1x1152x14x14xf16, 1x1152x1x1xf16)
        add__84 = paddle._C_ops.add_(conv2d_82, reshape_68)

        # pd_op.relu_: (-1x1152x14x14xf16) <- (-1x1152x14x14xf16)
        relu__39 = paddle._C_ops.relu_(add__84)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x1152x14x14xf16, 384x1152x1x1xf16)
        conv2d_83 = paddle._C_ops.conv2d(relu__39, parameter_358, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_64 = [1, 384, 1, 1]

        # pd_op.reshape: (1x384x1x1xf16, 0x384xf16) <- (384xf16, 4xi64)
        reshape_70, reshape_71 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_359, full_int_array_64), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 1x384x1x1xf16)
        add__85 = paddle._C_ops.add_(conv2d_83, reshape_70)

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__86 = paddle._C_ops.add_(add__83, add__85)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 384x32x3x3xf16)
        conv2d_84 = paddle._C_ops.conv2d(add__86, parameter_360, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 12, 'NCHW')

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__324, batch_norm__325, batch_norm__326, batch_norm__327, batch_norm__328, batch_norm__329 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_84, parameter_361, parameter_362, parameter_363, parameter_364, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x384x14x14xf16) <- (-1x384x14x14xf16)
        relu__40 = paddle._C_ops.relu_(batch_norm__324)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 384x384x1x1xf16)
        conv2d_85 = paddle._C_ops.conv2d(relu__40, parameter_365, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__87 = paddle._C_ops.add_(add__86, conv2d_85)

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__330, batch_norm__331, batch_norm__332, batch_norm__333, batch_norm__334, batch_norm__335 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__87, parameter_366, parameter_367, parameter_368, parameter_369, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x1152x14x14xf16) <- (-1x384x14x14xf16, 1152x384x1x1xf16)
        conv2d_86 = paddle._C_ops.conv2d(batch_norm__330, parameter_370, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_65 = [1, 1152, 1, 1]

        # pd_op.reshape: (1x1152x1x1xf16, 0x1152xf16) <- (1152xf16, 4xi64)
        reshape_72, reshape_73 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_371, full_int_array_65), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1152x14x14xf16) <- (-1x1152x14x14xf16, 1x1152x1x1xf16)
        add__88 = paddle._C_ops.add_(conv2d_86, reshape_72)

        # pd_op.relu_: (-1x1152x14x14xf16) <- (-1x1152x14x14xf16)
        relu__41 = paddle._C_ops.relu_(add__88)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x1152x14x14xf16, 384x1152x1x1xf16)
        conv2d_87 = paddle._C_ops.conv2d(relu__41, parameter_372, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_66 = [1, 384, 1, 1]

        # pd_op.reshape: (1x384x1x1xf16, 0x384xf16) <- (384xf16, 4xi64)
        reshape_74, reshape_75 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_373, full_int_array_66), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 1x384x1x1xf16)
        add__89 = paddle._C_ops.add_(conv2d_87, reshape_74)

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__90 = paddle._C_ops.add_(add__87, add__89)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 384x32x3x3xf16)
        conv2d_88 = paddle._C_ops.conv2d(add__90, parameter_374, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 12, 'NCHW')

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__336, batch_norm__337, batch_norm__338, batch_norm__339, batch_norm__340, batch_norm__341 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_88, parameter_375, parameter_376, parameter_377, parameter_378, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x384x14x14xf16) <- (-1x384x14x14xf16)
        relu__42 = paddle._C_ops.relu_(batch_norm__336)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 384x384x1x1xf16)
        conv2d_89 = paddle._C_ops.conv2d(relu__42, parameter_379, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__91 = paddle._C_ops.add_(add__90, conv2d_89)

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__342, batch_norm__343, batch_norm__344, batch_norm__345, batch_norm__346, batch_norm__347 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__91, parameter_380, parameter_381, parameter_382, parameter_383, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x1152x14x14xf16) <- (-1x384x14x14xf16, 1152x384x1x1xf16)
        conv2d_90 = paddle._C_ops.conv2d(batch_norm__342, parameter_384, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_67 = [1, 1152, 1, 1]

        # pd_op.reshape: (1x1152x1x1xf16, 0x1152xf16) <- (1152xf16, 4xi64)
        reshape_76, reshape_77 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_385, full_int_array_67), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1152x14x14xf16) <- (-1x1152x14x14xf16, 1x1152x1x1xf16)
        add__92 = paddle._C_ops.add_(conv2d_90, reshape_76)

        # pd_op.relu_: (-1x1152x14x14xf16) <- (-1x1152x14x14xf16)
        relu__43 = paddle._C_ops.relu_(add__92)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x1152x14x14xf16, 384x1152x1x1xf16)
        conv2d_91 = paddle._C_ops.conv2d(relu__43, parameter_386, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_68 = [1, 384, 1, 1]

        # pd_op.reshape: (1x384x1x1xf16, 0x384xf16) <- (384xf16, 4xi64)
        reshape_78, reshape_79 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_387, full_int_array_68), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 1x384x1x1xf16)
        add__93 = paddle._C_ops.add_(conv2d_91, reshape_78)

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__94 = paddle._C_ops.add_(add__91, add__93)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 384x32x3x3xf16)
        conv2d_92 = paddle._C_ops.conv2d(add__94, parameter_388, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 12, 'NCHW')

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__348, batch_norm__349, batch_norm__350, batch_norm__351, batch_norm__352, batch_norm__353 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_92, parameter_389, parameter_390, parameter_391, parameter_392, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x384x14x14xf16) <- (-1x384x14x14xf16)
        relu__44 = paddle._C_ops.relu_(batch_norm__348)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 384x384x1x1xf16)
        conv2d_93 = paddle._C_ops.conv2d(relu__44, parameter_393, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__95 = paddle._C_ops.add_(add__94, conv2d_93)

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__354, batch_norm__355, batch_norm__356, batch_norm__357, batch_norm__358, batch_norm__359 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__95, parameter_394, parameter_395, parameter_396, parameter_397, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x1152x14x14xf16) <- (-1x384x14x14xf16, 1152x384x1x1xf16)
        conv2d_94 = paddle._C_ops.conv2d(batch_norm__354, parameter_398, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_69 = [1, 1152, 1, 1]

        # pd_op.reshape: (1x1152x1x1xf16, 0x1152xf16) <- (1152xf16, 4xi64)
        reshape_80, reshape_81 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_399, full_int_array_69), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1152x14x14xf16) <- (-1x1152x14x14xf16, 1x1152x1x1xf16)
        add__96 = paddle._C_ops.add_(conv2d_94, reshape_80)

        # pd_op.relu_: (-1x1152x14x14xf16) <- (-1x1152x14x14xf16)
        relu__45 = paddle._C_ops.relu_(add__96)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x1152x14x14xf16, 384x1152x1x1xf16)
        conv2d_95 = paddle._C_ops.conv2d(relu__45, parameter_400, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_70 = [1, 384, 1, 1]

        # pd_op.reshape: (1x384x1x1xf16, 0x384xf16) <- (384xf16, 4xi64)
        reshape_82, reshape_83 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_401, full_int_array_70), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 1x384x1x1xf16)
        add__97 = paddle._C_ops.add_(conv2d_95, reshape_82)

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__98 = paddle._C_ops.add_(add__95, add__97)

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__360, batch_norm__361, batch_norm__362, batch_norm__363, batch_norm__364, batch_norm__365 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__98, parameter_402, parameter_403, parameter_404, parameter_405, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (4xi32) <- (-1x384x14x14xf16)
        shape_9 = paddle._C_ops.shape(paddle.cast(batch_norm__360, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_71 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_72 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(shape_9, [0], full_int_array_71, full_int_array_72, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_54 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_55 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_21 = [slice_9, full_54, full_55]

        # pd_op.reshape_: (-1x384x196xf16, 0x-1x384x14x14xf16) <- (-1x384x14x14xf16, [1xi32, 1xi32, 1xi32])
        reshape__36, reshape__37 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__360, [x.reshape([]) for x in combine_21]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x384xf16) <- (-1x384x196xf16)
        transpose_24 = paddle._C_ops.transpose(reshape__36, [0, 2, 1])

        # pd_op.shape: (3xi32) <- (-1x196x384xf16)
        shape_10 = paddle._C_ops.shape(paddle.cast(transpose_24, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_73 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_74 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(shape_10, [0], full_int_array_73, full_int_array_74, [1], [0])

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_18 = paddle._C_ops.matmul(transpose_24, parameter_406, False, False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__99 = paddle._C_ops.add_(matmul_18, parameter_407)

        # pd_op.full: (1xi32) <- ()
        full_56 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_57 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_58 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_22 = [slice_10, full_56, full_57, full_58]

        # pd_op.reshape_: (-1x196x12x32xf16, 0x-1x196x384xf16) <- (-1x196x384xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__38, reshape__39 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__99, [x.reshape([]) for x in combine_22]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x12x196x32xf16) <- (-1x196x12x32xf16)
        transpose_25 = paddle._C_ops.transpose(reshape__38, [0, 2, 1, 3])

        # pd_op.transpose: (-1x384x196xf16) <- (-1x196x384xf16)
        transpose_26 = paddle._C_ops.transpose(transpose_24, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_75 = [2]

        # pd_op.unsqueeze_: (-1x384x1x196xf16, None) <- (-1x384x196xf16, 1xi64)
        unsqueeze__6, unsqueeze__7 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(transpose_26, full_int_array_75), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_76 = [1, 4]

        # pd_op.pool2d: (-1x384x1x49xf16) <- (-1x384x1x196xf16, 2xi64)
        pool2d_5 = paddle._C_ops.pool2d(unsqueeze__6, full_int_array_76, [1, 4], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_77 = [2]

        # pd_op.squeeze_: (-1x384x49xf16, None) <- (-1x384x1x49xf16, 1xi64)
        squeeze__6, squeeze__7 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_5, full_int_array_77), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x384x49xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x49xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__366, batch_norm__367, batch_norm__368, batch_norm__369, batch_norm__370, batch_norm__371 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(squeeze__6, parameter_408, parameter_409, parameter_410, parameter_411, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.transpose: (-1x49x384xf16) <- (-1x384x49xf16)
        transpose_27 = paddle._C_ops.transpose(batch_norm__366, [0, 2, 1])

        # pd_op.matmul: (-1x49x384xf16) <- (-1x49x384xf16, 384x384xf16)
        matmul_19 = paddle._C_ops.matmul(transpose_27, parameter_412, False, False)

        # pd_op.add_: (-1x49x384xf16) <- (-1x49x384xf16, 384xf16)
        add__100 = paddle._C_ops.add_(matmul_19, parameter_413)

        # pd_op.full: (1xi32) <- ()
        full_59 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_60 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_61 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_23 = [slice_10, full_59, full_60, full_61]

        # pd_op.reshape_: (-1x49x12x32xf16, 0x-1x49x384xf16) <- (-1x49x384xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__40, reshape__41 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__100, [x.reshape([]) for x in combine_23]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x12x32x49xf16) <- (-1x49x12x32xf16)
        transpose_28 = paddle._C_ops.transpose(reshape__40, [0, 2, 3, 1])

        # pd_op.matmul: (-1x49x384xf16) <- (-1x49x384xf16, 384x384xf16)
        matmul_20 = paddle._C_ops.matmul(transpose_27, parameter_414, False, False)

        # pd_op.add_: (-1x49x384xf16) <- (-1x49x384xf16, 384xf16)
        add__101 = paddle._C_ops.add_(matmul_20, parameter_415)

        # pd_op.full: (1xi32) <- ()
        full_62 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_63 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_64 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_24 = [slice_10, full_62, full_63, full_64]

        # pd_op.reshape_: (-1x49x12x32xf16, 0x-1x49x384xf16) <- (-1x49x384xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__42, reshape__43 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__101, [x.reshape([]) for x in combine_24]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x12x49x32xf16) <- (-1x49x12x32xf16)
        transpose_29 = paddle._C_ops.transpose(reshape__42, [0, 2, 1, 3])

        # pd_op.matmul: (-1x12x196x49xf16) <- (-1x12x196x32xf16, -1x12x32x49xf16)
        matmul_21 = paddle._C_ops.matmul(transpose_25, transpose_28, False, False)

        # pd_op.full: (1xf32) <- ()
        full_65 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x12x196x49xf16) <- (-1x12x196x49xf16, 1xf32)
        scale__3 = paddle._C_ops.scale_(matmul_21, full_65, float('0'), True)

        # pd_op.softmax_: (-1x12x196x49xf16) <- (-1x12x196x49xf16)
        softmax__3 = paddle._C_ops.softmax_(scale__3, -1)

        # pd_op.matmul: (-1x12x196x32xf16) <- (-1x12x196x49xf16, -1x12x49x32xf16)
        matmul_22 = paddle._C_ops.matmul(softmax__3, transpose_29, False, False)

        # pd_op.transpose: (-1x196x12x32xf16) <- (-1x12x196x32xf16)
        transpose_30 = paddle._C_ops.transpose(matmul_22, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_66 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_67 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_25 = [slice_10, full_66, full_67]

        # pd_op.reshape_: (-1x196x384xf16, 0x-1x196x12x32xf16) <- (-1x196x12x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__44, reshape__45 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_30, [x.reshape([]) for x in combine_25]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_23 = paddle._C_ops.matmul(reshape__44, parameter_416, False, False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__102 = paddle._C_ops.add_(matmul_23, parameter_417)

        # pd_op.shape: (3xi32) <- (-1x196x384xf16)
        shape_11 = paddle._C_ops.shape(paddle.cast(add__102, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_78 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_79 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(shape_11, [0], full_int_array_78, full_int_array_79, [1], [0])

        # pd_op.transpose: (-1x384x196xf16) <- (-1x196x384xf16)
        transpose_31 = paddle._C_ops.transpose(add__102, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_68 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_69 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_70 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_26 = [slice_11, full_68, full_69, full_70]

        # pd_op.reshape_: (-1x384x14x14xf16, 0x-1x384x196xf16) <- (-1x384x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__46, reshape__47 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_31, [x.reshape([]) for x in combine_26]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__103 = paddle._C_ops.add_(add__98, reshape__46)

        # pd_op.conv2d: (-1x128x14x14xf16) <- (-1x384x14x14xf16, 128x384x1x1xf16)
        conv2d_96 = paddle._C_ops.conv2d(add__103, parameter_418, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x14x14xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x14x14xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__372, batch_norm__373, batch_norm__374, batch_norm__375, batch_norm__376, batch_norm__377 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_96, parameter_419, parameter_420, parameter_421, parameter_422, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x128x14x14xf16) <- (-1x128x14x14xf16, 128x32x3x3xf16)
        conv2d_97 = paddle._C_ops.conv2d(batch_norm__372, parameter_423, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 4, 'NCHW')

        # pd_op.batch_norm_: (-1x128x14x14xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x14x14xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__378, batch_norm__379, batch_norm__380, batch_norm__381, batch_norm__382, batch_norm__383 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_97, parameter_424, parameter_425, parameter_426, parameter_427, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x14x14xf16) <- (-1x128x14x14xf16)
        relu__46 = paddle._C_ops.relu_(batch_norm__378)

        # pd_op.conv2d: (-1x128x14x14xf16) <- (-1x128x14x14xf16, 128x128x1x1xf16)
        conv2d_98 = paddle._C_ops.conv2d(relu__46, parameter_428, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x14x14xf16) <- (-1x128x14x14xf16, -1x128x14x14xf16)
        add__104 = paddle._C_ops.add_(batch_norm__372, conv2d_98)

        # builtin.combine: ([-1x384x14x14xf16, -1x128x14x14xf16]) <- (-1x384x14x14xf16, -1x128x14x14xf16)
        combine_27 = [add__103, add__104]

        # pd_op.full: (1xi32) <- ()
        full_71 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x512x14x14xf16) <- ([-1x384x14x14xf16, -1x128x14x14xf16], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_27, full_71)

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__384, batch_norm__385, batch_norm__386, batch_norm__387, batch_norm__388, batch_norm__389 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_3, parameter_429, parameter_430, parameter_431, parameter_432, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x512x14x14xf16, 1024x512x1x1xf16)
        conv2d_99 = paddle._C_ops.conv2d(batch_norm__384, parameter_433, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_80 = [1, 1024, 1, 1]

        # pd_op.reshape: (1x1024x1x1xf16, 0x1024xf16) <- (1024xf16, 4xi64)
        reshape_84, reshape_85 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_434, full_int_array_80), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, 1x1024x1x1xf16)
        add__105 = paddle._C_ops.add_(conv2d_99, reshape_84)

        # pd_op.relu_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16)
        relu__47 = paddle._C_ops.relu_(add__105)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x1024x14x14xf16, 512x1024x1x1xf16)
        conv2d_100 = paddle._C_ops.conv2d(relu__47, parameter_435, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_81 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf16, 0x512xf16) <- (512xf16, 4xi64)
        reshape_86, reshape_87 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_436, full_int_array_81), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x14x14xf16) <- (-1x512x14x14xf16, 1x512x1x1xf16)
        add__106 = paddle._C_ops.add_(conv2d_100, reshape_86)

        # pd_op.add_: (-1x512x14x14xf16) <- (-1x512x14x14xf16, -1x512x14x14xf16)
        add__107 = paddle._C_ops.add_(concat_3, add__106)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x512x14x14xf16, 384x512x1x1xf16)
        conv2d_101 = paddle._C_ops.conv2d(add__107, parameter_437, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__390, batch_norm__391, batch_norm__392, batch_norm__393, batch_norm__394, batch_norm__395 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_101, parameter_438, parameter_439, parameter_440, parameter_441, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 384x32x3x3xf16)
        conv2d_102 = paddle._C_ops.conv2d(batch_norm__390, parameter_442, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 12, 'NCHW')

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__396, batch_norm__397, batch_norm__398, batch_norm__399, batch_norm__400, batch_norm__401 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_102, parameter_443, parameter_444, parameter_445, parameter_446, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x384x14x14xf16) <- (-1x384x14x14xf16)
        relu__48 = paddle._C_ops.relu_(batch_norm__396)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 384x384x1x1xf16)
        conv2d_103 = paddle._C_ops.conv2d(relu__48, parameter_447, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__108 = paddle._C_ops.add_(batch_norm__390, conv2d_103)

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__402, batch_norm__403, batch_norm__404, batch_norm__405, batch_norm__406, batch_norm__407 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__108, parameter_448, parameter_449, parameter_450, parameter_451, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x1152x14x14xf16) <- (-1x384x14x14xf16, 1152x384x1x1xf16)
        conv2d_104 = paddle._C_ops.conv2d(batch_norm__402, parameter_452, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_82 = [1, 1152, 1, 1]

        # pd_op.reshape: (1x1152x1x1xf16, 0x1152xf16) <- (1152xf16, 4xi64)
        reshape_88, reshape_89 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_453, full_int_array_82), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1152x14x14xf16) <- (-1x1152x14x14xf16, 1x1152x1x1xf16)
        add__109 = paddle._C_ops.add_(conv2d_104, reshape_88)

        # pd_op.relu_: (-1x1152x14x14xf16) <- (-1x1152x14x14xf16)
        relu__49 = paddle._C_ops.relu_(add__109)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x1152x14x14xf16, 384x1152x1x1xf16)
        conv2d_105 = paddle._C_ops.conv2d(relu__49, parameter_454, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_83 = [1, 384, 1, 1]

        # pd_op.reshape: (1x384x1x1xf16, 0x384xf16) <- (384xf16, 4xi64)
        reshape_90, reshape_91 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_455, full_int_array_83), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 1x384x1x1xf16)
        add__110 = paddle._C_ops.add_(conv2d_105, reshape_90)

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__111 = paddle._C_ops.add_(add__108, add__110)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 384x32x3x3xf16)
        conv2d_106 = paddle._C_ops.conv2d(add__111, parameter_456, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 12, 'NCHW')

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__408, batch_norm__409, batch_norm__410, batch_norm__411, batch_norm__412, batch_norm__413 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_106, parameter_457, parameter_458, parameter_459, parameter_460, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x384x14x14xf16) <- (-1x384x14x14xf16)
        relu__50 = paddle._C_ops.relu_(batch_norm__408)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 384x384x1x1xf16)
        conv2d_107 = paddle._C_ops.conv2d(relu__50, parameter_461, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__112 = paddle._C_ops.add_(add__111, conv2d_107)

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__414, batch_norm__415, batch_norm__416, batch_norm__417, batch_norm__418, batch_norm__419 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__112, parameter_462, parameter_463, parameter_464, parameter_465, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x1152x14x14xf16) <- (-1x384x14x14xf16, 1152x384x1x1xf16)
        conv2d_108 = paddle._C_ops.conv2d(batch_norm__414, parameter_466, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_84 = [1, 1152, 1, 1]

        # pd_op.reshape: (1x1152x1x1xf16, 0x1152xf16) <- (1152xf16, 4xi64)
        reshape_92, reshape_93 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_467, full_int_array_84), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1152x14x14xf16) <- (-1x1152x14x14xf16, 1x1152x1x1xf16)
        add__113 = paddle._C_ops.add_(conv2d_108, reshape_92)

        # pd_op.relu_: (-1x1152x14x14xf16) <- (-1x1152x14x14xf16)
        relu__51 = paddle._C_ops.relu_(add__113)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x1152x14x14xf16, 384x1152x1x1xf16)
        conv2d_109 = paddle._C_ops.conv2d(relu__51, parameter_468, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_85 = [1, 384, 1, 1]

        # pd_op.reshape: (1x384x1x1xf16, 0x384xf16) <- (384xf16, 4xi64)
        reshape_94, reshape_95 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_469, full_int_array_85), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 1x384x1x1xf16)
        add__114 = paddle._C_ops.add_(conv2d_109, reshape_94)

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__115 = paddle._C_ops.add_(add__112, add__114)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 384x32x3x3xf16)
        conv2d_110 = paddle._C_ops.conv2d(add__115, parameter_470, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 12, 'NCHW')

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__420, batch_norm__421, batch_norm__422, batch_norm__423, batch_norm__424, batch_norm__425 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_110, parameter_471, parameter_472, parameter_473, parameter_474, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x384x14x14xf16) <- (-1x384x14x14xf16)
        relu__52 = paddle._C_ops.relu_(batch_norm__420)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 384x384x1x1xf16)
        conv2d_111 = paddle._C_ops.conv2d(relu__52, parameter_475, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__116 = paddle._C_ops.add_(add__115, conv2d_111)

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__426, batch_norm__427, batch_norm__428, batch_norm__429, batch_norm__430, batch_norm__431 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__116, parameter_476, parameter_477, parameter_478, parameter_479, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x1152x14x14xf16) <- (-1x384x14x14xf16, 1152x384x1x1xf16)
        conv2d_112 = paddle._C_ops.conv2d(batch_norm__426, parameter_480, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_86 = [1, 1152, 1, 1]

        # pd_op.reshape: (1x1152x1x1xf16, 0x1152xf16) <- (1152xf16, 4xi64)
        reshape_96, reshape_97 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_481, full_int_array_86), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1152x14x14xf16) <- (-1x1152x14x14xf16, 1x1152x1x1xf16)
        add__117 = paddle._C_ops.add_(conv2d_112, reshape_96)

        # pd_op.relu_: (-1x1152x14x14xf16) <- (-1x1152x14x14xf16)
        relu__53 = paddle._C_ops.relu_(add__117)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x1152x14x14xf16, 384x1152x1x1xf16)
        conv2d_113 = paddle._C_ops.conv2d(relu__53, parameter_482, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_87 = [1, 384, 1, 1]

        # pd_op.reshape: (1x384x1x1xf16, 0x384xf16) <- (384xf16, 4xi64)
        reshape_98, reshape_99 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_483, full_int_array_87), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 1x384x1x1xf16)
        add__118 = paddle._C_ops.add_(conv2d_113, reshape_98)

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__119 = paddle._C_ops.add_(add__116, add__118)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 384x32x3x3xf16)
        conv2d_114 = paddle._C_ops.conv2d(add__119, parameter_484, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 12, 'NCHW')

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__432, batch_norm__433, batch_norm__434, batch_norm__435, batch_norm__436, batch_norm__437 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_114, parameter_485, parameter_486, parameter_487, parameter_488, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x384x14x14xf16) <- (-1x384x14x14xf16)
        relu__54 = paddle._C_ops.relu_(batch_norm__432)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 384x384x1x1xf16)
        conv2d_115 = paddle._C_ops.conv2d(relu__54, parameter_489, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__120 = paddle._C_ops.add_(add__119, conv2d_115)

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__438, batch_norm__439, batch_norm__440, batch_norm__441, batch_norm__442, batch_norm__443 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__120, parameter_490, parameter_491, parameter_492, parameter_493, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x1152x14x14xf16) <- (-1x384x14x14xf16, 1152x384x1x1xf16)
        conv2d_116 = paddle._C_ops.conv2d(batch_norm__438, parameter_494, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_88 = [1, 1152, 1, 1]

        # pd_op.reshape: (1x1152x1x1xf16, 0x1152xf16) <- (1152xf16, 4xi64)
        reshape_100, reshape_101 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_495, full_int_array_88), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1152x14x14xf16) <- (-1x1152x14x14xf16, 1x1152x1x1xf16)
        add__121 = paddle._C_ops.add_(conv2d_116, reshape_100)

        # pd_op.relu_: (-1x1152x14x14xf16) <- (-1x1152x14x14xf16)
        relu__55 = paddle._C_ops.relu_(add__121)

        # pd_op.conv2d: (-1x384x14x14xf16) <- (-1x1152x14x14xf16, 384x1152x1x1xf16)
        conv2d_117 = paddle._C_ops.conv2d(relu__55, parameter_496, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_89 = [1, 384, 1, 1]

        # pd_op.reshape: (1x384x1x1xf16, 0x384xf16) <- (384xf16, 4xi64)
        reshape_102, reshape_103 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_497, full_int_array_89), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, 1x384x1x1xf16)
        add__122 = paddle._C_ops.add_(conv2d_117, reshape_102)

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__123 = paddle._C_ops.add_(add__120, add__122)

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__444, batch_norm__445, batch_norm__446, batch_norm__447, batch_norm__448, batch_norm__449 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__123, parameter_498, parameter_499, parameter_500, parameter_501, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (4xi32) <- (-1x384x14x14xf16)
        shape_12 = paddle._C_ops.shape(paddle.cast(batch_norm__444, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_90 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_91 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(shape_12, [0], full_int_array_90, full_int_array_91, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_72 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_73 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_28 = [slice_12, full_72, full_73]

        # pd_op.reshape_: (-1x384x196xf16, 0x-1x384x14x14xf16) <- (-1x384x14x14xf16, [1xi32, 1xi32, 1xi32])
        reshape__48, reshape__49 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__444, [x.reshape([]) for x in combine_28]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x196x384xf16) <- (-1x384x196xf16)
        transpose_32 = paddle._C_ops.transpose(reshape__48, [0, 2, 1])

        # pd_op.shape: (3xi32) <- (-1x196x384xf16)
        shape_13 = paddle._C_ops.shape(paddle.cast(transpose_32, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_92 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_93 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(shape_13, [0], full_int_array_92, full_int_array_93, [1], [0])

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_24 = paddle._C_ops.matmul(transpose_32, parameter_502, False, False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__124 = paddle._C_ops.add_(matmul_24, parameter_503)

        # pd_op.full: (1xi32) <- ()
        full_74 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_75 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_76 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_29 = [slice_13, full_74, full_75, full_76]

        # pd_op.reshape_: (-1x196x12x32xf16, 0x-1x196x384xf16) <- (-1x196x384xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__50, reshape__51 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__124, [x.reshape([]) for x in combine_29]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x12x196x32xf16) <- (-1x196x12x32xf16)
        transpose_33 = paddle._C_ops.transpose(reshape__50, [0, 2, 1, 3])

        # pd_op.transpose: (-1x384x196xf16) <- (-1x196x384xf16)
        transpose_34 = paddle._C_ops.transpose(transpose_32, [0, 2, 1])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_94 = [2]

        # pd_op.unsqueeze_: (-1x384x1x196xf16, None) <- (-1x384x196xf16, 1xi64)
        unsqueeze__8, unsqueeze__9 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(transpose_34, full_int_array_94), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_95 = [1, 4]

        # pd_op.pool2d: (-1x384x1x49xf16) <- (-1x384x1x196xf16, 2xi64)
        pool2d_6 = paddle._C_ops.pool2d(unsqueeze__8, full_int_array_95, [1, 4], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_96 = [2]

        # pd_op.squeeze_: (-1x384x49xf16, None) <- (-1x384x1x49xf16, 1xi64)
        squeeze__8, squeeze__9 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_6, full_int_array_96), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x384x49xf16, 384xf32, 384xf32, xf32, xf32, None) <- (-1x384x49xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__450, batch_norm__451, batch_norm__452, batch_norm__453, batch_norm__454, batch_norm__455 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(squeeze__8, parameter_504, parameter_505, parameter_506, parameter_507, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.transpose: (-1x49x384xf16) <- (-1x384x49xf16)
        transpose_35 = paddle._C_ops.transpose(batch_norm__450, [0, 2, 1])

        # pd_op.matmul: (-1x49x384xf16) <- (-1x49x384xf16, 384x384xf16)
        matmul_25 = paddle._C_ops.matmul(transpose_35, parameter_508, False, False)

        # pd_op.add_: (-1x49x384xf16) <- (-1x49x384xf16, 384xf16)
        add__125 = paddle._C_ops.add_(matmul_25, parameter_509)

        # pd_op.full: (1xi32) <- ()
        full_77 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_78 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_79 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_30 = [slice_13, full_77, full_78, full_79]

        # pd_op.reshape_: (-1x49x12x32xf16, 0x-1x49x384xf16) <- (-1x49x384xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__52, reshape__53 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__125, [x.reshape([]) for x in combine_30]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x12x32x49xf16) <- (-1x49x12x32xf16)
        transpose_36 = paddle._C_ops.transpose(reshape__52, [0, 2, 3, 1])

        # pd_op.matmul: (-1x49x384xf16) <- (-1x49x384xf16, 384x384xf16)
        matmul_26 = paddle._C_ops.matmul(transpose_35, parameter_510, False, False)

        # pd_op.add_: (-1x49x384xf16) <- (-1x49x384xf16, 384xf16)
        add__126 = paddle._C_ops.add_(matmul_26, parameter_511)

        # pd_op.full: (1xi32) <- ()
        full_80 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_81 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_82 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_31 = [slice_13, full_80, full_81, full_82]

        # pd_op.reshape_: (-1x49x12x32xf16, 0x-1x49x384xf16) <- (-1x49x384xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__54, reshape__55 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__126, [x.reshape([]) for x in combine_31]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x12x49x32xf16) <- (-1x49x12x32xf16)
        transpose_37 = paddle._C_ops.transpose(reshape__54, [0, 2, 1, 3])

        # pd_op.matmul: (-1x12x196x49xf16) <- (-1x12x196x32xf16, -1x12x32x49xf16)
        matmul_27 = paddle._C_ops.matmul(transpose_33, transpose_36, False, False)

        # pd_op.full: (1xf32) <- ()
        full_83 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x12x196x49xf16) <- (-1x12x196x49xf16, 1xf32)
        scale__4 = paddle._C_ops.scale_(matmul_27, full_83, float('0'), True)

        # pd_op.softmax_: (-1x12x196x49xf16) <- (-1x12x196x49xf16)
        softmax__4 = paddle._C_ops.softmax_(scale__4, -1)

        # pd_op.matmul: (-1x12x196x32xf16) <- (-1x12x196x49xf16, -1x12x49x32xf16)
        matmul_28 = paddle._C_ops.matmul(softmax__4, transpose_37, False, False)

        # pd_op.transpose: (-1x196x12x32xf16) <- (-1x12x196x32xf16)
        transpose_38 = paddle._C_ops.transpose(matmul_28, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_84 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_85 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_32 = [slice_13, full_84, full_85]

        # pd_op.reshape_: (-1x196x384xf16, 0x-1x196x12x32xf16) <- (-1x196x12x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__56, reshape__57 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_38, [x.reshape([]) for x in combine_32]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x196x384xf16) <- (-1x196x384xf16, 384x384xf16)
        matmul_29 = paddle._C_ops.matmul(reshape__56, parameter_512, False, False)

        # pd_op.add_: (-1x196x384xf16) <- (-1x196x384xf16, 384xf16)
        add__127 = paddle._C_ops.add_(matmul_29, parameter_513)

        # pd_op.shape: (3xi32) <- (-1x196x384xf16)
        shape_14 = paddle._C_ops.shape(paddle.cast(add__127, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_97 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_98 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(shape_14, [0], full_int_array_97, full_int_array_98, [1], [0])

        # pd_op.transpose: (-1x384x196xf16) <- (-1x196x384xf16)
        transpose_39 = paddle._C_ops.transpose(add__127, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_86 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_87 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_88 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_33 = [slice_14, full_86, full_87, full_88]

        # pd_op.reshape_: (-1x384x14x14xf16, 0x-1x384x196xf16) <- (-1x384x196xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__58, reshape__59 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_39, [x.reshape([]) for x in combine_33]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x384x14x14xf16) <- (-1x384x14x14xf16, -1x384x14x14xf16)
        add__128 = paddle._C_ops.add_(add__123, reshape__58)

        # pd_op.conv2d: (-1x128x14x14xf16) <- (-1x384x14x14xf16, 128x384x1x1xf16)
        conv2d_118 = paddle._C_ops.conv2d(add__128, parameter_514, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x14x14xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x14x14xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__456, batch_norm__457, batch_norm__458, batch_norm__459, batch_norm__460, batch_norm__461 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_118, parameter_515, parameter_516, parameter_517, parameter_518, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x128x14x14xf16) <- (-1x128x14x14xf16, 128x32x3x3xf16)
        conv2d_119 = paddle._C_ops.conv2d(batch_norm__456, parameter_519, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 4, 'NCHW')

        # pd_op.batch_norm_: (-1x128x14x14xf16, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x14x14xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__462, batch_norm__463, batch_norm__464, batch_norm__465, batch_norm__466, batch_norm__467 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_119, parameter_520, parameter_521, parameter_522, parameter_523, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x14x14xf16) <- (-1x128x14x14xf16)
        relu__56 = paddle._C_ops.relu_(batch_norm__462)

        # pd_op.conv2d: (-1x128x14x14xf16) <- (-1x128x14x14xf16, 128x128x1x1xf16)
        conv2d_120 = paddle._C_ops.conv2d(relu__56, parameter_524, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x128x14x14xf16) <- (-1x128x14x14xf16, -1x128x14x14xf16)
        add__129 = paddle._C_ops.add_(batch_norm__456, conv2d_120)

        # builtin.combine: ([-1x384x14x14xf16, -1x128x14x14xf16]) <- (-1x384x14x14xf16, -1x128x14x14xf16)
        combine_34 = [add__128, add__129]

        # pd_op.full: (1xi32) <- ()
        full_89 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x512x14x14xf16) <- ([-1x384x14x14xf16, -1x128x14x14xf16], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_34, full_89)

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__468, batch_norm__469, batch_norm__470, batch_norm__471, batch_norm__472, batch_norm__473 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_4, parameter_525, parameter_526, parameter_527, parameter_528, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x512x14x14xf16, 1024x512x1x1xf16)
        conv2d_121 = paddle._C_ops.conv2d(batch_norm__468, parameter_529, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_99 = [1, 1024, 1, 1]

        # pd_op.reshape: (1x1024x1x1xf16, 0x1024xf16) <- (1024xf16, 4xi64)
        reshape_104, reshape_105 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_530, full_int_array_99), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, 1x1024x1x1xf16)
        add__130 = paddle._C_ops.add_(conv2d_121, reshape_104)

        # pd_op.relu_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16)
        relu__57 = paddle._C_ops.relu_(add__130)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x1024x14x14xf16, 512x1024x1x1xf16)
        conv2d_122 = paddle._C_ops.conv2d(relu__57, parameter_531, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_100 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf16, 0x512xf16) <- (512xf16, 4xi64)
        reshape_106, reshape_107 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_532, full_int_array_100), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x14x14xf16) <- (-1x512x14x14xf16, 1x512x1x1xf16)
        add__131 = paddle._C_ops.add_(conv2d_122, reshape_106)

        # pd_op.add_: (-1x512x14x14xf16) <- (-1x512x14x14xf16, -1x512x14x14xf16)
        add__132 = paddle._C_ops.add_(concat_4, add__131)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_101 = [2, 2]

        # pd_op.pool2d: (-1x512x7x7xf16) <- (-1x512x14x14xf16, 2xi64)
        pool2d_7 = paddle._C_ops.pool2d(add__132, full_int_array_101, [2, 2], [0, 0], True, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x768x7x7xf16) <- (-1x512x7x7xf16, 768x512x1x1xf16)
        conv2d_123 = paddle._C_ops.conv2d(pool2d_7, parameter_533, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x768x7x7xf16, 768xf32, 768xf32, xf32, xf32, None) <- (-1x768x7x7xf16, 768xf32, 768xf32, 768xf32, 768xf32)
        batch_norm__474, batch_norm__475, batch_norm__476, batch_norm__477, batch_norm__478, batch_norm__479 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_123, parameter_534, parameter_535, parameter_536, parameter_537, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x768x7x7xf16) <- (-1x768x7x7xf16, 768x32x3x3xf16)
        conv2d_124 = paddle._C_ops.conv2d(batch_norm__474, parameter_538, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 24, 'NCHW')

        # pd_op.batch_norm_: (-1x768x7x7xf16, 768xf32, 768xf32, xf32, xf32, None) <- (-1x768x7x7xf16, 768xf32, 768xf32, 768xf32, 768xf32)
        batch_norm__480, batch_norm__481, batch_norm__482, batch_norm__483, batch_norm__484, batch_norm__485 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_124, parameter_539, parameter_540, parameter_541, parameter_542, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x768x7x7xf16) <- (-1x768x7x7xf16)
        relu__58 = paddle._C_ops.relu_(batch_norm__480)

        # pd_op.conv2d: (-1x768x7x7xf16) <- (-1x768x7x7xf16, 768x768x1x1xf16)
        conv2d_125 = paddle._C_ops.conv2d(relu__58, parameter_543, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x768x7x7xf16) <- (-1x768x7x7xf16, -1x768x7x7xf16)
        add__133 = paddle._C_ops.add_(batch_norm__474, conv2d_125)

        # pd_op.batch_norm_: (-1x768x7x7xf16, 768xf32, 768xf32, xf32, xf32, None) <- (-1x768x7x7xf16, 768xf32, 768xf32, 768xf32, 768xf32)
        batch_norm__486, batch_norm__487, batch_norm__488, batch_norm__489, batch_norm__490, batch_norm__491 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__133, parameter_544, parameter_545, parameter_546, parameter_547, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x2304x7x7xf16) <- (-1x768x7x7xf16, 2304x768x1x1xf16)
        conv2d_126 = paddle._C_ops.conv2d(batch_norm__486, parameter_548, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_102 = [1, 2304, 1, 1]

        # pd_op.reshape: (1x2304x1x1xf16, 0x2304xf16) <- (2304xf16, 4xi64)
        reshape_108, reshape_109 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_549, full_int_array_102), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x2304x7x7xf16) <- (-1x2304x7x7xf16, 1x2304x1x1xf16)
        add__134 = paddle._C_ops.add_(conv2d_126, reshape_108)

        # pd_op.relu_: (-1x2304x7x7xf16) <- (-1x2304x7x7xf16)
        relu__59 = paddle._C_ops.relu_(add__134)

        # pd_op.conv2d: (-1x768x7x7xf16) <- (-1x2304x7x7xf16, 768x2304x1x1xf16)
        conv2d_127 = paddle._C_ops.conv2d(relu__59, parameter_550, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_103 = [1, 768, 1, 1]

        # pd_op.reshape: (1x768x1x1xf16, 0x768xf16) <- (768xf16, 4xi64)
        reshape_110, reshape_111 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_551, full_int_array_103), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x768x7x7xf16) <- (-1x768x7x7xf16, 1x768x1x1xf16)
        add__135 = paddle._C_ops.add_(conv2d_127, reshape_110)

        # pd_op.add_: (-1x768x7x7xf16) <- (-1x768x7x7xf16, -1x768x7x7xf16)
        add__136 = paddle._C_ops.add_(add__133, add__135)

        # pd_op.conv2d: (-1x768x7x7xf16) <- (-1x768x7x7xf16, 768x32x3x3xf16)
        conv2d_128 = paddle._C_ops.conv2d(add__136, parameter_552, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 24, 'NCHW')

        # pd_op.batch_norm_: (-1x768x7x7xf16, 768xf32, 768xf32, xf32, xf32, None) <- (-1x768x7x7xf16, 768xf32, 768xf32, 768xf32, 768xf32)
        batch_norm__492, batch_norm__493, batch_norm__494, batch_norm__495, batch_norm__496, batch_norm__497 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_128, parameter_553, parameter_554, parameter_555, parameter_556, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x768x7x7xf16) <- (-1x768x7x7xf16)
        relu__60 = paddle._C_ops.relu_(batch_norm__492)

        # pd_op.conv2d: (-1x768x7x7xf16) <- (-1x768x7x7xf16, 768x768x1x1xf16)
        conv2d_129 = paddle._C_ops.conv2d(relu__60, parameter_557, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x768x7x7xf16) <- (-1x768x7x7xf16, -1x768x7x7xf16)
        add__137 = paddle._C_ops.add_(add__136, conv2d_129)

        # pd_op.batch_norm_: (-1x768x7x7xf16, 768xf32, 768xf32, xf32, xf32, None) <- (-1x768x7x7xf16, 768xf32, 768xf32, 768xf32, 768xf32)
        batch_norm__498, batch_norm__499, batch_norm__500, batch_norm__501, batch_norm__502, batch_norm__503 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__137, parameter_558, parameter_559, parameter_560, parameter_561, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x2304x7x7xf16) <- (-1x768x7x7xf16, 2304x768x1x1xf16)
        conv2d_130 = paddle._C_ops.conv2d(batch_norm__498, parameter_562, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_104 = [1, 2304, 1, 1]

        # pd_op.reshape: (1x2304x1x1xf16, 0x2304xf16) <- (2304xf16, 4xi64)
        reshape_112, reshape_113 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_563, full_int_array_104), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x2304x7x7xf16) <- (-1x2304x7x7xf16, 1x2304x1x1xf16)
        add__138 = paddle._C_ops.add_(conv2d_130, reshape_112)

        # pd_op.relu_: (-1x2304x7x7xf16) <- (-1x2304x7x7xf16)
        relu__61 = paddle._C_ops.relu_(add__138)

        # pd_op.conv2d: (-1x768x7x7xf16) <- (-1x2304x7x7xf16, 768x2304x1x1xf16)
        conv2d_131 = paddle._C_ops.conv2d(relu__61, parameter_564, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_105 = [1, 768, 1, 1]

        # pd_op.reshape: (1x768x1x1xf16, 0x768xf16) <- (768xf16, 4xi64)
        reshape_114, reshape_115 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_565, full_int_array_105), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x768x7x7xf16) <- (-1x768x7x7xf16, 1x768x1x1xf16)
        add__139 = paddle._C_ops.add_(conv2d_131, reshape_114)

        # pd_op.add_: (-1x768x7x7xf16) <- (-1x768x7x7xf16, -1x768x7x7xf16)
        add__140 = paddle._C_ops.add_(add__137, add__139)

        # pd_op.batch_norm_: (-1x768x7x7xf16, 768xf32, 768xf32, xf32, xf32, None) <- (-1x768x7x7xf16, 768xf32, 768xf32, 768xf32, 768xf32)
        batch_norm__504, batch_norm__505, batch_norm__506, batch_norm__507, batch_norm__508, batch_norm__509 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__140, parameter_566, parameter_567, parameter_568, parameter_569, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.shape: (4xi32) <- (-1x768x7x7xf16)
        shape_15 = paddle._C_ops.shape(paddle.cast(batch_norm__504, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_106 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_107 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(shape_15, [0], full_int_array_106, full_int_array_107, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_90 = paddle._C_ops.full([1], float('768'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_91 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_35 = [slice_15, full_90, full_91]

        # pd_op.reshape_: (-1x768x49xf16, 0x-1x768x7x7xf16) <- (-1x768x7x7xf16, [1xi32, 1xi32, 1xi32])
        reshape__60, reshape__61 = (lambda x, f: f(x))(paddle._C_ops.reshape_(batch_norm__504, [x.reshape([]) for x in combine_35]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x49x768xf16) <- (-1x768x49xf16)
        transpose_40 = paddle._C_ops.transpose(reshape__60, [0, 2, 1])

        # pd_op.shape: (3xi32) <- (-1x49x768xf16)
        shape_16 = paddle._C_ops.shape(paddle.cast(transpose_40, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_108 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_109 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(shape_16, [0], full_int_array_108, full_int_array_109, [1], [0])

        # pd_op.matmul: (-1x49x768xf16) <- (-1x49x768xf16, 768x768xf16)
        matmul_30 = paddle._C_ops.matmul(transpose_40, parameter_570, False, False)

        # pd_op.add_: (-1x49x768xf16) <- (-1x49x768xf16, 768xf16)
        add__141 = paddle._C_ops.add_(matmul_30, parameter_571)

        # pd_op.full: (1xi32) <- ()
        full_92 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_93 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_94 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_36 = [slice_16, full_92, full_93, full_94]

        # pd_op.reshape_: (-1x49x24x32xf16, 0x-1x49x768xf16) <- (-1x49x768xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__62, reshape__63 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__141, [x.reshape([]) for x in combine_36]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x24x49x32xf16) <- (-1x49x24x32xf16)
        transpose_41 = paddle._C_ops.transpose(reshape__62, [0, 2, 1, 3])

        # pd_op.matmul: (-1x49x768xf16) <- (-1x49x768xf16, 768x768xf16)
        matmul_31 = paddle._C_ops.matmul(transpose_40, parameter_572, False, False)

        # pd_op.add_: (-1x49x768xf16) <- (-1x49x768xf16, 768xf16)
        add__142 = paddle._C_ops.add_(matmul_31, parameter_573)

        # pd_op.full: (1xi32) <- ()
        full_95 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_96 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_97 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_37 = [slice_16, full_95, full_96, full_97]

        # pd_op.reshape_: (-1x49x24x32xf16, 0x-1x49x768xf16) <- (-1x49x768xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__64, reshape__65 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__142, [x.reshape([]) for x in combine_37]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x24x32x49xf16) <- (-1x49x24x32xf16)
        transpose_42 = paddle._C_ops.transpose(reshape__64, [0, 2, 3, 1])

        # pd_op.matmul: (-1x49x768xf16) <- (-1x49x768xf16, 768x768xf16)
        matmul_32 = paddle._C_ops.matmul(transpose_40, parameter_574, False, False)

        # pd_op.add_: (-1x49x768xf16) <- (-1x49x768xf16, 768xf16)
        add__143 = paddle._C_ops.add_(matmul_32, parameter_575)

        # pd_op.full: (1xi32) <- ()
        full_98 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_99 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_100 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_38 = [slice_16, full_98, full_99, full_100]

        # pd_op.reshape_: (-1x49x24x32xf16, 0x-1x49x768xf16) <- (-1x49x768xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__66, reshape__67 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__143, [x.reshape([]) for x in combine_38]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x24x49x32xf16) <- (-1x49x24x32xf16)
        transpose_43 = paddle._C_ops.transpose(reshape__66, [0, 2, 1, 3])

        # pd_op.matmul: (-1x24x49x49xf16) <- (-1x24x49x32xf16, -1x24x32x49xf16)
        matmul_33 = paddle._C_ops.matmul(transpose_41, transpose_42, False, False)

        # pd_op.full: (1xf32) <- ()
        full_101 = paddle._C_ops.full([1], float('0.176777'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x24x49x49xf16) <- (-1x24x49x49xf16, 1xf32)
        scale__5 = paddle._C_ops.scale_(matmul_33, full_101, float('0'), True)

        # pd_op.softmax_: (-1x24x49x49xf16) <- (-1x24x49x49xf16)
        softmax__5 = paddle._C_ops.softmax_(scale__5, -1)

        # pd_op.matmul: (-1x24x49x32xf16) <- (-1x24x49x49xf16, -1x24x49x32xf16)
        matmul_34 = paddle._C_ops.matmul(softmax__5, transpose_43, False, False)

        # pd_op.transpose: (-1x49x24x32xf16) <- (-1x24x49x32xf16)
        transpose_44 = paddle._C_ops.transpose(matmul_34, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_102 = paddle._C_ops.full([1], float('49'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_103 = paddle._C_ops.full([1], float('768'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_39 = [slice_16, full_102, full_103]

        # pd_op.reshape_: (-1x49x768xf16, 0x-1x49x24x32xf16) <- (-1x49x24x32xf16, [1xi32, 1xi32, 1xi32])
        reshape__68, reshape__69 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_44, [x.reshape([]) for x in combine_39]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x49x768xf16) <- (-1x49x768xf16, 768x768xf16)
        matmul_35 = paddle._C_ops.matmul(reshape__68, parameter_576, False, False)

        # pd_op.add_: (-1x49x768xf16) <- (-1x49x768xf16, 768xf16)
        add__144 = paddle._C_ops.add_(matmul_35, parameter_577)

        # pd_op.shape: (3xi32) <- (-1x49x768xf16)
        shape_17 = paddle._C_ops.shape(paddle.cast(add__144, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_110 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_111 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(shape_17, [0], full_int_array_110, full_int_array_111, [1], [0])

        # pd_op.transpose: (-1x768x49xf16) <- (-1x49x768xf16)
        transpose_45 = paddle._C_ops.transpose(add__144, [0, 2, 1])

        # pd_op.full: (1xi32) <- ()
        full_104 = paddle._C_ops.full([1], float('768'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_105 = paddle._C_ops.full([1], float('7'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_106 = paddle._C_ops.full([1], float('7'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_40 = [slice_17, full_104, full_105, full_106]

        # pd_op.reshape_: (-1x768x7x7xf16, 0x-1x768x49xf16) <- (-1x768x49xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__70, reshape__71 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_45, [x.reshape([]) for x in combine_40]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x768x7x7xf16) <- (-1x768x7x7xf16, -1x768x7x7xf16)
        add__145 = paddle._C_ops.add_(add__140, reshape__70)

        # pd_op.conv2d: (-1x256x7x7xf16) <- (-1x768x7x7xf16, 256x768x1x1xf16)
        conv2d_132 = paddle._C_ops.conv2d(add__145, parameter_578, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x7x7xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x7x7xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__510, batch_norm__511, batch_norm__512, batch_norm__513, batch_norm__514, batch_norm__515 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_132, parameter_579, parameter_580, parameter_581, parameter_582, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x256x7x7xf16) <- (-1x256x7x7xf16, 256x32x3x3xf16)
        conv2d_133 = paddle._C_ops.conv2d(batch_norm__510, parameter_583, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 8, 'NCHW')

        # pd_op.batch_norm_: (-1x256x7x7xf16, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x7x7xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__516, batch_norm__517, batch_norm__518, batch_norm__519, batch_norm__520, batch_norm__521 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_133, parameter_584, parameter_585, parameter_586, parameter_587, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x7x7xf16) <- (-1x256x7x7xf16)
        relu__62 = paddle._C_ops.relu_(batch_norm__516)

        # pd_op.conv2d: (-1x256x7x7xf16) <- (-1x256x7x7xf16, 256x256x1x1xf16)
        conv2d_134 = paddle._C_ops.conv2d(relu__62, parameter_588, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.add_: (-1x256x7x7xf16) <- (-1x256x7x7xf16, -1x256x7x7xf16)
        add__146 = paddle._C_ops.add_(batch_norm__510, conv2d_134)

        # builtin.combine: ([-1x768x7x7xf16, -1x256x7x7xf16]) <- (-1x768x7x7xf16, -1x256x7x7xf16)
        combine_41 = [add__145, add__146]

        # pd_op.full: (1xi32) <- ()
        full_107 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x1024x7x7xf16) <- ([-1x768x7x7xf16, -1x256x7x7xf16], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_41, full_107)

        # pd_op.batch_norm_: (-1x1024x7x7xf16, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x7x7xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__522, batch_norm__523, batch_norm__524, batch_norm__525, batch_norm__526, batch_norm__527 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_5, parameter_589, parameter_590, parameter_591, parameter_592, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x2048x7x7xf16) <- (-1x1024x7x7xf16, 2048x1024x1x1xf16)
        conv2d_135 = paddle._C_ops.conv2d(batch_norm__522, parameter_593, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_112 = [1, 2048, 1, 1]

        # pd_op.reshape: (1x2048x1x1xf16, 0x2048xf16) <- (2048xf16, 4xi64)
        reshape_116, reshape_117 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_594, full_int_array_112), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x2048x7x7xf16) <- (-1x2048x7x7xf16, 1x2048x1x1xf16)
        add__147 = paddle._C_ops.add_(conv2d_135, reshape_116)

        # pd_op.relu_: (-1x2048x7x7xf16) <- (-1x2048x7x7xf16)
        relu__63 = paddle._C_ops.relu_(add__147)

        # pd_op.conv2d: (-1x1024x7x7xf16) <- (-1x2048x7x7xf16, 1024x2048x1x1xf16)
        conv2d_136 = paddle._C_ops.conv2d(relu__63, parameter_595, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_113 = [1, 1024, 1, 1]

        # pd_op.reshape: (1x1024x1x1xf16, 0x1024xf16) <- (1024xf16, 4xi64)
        reshape_118, reshape_119 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_596, full_int_array_113), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1024x7x7xf16) <- (-1x1024x7x7xf16, 1x1024x1x1xf16)
        add__148 = paddle._C_ops.add_(conv2d_136, reshape_118)

        # pd_op.add_: (-1x1024x7x7xf16) <- (-1x1024x7x7xf16, -1x1024x7x7xf16)
        add__149 = paddle._C_ops.add_(concat_5, add__148)

        # pd_op.batch_norm_: (-1x1024x7x7xf16, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x7x7xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__528, batch_norm__529, batch_norm__530, batch_norm__531, batch_norm__532, batch_norm__533 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(add__149, parameter_597, parameter_598, parameter_599, parameter_600, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_114 = [1, 1]

        # pd_op.pool2d: (-1x1024x1x1xf16) <- (-1x1024x7x7xf16, 2xi64)
        pool2d_8 = paddle._C_ops.pool2d(batch_norm__528, full_int_array_114, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.flatten_: (-1x1024xf16, None) <- (-1x1024x1x1xf16)
        flatten__0, flatten__1 = (lambda x, f: f(x))(paddle._C_ops.flatten_(pool2d_8, 1, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x1000xf16) <- (-1x1024xf16, 1024x1000xf16)
        matmul_36 = paddle._C_ops.matmul(flatten__0, parameter_601, False, False)

        # pd_op.add_: (-1x1000xf16) <- (-1x1000xf16, 1000xf16)
        add__150 = paddle._C_ops.add_(matmul_36, parameter_602)

        # pd_op.softmax_: (-1x1000xf16) <- (-1x1000xf16)
        softmax__6 = paddle._C_ops.softmax_(add__150, -1)

        # pd_op.cast: (-1x1000xf32) <- (-1x1000xf16)
        cast_1 = paddle._C_ops.cast(softmax__6, paddle.float32)
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

    def forward(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_36, parameter_37, parameter_38, parameter_39, parameter_43, parameter_40, parameter_42, parameter_41, parameter_44, parameter_48, parameter_45, parameter_47, parameter_46, parameter_49, parameter_50, parameter_51, parameter_52, parameter_53, parameter_57, parameter_54, parameter_56, parameter_55, parameter_58, parameter_62, parameter_59, parameter_61, parameter_60, parameter_63, parameter_64, parameter_65, parameter_66, parameter_67, parameter_71, parameter_68, parameter_70, parameter_69, parameter_72, parameter_76, parameter_73, parameter_75, parameter_74, parameter_77, parameter_81, parameter_78, parameter_80, parameter_79, parameter_82, parameter_83, parameter_84, parameter_85, parameter_86, parameter_90, parameter_87, parameter_89, parameter_88, parameter_91, parameter_95, parameter_92, parameter_94, parameter_93, parameter_96, parameter_97, parameter_98, parameter_99, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_111, parameter_112, parameter_113, parameter_117, parameter_114, parameter_116, parameter_115, parameter_118, parameter_119, parameter_123, parameter_120, parameter_122, parameter_121, parameter_124, parameter_125, parameter_126, parameter_127, parameter_128, parameter_129, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_146, parameter_147, parameter_148, parameter_149, parameter_153, parameter_150, parameter_152, parameter_151, parameter_154, parameter_158, parameter_155, parameter_157, parameter_156, parameter_159, parameter_163, parameter_160, parameter_162, parameter_161, parameter_164, parameter_165, parameter_166, parameter_167, parameter_168, parameter_172, parameter_169, parameter_171, parameter_170, parameter_173, parameter_177, parameter_174, parameter_176, parameter_175, parameter_178, parameter_179, parameter_180, parameter_181, parameter_182, parameter_186, parameter_183, parameter_185, parameter_184, parameter_187, parameter_191, parameter_188, parameter_190, parameter_189, parameter_192, parameter_193, parameter_194, parameter_195, parameter_196, parameter_200, parameter_197, parameter_199, parameter_198, parameter_201, parameter_205, parameter_202, parameter_204, parameter_203, parameter_206, parameter_207, parameter_208, parameter_209, parameter_213, parameter_210, parameter_212, parameter_211, parameter_214, parameter_215, parameter_219, parameter_216, parameter_218, parameter_217, parameter_220, parameter_221, parameter_222, parameter_223, parameter_224, parameter_225, parameter_226, parameter_230, parameter_227, parameter_229, parameter_228, parameter_231, parameter_235, parameter_232, parameter_234, parameter_233, parameter_236, parameter_240, parameter_237, parameter_239, parameter_238, parameter_241, parameter_242, parameter_243, parameter_244, parameter_245, parameter_249, parameter_246, parameter_248, parameter_247, parameter_250, parameter_254, parameter_251, parameter_253, parameter_252, parameter_255, parameter_259, parameter_256, parameter_258, parameter_257, parameter_260, parameter_261, parameter_262, parameter_263, parameter_264, parameter_268, parameter_265, parameter_267, parameter_266, parameter_269, parameter_273, parameter_270, parameter_272, parameter_271, parameter_274, parameter_275, parameter_276, parameter_277, parameter_278, parameter_282, parameter_279, parameter_281, parameter_280, parameter_283, parameter_287, parameter_284, parameter_286, parameter_285, parameter_288, parameter_289, parameter_290, parameter_291, parameter_292, parameter_296, parameter_293, parameter_295, parameter_294, parameter_297, parameter_301, parameter_298, parameter_300, parameter_299, parameter_302, parameter_303, parameter_304, parameter_305, parameter_309, parameter_306, parameter_308, parameter_307, parameter_310, parameter_311, parameter_315, parameter_312, parameter_314, parameter_313, parameter_316, parameter_317, parameter_318, parameter_319, parameter_320, parameter_321, parameter_322, parameter_326, parameter_323, parameter_325, parameter_324, parameter_327, parameter_331, parameter_328, parameter_330, parameter_329, parameter_332, parameter_336, parameter_333, parameter_335, parameter_334, parameter_337, parameter_338, parameter_339, parameter_340, parameter_341, parameter_345, parameter_342, parameter_344, parameter_343, parameter_346, parameter_350, parameter_347, parameter_349, parameter_348, parameter_351, parameter_355, parameter_352, parameter_354, parameter_353, parameter_356, parameter_357, parameter_358, parameter_359, parameter_360, parameter_364, parameter_361, parameter_363, parameter_362, parameter_365, parameter_369, parameter_366, parameter_368, parameter_367, parameter_370, parameter_371, parameter_372, parameter_373, parameter_374, parameter_378, parameter_375, parameter_377, parameter_376, parameter_379, parameter_383, parameter_380, parameter_382, parameter_381, parameter_384, parameter_385, parameter_386, parameter_387, parameter_388, parameter_392, parameter_389, parameter_391, parameter_390, parameter_393, parameter_397, parameter_394, parameter_396, parameter_395, parameter_398, parameter_399, parameter_400, parameter_401, parameter_405, parameter_402, parameter_404, parameter_403, parameter_406, parameter_407, parameter_411, parameter_408, parameter_410, parameter_409, parameter_412, parameter_413, parameter_414, parameter_415, parameter_416, parameter_417, parameter_418, parameter_422, parameter_419, parameter_421, parameter_420, parameter_423, parameter_427, parameter_424, parameter_426, parameter_425, parameter_428, parameter_432, parameter_429, parameter_431, parameter_430, parameter_433, parameter_434, parameter_435, parameter_436, parameter_437, parameter_441, parameter_438, parameter_440, parameter_439, parameter_442, parameter_446, parameter_443, parameter_445, parameter_444, parameter_447, parameter_451, parameter_448, parameter_450, parameter_449, parameter_452, parameter_453, parameter_454, parameter_455, parameter_456, parameter_460, parameter_457, parameter_459, parameter_458, parameter_461, parameter_465, parameter_462, parameter_464, parameter_463, parameter_466, parameter_467, parameter_468, parameter_469, parameter_470, parameter_474, parameter_471, parameter_473, parameter_472, parameter_475, parameter_479, parameter_476, parameter_478, parameter_477, parameter_480, parameter_481, parameter_482, parameter_483, parameter_484, parameter_488, parameter_485, parameter_487, parameter_486, parameter_489, parameter_493, parameter_490, parameter_492, parameter_491, parameter_494, parameter_495, parameter_496, parameter_497, parameter_501, parameter_498, parameter_500, parameter_499, parameter_502, parameter_503, parameter_507, parameter_504, parameter_506, parameter_505, parameter_508, parameter_509, parameter_510, parameter_511, parameter_512, parameter_513, parameter_514, parameter_518, parameter_515, parameter_517, parameter_516, parameter_519, parameter_523, parameter_520, parameter_522, parameter_521, parameter_524, parameter_528, parameter_525, parameter_527, parameter_526, parameter_529, parameter_530, parameter_531, parameter_532, parameter_533, parameter_537, parameter_534, parameter_536, parameter_535, parameter_538, parameter_542, parameter_539, parameter_541, parameter_540, parameter_543, parameter_547, parameter_544, parameter_546, parameter_545, parameter_548, parameter_549, parameter_550, parameter_551, parameter_552, parameter_556, parameter_553, parameter_555, parameter_554, parameter_557, parameter_561, parameter_558, parameter_560, parameter_559, parameter_562, parameter_563, parameter_564, parameter_565, parameter_569, parameter_566, parameter_568, parameter_567, parameter_570, parameter_571, parameter_572, parameter_573, parameter_574, parameter_575, parameter_576, parameter_577, parameter_578, parameter_582, parameter_579, parameter_581, parameter_580, parameter_583, parameter_587, parameter_584, parameter_586, parameter_585, parameter_588, parameter_592, parameter_589, parameter_591, parameter_590, parameter_593, parameter_594, parameter_595, parameter_596, parameter_600, parameter_597, parameter_599, parameter_598, parameter_601, parameter_602, feed_0):
        return self.builtin_module_1572_0_0(parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_36, parameter_37, parameter_38, parameter_39, parameter_43, parameter_40, parameter_42, parameter_41, parameter_44, parameter_48, parameter_45, parameter_47, parameter_46, parameter_49, parameter_50, parameter_51, parameter_52, parameter_53, parameter_57, parameter_54, parameter_56, parameter_55, parameter_58, parameter_62, parameter_59, parameter_61, parameter_60, parameter_63, parameter_64, parameter_65, parameter_66, parameter_67, parameter_71, parameter_68, parameter_70, parameter_69, parameter_72, parameter_76, parameter_73, parameter_75, parameter_74, parameter_77, parameter_81, parameter_78, parameter_80, parameter_79, parameter_82, parameter_83, parameter_84, parameter_85, parameter_86, parameter_90, parameter_87, parameter_89, parameter_88, parameter_91, parameter_95, parameter_92, parameter_94, parameter_93, parameter_96, parameter_97, parameter_98, parameter_99, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_111, parameter_112, parameter_113, parameter_117, parameter_114, parameter_116, parameter_115, parameter_118, parameter_119, parameter_123, parameter_120, parameter_122, parameter_121, parameter_124, parameter_125, parameter_126, parameter_127, parameter_128, parameter_129, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_146, parameter_147, parameter_148, parameter_149, parameter_153, parameter_150, parameter_152, parameter_151, parameter_154, parameter_158, parameter_155, parameter_157, parameter_156, parameter_159, parameter_163, parameter_160, parameter_162, parameter_161, parameter_164, parameter_165, parameter_166, parameter_167, parameter_168, parameter_172, parameter_169, parameter_171, parameter_170, parameter_173, parameter_177, parameter_174, parameter_176, parameter_175, parameter_178, parameter_179, parameter_180, parameter_181, parameter_182, parameter_186, parameter_183, parameter_185, parameter_184, parameter_187, parameter_191, parameter_188, parameter_190, parameter_189, parameter_192, parameter_193, parameter_194, parameter_195, parameter_196, parameter_200, parameter_197, parameter_199, parameter_198, parameter_201, parameter_205, parameter_202, parameter_204, parameter_203, parameter_206, parameter_207, parameter_208, parameter_209, parameter_213, parameter_210, parameter_212, parameter_211, parameter_214, parameter_215, parameter_219, parameter_216, parameter_218, parameter_217, parameter_220, parameter_221, parameter_222, parameter_223, parameter_224, parameter_225, parameter_226, parameter_230, parameter_227, parameter_229, parameter_228, parameter_231, parameter_235, parameter_232, parameter_234, parameter_233, parameter_236, parameter_240, parameter_237, parameter_239, parameter_238, parameter_241, parameter_242, parameter_243, parameter_244, parameter_245, parameter_249, parameter_246, parameter_248, parameter_247, parameter_250, parameter_254, parameter_251, parameter_253, parameter_252, parameter_255, parameter_259, parameter_256, parameter_258, parameter_257, parameter_260, parameter_261, parameter_262, parameter_263, parameter_264, parameter_268, parameter_265, parameter_267, parameter_266, parameter_269, parameter_273, parameter_270, parameter_272, parameter_271, parameter_274, parameter_275, parameter_276, parameter_277, parameter_278, parameter_282, parameter_279, parameter_281, parameter_280, parameter_283, parameter_287, parameter_284, parameter_286, parameter_285, parameter_288, parameter_289, parameter_290, parameter_291, parameter_292, parameter_296, parameter_293, parameter_295, parameter_294, parameter_297, parameter_301, parameter_298, parameter_300, parameter_299, parameter_302, parameter_303, parameter_304, parameter_305, parameter_309, parameter_306, parameter_308, parameter_307, parameter_310, parameter_311, parameter_315, parameter_312, parameter_314, parameter_313, parameter_316, parameter_317, parameter_318, parameter_319, parameter_320, parameter_321, parameter_322, parameter_326, parameter_323, parameter_325, parameter_324, parameter_327, parameter_331, parameter_328, parameter_330, parameter_329, parameter_332, parameter_336, parameter_333, parameter_335, parameter_334, parameter_337, parameter_338, parameter_339, parameter_340, parameter_341, parameter_345, parameter_342, parameter_344, parameter_343, parameter_346, parameter_350, parameter_347, parameter_349, parameter_348, parameter_351, parameter_355, parameter_352, parameter_354, parameter_353, parameter_356, parameter_357, parameter_358, parameter_359, parameter_360, parameter_364, parameter_361, parameter_363, parameter_362, parameter_365, parameter_369, parameter_366, parameter_368, parameter_367, parameter_370, parameter_371, parameter_372, parameter_373, parameter_374, parameter_378, parameter_375, parameter_377, parameter_376, parameter_379, parameter_383, parameter_380, parameter_382, parameter_381, parameter_384, parameter_385, parameter_386, parameter_387, parameter_388, parameter_392, parameter_389, parameter_391, parameter_390, parameter_393, parameter_397, parameter_394, parameter_396, parameter_395, parameter_398, parameter_399, parameter_400, parameter_401, parameter_405, parameter_402, parameter_404, parameter_403, parameter_406, parameter_407, parameter_411, parameter_408, parameter_410, parameter_409, parameter_412, parameter_413, parameter_414, parameter_415, parameter_416, parameter_417, parameter_418, parameter_422, parameter_419, parameter_421, parameter_420, parameter_423, parameter_427, parameter_424, parameter_426, parameter_425, parameter_428, parameter_432, parameter_429, parameter_431, parameter_430, parameter_433, parameter_434, parameter_435, parameter_436, parameter_437, parameter_441, parameter_438, parameter_440, parameter_439, parameter_442, parameter_446, parameter_443, parameter_445, parameter_444, parameter_447, parameter_451, parameter_448, parameter_450, parameter_449, parameter_452, parameter_453, parameter_454, parameter_455, parameter_456, parameter_460, parameter_457, parameter_459, parameter_458, parameter_461, parameter_465, parameter_462, parameter_464, parameter_463, parameter_466, parameter_467, parameter_468, parameter_469, parameter_470, parameter_474, parameter_471, parameter_473, parameter_472, parameter_475, parameter_479, parameter_476, parameter_478, parameter_477, parameter_480, parameter_481, parameter_482, parameter_483, parameter_484, parameter_488, parameter_485, parameter_487, parameter_486, parameter_489, parameter_493, parameter_490, parameter_492, parameter_491, parameter_494, parameter_495, parameter_496, parameter_497, parameter_501, parameter_498, parameter_500, parameter_499, parameter_502, parameter_503, parameter_507, parameter_504, parameter_506, parameter_505, parameter_508, parameter_509, parameter_510, parameter_511, parameter_512, parameter_513, parameter_514, parameter_518, parameter_515, parameter_517, parameter_516, parameter_519, parameter_523, parameter_520, parameter_522, parameter_521, parameter_524, parameter_528, parameter_525, parameter_527, parameter_526, parameter_529, parameter_530, parameter_531, parameter_532, parameter_533, parameter_537, parameter_534, parameter_536, parameter_535, parameter_538, parameter_542, parameter_539, parameter_541, parameter_540, parameter_543, parameter_547, parameter_544, parameter_546, parameter_545, parameter_548, parameter_549, parameter_550, parameter_551, parameter_552, parameter_556, parameter_553, parameter_555, parameter_554, parameter_557, parameter_561, parameter_558, parameter_560, parameter_559, parameter_562, parameter_563, parameter_564, parameter_565, parameter_569, parameter_566, parameter_568, parameter_567, parameter_570, parameter_571, parameter_572, parameter_573, parameter_574, parameter_575, parameter_576, parameter_577, parameter_578, parameter_582, parameter_579, parameter_581, parameter_580, parameter_583, parameter_587, parameter_584, parameter_586, parameter_585, parameter_588, parameter_592, parameter_589, parameter_591, parameter_590, parameter_593, parameter_594, parameter_595, parameter_596, parameter_600, parameter_597, parameter_599, parameter_598, parameter_601, parameter_602, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_1572_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([64, 3, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_4
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([32, 64, 3, 3], dtype='float16', min=0, max=0.5),
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
            paddle.uniform([64, 64, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_19
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([96, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_24
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([96, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_29
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([96, 96, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_34
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([288, 96, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_36
            paddle.uniform([288], dtype='float16', min=0, max=0.5),
            # parameter_37
            paddle.uniform([96, 288, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_38
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_39
            paddle.uniform([96, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_43
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([96, 96, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_48
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([288, 96, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_50
            paddle.uniform([288], dtype='float16', min=0, max=0.5),
            # parameter_51
            paddle.uniform([96, 288, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_52
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_53
            paddle.uniform([96, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_57
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([96, 96, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_62
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([288, 96, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_64
            paddle.uniform([288], dtype='float16', min=0, max=0.5),
            # parameter_65
            paddle.uniform([96, 288, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_66
            paddle.uniform([96], dtype='float16', min=0, max=0.5),
            # parameter_67
            paddle.uniform([192, 96, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_71
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([192, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_76
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([192, 192, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_81
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([576, 192, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_83
            paddle.uniform([576], dtype='float16', min=0, max=0.5),
            # parameter_84
            paddle.uniform([192, 576, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_85
            paddle.uniform([192], dtype='float16', min=0, max=0.5),
            # parameter_86
            paddle.uniform([192, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_90
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([192, 192, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_95
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([576, 192, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_97
            paddle.uniform([576], dtype='float16', min=0, max=0.5),
            # parameter_98
            paddle.uniform([192, 576, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_99
            paddle.uniform([192], dtype='float16', min=0, max=0.5),
            # parameter_100
            paddle.uniform([192, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_104
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([192, 192, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_109
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([576, 192, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_111
            paddle.uniform([576], dtype='float16', min=0, max=0.5),
            # parameter_112
            paddle.uniform([192, 576, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_113
            paddle.uniform([192], dtype='float16', min=0, max=0.5),
            # parameter_117
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([192, 192], dtype='float16', min=0, max=0.5),
            # parameter_119
            paddle.uniform([192], dtype='float16', min=0, max=0.5),
            # parameter_123
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([192, 192], dtype='float16', min=0, max=0.5),
            # parameter_125
            paddle.uniform([192], dtype='float16', min=0, max=0.5),
            # parameter_126
            paddle.uniform([192, 192], dtype='float16', min=0, max=0.5),
            # parameter_127
            paddle.uniform([192], dtype='float16', min=0, max=0.5),
            # parameter_128
            paddle.uniform([192, 192], dtype='float16', min=0, max=0.5),
            # parameter_129
            paddle.uniform([192], dtype='float16', min=0, max=0.5),
            # parameter_130
            paddle.uniform([64, 192, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_134
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([64, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_139
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_144
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([512, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_146
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_147
            paddle.uniform([256, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_148
            paddle.uniform([256], dtype='float16', min=0, max=0.5),
            # parameter_149
            paddle.uniform([384, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_153
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_154
            paddle.uniform([384, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_158
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.uniform([384, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_163
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([1152, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_165
            paddle.uniform([1152], dtype='float16', min=0, max=0.5),
            # parameter_166
            paddle.uniform([384, 1152, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_167
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_168
            paddle.uniform([384, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_172
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([384, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_177
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([1152, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_179
            paddle.uniform([1152], dtype='float16', min=0, max=0.5),
            # parameter_180
            paddle.uniform([384, 1152, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_181
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_182
            paddle.uniform([384, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_186
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([384, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_191
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_189
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([1152, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_193
            paddle.uniform([1152], dtype='float16', min=0, max=0.5),
            # parameter_194
            paddle.uniform([384, 1152, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_195
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_196
            paddle.uniform([384, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_200
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([384, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_205
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([1152, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_207
            paddle.uniform([1152], dtype='float16', min=0, max=0.5),
            # parameter_208
            paddle.uniform([384, 1152, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_209
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_213
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_214
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_215
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_219
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_218
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_217
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_221
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_222
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_223
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_224
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_225
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_226
            paddle.uniform([128, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_230
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_227
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_229
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([128, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_235
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_234
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_233
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_240
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_239
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_238
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_241
            paddle.uniform([1024, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_242
            paddle.uniform([1024], dtype='float16', min=0, max=0.5),
            # parameter_243
            paddle.uniform([512, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_244
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_245
            paddle.uniform([384, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_249
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_246
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_248
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_250
            paddle.uniform([384, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_254
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_251
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_253
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_252
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_255
            paddle.uniform([384, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_259
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_256
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_258
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_257
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([1152, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_261
            paddle.uniform([1152], dtype='float16', min=0, max=0.5),
            # parameter_262
            paddle.uniform([384, 1152, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_263
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_264
            paddle.uniform([384, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_268
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_265
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_267
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_266
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_269
            paddle.uniform([384, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_273
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_270
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_272
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_271
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_274
            paddle.uniform([1152, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_275
            paddle.uniform([1152], dtype='float16', min=0, max=0.5),
            # parameter_276
            paddle.uniform([384, 1152, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_277
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_278
            paddle.uniform([384, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_282
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_279
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_281
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_280
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_283
            paddle.uniform([384, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_287
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_284
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_286
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_285
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_288
            paddle.uniform([1152, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_289
            paddle.uniform([1152], dtype='float16', min=0, max=0.5),
            # parameter_290
            paddle.uniform([384, 1152, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_291
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_292
            paddle.uniform([384, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_296
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_293
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_295
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_294
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_297
            paddle.uniform([384, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_301
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_298
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_300
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_299
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_302
            paddle.uniform([1152, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_303
            paddle.uniform([1152], dtype='float16', min=0, max=0.5),
            # parameter_304
            paddle.uniform([384, 1152, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_305
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_309
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_306
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_308
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_307
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_310
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_311
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_315
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_312
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_314
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_313
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_316
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_317
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_318
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_319
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_320
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_321
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_322
            paddle.uniform([128, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_326
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_323
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_325
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_324
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_327
            paddle.uniform([128, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_331
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_328
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_330
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_329
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_332
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_336
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_333
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_335
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_334
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_337
            paddle.uniform([1024, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_338
            paddle.uniform([1024], dtype='float16', min=0, max=0.5),
            # parameter_339
            paddle.uniform([512, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_340
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_341
            paddle.uniform([384, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_345
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_342
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_344
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_343
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_346
            paddle.uniform([384, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_350
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_347
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_349
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_348
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_351
            paddle.uniform([384, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_355
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_352
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_354
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_353
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_356
            paddle.uniform([1152, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_357
            paddle.uniform([1152], dtype='float16', min=0, max=0.5),
            # parameter_358
            paddle.uniform([384, 1152, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_359
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_360
            paddle.uniform([384, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_364
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_361
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_363
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_362
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_365
            paddle.uniform([384, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_369
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_366
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_368
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_367
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_370
            paddle.uniform([1152, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_371
            paddle.uniform([1152], dtype='float16', min=0, max=0.5),
            # parameter_372
            paddle.uniform([384, 1152, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_373
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_374
            paddle.uniform([384, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_378
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_375
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_377
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_376
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_379
            paddle.uniform([384, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_383
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_380
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_382
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_381
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_384
            paddle.uniform([1152, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_385
            paddle.uniform([1152], dtype='float16', min=0, max=0.5),
            # parameter_386
            paddle.uniform([384, 1152, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_387
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_388
            paddle.uniform([384, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_392
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_389
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_391
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_390
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_393
            paddle.uniform([384, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_397
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_394
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_396
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_395
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_398
            paddle.uniform([1152, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_399
            paddle.uniform([1152], dtype='float16', min=0, max=0.5),
            # parameter_400
            paddle.uniform([384, 1152, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_401
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_405
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_402
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_404
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_403
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_406
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_407
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_411
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_408
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_410
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_409
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_412
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_413
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_414
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_415
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_416
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_417
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_418
            paddle.uniform([128, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_422
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_419
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_421
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_420
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_423
            paddle.uniform([128, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_427
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_424
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_426
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_425
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_428
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_432
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_429
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_431
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_430
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_433
            paddle.uniform([1024, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_434
            paddle.uniform([1024], dtype='float16', min=0, max=0.5),
            # parameter_435
            paddle.uniform([512, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_436
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_437
            paddle.uniform([384, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_441
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_438
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_440
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_439
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_442
            paddle.uniform([384, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_446
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_443
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_445
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_444
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_447
            paddle.uniform([384, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_451
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_448
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_450
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_449
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_452
            paddle.uniform([1152, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_453
            paddle.uniform([1152], dtype='float16', min=0, max=0.5),
            # parameter_454
            paddle.uniform([384, 1152, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_455
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_456
            paddle.uniform([384, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_460
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_457
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_459
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_458
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_461
            paddle.uniform([384, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_465
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_462
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_464
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_463
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_466
            paddle.uniform([1152, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_467
            paddle.uniform([1152], dtype='float16', min=0, max=0.5),
            # parameter_468
            paddle.uniform([384, 1152, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_469
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_470
            paddle.uniform([384, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_474
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_471
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_473
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_472
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_475
            paddle.uniform([384, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_479
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_476
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_478
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_477
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_480
            paddle.uniform([1152, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_481
            paddle.uniform([1152], dtype='float16', min=0, max=0.5),
            # parameter_482
            paddle.uniform([384, 1152, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_483
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_484
            paddle.uniform([384, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_488
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_485
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_487
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_486
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_489
            paddle.uniform([384, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_493
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_490
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_492
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_491
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_494
            paddle.uniform([1152, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_495
            paddle.uniform([1152], dtype='float16', min=0, max=0.5),
            # parameter_496
            paddle.uniform([384, 1152, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_497
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_501
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_498
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_500
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_499
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_502
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_503
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_507
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_504
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_506
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_505
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_508
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_509
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_510
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_511
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_512
            paddle.uniform([384, 384], dtype='float16', min=0, max=0.5),
            # parameter_513
            paddle.uniform([384], dtype='float16', min=0, max=0.5),
            # parameter_514
            paddle.uniform([128, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_518
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_515
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_517
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_516
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_519
            paddle.uniform([128, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_523
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_520
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_522
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_521
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_524
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_528
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_525
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_527
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_526
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_529
            paddle.uniform([1024, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_530
            paddle.uniform([1024], dtype='float16', min=0, max=0.5),
            # parameter_531
            paddle.uniform([512, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_532
            paddle.uniform([512], dtype='float16', min=0, max=0.5),
            # parameter_533
            paddle.uniform([768, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_537
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_534
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_536
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_535
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_538
            paddle.uniform([768, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_542
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_539
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_541
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_540
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_543
            paddle.uniform([768, 768, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_547
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_544
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_546
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_545
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_548
            paddle.uniform([2304, 768, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_549
            paddle.uniform([2304], dtype='float16', min=0, max=0.5),
            # parameter_550
            paddle.uniform([768, 2304, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_551
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_552
            paddle.uniform([768, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_556
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_553
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_555
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_554
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_557
            paddle.uniform([768, 768, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_561
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_558
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_560
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_559
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_562
            paddle.uniform([2304, 768, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_563
            paddle.uniform([2304], dtype='float16', min=0, max=0.5),
            # parameter_564
            paddle.uniform([768, 2304, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_565
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_569
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_566
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_568
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_567
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_570
            paddle.uniform([768, 768], dtype='float16', min=0, max=0.5),
            # parameter_571
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_572
            paddle.uniform([768, 768], dtype='float16', min=0, max=0.5),
            # parameter_573
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_574
            paddle.uniform([768, 768], dtype='float16', min=0, max=0.5),
            # parameter_575
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_576
            paddle.uniform([768, 768], dtype='float16', min=0, max=0.5),
            # parameter_577
            paddle.uniform([768], dtype='float16', min=0, max=0.5),
            # parameter_578
            paddle.uniform([256, 768, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_582
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_579
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_581
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_580
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_583
            paddle.uniform([256, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_587
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_584
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_586
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_585
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_588
            paddle.uniform([256, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_592
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_589
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_591
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_590
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_593
            paddle.uniform([2048, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_594
            paddle.uniform([2048], dtype='float16', min=0, max=0.5),
            # parameter_595
            paddle.uniform([1024, 2048, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_596
            paddle.uniform([1024], dtype='float16', min=0, max=0.5),
            # parameter_600
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_597
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_599
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_598
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_601
            paddle.uniform([1024, 1000], dtype='float16', min=0, max=0.5),
            # parameter_602
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
            paddle.static.InputSpec(shape=[64, 3, 3, 3], dtype='float16'),
            # parameter_4
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[32, 64, 3, 3], dtype='float16'),
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
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float16'),
            # parameter_19
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[96, 64, 1, 1], dtype='float16'),
            # parameter_24
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[96, 32, 3, 3], dtype='float16'),
            # parameter_29
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float16'),
            # parameter_34
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[288, 96, 1, 1], dtype='float16'),
            # parameter_36
            paddle.static.InputSpec(shape=[288], dtype='float16'),
            # parameter_37
            paddle.static.InputSpec(shape=[96, 288, 1, 1], dtype='float16'),
            # parameter_38
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_39
            paddle.static.InputSpec(shape=[96, 32, 3, 3], dtype='float16'),
            # parameter_43
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float16'),
            # parameter_48
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[288, 96, 1, 1], dtype='float16'),
            # parameter_50
            paddle.static.InputSpec(shape=[288], dtype='float16'),
            # parameter_51
            paddle.static.InputSpec(shape=[96, 288, 1, 1], dtype='float16'),
            # parameter_52
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_53
            paddle.static.InputSpec(shape=[96, 32, 3, 3], dtype='float16'),
            # parameter_57
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[96, 96, 1, 1], dtype='float16'),
            # parameter_62
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[288, 96, 1, 1], dtype='float16'),
            # parameter_64
            paddle.static.InputSpec(shape=[288], dtype='float16'),
            # parameter_65
            paddle.static.InputSpec(shape=[96, 288, 1, 1], dtype='float16'),
            # parameter_66
            paddle.static.InputSpec(shape=[96], dtype='float16'),
            # parameter_67
            paddle.static.InputSpec(shape=[192, 96, 1, 1], dtype='float16'),
            # parameter_71
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[192, 32, 3, 3], dtype='float16'),
            # parameter_76
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[192, 192, 1, 1], dtype='float16'),
            # parameter_81
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[576, 192, 1, 1], dtype='float16'),
            # parameter_83
            paddle.static.InputSpec(shape=[576], dtype='float16'),
            # parameter_84
            paddle.static.InputSpec(shape=[192, 576, 1, 1], dtype='float16'),
            # parameter_85
            paddle.static.InputSpec(shape=[192], dtype='float16'),
            # parameter_86
            paddle.static.InputSpec(shape=[192, 32, 3, 3], dtype='float16'),
            # parameter_90
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[192, 192, 1, 1], dtype='float16'),
            # parameter_95
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[576, 192, 1, 1], dtype='float16'),
            # parameter_97
            paddle.static.InputSpec(shape=[576], dtype='float16'),
            # parameter_98
            paddle.static.InputSpec(shape=[192, 576, 1, 1], dtype='float16'),
            # parameter_99
            paddle.static.InputSpec(shape=[192], dtype='float16'),
            # parameter_100
            paddle.static.InputSpec(shape=[192, 32, 3, 3], dtype='float16'),
            # parameter_104
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[192, 192, 1, 1], dtype='float16'),
            # parameter_109
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[576, 192, 1, 1], dtype='float16'),
            # parameter_111
            paddle.static.InputSpec(shape=[576], dtype='float16'),
            # parameter_112
            paddle.static.InputSpec(shape=[192, 576, 1, 1], dtype='float16'),
            # parameter_113
            paddle.static.InputSpec(shape=[192], dtype='float16'),
            # parameter_117
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[192, 192], dtype='float16'),
            # parameter_119
            paddle.static.InputSpec(shape=[192], dtype='float16'),
            # parameter_123
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[192, 192], dtype='float16'),
            # parameter_125
            paddle.static.InputSpec(shape=[192], dtype='float16'),
            # parameter_126
            paddle.static.InputSpec(shape=[192, 192], dtype='float16'),
            # parameter_127
            paddle.static.InputSpec(shape=[192], dtype='float16'),
            # parameter_128
            paddle.static.InputSpec(shape=[192, 192], dtype='float16'),
            # parameter_129
            paddle.static.InputSpec(shape=[192], dtype='float16'),
            # parameter_130
            paddle.static.InputSpec(shape=[64, 192, 1, 1], dtype='float16'),
            # parameter_134
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[64, 32, 3, 3], dtype='float16'),
            # parameter_139
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_144
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[512, 256, 1, 1], dtype='float16'),
            # parameter_146
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_147
            paddle.static.InputSpec(shape=[256, 512, 1, 1], dtype='float16'),
            # parameter_148
            paddle.static.InputSpec(shape=[256], dtype='float16'),
            # parameter_149
            paddle.static.InputSpec(shape=[384, 256, 1, 1], dtype='float16'),
            # parameter_153
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_154
            paddle.static.InputSpec(shape=[384, 32, 3, 3], dtype='float16'),
            # parameter_158
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[384, 384, 1, 1], dtype='float16'),
            # parameter_163
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[1152, 384, 1, 1], dtype='float16'),
            # parameter_165
            paddle.static.InputSpec(shape=[1152], dtype='float16'),
            # parameter_166
            paddle.static.InputSpec(shape=[384, 1152, 1, 1], dtype='float16'),
            # parameter_167
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_168
            paddle.static.InputSpec(shape=[384, 32, 3, 3], dtype='float16'),
            # parameter_172
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[384, 384, 1, 1], dtype='float16'),
            # parameter_177
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[1152, 384, 1, 1], dtype='float16'),
            # parameter_179
            paddle.static.InputSpec(shape=[1152], dtype='float16'),
            # parameter_180
            paddle.static.InputSpec(shape=[384, 1152, 1, 1], dtype='float16'),
            # parameter_181
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_182
            paddle.static.InputSpec(shape=[384, 32, 3, 3], dtype='float16'),
            # parameter_186
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[384, 384, 1, 1], dtype='float16'),
            # parameter_191
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_189
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[1152, 384, 1, 1], dtype='float16'),
            # parameter_193
            paddle.static.InputSpec(shape=[1152], dtype='float16'),
            # parameter_194
            paddle.static.InputSpec(shape=[384, 1152, 1, 1], dtype='float16'),
            # parameter_195
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_196
            paddle.static.InputSpec(shape=[384, 32, 3, 3], dtype='float16'),
            # parameter_200
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[384, 384, 1, 1], dtype='float16'),
            # parameter_205
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[1152, 384, 1, 1], dtype='float16'),
            # parameter_207
            paddle.static.InputSpec(shape=[1152], dtype='float16'),
            # parameter_208
            paddle.static.InputSpec(shape=[384, 1152, 1, 1], dtype='float16'),
            # parameter_209
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_213
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_214
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_215
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_219
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_218
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_217
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_221
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_222
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_223
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_224
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_225
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_226
            paddle.static.InputSpec(shape=[128, 384, 1, 1], dtype='float16'),
            # parameter_230
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_227
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_229
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[128, 32, 3, 3], dtype='float16'),
            # parameter_235
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_234
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_233
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_240
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_239
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_238
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_241
            paddle.static.InputSpec(shape=[1024, 512, 1, 1], dtype='float16'),
            # parameter_242
            paddle.static.InputSpec(shape=[1024], dtype='float16'),
            # parameter_243
            paddle.static.InputSpec(shape=[512, 1024, 1, 1], dtype='float16'),
            # parameter_244
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_245
            paddle.static.InputSpec(shape=[384, 512, 1, 1], dtype='float16'),
            # parameter_249
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_246
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_248
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_250
            paddle.static.InputSpec(shape=[384, 32, 3, 3], dtype='float16'),
            # parameter_254
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_251
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_253
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_252
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_255
            paddle.static.InputSpec(shape=[384, 384, 1, 1], dtype='float16'),
            # parameter_259
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_256
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_258
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_257
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[1152, 384, 1, 1], dtype='float16'),
            # parameter_261
            paddle.static.InputSpec(shape=[1152], dtype='float16'),
            # parameter_262
            paddle.static.InputSpec(shape=[384, 1152, 1, 1], dtype='float16'),
            # parameter_263
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_264
            paddle.static.InputSpec(shape=[384, 32, 3, 3], dtype='float16'),
            # parameter_268
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_265
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_267
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_266
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_269
            paddle.static.InputSpec(shape=[384, 384, 1, 1], dtype='float16'),
            # parameter_273
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_270
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_272
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_271
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_274
            paddle.static.InputSpec(shape=[1152, 384, 1, 1], dtype='float16'),
            # parameter_275
            paddle.static.InputSpec(shape=[1152], dtype='float16'),
            # parameter_276
            paddle.static.InputSpec(shape=[384, 1152, 1, 1], dtype='float16'),
            # parameter_277
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_278
            paddle.static.InputSpec(shape=[384, 32, 3, 3], dtype='float16'),
            # parameter_282
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_279
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_281
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_280
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_283
            paddle.static.InputSpec(shape=[384, 384, 1, 1], dtype='float16'),
            # parameter_287
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_284
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_286
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_285
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_288
            paddle.static.InputSpec(shape=[1152, 384, 1, 1], dtype='float16'),
            # parameter_289
            paddle.static.InputSpec(shape=[1152], dtype='float16'),
            # parameter_290
            paddle.static.InputSpec(shape=[384, 1152, 1, 1], dtype='float16'),
            # parameter_291
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_292
            paddle.static.InputSpec(shape=[384, 32, 3, 3], dtype='float16'),
            # parameter_296
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_293
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_295
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_294
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_297
            paddle.static.InputSpec(shape=[384, 384, 1, 1], dtype='float16'),
            # parameter_301
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_298
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_300
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_299
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_302
            paddle.static.InputSpec(shape=[1152, 384, 1, 1], dtype='float16'),
            # parameter_303
            paddle.static.InputSpec(shape=[1152], dtype='float16'),
            # parameter_304
            paddle.static.InputSpec(shape=[384, 1152, 1, 1], dtype='float16'),
            # parameter_305
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_309
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_306
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_308
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_307
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_310
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_311
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_315
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_312
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_314
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_313
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_316
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_317
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_318
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_319
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_320
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_321
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_322
            paddle.static.InputSpec(shape=[128, 384, 1, 1], dtype='float16'),
            # parameter_326
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_323
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_325
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_324
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_327
            paddle.static.InputSpec(shape=[128, 32, 3, 3], dtype='float16'),
            # parameter_331
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_328
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_330
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_329
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_332
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_336
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_333
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_335
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_334
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_337
            paddle.static.InputSpec(shape=[1024, 512, 1, 1], dtype='float16'),
            # parameter_338
            paddle.static.InputSpec(shape=[1024], dtype='float16'),
            # parameter_339
            paddle.static.InputSpec(shape=[512, 1024, 1, 1], dtype='float16'),
            # parameter_340
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_341
            paddle.static.InputSpec(shape=[384, 512, 1, 1], dtype='float16'),
            # parameter_345
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_342
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_344
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_343
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_346
            paddle.static.InputSpec(shape=[384, 32, 3, 3], dtype='float16'),
            # parameter_350
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_347
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_349
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_348
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_351
            paddle.static.InputSpec(shape=[384, 384, 1, 1], dtype='float16'),
            # parameter_355
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_352
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_354
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_353
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_356
            paddle.static.InputSpec(shape=[1152, 384, 1, 1], dtype='float16'),
            # parameter_357
            paddle.static.InputSpec(shape=[1152], dtype='float16'),
            # parameter_358
            paddle.static.InputSpec(shape=[384, 1152, 1, 1], dtype='float16'),
            # parameter_359
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_360
            paddle.static.InputSpec(shape=[384, 32, 3, 3], dtype='float16'),
            # parameter_364
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_361
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_363
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_362
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_365
            paddle.static.InputSpec(shape=[384, 384, 1, 1], dtype='float16'),
            # parameter_369
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_366
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_368
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_367
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_370
            paddle.static.InputSpec(shape=[1152, 384, 1, 1], dtype='float16'),
            # parameter_371
            paddle.static.InputSpec(shape=[1152], dtype='float16'),
            # parameter_372
            paddle.static.InputSpec(shape=[384, 1152, 1, 1], dtype='float16'),
            # parameter_373
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_374
            paddle.static.InputSpec(shape=[384, 32, 3, 3], dtype='float16'),
            # parameter_378
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_375
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_377
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_376
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_379
            paddle.static.InputSpec(shape=[384, 384, 1, 1], dtype='float16'),
            # parameter_383
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_380
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_382
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_381
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_384
            paddle.static.InputSpec(shape=[1152, 384, 1, 1], dtype='float16'),
            # parameter_385
            paddle.static.InputSpec(shape=[1152], dtype='float16'),
            # parameter_386
            paddle.static.InputSpec(shape=[384, 1152, 1, 1], dtype='float16'),
            # parameter_387
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_388
            paddle.static.InputSpec(shape=[384, 32, 3, 3], dtype='float16'),
            # parameter_392
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_389
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_391
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_390
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_393
            paddle.static.InputSpec(shape=[384, 384, 1, 1], dtype='float16'),
            # parameter_397
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_394
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_396
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_395
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_398
            paddle.static.InputSpec(shape=[1152, 384, 1, 1], dtype='float16'),
            # parameter_399
            paddle.static.InputSpec(shape=[1152], dtype='float16'),
            # parameter_400
            paddle.static.InputSpec(shape=[384, 1152, 1, 1], dtype='float16'),
            # parameter_401
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_405
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_402
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_404
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_403
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_406
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_407
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_411
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_408
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_410
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_409
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_412
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_413
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_414
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_415
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_416
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_417
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_418
            paddle.static.InputSpec(shape=[128, 384, 1, 1], dtype='float16'),
            # parameter_422
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_419
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_421
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_420
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_423
            paddle.static.InputSpec(shape=[128, 32, 3, 3], dtype='float16'),
            # parameter_427
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_424
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_426
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_425
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_428
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_432
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_429
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_431
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_430
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_433
            paddle.static.InputSpec(shape=[1024, 512, 1, 1], dtype='float16'),
            # parameter_434
            paddle.static.InputSpec(shape=[1024], dtype='float16'),
            # parameter_435
            paddle.static.InputSpec(shape=[512, 1024, 1, 1], dtype='float16'),
            # parameter_436
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_437
            paddle.static.InputSpec(shape=[384, 512, 1, 1], dtype='float16'),
            # parameter_441
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_438
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_440
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_439
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_442
            paddle.static.InputSpec(shape=[384, 32, 3, 3], dtype='float16'),
            # parameter_446
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_443
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_445
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_444
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_447
            paddle.static.InputSpec(shape=[384, 384, 1, 1], dtype='float16'),
            # parameter_451
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_448
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_450
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_449
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_452
            paddle.static.InputSpec(shape=[1152, 384, 1, 1], dtype='float16'),
            # parameter_453
            paddle.static.InputSpec(shape=[1152], dtype='float16'),
            # parameter_454
            paddle.static.InputSpec(shape=[384, 1152, 1, 1], dtype='float16'),
            # parameter_455
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_456
            paddle.static.InputSpec(shape=[384, 32, 3, 3], dtype='float16'),
            # parameter_460
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_457
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_459
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_458
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_461
            paddle.static.InputSpec(shape=[384, 384, 1, 1], dtype='float16'),
            # parameter_465
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_462
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_464
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_463
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_466
            paddle.static.InputSpec(shape=[1152, 384, 1, 1], dtype='float16'),
            # parameter_467
            paddle.static.InputSpec(shape=[1152], dtype='float16'),
            # parameter_468
            paddle.static.InputSpec(shape=[384, 1152, 1, 1], dtype='float16'),
            # parameter_469
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_470
            paddle.static.InputSpec(shape=[384, 32, 3, 3], dtype='float16'),
            # parameter_474
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_471
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_473
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_472
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_475
            paddle.static.InputSpec(shape=[384, 384, 1, 1], dtype='float16'),
            # parameter_479
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_476
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_478
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_477
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_480
            paddle.static.InputSpec(shape=[1152, 384, 1, 1], dtype='float16'),
            # parameter_481
            paddle.static.InputSpec(shape=[1152], dtype='float16'),
            # parameter_482
            paddle.static.InputSpec(shape=[384, 1152, 1, 1], dtype='float16'),
            # parameter_483
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_484
            paddle.static.InputSpec(shape=[384, 32, 3, 3], dtype='float16'),
            # parameter_488
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_485
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_487
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_486
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_489
            paddle.static.InputSpec(shape=[384, 384, 1, 1], dtype='float16'),
            # parameter_493
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_490
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_492
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_491
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_494
            paddle.static.InputSpec(shape=[1152, 384, 1, 1], dtype='float16'),
            # parameter_495
            paddle.static.InputSpec(shape=[1152], dtype='float16'),
            # parameter_496
            paddle.static.InputSpec(shape=[384, 1152, 1, 1], dtype='float16'),
            # parameter_497
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_501
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_498
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_500
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_499
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_502
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_503
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_507
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_504
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_506
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_505
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_508
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_509
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_510
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_511
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_512
            paddle.static.InputSpec(shape=[384, 384], dtype='float16'),
            # parameter_513
            paddle.static.InputSpec(shape=[384], dtype='float16'),
            # parameter_514
            paddle.static.InputSpec(shape=[128, 384, 1, 1], dtype='float16'),
            # parameter_518
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_515
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_517
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_516
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_519
            paddle.static.InputSpec(shape=[128, 32, 3, 3], dtype='float16'),
            # parameter_523
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_520
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_522
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_521
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_524
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_528
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_525
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_527
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_526
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_529
            paddle.static.InputSpec(shape=[1024, 512, 1, 1], dtype='float16'),
            # parameter_530
            paddle.static.InputSpec(shape=[1024], dtype='float16'),
            # parameter_531
            paddle.static.InputSpec(shape=[512, 1024, 1, 1], dtype='float16'),
            # parameter_532
            paddle.static.InputSpec(shape=[512], dtype='float16'),
            # parameter_533
            paddle.static.InputSpec(shape=[768, 512, 1, 1], dtype='float16'),
            # parameter_537
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_534
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_536
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_535
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_538
            paddle.static.InputSpec(shape=[768, 32, 3, 3], dtype='float16'),
            # parameter_542
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_539
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_541
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_540
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_543
            paddle.static.InputSpec(shape=[768, 768, 1, 1], dtype='float16'),
            # parameter_547
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_544
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_546
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_545
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_548
            paddle.static.InputSpec(shape=[2304, 768, 1, 1], dtype='float16'),
            # parameter_549
            paddle.static.InputSpec(shape=[2304], dtype='float16'),
            # parameter_550
            paddle.static.InputSpec(shape=[768, 2304, 1, 1], dtype='float16'),
            # parameter_551
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_552
            paddle.static.InputSpec(shape=[768, 32, 3, 3], dtype='float16'),
            # parameter_556
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_553
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_555
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_554
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_557
            paddle.static.InputSpec(shape=[768, 768, 1, 1], dtype='float16'),
            # parameter_561
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_558
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_560
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_559
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_562
            paddle.static.InputSpec(shape=[2304, 768, 1, 1], dtype='float16'),
            # parameter_563
            paddle.static.InputSpec(shape=[2304], dtype='float16'),
            # parameter_564
            paddle.static.InputSpec(shape=[768, 2304, 1, 1], dtype='float16'),
            # parameter_565
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_569
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_566
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_568
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_567
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_570
            paddle.static.InputSpec(shape=[768, 768], dtype='float16'),
            # parameter_571
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_572
            paddle.static.InputSpec(shape=[768, 768], dtype='float16'),
            # parameter_573
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_574
            paddle.static.InputSpec(shape=[768, 768], dtype='float16'),
            # parameter_575
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_576
            paddle.static.InputSpec(shape=[768, 768], dtype='float16'),
            # parameter_577
            paddle.static.InputSpec(shape=[768], dtype='float16'),
            # parameter_578
            paddle.static.InputSpec(shape=[256, 768, 1, 1], dtype='float16'),
            # parameter_582
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_579
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_581
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_580
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_583
            paddle.static.InputSpec(shape=[256, 32, 3, 3], dtype='float16'),
            # parameter_587
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_584
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_586
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_585
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_588
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float16'),
            # parameter_592
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_589
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_591
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_590
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_593
            paddle.static.InputSpec(shape=[2048, 1024, 1, 1], dtype='float16'),
            # parameter_594
            paddle.static.InputSpec(shape=[2048], dtype='float16'),
            # parameter_595
            paddle.static.InputSpec(shape=[1024, 2048, 1, 1], dtype='float16'),
            # parameter_596
            paddle.static.InputSpec(shape=[1024], dtype='float16'),
            # parameter_600
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_597
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_599
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_598
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_601
            paddle.static.InputSpec(shape=[1024, 1000], dtype='float16'),
            # parameter_602
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