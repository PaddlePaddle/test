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
    return [1657][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_2427_0_0(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_31, parameter_32, parameter_36, parameter_33, parameter_35, parameter_34, parameter_37, parameter_41, parameter_38, parameter_40, parameter_39, parameter_42, parameter_46, parameter_43, parameter_45, parameter_44, parameter_47, parameter_51, parameter_48, parameter_50, parameter_49, parameter_52, parameter_56, parameter_53, parameter_55, parameter_54, parameter_57, parameter_58, parameter_59, parameter_63, parameter_60, parameter_62, parameter_61, parameter_64, parameter_68, parameter_65, parameter_67, parameter_66, parameter_69, parameter_73, parameter_70, parameter_72, parameter_71, parameter_74, parameter_78, parameter_75, parameter_77, parameter_76, parameter_79, parameter_80, parameter_81, parameter_85, parameter_82, parameter_84, parameter_83, parameter_86, parameter_90, parameter_87, parameter_89, parameter_88, parameter_91, parameter_95, parameter_92, parameter_94, parameter_93, parameter_96, parameter_100, parameter_97, parameter_99, parameter_98, parameter_101, parameter_102, parameter_103, parameter_107, parameter_104, parameter_106, parameter_105, parameter_108, parameter_112, parameter_109, parameter_111, parameter_110, parameter_113, parameter_117, parameter_114, parameter_116, parameter_115, parameter_118, parameter_122, parameter_119, parameter_121, parameter_120, parameter_123, parameter_127, parameter_124, parameter_126, parameter_125, parameter_128, parameter_129, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_151, parameter_152, parameter_156, parameter_153, parameter_155, parameter_154, parameter_157, parameter_161, parameter_158, parameter_160, parameter_159, parameter_162, parameter_166, parameter_163, parameter_165, parameter_164, parameter_167, parameter_171, parameter_168, parameter_170, parameter_169, parameter_172, parameter_173, parameter_174, parameter_178, parameter_175, parameter_177, parameter_176, parameter_179, parameter_183, parameter_180, parameter_182, parameter_181, parameter_184, parameter_188, parameter_185, parameter_187, parameter_186, parameter_189, parameter_193, parameter_190, parameter_192, parameter_191, parameter_194, parameter_195, parameter_196, parameter_200, parameter_197, parameter_199, parameter_198, parameter_201, parameter_205, parameter_202, parameter_204, parameter_203, parameter_206, parameter_210, parameter_207, parameter_209, parameter_208, parameter_211, parameter_215, parameter_212, parameter_214, parameter_213, parameter_216, parameter_220, parameter_217, parameter_219, parameter_218, parameter_221, parameter_222, parameter_223, parameter_227, parameter_224, parameter_226, parameter_225, parameter_228, parameter_232, parameter_229, parameter_231, parameter_230, parameter_233, parameter_237, parameter_234, parameter_236, parameter_235, parameter_238, parameter_242, parameter_239, parameter_241, parameter_240, parameter_243, parameter_244, parameter_245, parameter_249, parameter_246, parameter_248, parameter_247, parameter_250, parameter_254, parameter_251, parameter_253, parameter_252, parameter_255, parameter_259, parameter_256, parameter_258, parameter_257, parameter_260, parameter_264, parameter_261, parameter_263, parameter_262, parameter_265, parameter_266, parameter_267, parameter_271, parameter_268, parameter_270, parameter_269, parameter_272, parameter_276, parameter_273, parameter_275, parameter_274, parameter_277, parameter_281, parameter_278, parameter_280, parameter_279, parameter_282, parameter_286, parameter_283, parameter_285, parameter_284, parameter_287, parameter_288, parameter_289, parameter_293, parameter_290, parameter_292, parameter_291, parameter_294, parameter_298, parameter_295, parameter_297, parameter_296, parameter_299, parameter_303, parameter_300, parameter_302, parameter_301, parameter_304, parameter_308, parameter_305, parameter_307, parameter_306, parameter_309, parameter_310, parameter_311, parameter_315, parameter_312, parameter_314, parameter_313, parameter_316, parameter_320, parameter_317, parameter_319, parameter_318, parameter_321, parameter_325, parameter_322, parameter_324, parameter_323, parameter_326, parameter_330, parameter_327, parameter_329, parameter_328, parameter_331, parameter_332, parameter_333, parameter_337, parameter_334, parameter_336, parameter_335, parameter_338, parameter_342, parameter_339, parameter_341, parameter_340, parameter_343, parameter_347, parameter_344, parameter_346, parameter_345, parameter_348, parameter_352, parameter_349, parameter_351, parameter_350, parameter_353, parameter_354, parameter_355, parameter_359, parameter_356, parameter_358, parameter_357, parameter_360, parameter_364, parameter_361, parameter_363, parameter_362, parameter_365, parameter_369, parameter_366, parameter_368, parameter_367, parameter_370, parameter_374, parameter_371, parameter_373, parameter_372, parameter_375, parameter_376, parameter_377, parameter_381, parameter_378, parameter_380, parameter_379, parameter_382, parameter_386, parameter_383, parameter_385, parameter_384, parameter_387, parameter_391, parameter_388, parameter_390, parameter_389, parameter_392, parameter_396, parameter_393, parameter_395, parameter_394, parameter_397, parameter_398, parameter_399, parameter_403, parameter_400, parameter_402, parameter_401, parameter_404, parameter_408, parameter_405, parameter_407, parameter_406, parameter_409, parameter_413, parameter_410, parameter_412, parameter_411, parameter_414, parameter_418, parameter_415, parameter_417, parameter_416, parameter_419, parameter_420, parameter_421, parameter_425, parameter_422, parameter_424, parameter_423, parameter_426, parameter_430, parameter_427, parameter_429, parameter_428, parameter_431, parameter_435, parameter_432, parameter_434, parameter_433, parameter_436, parameter_440, parameter_437, parameter_439, parameter_438, parameter_441, parameter_442, parameter_443, parameter_447, parameter_444, parameter_446, parameter_445, parameter_448, parameter_452, parameter_449, parameter_451, parameter_450, parameter_453, parameter_457, parameter_454, parameter_456, parameter_455, parameter_458, parameter_462, parameter_459, parameter_461, parameter_460, parameter_463, parameter_464, parameter_465, parameter_469, parameter_466, parameter_468, parameter_467, parameter_470, parameter_474, parameter_471, parameter_473, parameter_472, parameter_475, parameter_479, parameter_476, parameter_478, parameter_477, parameter_480, parameter_484, parameter_481, parameter_483, parameter_482, parameter_485, parameter_486, parameter_487, parameter_491, parameter_488, parameter_490, parameter_489, parameter_492, parameter_496, parameter_493, parameter_495, parameter_494, parameter_497, parameter_501, parameter_498, parameter_500, parameter_499, parameter_502, parameter_506, parameter_503, parameter_505, parameter_504, parameter_507, parameter_508, parameter_509, parameter_513, parameter_510, parameter_512, parameter_511, parameter_514, parameter_518, parameter_515, parameter_517, parameter_516, parameter_519, parameter_523, parameter_520, parameter_522, parameter_521, parameter_524, parameter_528, parameter_525, parameter_527, parameter_526, parameter_529, parameter_530, parameter_531, parameter_535, parameter_532, parameter_534, parameter_533, parameter_536, parameter_540, parameter_537, parameter_539, parameter_538, parameter_541, parameter_545, parameter_542, parameter_544, parameter_543, parameter_546, parameter_550, parameter_547, parameter_549, parameter_548, parameter_551, parameter_552, parameter_553, parameter_557, parameter_554, parameter_556, parameter_555, parameter_558, parameter_562, parameter_559, parameter_561, parameter_560, parameter_563, parameter_567, parameter_564, parameter_566, parameter_565, parameter_568, parameter_572, parameter_569, parameter_571, parameter_570, parameter_573, parameter_574, parameter_575, parameter_579, parameter_576, parameter_578, parameter_577, parameter_580, parameter_584, parameter_581, parameter_583, parameter_582, parameter_585, parameter_589, parameter_586, parameter_588, parameter_587, parameter_590, parameter_594, parameter_591, parameter_593, parameter_592, parameter_595, parameter_596, parameter_597, parameter_601, parameter_598, parameter_600, parameter_599, parameter_602, parameter_606, parameter_603, parameter_605, parameter_604, parameter_607, parameter_611, parameter_608, parameter_610, parameter_609, parameter_612, parameter_616, parameter_613, parameter_615, parameter_614, parameter_617, parameter_618, parameter_619, parameter_623, parameter_620, parameter_622, parameter_621, parameter_624, parameter_628, parameter_625, parameter_627, parameter_626, parameter_629, parameter_633, parameter_630, parameter_632, parameter_631, parameter_634, parameter_638, parameter_635, parameter_637, parameter_636, parameter_639, parameter_640, parameter_641, parameter_645, parameter_642, parameter_644, parameter_643, parameter_646, parameter_650, parameter_647, parameter_649, parameter_648, parameter_651, parameter_655, parameter_652, parameter_654, parameter_653, parameter_656, parameter_660, parameter_657, parameter_659, parameter_658, parameter_661, parameter_662, parameter_663, parameter_667, parameter_664, parameter_666, parameter_665, parameter_668, parameter_672, parameter_669, parameter_671, parameter_670, parameter_673, parameter_677, parameter_674, parameter_676, parameter_675, parameter_678, parameter_682, parameter_679, parameter_681, parameter_680, parameter_683, parameter_684, parameter_685, parameter_689, parameter_686, parameter_688, parameter_687, parameter_690, parameter_694, parameter_691, parameter_693, parameter_692, parameter_695, parameter_699, parameter_696, parameter_698, parameter_697, parameter_700, parameter_704, parameter_701, parameter_703, parameter_702, parameter_705, parameter_706, parameter_707, parameter_711, parameter_708, parameter_710, parameter_709, parameter_712, parameter_716, parameter_713, parameter_715, parameter_714, parameter_717, parameter_721, parameter_718, parameter_720, parameter_719, parameter_722, parameter_726, parameter_723, parameter_725, parameter_724, parameter_727, parameter_731, parameter_728, parameter_730, parameter_729, parameter_732, parameter_733, parameter_734, parameter_738, parameter_735, parameter_737, parameter_736, parameter_739, parameter_743, parameter_740, parameter_742, parameter_741, parameter_744, parameter_748, parameter_745, parameter_747, parameter_746, parameter_749, parameter_753, parameter_750, parameter_752, parameter_751, parameter_754, parameter_755, parameter_756, parameter_760, parameter_757, parameter_759, parameter_758, parameter_761, parameter_762, feed_0):

        # pd_op.conv2d: (-1x64x128x128xf32) <- (-1x3x256x256xf32, 64x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(feed_0, parameter_0, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x128x128xf32, 64xf32, 64xf32, 64xf32, 64xf32, None) <- (-1x64x128x128xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_0, parameter_1, parameter_2, parameter_3, parameter_4, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x128x128xf32) <- (-1x64x128x128xf32)
        relu__0 = paddle._C_ops.relu_(batch_norm__0)

        # pd_op.conv2d: (-1x64x128x128xf32) <- (-1x64x128x128xf32, 64x64x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(relu__0, parameter_5, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x128x128xf32, 64xf32, 64xf32, 64xf32, 64xf32, None) <- (-1x64x128x128xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_1, parameter_6, parameter_7, parameter_8, parameter_9, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x128x128xf32) <- (-1x64x128x128xf32)
        relu__1 = paddle._C_ops.relu_(batch_norm__6)

        # pd_op.conv2d: (-1x128x128x128xf32) <- (-1x64x128x128xf32, 128x64x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(relu__1, parameter_10, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x128x128xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x128x128xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_2, parameter_11, parameter_12, parameter_13, parameter_14, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x128x128xf32) <- (-1x128x128x128xf32)
        relu__2 = paddle._C_ops.relu_(batch_norm__12)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [3, 3]

        # pd_op.pool2d: (-1x128x64x64xf32) <- (-1x128x128x128xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(relu__2, full_int_array_0, [2, 2], [1, 1], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x64x64x64xf32) <- (-1x128x64x64xf32, 64x128x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(pool2d_0, parameter_15, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x64x64xf32, 64xf32, 64xf32, 64xf32, 64xf32, None) <- (-1x64x64x64xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_3, parameter_16, parameter_17, parameter_18, parameter_19, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x64x64xf32) <- (-1x64x64x64xf32)
        relu__3 = paddle._C_ops.relu_(batch_norm__18)

        # pd_op.conv2d: (-1x128x64x64xf32) <- (-1x64x64x64xf32, 128x32x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(relu__3, parameter_20, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_4, parameter_21, parameter_22, parameter_23, parameter_24, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x64x64xf32) <- (-1x128x64x64xf32)
        relu__4 = paddle._C_ops.relu_(batch_norm__24)

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x64x64x64xf32, -1x64x64x64xf32]) <- (-1x128x64x64xf32, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(relu__4, 2, full_0)

        # builtin.slice: (-1x64x64x64xf32) <- ([-1x64x64x64xf32, -1x64x64x64xf32])
        slice_0 = split_with_num_0[0]

        # builtin.slice: (-1x64x64x64xf32) <- ([-1x64x64x64xf32, -1x64x64x64xf32])
        slice_1 = split_with_num_0[1]

        # builtin.combine: ([-1x64x64x64xf32, -1x64x64x64xf32]) <- (-1x64x64x64xf32, -1x64x64x64xf32)
        combine_0 = [slice_0, slice_1]

        # pd_op.add_n: (-1x64x64x64xf32) <- ([-1x64x64x64xf32, -1x64x64x64xf32])
        add_n_0 = paddle._C_ops.add_n(combine_0)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [1, 1]

        # pd_op.pool2d: (-1x64x1x1xf32) <- (-1x64x64x64xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(add_n_0, full_int_array_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x32x1x1xf32) <- (-1x64x1x1xf32, 32x64x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(pool2d_1, parameter_25, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x1x1xf32, 32xf32, 32xf32, 32xf32, 32xf32, None) <- (-1x32x1x1xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_5, parameter_26, parameter_27, parameter_28, parameter_29, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x32x1x1xf32) <- (-1x32x1x1xf32)
        relu__5 = paddle._C_ops.relu_(batch_norm__30)

        # pd_op.conv2d: (-1x128x1x1xf32) <- (-1x32x1x1xf32, 128x32x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(relu__5, parameter_30, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_2 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_31, full_int_array_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x1x1xf32) <- (-1x128x1x1xf32, 1x128x1x1xf32)
        add__0 = paddle._C_ops.add_(conv2d_6, reshape_0)

        # pd_op.shape: (4xi32) <- (-1x128x1x1xf32)
        shape_0 = paddle._C_ops.shape(add__0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(shape_0, [0], full_int_array_3, full_int_array_4, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_1 = [slice_2, full_1, full_2, full_3]

        # pd_op.reshape_: (-1x1x2x64xf32, 0x-1x128x1x1xf32) <- (-1x128x1x1xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__0, [x.reshape([]) for x in combine_1]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x64xf32) <- (-1x1x2x64xf32)
        transpose_0 = paddle._C_ops.transpose(reshape__0, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x64xf32) <- (-1x2x1x64xf32)
        softmax__0 = paddle._C_ops.softmax_(transpose_0, 1)

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_2 = [slice_2, full_4, full_5, full_6]

        # pd_op.reshape_: (-1x128x1x1xf32, 0x-1x2x1x64xf32) <- (-1x2x1x64xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__0, [x.reshape([]) for x in combine_2]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x64x1x1xf32, -1x64x1x1xf32]) <- (-1x128x1x1xf32, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(reshape__2, 2, full_7)

        # builtin.slice: (-1x64x1x1xf32) <- ([-1x64x1x1xf32, -1x64x1x1xf32])
        slice_3 = split_with_num_1[0]

        # pd_op.multiply_: (-1x64x64x64xf32) <- (-1x64x64x64xf32, -1x64x1x1xf32)
        multiply__0 = paddle._C_ops.multiply_(slice_0, slice_3)

        # builtin.slice: (-1x64x1x1xf32) <- ([-1x64x1x1xf32, -1x64x1x1xf32])
        slice_4 = split_with_num_1[1]

        # pd_op.multiply_: (-1x64x64x64xf32) <- (-1x64x64x64xf32, -1x64x1x1xf32)
        multiply__1 = paddle._C_ops.multiply_(slice_1, slice_4)

        # builtin.combine: ([-1x64x64x64xf32, -1x64x64x64xf32]) <- (-1x64x64x64xf32, -1x64x64x64xf32)
        combine_3 = [multiply__0, multiply__1]

        # pd_op.add_n: (-1x64x64x64xf32) <- ([-1x64x64x64xf32, -1x64x64x64xf32])
        add_n_1 = paddle._C_ops.add_n(combine_3)

        # pd_op.conv2d: (-1x256x64x64xf32) <- (-1x64x64x64xf32, 256x64x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(add_n_1, parameter_32, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x64x64xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x64x64xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_7, parameter_33, parameter_34, parameter_35, parameter_36, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [1, 1]

        # pd_op.pool2d: (-1x128x64x64xf32) <- (-1x128x64x64xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(pool2d_0, full_int_array_5, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x256x64x64xf32) <- (-1x128x64x64xf32, 256x128x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(pool2d_2, parameter_37, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x64x64xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x64x64xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_8, parameter_38, parameter_39, parameter_40, parameter_41, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x256x64x64xf32) <- (-1x256x64x64xf32, -1x256x64x64xf32)
        add__1 = paddle._C_ops.add_(batch_norm__42, batch_norm__36)

        # pd_op.relu_: (-1x256x64x64xf32) <- (-1x256x64x64xf32)
        relu__6 = paddle._C_ops.relu_(add__1)

        # pd_op.conv2d: (-1x64x64x64xf32) <- (-1x256x64x64xf32, 64x256x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(relu__6, parameter_42, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x64x64xf32, 64xf32, 64xf32, 64xf32, 64xf32, None) <- (-1x64x64x64xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_9, parameter_43, parameter_44, parameter_45, parameter_46, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x64x64xf32) <- (-1x64x64x64xf32)
        relu__7 = paddle._C_ops.relu_(batch_norm__48)

        # pd_op.conv2d: (-1x128x64x64xf32) <- (-1x64x64x64xf32, 128x32x3x3xf32)
        conv2d_10 = paddle._C_ops.conv2d(relu__7, parameter_47, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_10, parameter_48, parameter_49, parameter_50, parameter_51, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x64x64xf32) <- (-1x128x64x64xf32)
        relu__8 = paddle._C_ops.relu_(batch_norm__54)

        # pd_op.full: (1xi32) <- ()
        full_8 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x64x64x64xf32, -1x64x64x64xf32]) <- (-1x128x64x64xf32, 1xi32)
        split_with_num_2 = paddle._C_ops.split_with_num(relu__8, 2, full_8)

        # builtin.slice: (-1x64x64x64xf32) <- ([-1x64x64x64xf32, -1x64x64x64xf32])
        slice_5 = split_with_num_2[0]

        # builtin.slice: (-1x64x64x64xf32) <- ([-1x64x64x64xf32, -1x64x64x64xf32])
        slice_6 = split_with_num_2[1]

        # builtin.combine: ([-1x64x64x64xf32, -1x64x64x64xf32]) <- (-1x64x64x64xf32, -1x64x64x64xf32)
        combine_4 = [slice_5, slice_6]

        # pd_op.add_n: (-1x64x64x64xf32) <- ([-1x64x64x64xf32, -1x64x64x64xf32])
        add_n_2 = paddle._C_ops.add_n(combine_4)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_6 = [1, 1]

        # pd_op.pool2d: (-1x64x1x1xf32) <- (-1x64x64x64xf32, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(add_n_2, full_int_array_6, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x32x1x1xf32) <- (-1x64x1x1xf32, 32x64x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(pool2d_3, parameter_52, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x1x1xf32, 32xf32, 32xf32, 32xf32, 32xf32, None) <- (-1x32x1x1xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_11, parameter_53, parameter_54, parameter_55, parameter_56, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x32x1x1xf32) <- (-1x32x1x1xf32)
        relu__9 = paddle._C_ops.relu_(batch_norm__60)

        # pd_op.conv2d: (-1x128x1x1xf32) <- (-1x32x1x1xf32, 128x32x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(relu__9, parameter_57, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_7 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_58, full_int_array_7), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x1x1xf32) <- (-1x128x1x1xf32, 1x128x1x1xf32)
        add__2 = paddle._C_ops.add_(conv2d_12, reshape_2)

        # pd_op.shape: (4xi32) <- (-1x128x1x1xf32)
        shape_1 = paddle._C_ops.shape(add__2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(shape_1, [0], full_int_array_8, full_int_array_9, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_9 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_10 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_11 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_5 = [slice_7, full_9, full_10, full_11]

        # pd_op.reshape_: (-1x1x2x64xf32, 0x-1x128x1x1xf32) <- (-1x128x1x1xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__2, [x.reshape([]) for x in combine_5]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x64xf32) <- (-1x1x2x64xf32)
        transpose_1 = paddle._C_ops.transpose(reshape__4, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x64xf32) <- (-1x2x1x64xf32)
        softmax__1 = paddle._C_ops.softmax_(transpose_1, 1)

        # pd_op.full: (1xi32) <- ()
        full_12 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_13 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_14 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_6 = [slice_7, full_12, full_13, full_14]

        # pd_op.reshape_: (-1x128x1x1xf32, 0x-1x2x1x64xf32) <- (-1x2x1x64xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__1, [x.reshape([]) for x in combine_6]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_15 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x64x1x1xf32, -1x64x1x1xf32]) <- (-1x128x1x1xf32, 1xi32)
        split_with_num_3 = paddle._C_ops.split_with_num(reshape__6, 2, full_15)

        # builtin.slice: (-1x64x1x1xf32) <- ([-1x64x1x1xf32, -1x64x1x1xf32])
        slice_8 = split_with_num_3[0]

        # pd_op.multiply_: (-1x64x64x64xf32) <- (-1x64x64x64xf32, -1x64x1x1xf32)
        multiply__2 = paddle._C_ops.multiply_(slice_5, slice_8)

        # builtin.slice: (-1x64x1x1xf32) <- ([-1x64x1x1xf32, -1x64x1x1xf32])
        slice_9 = split_with_num_3[1]

        # pd_op.multiply_: (-1x64x64x64xf32) <- (-1x64x64x64xf32, -1x64x1x1xf32)
        multiply__3 = paddle._C_ops.multiply_(slice_6, slice_9)

        # builtin.combine: ([-1x64x64x64xf32, -1x64x64x64xf32]) <- (-1x64x64x64xf32, -1x64x64x64xf32)
        combine_7 = [multiply__2, multiply__3]

        # pd_op.add_n: (-1x64x64x64xf32) <- ([-1x64x64x64xf32, -1x64x64x64xf32])
        add_n_3 = paddle._C_ops.add_n(combine_7)

        # pd_op.conv2d: (-1x256x64x64xf32) <- (-1x64x64x64xf32, 256x64x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(add_n_3, parameter_59, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x64x64xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x64x64xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_13, parameter_60, parameter_61, parameter_62, parameter_63, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x256x64x64xf32) <- (-1x256x64x64xf32, -1x256x64x64xf32)
        add__3 = paddle._C_ops.add_(relu__6, batch_norm__66)

        # pd_op.relu_: (-1x256x64x64xf32) <- (-1x256x64x64xf32)
        relu__10 = paddle._C_ops.relu_(add__3)

        # pd_op.conv2d: (-1x64x64x64xf32) <- (-1x256x64x64xf32, 64x256x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(relu__10, parameter_64, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x64x64xf32, 64xf32, 64xf32, 64xf32, 64xf32, None) <- (-1x64x64x64xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_14, parameter_65, parameter_66, parameter_67, parameter_68, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x64x64xf32) <- (-1x64x64x64xf32)
        relu__11 = paddle._C_ops.relu_(batch_norm__72)

        # pd_op.conv2d: (-1x128x64x64xf32) <- (-1x64x64x64xf32, 128x32x3x3xf32)
        conv2d_15 = paddle._C_ops.conv2d(relu__11, parameter_69, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_15, parameter_70, parameter_71, parameter_72, parameter_73, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x64x64xf32) <- (-1x128x64x64xf32)
        relu__12 = paddle._C_ops.relu_(batch_norm__78)

        # pd_op.full: (1xi32) <- ()
        full_16 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x64x64x64xf32, -1x64x64x64xf32]) <- (-1x128x64x64xf32, 1xi32)
        split_with_num_4 = paddle._C_ops.split_with_num(relu__12, 2, full_16)

        # builtin.slice: (-1x64x64x64xf32) <- ([-1x64x64x64xf32, -1x64x64x64xf32])
        slice_10 = split_with_num_4[0]

        # builtin.slice: (-1x64x64x64xf32) <- ([-1x64x64x64xf32, -1x64x64x64xf32])
        slice_11 = split_with_num_4[1]

        # builtin.combine: ([-1x64x64x64xf32, -1x64x64x64xf32]) <- (-1x64x64x64xf32, -1x64x64x64xf32)
        combine_8 = [slice_10, slice_11]

        # pd_op.add_n: (-1x64x64x64xf32) <- ([-1x64x64x64xf32, -1x64x64x64xf32])
        add_n_4 = paddle._C_ops.add_n(combine_8)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_10 = [1, 1]

        # pd_op.pool2d: (-1x64x1x1xf32) <- (-1x64x64x64xf32, 2xi64)
        pool2d_4 = paddle._C_ops.pool2d(add_n_4, full_int_array_10, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x32x1x1xf32) <- (-1x64x1x1xf32, 32x64x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(pool2d_4, parameter_74, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x1x1xf32, 32xf32, 32xf32, 32xf32, 32xf32, None) <- (-1x32x1x1xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_16, parameter_75, parameter_76, parameter_77, parameter_78, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x32x1x1xf32) <- (-1x32x1x1xf32)
        relu__13 = paddle._C_ops.relu_(batch_norm__84)

        # pd_op.conv2d: (-1x128x1x1xf32) <- (-1x32x1x1xf32, 128x32x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(relu__13, parameter_79, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_11 = [1, 128, 1, 1]

        # pd_op.reshape: (1x128x1x1xf32, 0x128xf32) <- (128xf32, 4xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_80, full_int_array_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x128x1x1xf32) <- (-1x128x1x1xf32, 1x128x1x1xf32)
        add__4 = paddle._C_ops.add_(conv2d_17, reshape_4)

        # pd_op.shape: (4xi32) <- (-1x128x1x1xf32)
        shape_2 = paddle._C_ops.shape(add__4)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_12 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_13 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(shape_2, [0], full_int_array_12, full_int_array_13, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_17 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_18 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_19 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_9 = [slice_12, full_17, full_18, full_19]

        # pd_op.reshape_: (-1x1x2x64xf32, 0x-1x128x1x1xf32) <- (-1x128x1x1xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__4, [x.reshape([]) for x in combine_9]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x64xf32) <- (-1x1x2x64xf32)
        transpose_2 = paddle._C_ops.transpose(reshape__8, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x64xf32) <- (-1x2x1x64xf32)
        softmax__2 = paddle._C_ops.softmax_(transpose_2, 1)

        # pd_op.full: (1xi32) <- ()
        full_20 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_21 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_22 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_10 = [slice_12, full_20, full_21, full_22]

        # pd_op.reshape_: (-1x128x1x1xf32, 0x-1x2x1x64xf32) <- (-1x2x1x64xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__2, [x.reshape([]) for x in combine_10]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_23 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x64x1x1xf32, -1x64x1x1xf32]) <- (-1x128x1x1xf32, 1xi32)
        split_with_num_5 = paddle._C_ops.split_with_num(reshape__10, 2, full_23)

        # builtin.slice: (-1x64x1x1xf32) <- ([-1x64x1x1xf32, -1x64x1x1xf32])
        slice_13 = split_with_num_5[0]

        # pd_op.multiply_: (-1x64x64x64xf32) <- (-1x64x64x64xf32, -1x64x1x1xf32)
        multiply__4 = paddle._C_ops.multiply_(slice_10, slice_13)

        # builtin.slice: (-1x64x1x1xf32) <- ([-1x64x1x1xf32, -1x64x1x1xf32])
        slice_14 = split_with_num_5[1]

        # pd_op.multiply_: (-1x64x64x64xf32) <- (-1x64x64x64xf32, -1x64x1x1xf32)
        multiply__5 = paddle._C_ops.multiply_(slice_11, slice_14)

        # builtin.combine: ([-1x64x64x64xf32, -1x64x64x64xf32]) <- (-1x64x64x64xf32, -1x64x64x64xf32)
        combine_11 = [multiply__4, multiply__5]

        # pd_op.add_n: (-1x64x64x64xf32) <- ([-1x64x64x64xf32, -1x64x64x64xf32])
        add_n_5 = paddle._C_ops.add_n(combine_11)

        # pd_op.conv2d: (-1x256x64x64xf32) <- (-1x64x64x64xf32, 256x64x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(add_n_5, parameter_81, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x64x64xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x64x64xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_18, parameter_82, parameter_83, parameter_84, parameter_85, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x256x64x64xf32) <- (-1x256x64x64xf32, -1x256x64x64xf32)
        add__5 = paddle._C_ops.add_(relu__10, batch_norm__90)

        # pd_op.relu_: (-1x256x64x64xf32) <- (-1x256x64x64xf32)
        relu__14 = paddle._C_ops.relu_(add__5)

        # pd_op.conv2d: (-1x128x64x64xf32) <- (-1x256x64x64xf32, 128x256x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(relu__14, parameter_86, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x64x64xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_19, parameter_87, parameter_88, parameter_89, parameter_90, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x64x64xf32) <- (-1x128x64x64xf32)
        relu__15 = paddle._C_ops.relu_(batch_norm__96)

        # pd_op.conv2d: (-1x256x64x64xf32) <- (-1x128x64x64xf32, 256x64x3x3xf32)
        conv2d_20 = paddle._C_ops.conv2d(relu__15, parameter_91, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x256x64x64xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x64x64xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_20, parameter_92, parameter_93, parameter_94, parameter_95, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x64x64xf32) <- (-1x256x64x64xf32)
        relu__16 = paddle._C_ops.relu_(batch_norm__102)

        # pd_op.full: (1xi32) <- ()
        full_24 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x128x64x64xf32, -1x128x64x64xf32]) <- (-1x256x64x64xf32, 1xi32)
        split_with_num_6 = paddle._C_ops.split_with_num(relu__16, 2, full_24)

        # builtin.slice: (-1x128x64x64xf32) <- ([-1x128x64x64xf32, -1x128x64x64xf32])
        slice_15 = split_with_num_6[0]

        # builtin.slice: (-1x128x64x64xf32) <- ([-1x128x64x64xf32, -1x128x64x64xf32])
        slice_16 = split_with_num_6[1]

        # builtin.combine: ([-1x128x64x64xf32, -1x128x64x64xf32]) <- (-1x128x64x64xf32, -1x128x64x64xf32)
        combine_12 = [slice_15, slice_16]

        # pd_op.add_n: (-1x128x64x64xf32) <- ([-1x128x64x64xf32, -1x128x64x64xf32])
        add_n_6 = paddle._C_ops.add_n(combine_12)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_14 = [1, 1]

        # pd_op.pool2d: (-1x128x1x1xf32) <- (-1x128x64x64xf32, 2xi64)
        pool2d_5 = paddle._C_ops.pool2d(add_n_6, full_int_array_14, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x64x1x1xf32) <- (-1x128x1x1xf32, 64x128x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(pool2d_5, parameter_96, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x1x1xf32, 64xf32, 64xf32, 64xf32, 64xf32, None) <- (-1x64x1x1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_21, parameter_97, parameter_98, parameter_99, parameter_100, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x1x1xf32) <- (-1x64x1x1xf32)
        relu__17 = paddle._C_ops.relu_(batch_norm__108)

        # pd_op.conv2d: (-1x256x1x1xf32) <- (-1x64x1x1xf32, 256x64x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(relu__17, parameter_101, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_15 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_102, full_int_array_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x1x1xf32) <- (-1x256x1x1xf32, 1x256x1x1xf32)
        add__6 = paddle._C_ops.add_(conv2d_22, reshape_6)

        # pd_op.shape: (4xi32) <- (-1x256x1x1xf32)
        shape_3 = paddle._C_ops.shape(add__6)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_16 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_17 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(shape_3, [0], full_int_array_16, full_int_array_17, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_25 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_26 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_27 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_13 = [slice_17, full_25, full_26, full_27]

        # pd_op.reshape_: (-1x1x2x128xf32, 0x-1x256x1x1xf32) <- (-1x256x1x1xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__6, [x.reshape([]) for x in combine_13]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x128xf32) <- (-1x1x2x128xf32)
        transpose_3 = paddle._C_ops.transpose(reshape__12, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x128xf32) <- (-1x2x1x128xf32)
        softmax__3 = paddle._C_ops.softmax_(transpose_3, 1)

        # pd_op.full: (1xi32) <- ()
        full_28 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_29 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_30 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_14 = [slice_17, full_28, full_29, full_30]

        # pd_op.reshape_: (-1x256x1x1xf32, 0x-1x2x1x128xf32) <- (-1x2x1x128xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__3, [x.reshape([]) for x in combine_14]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_31 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x128x1x1xf32, -1x128x1x1xf32]) <- (-1x256x1x1xf32, 1xi32)
        split_with_num_7 = paddle._C_ops.split_with_num(reshape__14, 2, full_31)

        # builtin.slice: (-1x128x1x1xf32) <- ([-1x128x1x1xf32, -1x128x1x1xf32])
        slice_18 = split_with_num_7[0]

        # pd_op.multiply_: (-1x128x64x64xf32) <- (-1x128x64x64xf32, -1x128x1x1xf32)
        multiply__6 = paddle._C_ops.multiply_(slice_15, slice_18)

        # builtin.slice: (-1x128x1x1xf32) <- ([-1x128x1x1xf32, -1x128x1x1xf32])
        slice_19 = split_with_num_7[1]

        # pd_op.multiply_: (-1x128x64x64xf32) <- (-1x128x64x64xf32, -1x128x1x1xf32)
        multiply__7 = paddle._C_ops.multiply_(slice_16, slice_19)

        # builtin.combine: ([-1x128x64x64xf32, -1x128x64x64xf32]) <- (-1x128x64x64xf32, -1x128x64x64xf32)
        combine_15 = [multiply__6, multiply__7]

        # pd_op.add_n: (-1x128x64x64xf32) <- ([-1x128x64x64xf32, -1x128x64x64xf32])
        add_n_7 = paddle._C_ops.add_n(combine_15)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_18 = [3, 3]

        # pd_op.pool2d: (-1x128x32x32xf32) <- (-1x128x64x64xf32, 2xi64)
        pool2d_6 = paddle._C_ops.pool2d(add_n_7, full_int_array_18, [2, 2], [1, 1], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x512x32x32xf32) <- (-1x128x32x32xf32, 512x128x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(pool2d_6, parameter_103, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x32x32xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x32x32xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_23, parameter_104, parameter_105, parameter_106, parameter_107, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_19 = [2, 2]

        # pd_op.pool2d: (-1x256x32x32xf32) <- (-1x256x64x64xf32, 2xi64)
        pool2d_7 = paddle._C_ops.pool2d(relu__14, full_int_array_19, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x512x32x32xf32) <- (-1x256x32x32xf32, 512x256x1x1xf32)
        conv2d_24 = paddle._C_ops.conv2d(pool2d_7, parameter_108, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x32x32xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x32x32xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_24, parameter_109, parameter_110, parameter_111, parameter_112, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x32x32xf32) <- (-1x512x32x32xf32, -1x512x32x32xf32)
        add__7 = paddle._C_ops.add_(batch_norm__120, batch_norm__114)

        # pd_op.relu_: (-1x512x32x32xf32) <- (-1x512x32x32xf32)
        relu__18 = paddle._C_ops.relu_(add__7)

        # pd_op.conv2d: (-1x128x32x32xf32) <- (-1x512x32x32xf32, 128x512x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(relu__18, parameter_113, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x32x32xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x32x32xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_25, parameter_114, parameter_115, parameter_116, parameter_117, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x32x32xf32) <- (-1x128x32x32xf32)
        relu__19 = paddle._C_ops.relu_(batch_norm__126)

        # pd_op.conv2d: (-1x256x32x32xf32) <- (-1x128x32x32xf32, 256x64x3x3xf32)
        conv2d_26 = paddle._C_ops.conv2d(relu__19, parameter_118, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_26, parameter_119, parameter_120, parameter_121, parameter_122, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x32x32xf32) <- (-1x256x32x32xf32)
        relu__20 = paddle._C_ops.relu_(batch_norm__132)

        # pd_op.full: (1xi32) <- ()
        full_32 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x128x32x32xf32, -1x128x32x32xf32]) <- (-1x256x32x32xf32, 1xi32)
        split_with_num_8 = paddle._C_ops.split_with_num(relu__20, 2, full_32)

        # builtin.slice: (-1x128x32x32xf32) <- ([-1x128x32x32xf32, -1x128x32x32xf32])
        slice_20 = split_with_num_8[0]

        # builtin.slice: (-1x128x32x32xf32) <- ([-1x128x32x32xf32, -1x128x32x32xf32])
        slice_21 = split_with_num_8[1]

        # builtin.combine: ([-1x128x32x32xf32, -1x128x32x32xf32]) <- (-1x128x32x32xf32, -1x128x32x32xf32)
        combine_16 = [slice_20, slice_21]

        # pd_op.add_n: (-1x128x32x32xf32) <- ([-1x128x32x32xf32, -1x128x32x32xf32])
        add_n_8 = paddle._C_ops.add_n(combine_16)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_20 = [1, 1]

        # pd_op.pool2d: (-1x128x1x1xf32) <- (-1x128x32x32xf32, 2xi64)
        pool2d_8 = paddle._C_ops.pool2d(add_n_8, full_int_array_20, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x64x1x1xf32) <- (-1x128x1x1xf32, 64x128x1x1xf32)
        conv2d_27 = paddle._C_ops.conv2d(pool2d_8, parameter_123, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x1x1xf32, 64xf32, 64xf32, 64xf32, 64xf32, None) <- (-1x64x1x1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_27, parameter_124, parameter_125, parameter_126, parameter_127, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x1x1xf32) <- (-1x64x1x1xf32)
        relu__21 = paddle._C_ops.relu_(batch_norm__138)

        # pd_op.conv2d: (-1x256x1x1xf32) <- (-1x64x1x1xf32, 256x64x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(relu__21, parameter_128, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_21 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_129, full_int_array_21), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x1x1xf32) <- (-1x256x1x1xf32, 1x256x1x1xf32)
        add__8 = paddle._C_ops.add_(conv2d_28, reshape_8)

        # pd_op.shape: (4xi32) <- (-1x256x1x1xf32)
        shape_4 = paddle._C_ops.shape(add__8)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_22 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_23 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(shape_4, [0], full_int_array_22, full_int_array_23, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_33 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_34 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_35 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_17 = [slice_22, full_33, full_34, full_35]

        # pd_op.reshape_: (-1x1x2x128xf32, 0x-1x256x1x1xf32) <- (-1x256x1x1xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__16, reshape__17 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__8, [x.reshape([]) for x in combine_17]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x128xf32) <- (-1x1x2x128xf32)
        transpose_4 = paddle._C_ops.transpose(reshape__16, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x128xf32) <- (-1x2x1x128xf32)
        softmax__4 = paddle._C_ops.softmax_(transpose_4, 1)

        # pd_op.full: (1xi32) <- ()
        full_36 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_37 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_38 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_18 = [slice_22, full_36, full_37, full_38]

        # pd_op.reshape_: (-1x256x1x1xf32, 0x-1x2x1x128xf32) <- (-1x2x1x128xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__18, reshape__19 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__4, [x.reshape([]) for x in combine_18]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_39 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x128x1x1xf32, -1x128x1x1xf32]) <- (-1x256x1x1xf32, 1xi32)
        split_with_num_9 = paddle._C_ops.split_with_num(reshape__18, 2, full_39)

        # builtin.slice: (-1x128x1x1xf32) <- ([-1x128x1x1xf32, -1x128x1x1xf32])
        slice_23 = split_with_num_9[0]

        # pd_op.multiply_: (-1x128x32x32xf32) <- (-1x128x32x32xf32, -1x128x1x1xf32)
        multiply__8 = paddle._C_ops.multiply_(slice_20, slice_23)

        # builtin.slice: (-1x128x1x1xf32) <- ([-1x128x1x1xf32, -1x128x1x1xf32])
        slice_24 = split_with_num_9[1]

        # pd_op.multiply_: (-1x128x32x32xf32) <- (-1x128x32x32xf32, -1x128x1x1xf32)
        multiply__9 = paddle._C_ops.multiply_(slice_21, slice_24)

        # builtin.combine: ([-1x128x32x32xf32, -1x128x32x32xf32]) <- (-1x128x32x32xf32, -1x128x32x32xf32)
        combine_19 = [multiply__8, multiply__9]

        # pd_op.add_n: (-1x128x32x32xf32) <- ([-1x128x32x32xf32, -1x128x32x32xf32])
        add_n_9 = paddle._C_ops.add_n(combine_19)

        # pd_op.conv2d: (-1x512x32x32xf32) <- (-1x128x32x32xf32, 512x128x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(add_n_9, parameter_130, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x32x32xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x32x32xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_29, parameter_131, parameter_132, parameter_133, parameter_134, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x32x32xf32) <- (-1x512x32x32xf32, -1x512x32x32xf32)
        add__9 = paddle._C_ops.add_(relu__18, batch_norm__144)

        # pd_op.relu_: (-1x512x32x32xf32) <- (-1x512x32x32xf32)
        relu__22 = paddle._C_ops.relu_(add__9)

        # pd_op.conv2d: (-1x128x32x32xf32) <- (-1x512x32x32xf32, 128x512x1x1xf32)
        conv2d_30 = paddle._C_ops.conv2d(relu__22, parameter_135, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x32x32xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x32x32xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_30, parameter_136, parameter_137, parameter_138, parameter_139, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x32x32xf32) <- (-1x128x32x32xf32)
        relu__23 = paddle._C_ops.relu_(batch_norm__150)

        # pd_op.conv2d: (-1x256x32x32xf32) <- (-1x128x32x32xf32, 256x64x3x3xf32)
        conv2d_31 = paddle._C_ops.conv2d(relu__23, parameter_140, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_31, parameter_141, parameter_142, parameter_143, parameter_144, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x32x32xf32) <- (-1x256x32x32xf32)
        relu__24 = paddle._C_ops.relu_(batch_norm__156)

        # pd_op.full: (1xi32) <- ()
        full_40 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x128x32x32xf32, -1x128x32x32xf32]) <- (-1x256x32x32xf32, 1xi32)
        split_with_num_10 = paddle._C_ops.split_with_num(relu__24, 2, full_40)

        # builtin.slice: (-1x128x32x32xf32) <- ([-1x128x32x32xf32, -1x128x32x32xf32])
        slice_25 = split_with_num_10[0]

        # builtin.slice: (-1x128x32x32xf32) <- ([-1x128x32x32xf32, -1x128x32x32xf32])
        slice_26 = split_with_num_10[1]

        # builtin.combine: ([-1x128x32x32xf32, -1x128x32x32xf32]) <- (-1x128x32x32xf32, -1x128x32x32xf32)
        combine_20 = [slice_25, slice_26]

        # pd_op.add_n: (-1x128x32x32xf32) <- ([-1x128x32x32xf32, -1x128x32x32xf32])
        add_n_10 = paddle._C_ops.add_n(combine_20)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_24 = [1, 1]

        # pd_op.pool2d: (-1x128x1x1xf32) <- (-1x128x32x32xf32, 2xi64)
        pool2d_9 = paddle._C_ops.pool2d(add_n_10, full_int_array_24, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x64x1x1xf32) <- (-1x128x1x1xf32, 64x128x1x1xf32)
        conv2d_32 = paddle._C_ops.conv2d(pool2d_9, parameter_145, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x1x1xf32, 64xf32, 64xf32, 64xf32, 64xf32, None) <- (-1x64x1x1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_32, parameter_146, parameter_147, parameter_148, parameter_149, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x1x1xf32) <- (-1x64x1x1xf32)
        relu__25 = paddle._C_ops.relu_(batch_norm__162)

        # pd_op.conv2d: (-1x256x1x1xf32) <- (-1x64x1x1xf32, 256x64x1x1xf32)
        conv2d_33 = paddle._C_ops.conv2d(relu__25, parameter_150, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_25 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_151, full_int_array_25), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x1x1xf32) <- (-1x256x1x1xf32, 1x256x1x1xf32)
        add__10 = paddle._C_ops.add_(conv2d_33, reshape_10)

        # pd_op.shape: (4xi32) <- (-1x256x1x1xf32)
        shape_5 = paddle._C_ops.shape(add__10)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_26 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_27 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(shape_5, [0], full_int_array_26, full_int_array_27, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_41 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_42 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_43 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_21 = [slice_27, full_41, full_42, full_43]

        # pd_op.reshape_: (-1x1x2x128xf32, 0x-1x256x1x1xf32) <- (-1x256x1x1xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__20, reshape__21 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__10, [x.reshape([]) for x in combine_21]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x128xf32) <- (-1x1x2x128xf32)
        transpose_5 = paddle._C_ops.transpose(reshape__20, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x128xf32) <- (-1x2x1x128xf32)
        softmax__5 = paddle._C_ops.softmax_(transpose_5, 1)

        # pd_op.full: (1xi32) <- ()
        full_44 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_45 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_46 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_22 = [slice_27, full_44, full_45, full_46]

        # pd_op.reshape_: (-1x256x1x1xf32, 0x-1x2x1x128xf32) <- (-1x2x1x128xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__22, reshape__23 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__5, [x.reshape([]) for x in combine_22]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_47 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x128x1x1xf32, -1x128x1x1xf32]) <- (-1x256x1x1xf32, 1xi32)
        split_with_num_11 = paddle._C_ops.split_with_num(reshape__22, 2, full_47)

        # builtin.slice: (-1x128x1x1xf32) <- ([-1x128x1x1xf32, -1x128x1x1xf32])
        slice_28 = split_with_num_11[0]

        # pd_op.multiply_: (-1x128x32x32xf32) <- (-1x128x32x32xf32, -1x128x1x1xf32)
        multiply__10 = paddle._C_ops.multiply_(slice_25, slice_28)

        # builtin.slice: (-1x128x1x1xf32) <- ([-1x128x1x1xf32, -1x128x1x1xf32])
        slice_29 = split_with_num_11[1]

        # pd_op.multiply_: (-1x128x32x32xf32) <- (-1x128x32x32xf32, -1x128x1x1xf32)
        multiply__11 = paddle._C_ops.multiply_(slice_26, slice_29)

        # builtin.combine: ([-1x128x32x32xf32, -1x128x32x32xf32]) <- (-1x128x32x32xf32, -1x128x32x32xf32)
        combine_23 = [multiply__10, multiply__11]

        # pd_op.add_n: (-1x128x32x32xf32) <- ([-1x128x32x32xf32, -1x128x32x32xf32])
        add_n_11 = paddle._C_ops.add_n(combine_23)

        # pd_op.conv2d: (-1x512x32x32xf32) <- (-1x128x32x32xf32, 512x128x1x1xf32)
        conv2d_34 = paddle._C_ops.conv2d(add_n_11, parameter_152, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x32x32xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x32x32xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_34, parameter_153, parameter_154, parameter_155, parameter_156, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x32x32xf32) <- (-1x512x32x32xf32, -1x512x32x32xf32)
        add__11 = paddle._C_ops.add_(relu__22, batch_norm__168)

        # pd_op.relu_: (-1x512x32x32xf32) <- (-1x512x32x32xf32)
        relu__26 = paddle._C_ops.relu_(add__11)

        # pd_op.conv2d: (-1x128x32x32xf32) <- (-1x512x32x32xf32, 128x512x1x1xf32)
        conv2d_35 = paddle._C_ops.conv2d(relu__26, parameter_157, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x32x32xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x32x32xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_35, parameter_158, parameter_159, parameter_160, parameter_161, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x32x32xf32) <- (-1x128x32x32xf32)
        relu__27 = paddle._C_ops.relu_(batch_norm__174)

        # pd_op.conv2d: (-1x256x32x32xf32) <- (-1x128x32x32xf32, 256x64x3x3xf32)
        conv2d_36 = paddle._C_ops.conv2d(relu__27, parameter_162, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__180, batch_norm__181, batch_norm__182, batch_norm__183, batch_norm__184, batch_norm__185 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_36, parameter_163, parameter_164, parameter_165, parameter_166, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x32x32xf32) <- (-1x256x32x32xf32)
        relu__28 = paddle._C_ops.relu_(batch_norm__180)

        # pd_op.full: (1xi32) <- ()
        full_48 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x128x32x32xf32, -1x128x32x32xf32]) <- (-1x256x32x32xf32, 1xi32)
        split_with_num_12 = paddle._C_ops.split_with_num(relu__28, 2, full_48)

        # builtin.slice: (-1x128x32x32xf32) <- ([-1x128x32x32xf32, -1x128x32x32xf32])
        slice_30 = split_with_num_12[0]

        # builtin.slice: (-1x128x32x32xf32) <- ([-1x128x32x32xf32, -1x128x32x32xf32])
        slice_31 = split_with_num_12[1]

        # builtin.combine: ([-1x128x32x32xf32, -1x128x32x32xf32]) <- (-1x128x32x32xf32, -1x128x32x32xf32)
        combine_24 = [slice_30, slice_31]

        # pd_op.add_n: (-1x128x32x32xf32) <- ([-1x128x32x32xf32, -1x128x32x32xf32])
        add_n_12 = paddle._C_ops.add_n(combine_24)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_28 = [1, 1]

        # pd_op.pool2d: (-1x128x1x1xf32) <- (-1x128x32x32xf32, 2xi64)
        pool2d_10 = paddle._C_ops.pool2d(add_n_12, full_int_array_28, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x64x1x1xf32) <- (-1x128x1x1xf32, 64x128x1x1xf32)
        conv2d_37 = paddle._C_ops.conv2d(pool2d_10, parameter_167, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x1x1xf32, 64xf32, 64xf32, 64xf32, 64xf32, None) <- (-1x64x1x1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__186, batch_norm__187, batch_norm__188, batch_norm__189, batch_norm__190, batch_norm__191 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_37, parameter_168, parameter_169, parameter_170, parameter_171, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x1x1xf32) <- (-1x64x1x1xf32)
        relu__29 = paddle._C_ops.relu_(batch_norm__186)

        # pd_op.conv2d: (-1x256x1x1xf32) <- (-1x64x1x1xf32, 256x64x1x1xf32)
        conv2d_38 = paddle._C_ops.conv2d(relu__29, parameter_172, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_29 = [1, 256, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xf32) <- (256xf32, 4xi64)
        reshape_12, reshape_13 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_173, full_int_array_29), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x256x1x1xf32) <- (-1x256x1x1xf32, 1x256x1x1xf32)
        add__12 = paddle._C_ops.add_(conv2d_38, reshape_12)

        # pd_op.shape: (4xi32) <- (-1x256x1x1xf32)
        shape_6 = paddle._C_ops.shape(add__12)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_30 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_31 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(shape_6, [0], full_int_array_30, full_int_array_31, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_49 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_50 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_51 = paddle._C_ops.full([1], float('128'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_25 = [slice_32, full_49, full_50, full_51]

        # pd_op.reshape_: (-1x1x2x128xf32, 0x-1x256x1x1xf32) <- (-1x256x1x1xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__24, reshape__25 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__12, [x.reshape([]) for x in combine_25]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x128xf32) <- (-1x1x2x128xf32)
        transpose_6 = paddle._C_ops.transpose(reshape__24, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x128xf32) <- (-1x2x1x128xf32)
        softmax__6 = paddle._C_ops.softmax_(transpose_6, 1)

        # pd_op.full: (1xi32) <- ()
        full_52 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_53 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_54 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_26 = [slice_32, full_52, full_53, full_54]

        # pd_op.reshape_: (-1x256x1x1xf32, 0x-1x2x1x128xf32) <- (-1x2x1x128xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__26, reshape__27 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__6, [x.reshape([]) for x in combine_26]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_55 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x128x1x1xf32, -1x128x1x1xf32]) <- (-1x256x1x1xf32, 1xi32)
        split_with_num_13 = paddle._C_ops.split_with_num(reshape__26, 2, full_55)

        # builtin.slice: (-1x128x1x1xf32) <- ([-1x128x1x1xf32, -1x128x1x1xf32])
        slice_33 = split_with_num_13[0]

        # pd_op.multiply_: (-1x128x32x32xf32) <- (-1x128x32x32xf32, -1x128x1x1xf32)
        multiply__12 = paddle._C_ops.multiply_(slice_30, slice_33)

        # builtin.slice: (-1x128x1x1xf32) <- ([-1x128x1x1xf32, -1x128x1x1xf32])
        slice_34 = split_with_num_13[1]

        # pd_op.multiply_: (-1x128x32x32xf32) <- (-1x128x32x32xf32, -1x128x1x1xf32)
        multiply__13 = paddle._C_ops.multiply_(slice_31, slice_34)

        # builtin.combine: ([-1x128x32x32xf32, -1x128x32x32xf32]) <- (-1x128x32x32xf32, -1x128x32x32xf32)
        combine_27 = [multiply__12, multiply__13]

        # pd_op.add_n: (-1x128x32x32xf32) <- ([-1x128x32x32xf32, -1x128x32x32xf32])
        add_n_13 = paddle._C_ops.add_n(combine_27)

        # pd_op.conv2d: (-1x512x32x32xf32) <- (-1x128x32x32xf32, 512x128x1x1xf32)
        conv2d_39 = paddle._C_ops.conv2d(add_n_13, parameter_174, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x32x32xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x32x32xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__192, batch_norm__193, batch_norm__194, batch_norm__195, batch_norm__196, batch_norm__197 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_39, parameter_175, parameter_176, parameter_177, parameter_178, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x32x32xf32) <- (-1x512x32x32xf32, -1x512x32x32xf32)
        add__13 = paddle._C_ops.add_(relu__26, batch_norm__192)

        # pd_op.relu_: (-1x512x32x32xf32) <- (-1x512x32x32xf32)
        relu__30 = paddle._C_ops.relu_(add__13)

        # pd_op.conv2d: (-1x256x32x32xf32) <- (-1x512x32x32xf32, 256x512x1x1xf32)
        conv2d_40 = paddle._C_ops.conv2d(relu__30, parameter_179, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x32x32xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__198, batch_norm__199, batch_norm__200, batch_norm__201, batch_norm__202, batch_norm__203 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_40, parameter_180, parameter_181, parameter_182, parameter_183, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x32x32xf32) <- (-1x256x32x32xf32)
        relu__31 = paddle._C_ops.relu_(batch_norm__198)

        # pd_op.conv2d: (-1x512x32x32xf32) <- (-1x256x32x32xf32, 512x128x3x3xf32)
        conv2d_41 = paddle._C_ops.conv2d(relu__31, parameter_184, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x512x32x32xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x32x32xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__204, batch_norm__205, batch_norm__206, batch_norm__207, batch_norm__208, batch_norm__209 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_41, parameter_185, parameter_186, parameter_187, parameter_188, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x32x32xf32) <- (-1x512x32x32xf32)
        relu__32 = paddle._C_ops.relu_(batch_norm__204)

        # pd_op.full: (1xi32) <- ()
        full_56 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x32x32xf32, -1x256x32x32xf32]) <- (-1x512x32x32xf32, 1xi32)
        split_with_num_14 = paddle._C_ops.split_with_num(relu__32, 2, full_56)

        # builtin.slice: (-1x256x32x32xf32) <- ([-1x256x32x32xf32, -1x256x32x32xf32])
        slice_35 = split_with_num_14[0]

        # builtin.slice: (-1x256x32x32xf32) <- ([-1x256x32x32xf32, -1x256x32x32xf32])
        slice_36 = split_with_num_14[1]

        # builtin.combine: ([-1x256x32x32xf32, -1x256x32x32xf32]) <- (-1x256x32x32xf32, -1x256x32x32xf32)
        combine_28 = [slice_35, slice_36]

        # pd_op.add_n: (-1x256x32x32xf32) <- ([-1x256x32x32xf32, -1x256x32x32xf32])
        add_n_14 = paddle._C_ops.add_n(combine_28)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_32 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf32) <- (-1x256x32x32xf32, 2xi64)
        pool2d_11 = paddle._C_ops.pool2d(add_n_14, full_int_array_32, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x1x1xf32) <- (-1x256x1x1xf32, 128x256x1x1xf32)
        conv2d_42 = paddle._C_ops.conv2d(pool2d_11, parameter_189, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__210, batch_norm__211, batch_norm__212, batch_norm__213, batch_norm__214, batch_norm__215 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_42, parameter_190, parameter_191, parameter_192, parameter_193, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x1x1xf32) <- (-1x128x1x1xf32)
        relu__33 = paddle._C_ops.relu_(batch_norm__210)

        # pd_op.conv2d: (-1x512x1x1xf32) <- (-1x128x1x1xf32, 512x128x1x1xf32)
        conv2d_43 = paddle._C_ops.conv2d(relu__33, parameter_194, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_33 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32, 0x512xf32) <- (512xf32, 4xi64)
        reshape_14, reshape_15 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_195, full_int_array_33), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x1x1xf32) <- (-1x512x1x1xf32, 1x512x1x1xf32)
        add__14 = paddle._C_ops.add_(conv2d_43, reshape_14)

        # pd_op.shape: (4xi32) <- (-1x512x1x1xf32)
        shape_7 = paddle._C_ops.shape(add__14)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_34 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_35 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(shape_7, [0], full_int_array_34, full_int_array_35, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_57 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_58 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_59 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_29 = [slice_37, full_57, full_58, full_59]

        # pd_op.reshape_: (-1x1x2x256xf32, 0x-1x512x1x1xf32) <- (-1x512x1x1xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__28, reshape__29 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__14, [x.reshape([]) for x in combine_29]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x256xf32) <- (-1x1x2x256xf32)
        transpose_7 = paddle._C_ops.transpose(reshape__28, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x256xf32) <- (-1x2x1x256xf32)
        softmax__7 = paddle._C_ops.softmax_(transpose_7, 1)

        # pd_op.full: (1xi32) <- ()
        full_60 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_61 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_62 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_30 = [slice_37, full_60, full_61, full_62]

        # pd_op.reshape_: (-1x512x1x1xf32, 0x-1x2x1x256xf32) <- (-1x2x1x256xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__30, reshape__31 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__7, [x.reshape([]) for x in combine_30]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_63 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x1x1xf32, -1x256x1x1xf32]) <- (-1x512x1x1xf32, 1xi32)
        split_with_num_15 = paddle._C_ops.split_with_num(reshape__30, 2, full_63)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_38 = split_with_num_15[0]

        # pd_op.multiply_: (-1x256x32x32xf32) <- (-1x256x32x32xf32, -1x256x1x1xf32)
        multiply__14 = paddle._C_ops.multiply_(slice_35, slice_38)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_39 = split_with_num_15[1]

        # pd_op.multiply_: (-1x256x32x32xf32) <- (-1x256x32x32xf32, -1x256x1x1xf32)
        multiply__15 = paddle._C_ops.multiply_(slice_36, slice_39)

        # builtin.combine: ([-1x256x32x32xf32, -1x256x32x32xf32]) <- (-1x256x32x32xf32, -1x256x32x32xf32)
        combine_31 = [multiply__14, multiply__15]

        # pd_op.add_n: (-1x256x32x32xf32) <- ([-1x256x32x32xf32, -1x256x32x32xf32])
        add_n_15 = paddle._C_ops.add_n(combine_31)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_36 = [3, 3]

        # pd_op.pool2d: (-1x256x16x16xf32) <- (-1x256x32x32xf32, 2xi64)
        pool2d_12 = paddle._C_ops.pool2d(add_n_15, full_int_array_36, [2, 2], [1, 1], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x1024x16x16xf32) <- (-1x256x16x16xf32, 1024x256x1x1xf32)
        conv2d_44 = paddle._C_ops.conv2d(pool2d_12, parameter_196, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__216, batch_norm__217, batch_norm__218, batch_norm__219, batch_norm__220, batch_norm__221 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_44, parameter_197, parameter_198, parameter_199, parameter_200, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_37 = [2, 2]

        # pd_op.pool2d: (-1x512x16x16xf32) <- (-1x512x32x32xf32, 2xi64)
        pool2d_13 = paddle._C_ops.pool2d(relu__30, full_int_array_37, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x1024x16x16xf32) <- (-1x512x16x16xf32, 1024x512x1x1xf32)
        conv2d_45 = paddle._C_ops.conv2d(pool2d_13, parameter_201, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__222, batch_norm__223, batch_norm__224, batch_norm__225, batch_norm__226, batch_norm__227 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_45, parameter_202, parameter_203, parameter_204, parameter_205, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32, -1x1024x16x16xf32)
        add__15 = paddle._C_ops.add_(batch_norm__222, batch_norm__216)

        # pd_op.relu_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32)
        relu__34 = paddle._C_ops.relu_(add__15)

        # pd_op.conv2d: (-1x256x16x16xf32) <- (-1x1024x16x16xf32, 256x1024x1x1xf32)
        conv2d_46 = paddle._C_ops.conv2d(relu__34, parameter_206, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__228, batch_norm__229, batch_norm__230, batch_norm__231, batch_norm__232, batch_norm__233 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_46, parameter_207, parameter_208, parameter_209, parameter_210, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x16x16xf32) <- (-1x256x16x16xf32)
        relu__35 = paddle._C_ops.relu_(batch_norm__228)

        # pd_op.conv2d: (-1x512x16x16xf32) <- (-1x256x16x16xf32, 512x128x3x3xf32)
        conv2d_47 = paddle._C_ops.conv2d(relu__35, parameter_211, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__234, batch_norm__235, batch_norm__236, batch_norm__237, batch_norm__238, batch_norm__239 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_47, parameter_212, parameter_213, parameter_214, parameter_215, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x16x16xf32) <- (-1x512x16x16xf32)
        relu__36 = paddle._C_ops.relu_(batch_norm__234)

        # pd_op.full: (1xi32) <- ()
        full_64 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x512x16x16xf32, 1xi32)
        split_with_num_16 = paddle._C_ops.split_with_num(relu__36, 2, full_64)

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_40 = split_with_num_16[0]

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_41 = split_with_num_16[1]

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_32 = [slice_40, slice_41]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_16 = paddle._C_ops.add_n(combine_32)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_38 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf32) <- (-1x256x16x16xf32, 2xi64)
        pool2d_14 = paddle._C_ops.pool2d(add_n_16, full_int_array_38, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x1x1xf32) <- (-1x256x1x1xf32, 128x256x1x1xf32)
        conv2d_48 = paddle._C_ops.conv2d(pool2d_14, parameter_216, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__240, batch_norm__241, batch_norm__242, batch_norm__243, batch_norm__244, batch_norm__245 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_48, parameter_217, parameter_218, parameter_219, parameter_220, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x1x1xf32) <- (-1x128x1x1xf32)
        relu__37 = paddle._C_ops.relu_(batch_norm__240)

        # pd_op.conv2d: (-1x512x1x1xf32) <- (-1x128x1x1xf32, 512x128x1x1xf32)
        conv2d_49 = paddle._C_ops.conv2d(relu__37, parameter_221, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_39 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32, 0x512xf32) <- (512xf32, 4xi64)
        reshape_16, reshape_17 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_222, full_int_array_39), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x1x1xf32) <- (-1x512x1x1xf32, 1x512x1x1xf32)
        add__16 = paddle._C_ops.add_(conv2d_49, reshape_16)

        # pd_op.shape: (4xi32) <- (-1x512x1x1xf32)
        shape_8 = paddle._C_ops.shape(add__16)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_40 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_41 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_42 = paddle._C_ops.slice(shape_8, [0], full_int_array_40, full_int_array_41, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_65 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_66 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_67 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_33 = [slice_42, full_65, full_66, full_67]

        # pd_op.reshape_: (-1x1x2x256xf32, 0x-1x512x1x1xf32) <- (-1x512x1x1xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__32, reshape__33 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__16, [x.reshape([]) for x in combine_33]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x256xf32) <- (-1x1x2x256xf32)
        transpose_8 = paddle._C_ops.transpose(reshape__32, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x256xf32) <- (-1x2x1x256xf32)
        softmax__8 = paddle._C_ops.softmax_(transpose_8, 1)

        # pd_op.full: (1xi32) <- ()
        full_68 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_69 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_70 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_34 = [slice_42, full_68, full_69, full_70]

        # pd_op.reshape_: (-1x512x1x1xf32, 0x-1x2x1x256xf32) <- (-1x2x1x256xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__34, reshape__35 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__8, [x.reshape([]) for x in combine_34]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_71 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x1x1xf32, -1x256x1x1xf32]) <- (-1x512x1x1xf32, 1xi32)
        split_with_num_17 = paddle._C_ops.split_with_num(reshape__34, 2, full_71)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_43 = split_with_num_17[0]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__16 = paddle._C_ops.multiply_(slice_40, slice_43)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_44 = split_with_num_17[1]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__17 = paddle._C_ops.multiply_(slice_41, slice_44)

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_35 = [multiply__16, multiply__17]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_17 = paddle._C_ops.add_n(combine_35)

        # pd_op.conv2d: (-1x1024x16x16xf32) <- (-1x256x16x16xf32, 1024x256x1x1xf32)
        conv2d_50 = paddle._C_ops.conv2d(add_n_17, parameter_223, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__246, batch_norm__247, batch_norm__248, batch_norm__249, batch_norm__250, batch_norm__251 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_50, parameter_224, parameter_225, parameter_226, parameter_227, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32, -1x1024x16x16xf32)
        add__17 = paddle._C_ops.add_(relu__34, batch_norm__246)

        # pd_op.relu_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32)
        relu__38 = paddle._C_ops.relu_(add__17)

        # pd_op.conv2d: (-1x256x16x16xf32) <- (-1x1024x16x16xf32, 256x1024x1x1xf32)
        conv2d_51 = paddle._C_ops.conv2d(relu__38, parameter_228, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__252, batch_norm__253, batch_norm__254, batch_norm__255, batch_norm__256, batch_norm__257 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_51, parameter_229, parameter_230, parameter_231, parameter_232, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x16x16xf32) <- (-1x256x16x16xf32)
        relu__39 = paddle._C_ops.relu_(batch_norm__252)

        # pd_op.conv2d: (-1x512x16x16xf32) <- (-1x256x16x16xf32, 512x128x3x3xf32)
        conv2d_52 = paddle._C_ops.conv2d(relu__39, parameter_233, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__258, batch_norm__259, batch_norm__260, batch_norm__261, batch_norm__262, batch_norm__263 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_52, parameter_234, parameter_235, parameter_236, parameter_237, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x16x16xf32) <- (-1x512x16x16xf32)
        relu__40 = paddle._C_ops.relu_(batch_norm__258)

        # pd_op.full: (1xi32) <- ()
        full_72 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x512x16x16xf32, 1xi32)
        split_with_num_18 = paddle._C_ops.split_with_num(relu__40, 2, full_72)

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_45 = split_with_num_18[0]

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_46 = split_with_num_18[1]

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_36 = [slice_45, slice_46]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_18 = paddle._C_ops.add_n(combine_36)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_42 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf32) <- (-1x256x16x16xf32, 2xi64)
        pool2d_15 = paddle._C_ops.pool2d(add_n_18, full_int_array_42, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x1x1xf32) <- (-1x256x1x1xf32, 128x256x1x1xf32)
        conv2d_53 = paddle._C_ops.conv2d(pool2d_15, parameter_238, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__264, batch_norm__265, batch_norm__266, batch_norm__267, batch_norm__268, batch_norm__269 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_53, parameter_239, parameter_240, parameter_241, parameter_242, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x1x1xf32) <- (-1x128x1x1xf32)
        relu__41 = paddle._C_ops.relu_(batch_norm__264)

        # pd_op.conv2d: (-1x512x1x1xf32) <- (-1x128x1x1xf32, 512x128x1x1xf32)
        conv2d_54 = paddle._C_ops.conv2d(relu__41, parameter_243, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_43 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32, 0x512xf32) <- (512xf32, 4xi64)
        reshape_18, reshape_19 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_244, full_int_array_43), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x1x1xf32) <- (-1x512x1x1xf32, 1x512x1x1xf32)
        add__18 = paddle._C_ops.add_(conv2d_54, reshape_18)

        # pd_op.shape: (4xi32) <- (-1x512x1x1xf32)
        shape_9 = paddle._C_ops.shape(add__18)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_44 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_45 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_47 = paddle._C_ops.slice(shape_9, [0], full_int_array_44, full_int_array_45, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_73 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_74 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_75 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_37 = [slice_47, full_73, full_74, full_75]

        # pd_op.reshape_: (-1x1x2x256xf32, 0x-1x512x1x1xf32) <- (-1x512x1x1xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__36, reshape__37 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__18, [x.reshape([]) for x in combine_37]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x256xf32) <- (-1x1x2x256xf32)
        transpose_9 = paddle._C_ops.transpose(reshape__36, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x256xf32) <- (-1x2x1x256xf32)
        softmax__9 = paddle._C_ops.softmax_(transpose_9, 1)

        # pd_op.full: (1xi32) <- ()
        full_76 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_77 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_78 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_38 = [slice_47, full_76, full_77, full_78]

        # pd_op.reshape_: (-1x512x1x1xf32, 0x-1x2x1x256xf32) <- (-1x2x1x256xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__38, reshape__39 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__9, [x.reshape([]) for x in combine_38]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_79 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x1x1xf32, -1x256x1x1xf32]) <- (-1x512x1x1xf32, 1xi32)
        split_with_num_19 = paddle._C_ops.split_with_num(reshape__38, 2, full_79)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_48 = split_with_num_19[0]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__18 = paddle._C_ops.multiply_(slice_45, slice_48)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_49 = split_with_num_19[1]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__19 = paddle._C_ops.multiply_(slice_46, slice_49)

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_39 = [multiply__18, multiply__19]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_19 = paddle._C_ops.add_n(combine_39)

        # pd_op.conv2d: (-1x1024x16x16xf32) <- (-1x256x16x16xf32, 1024x256x1x1xf32)
        conv2d_55 = paddle._C_ops.conv2d(add_n_19, parameter_245, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__270, batch_norm__271, batch_norm__272, batch_norm__273, batch_norm__274, batch_norm__275 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_55, parameter_246, parameter_247, parameter_248, parameter_249, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32, -1x1024x16x16xf32)
        add__19 = paddle._C_ops.add_(relu__38, batch_norm__270)

        # pd_op.relu_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32)
        relu__42 = paddle._C_ops.relu_(add__19)

        # pd_op.conv2d: (-1x256x16x16xf32) <- (-1x1024x16x16xf32, 256x1024x1x1xf32)
        conv2d_56 = paddle._C_ops.conv2d(relu__42, parameter_250, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__276, batch_norm__277, batch_norm__278, batch_norm__279, batch_norm__280, batch_norm__281 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_56, parameter_251, parameter_252, parameter_253, parameter_254, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x16x16xf32) <- (-1x256x16x16xf32)
        relu__43 = paddle._C_ops.relu_(batch_norm__276)

        # pd_op.conv2d: (-1x512x16x16xf32) <- (-1x256x16x16xf32, 512x128x3x3xf32)
        conv2d_57 = paddle._C_ops.conv2d(relu__43, parameter_255, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__282, batch_norm__283, batch_norm__284, batch_norm__285, batch_norm__286, batch_norm__287 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_57, parameter_256, parameter_257, parameter_258, parameter_259, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x16x16xf32) <- (-1x512x16x16xf32)
        relu__44 = paddle._C_ops.relu_(batch_norm__282)

        # pd_op.full: (1xi32) <- ()
        full_80 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x512x16x16xf32, 1xi32)
        split_with_num_20 = paddle._C_ops.split_with_num(relu__44, 2, full_80)

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_50 = split_with_num_20[0]

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_51 = split_with_num_20[1]

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_40 = [slice_50, slice_51]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_20 = paddle._C_ops.add_n(combine_40)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_46 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf32) <- (-1x256x16x16xf32, 2xi64)
        pool2d_16 = paddle._C_ops.pool2d(add_n_20, full_int_array_46, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x1x1xf32) <- (-1x256x1x1xf32, 128x256x1x1xf32)
        conv2d_58 = paddle._C_ops.conv2d(pool2d_16, parameter_260, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__288, batch_norm__289, batch_norm__290, batch_norm__291, batch_norm__292, batch_norm__293 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_58, parameter_261, parameter_262, parameter_263, parameter_264, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x1x1xf32) <- (-1x128x1x1xf32)
        relu__45 = paddle._C_ops.relu_(batch_norm__288)

        # pd_op.conv2d: (-1x512x1x1xf32) <- (-1x128x1x1xf32, 512x128x1x1xf32)
        conv2d_59 = paddle._C_ops.conv2d(relu__45, parameter_265, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_47 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32, 0x512xf32) <- (512xf32, 4xi64)
        reshape_20, reshape_21 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_266, full_int_array_47), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x1x1xf32) <- (-1x512x1x1xf32, 1x512x1x1xf32)
        add__20 = paddle._C_ops.add_(conv2d_59, reshape_20)

        # pd_op.shape: (4xi32) <- (-1x512x1x1xf32)
        shape_10 = paddle._C_ops.shape(add__20)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_48 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_49 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_52 = paddle._C_ops.slice(shape_10, [0], full_int_array_48, full_int_array_49, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_81 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_82 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_83 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_41 = [slice_52, full_81, full_82, full_83]

        # pd_op.reshape_: (-1x1x2x256xf32, 0x-1x512x1x1xf32) <- (-1x512x1x1xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__40, reshape__41 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__20, [x.reshape([]) for x in combine_41]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x256xf32) <- (-1x1x2x256xf32)
        transpose_10 = paddle._C_ops.transpose(reshape__40, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x256xf32) <- (-1x2x1x256xf32)
        softmax__10 = paddle._C_ops.softmax_(transpose_10, 1)

        # pd_op.full: (1xi32) <- ()
        full_84 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_85 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_86 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_42 = [slice_52, full_84, full_85, full_86]

        # pd_op.reshape_: (-1x512x1x1xf32, 0x-1x2x1x256xf32) <- (-1x2x1x256xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__42, reshape__43 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__10, [x.reshape([]) for x in combine_42]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_87 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x1x1xf32, -1x256x1x1xf32]) <- (-1x512x1x1xf32, 1xi32)
        split_with_num_21 = paddle._C_ops.split_with_num(reshape__42, 2, full_87)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_53 = split_with_num_21[0]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__20 = paddle._C_ops.multiply_(slice_50, slice_53)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_54 = split_with_num_21[1]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__21 = paddle._C_ops.multiply_(slice_51, slice_54)

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_43 = [multiply__20, multiply__21]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_21 = paddle._C_ops.add_n(combine_43)

        # pd_op.conv2d: (-1x1024x16x16xf32) <- (-1x256x16x16xf32, 1024x256x1x1xf32)
        conv2d_60 = paddle._C_ops.conv2d(add_n_21, parameter_267, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__294, batch_norm__295, batch_norm__296, batch_norm__297, batch_norm__298, batch_norm__299 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_60, parameter_268, parameter_269, parameter_270, parameter_271, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32, -1x1024x16x16xf32)
        add__21 = paddle._C_ops.add_(relu__42, batch_norm__294)

        # pd_op.relu_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32)
        relu__46 = paddle._C_ops.relu_(add__21)

        # pd_op.conv2d: (-1x256x16x16xf32) <- (-1x1024x16x16xf32, 256x1024x1x1xf32)
        conv2d_61 = paddle._C_ops.conv2d(relu__46, parameter_272, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__300, batch_norm__301, batch_norm__302, batch_norm__303, batch_norm__304, batch_norm__305 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_61, parameter_273, parameter_274, parameter_275, parameter_276, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x16x16xf32) <- (-1x256x16x16xf32)
        relu__47 = paddle._C_ops.relu_(batch_norm__300)

        # pd_op.conv2d: (-1x512x16x16xf32) <- (-1x256x16x16xf32, 512x128x3x3xf32)
        conv2d_62 = paddle._C_ops.conv2d(relu__47, parameter_277, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__306, batch_norm__307, batch_norm__308, batch_norm__309, batch_norm__310, batch_norm__311 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_62, parameter_278, parameter_279, parameter_280, parameter_281, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x16x16xf32) <- (-1x512x16x16xf32)
        relu__48 = paddle._C_ops.relu_(batch_norm__306)

        # pd_op.full: (1xi32) <- ()
        full_88 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x512x16x16xf32, 1xi32)
        split_with_num_22 = paddle._C_ops.split_with_num(relu__48, 2, full_88)

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_55 = split_with_num_22[0]

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_56 = split_with_num_22[1]

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_44 = [slice_55, slice_56]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_22 = paddle._C_ops.add_n(combine_44)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_50 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf32) <- (-1x256x16x16xf32, 2xi64)
        pool2d_17 = paddle._C_ops.pool2d(add_n_22, full_int_array_50, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x1x1xf32) <- (-1x256x1x1xf32, 128x256x1x1xf32)
        conv2d_63 = paddle._C_ops.conv2d(pool2d_17, parameter_282, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__312, batch_norm__313, batch_norm__314, batch_norm__315, batch_norm__316, batch_norm__317 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_63, parameter_283, parameter_284, parameter_285, parameter_286, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x1x1xf32) <- (-1x128x1x1xf32)
        relu__49 = paddle._C_ops.relu_(batch_norm__312)

        # pd_op.conv2d: (-1x512x1x1xf32) <- (-1x128x1x1xf32, 512x128x1x1xf32)
        conv2d_64 = paddle._C_ops.conv2d(relu__49, parameter_287, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_51 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32, 0x512xf32) <- (512xf32, 4xi64)
        reshape_22, reshape_23 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_288, full_int_array_51), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x1x1xf32) <- (-1x512x1x1xf32, 1x512x1x1xf32)
        add__22 = paddle._C_ops.add_(conv2d_64, reshape_22)

        # pd_op.shape: (4xi32) <- (-1x512x1x1xf32)
        shape_11 = paddle._C_ops.shape(add__22)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_52 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_53 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_57 = paddle._C_ops.slice(shape_11, [0], full_int_array_52, full_int_array_53, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_89 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_90 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_91 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_45 = [slice_57, full_89, full_90, full_91]

        # pd_op.reshape_: (-1x1x2x256xf32, 0x-1x512x1x1xf32) <- (-1x512x1x1xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__44, reshape__45 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__22, [x.reshape([]) for x in combine_45]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x256xf32) <- (-1x1x2x256xf32)
        transpose_11 = paddle._C_ops.transpose(reshape__44, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x256xf32) <- (-1x2x1x256xf32)
        softmax__11 = paddle._C_ops.softmax_(transpose_11, 1)

        # pd_op.full: (1xi32) <- ()
        full_92 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_93 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_94 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_46 = [slice_57, full_92, full_93, full_94]

        # pd_op.reshape_: (-1x512x1x1xf32, 0x-1x2x1x256xf32) <- (-1x2x1x256xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__46, reshape__47 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__11, [x.reshape([]) for x in combine_46]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_95 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x1x1xf32, -1x256x1x1xf32]) <- (-1x512x1x1xf32, 1xi32)
        split_with_num_23 = paddle._C_ops.split_with_num(reshape__46, 2, full_95)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_58 = split_with_num_23[0]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__22 = paddle._C_ops.multiply_(slice_55, slice_58)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_59 = split_with_num_23[1]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__23 = paddle._C_ops.multiply_(slice_56, slice_59)

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_47 = [multiply__22, multiply__23]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_23 = paddle._C_ops.add_n(combine_47)

        # pd_op.conv2d: (-1x1024x16x16xf32) <- (-1x256x16x16xf32, 1024x256x1x1xf32)
        conv2d_65 = paddle._C_ops.conv2d(add_n_23, parameter_289, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__318, batch_norm__319, batch_norm__320, batch_norm__321, batch_norm__322, batch_norm__323 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_65, parameter_290, parameter_291, parameter_292, parameter_293, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32, -1x1024x16x16xf32)
        add__23 = paddle._C_ops.add_(relu__46, batch_norm__318)

        # pd_op.relu_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32)
        relu__50 = paddle._C_ops.relu_(add__23)

        # pd_op.conv2d: (-1x256x16x16xf32) <- (-1x1024x16x16xf32, 256x1024x1x1xf32)
        conv2d_66 = paddle._C_ops.conv2d(relu__50, parameter_294, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__324, batch_norm__325, batch_norm__326, batch_norm__327, batch_norm__328, batch_norm__329 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_66, parameter_295, parameter_296, parameter_297, parameter_298, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x16x16xf32) <- (-1x256x16x16xf32)
        relu__51 = paddle._C_ops.relu_(batch_norm__324)

        # pd_op.conv2d: (-1x512x16x16xf32) <- (-1x256x16x16xf32, 512x128x3x3xf32)
        conv2d_67 = paddle._C_ops.conv2d(relu__51, parameter_299, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__330, batch_norm__331, batch_norm__332, batch_norm__333, batch_norm__334, batch_norm__335 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_67, parameter_300, parameter_301, parameter_302, parameter_303, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x16x16xf32) <- (-1x512x16x16xf32)
        relu__52 = paddle._C_ops.relu_(batch_norm__330)

        # pd_op.full: (1xi32) <- ()
        full_96 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x512x16x16xf32, 1xi32)
        split_with_num_24 = paddle._C_ops.split_with_num(relu__52, 2, full_96)

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_60 = split_with_num_24[0]

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_61 = split_with_num_24[1]

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_48 = [slice_60, slice_61]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_24 = paddle._C_ops.add_n(combine_48)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_54 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf32) <- (-1x256x16x16xf32, 2xi64)
        pool2d_18 = paddle._C_ops.pool2d(add_n_24, full_int_array_54, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x1x1xf32) <- (-1x256x1x1xf32, 128x256x1x1xf32)
        conv2d_68 = paddle._C_ops.conv2d(pool2d_18, parameter_304, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__336, batch_norm__337, batch_norm__338, batch_norm__339, batch_norm__340, batch_norm__341 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_68, parameter_305, parameter_306, parameter_307, parameter_308, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x1x1xf32) <- (-1x128x1x1xf32)
        relu__53 = paddle._C_ops.relu_(batch_norm__336)

        # pd_op.conv2d: (-1x512x1x1xf32) <- (-1x128x1x1xf32, 512x128x1x1xf32)
        conv2d_69 = paddle._C_ops.conv2d(relu__53, parameter_309, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_55 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32, 0x512xf32) <- (512xf32, 4xi64)
        reshape_24, reshape_25 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_310, full_int_array_55), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x1x1xf32) <- (-1x512x1x1xf32, 1x512x1x1xf32)
        add__24 = paddle._C_ops.add_(conv2d_69, reshape_24)

        # pd_op.shape: (4xi32) <- (-1x512x1x1xf32)
        shape_12 = paddle._C_ops.shape(add__24)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_56 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_57 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_62 = paddle._C_ops.slice(shape_12, [0], full_int_array_56, full_int_array_57, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_97 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_98 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_99 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_49 = [slice_62, full_97, full_98, full_99]

        # pd_op.reshape_: (-1x1x2x256xf32, 0x-1x512x1x1xf32) <- (-1x512x1x1xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__48, reshape__49 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__24, [x.reshape([]) for x in combine_49]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x256xf32) <- (-1x1x2x256xf32)
        transpose_12 = paddle._C_ops.transpose(reshape__48, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x256xf32) <- (-1x2x1x256xf32)
        softmax__12 = paddle._C_ops.softmax_(transpose_12, 1)

        # pd_op.full: (1xi32) <- ()
        full_100 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_101 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_102 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_50 = [slice_62, full_100, full_101, full_102]

        # pd_op.reshape_: (-1x512x1x1xf32, 0x-1x2x1x256xf32) <- (-1x2x1x256xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__50, reshape__51 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__12, [x.reshape([]) for x in combine_50]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_103 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x1x1xf32, -1x256x1x1xf32]) <- (-1x512x1x1xf32, 1xi32)
        split_with_num_25 = paddle._C_ops.split_with_num(reshape__50, 2, full_103)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_63 = split_with_num_25[0]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__24 = paddle._C_ops.multiply_(slice_60, slice_63)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_64 = split_with_num_25[1]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__25 = paddle._C_ops.multiply_(slice_61, slice_64)

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_51 = [multiply__24, multiply__25]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_25 = paddle._C_ops.add_n(combine_51)

        # pd_op.conv2d: (-1x1024x16x16xf32) <- (-1x256x16x16xf32, 1024x256x1x1xf32)
        conv2d_70 = paddle._C_ops.conv2d(add_n_25, parameter_311, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__342, batch_norm__343, batch_norm__344, batch_norm__345, batch_norm__346, batch_norm__347 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_70, parameter_312, parameter_313, parameter_314, parameter_315, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32, -1x1024x16x16xf32)
        add__25 = paddle._C_ops.add_(relu__50, batch_norm__342)

        # pd_op.relu_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32)
        relu__54 = paddle._C_ops.relu_(add__25)

        # pd_op.conv2d: (-1x256x16x16xf32) <- (-1x1024x16x16xf32, 256x1024x1x1xf32)
        conv2d_71 = paddle._C_ops.conv2d(relu__54, parameter_316, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__348, batch_norm__349, batch_norm__350, batch_norm__351, batch_norm__352, batch_norm__353 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_71, parameter_317, parameter_318, parameter_319, parameter_320, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x16x16xf32) <- (-1x256x16x16xf32)
        relu__55 = paddle._C_ops.relu_(batch_norm__348)

        # pd_op.conv2d: (-1x512x16x16xf32) <- (-1x256x16x16xf32, 512x128x3x3xf32)
        conv2d_72 = paddle._C_ops.conv2d(relu__55, parameter_321, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__354, batch_norm__355, batch_norm__356, batch_norm__357, batch_norm__358, batch_norm__359 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_72, parameter_322, parameter_323, parameter_324, parameter_325, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x16x16xf32) <- (-1x512x16x16xf32)
        relu__56 = paddle._C_ops.relu_(batch_norm__354)

        # pd_op.full: (1xi32) <- ()
        full_104 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x512x16x16xf32, 1xi32)
        split_with_num_26 = paddle._C_ops.split_with_num(relu__56, 2, full_104)

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_65 = split_with_num_26[0]

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_66 = split_with_num_26[1]

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_52 = [slice_65, slice_66]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_26 = paddle._C_ops.add_n(combine_52)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_58 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf32) <- (-1x256x16x16xf32, 2xi64)
        pool2d_19 = paddle._C_ops.pool2d(add_n_26, full_int_array_58, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x1x1xf32) <- (-1x256x1x1xf32, 128x256x1x1xf32)
        conv2d_73 = paddle._C_ops.conv2d(pool2d_19, parameter_326, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__360, batch_norm__361, batch_norm__362, batch_norm__363, batch_norm__364, batch_norm__365 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_73, parameter_327, parameter_328, parameter_329, parameter_330, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x1x1xf32) <- (-1x128x1x1xf32)
        relu__57 = paddle._C_ops.relu_(batch_norm__360)

        # pd_op.conv2d: (-1x512x1x1xf32) <- (-1x128x1x1xf32, 512x128x1x1xf32)
        conv2d_74 = paddle._C_ops.conv2d(relu__57, parameter_331, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_59 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32, 0x512xf32) <- (512xf32, 4xi64)
        reshape_26, reshape_27 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_332, full_int_array_59), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x1x1xf32) <- (-1x512x1x1xf32, 1x512x1x1xf32)
        add__26 = paddle._C_ops.add_(conv2d_74, reshape_26)

        # pd_op.shape: (4xi32) <- (-1x512x1x1xf32)
        shape_13 = paddle._C_ops.shape(add__26)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_60 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_61 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_67 = paddle._C_ops.slice(shape_13, [0], full_int_array_60, full_int_array_61, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_105 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_106 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_107 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_53 = [slice_67, full_105, full_106, full_107]

        # pd_op.reshape_: (-1x1x2x256xf32, 0x-1x512x1x1xf32) <- (-1x512x1x1xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__52, reshape__53 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__26, [x.reshape([]) for x in combine_53]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x256xf32) <- (-1x1x2x256xf32)
        transpose_13 = paddle._C_ops.transpose(reshape__52, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x256xf32) <- (-1x2x1x256xf32)
        softmax__13 = paddle._C_ops.softmax_(transpose_13, 1)

        # pd_op.full: (1xi32) <- ()
        full_108 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_109 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_110 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_54 = [slice_67, full_108, full_109, full_110]

        # pd_op.reshape_: (-1x512x1x1xf32, 0x-1x2x1x256xf32) <- (-1x2x1x256xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__54, reshape__55 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__13, [x.reshape([]) for x in combine_54]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_111 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x1x1xf32, -1x256x1x1xf32]) <- (-1x512x1x1xf32, 1xi32)
        split_with_num_27 = paddle._C_ops.split_with_num(reshape__54, 2, full_111)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_68 = split_with_num_27[0]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__26 = paddle._C_ops.multiply_(slice_65, slice_68)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_69 = split_with_num_27[1]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__27 = paddle._C_ops.multiply_(slice_66, slice_69)

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_55 = [multiply__26, multiply__27]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_27 = paddle._C_ops.add_n(combine_55)

        # pd_op.conv2d: (-1x1024x16x16xf32) <- (-1x256x16x16xf32, 1024x256x1x1xf32)
        conv2d_75 = paddle._C_ops.conv2d(add_n_27, parameter_333, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__366, batch_norm__367, batch_norm__368, batch_norm__369, batch_norm__370, batch_norm__371 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_75, parameter_334, parameter_335, parameter_336, parameter_337, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32, -1x1024x16x16xf32)
        add__27 = paddle._C_ops.add_(relu__54, batch_norm__366)

        # pd_op.relu_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32)
        relu__58 = paddle._C_ops.relu_(add__27)

        # pd_op.conv2d: (-1x256x16x16xf32) <- (-1x1024x16x16xf32, 256x1024x1x1xf32)
        conv2d_76 = paddle._C_ops.conv2d(relu__58, parameter_338, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__372, batch_norm__373, batch_norm__374, batch_norm__375, batch_norm__376, batch_norm__377 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_76, parameter_339, parameter_340, parameter_341, parameter_342, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x16x16xf32) <- (-1x256x16x16xf32)
        relu__59 = paddle._C_ops.relu_(batch_norm__372)

        # pd_op.conv2d: (-1x512x16x16xf32) <- (-1x256x16x16xf32, 512x128x3x3xf32)
        conv2d_77 = paddle._C_ops.conv2d(relu__59, parameter_343, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__378, batch_norm__379, batch_norm__380, batch_norm__381, batch_norm__382, batch_norm__383 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_77, parameter_344, parameter_345, parameter_346, parameter_347, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x16x16xf32) <- (-1x512x16x16xf32)
        relu__60 = paddle._C_ops.relu_(batch_norm__378)

        # pd_op.full: (1xi32) <- ()
        full_112 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x512x16x16xf32, 1xi32)
        split_with_num_28 = paddle._C_ops.split_with_num(relu__60, 2, full_112)

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_70 = split_with_num_28[0]

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_71 = split_with_num_28[1]

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_56 = [slice_70, slice_71]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_28 = paddle._C_ops.add_n(combine_56)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_62 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf32) <- (-1x256x16x16xf32, 2xi64)
        pool2d_20 = paddle._C_ops.pool2d(add_n_28, full_int_array_62, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x1x1xf32) <- (-1x256x1x1xf32, 128x256x1x1xf32)
        conv2d_78 = paddle._C_ops.conv2d(pool2d_20, parameter_348, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__384, batch_norm__385, batch_norm__386, batch_norm__387, batch_norm__388, batch_norm__389 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_78, parameter_349, parameter_350, parameter_351, parameter_352, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x1x1xf32) <- (-1x128x1x1xf32)
        relu__61 = paddle._C_ops.relu_(batch_norm__384)

        # pd_op.conv2d: (-1x512x1x1xf32) <- (-1x128x1x1xf32, 512x128x1x1xf32)
        conv2d_79 = paddle._C_ops.conv2d(relu__61, parameter_353, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_63 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32, 0x512xf32) <- (512xf32, 4xi64)
        reshape_28, reshape_29 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_354, full_int_array_63), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x1x1xf32) <- (-1x512x1x1xf32, 1x512x1x1xf32)
        add__28 = paddle._C_ops.add_(conv2d_79, reshape_28)

        # pd_op.shape: (4xi32) <- (-1x512x1x1xf32)
        shape_14 = paddle._C_ops.shape(add__28)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_64 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_65 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_72 = paddle._C_ops.slice(shape_14, [0], full_int_array_64, full_int_array_65, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_113 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_114 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_115 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_57 = [slice_72, full_113, full_114, full_115]

        # pd_op.reshape_: (-1x1x2x256xf32, 0x-1x512x1x1xf32) <- (-1x512x1x1xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__56, reshape__57 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__28, [x.reshape([]) for x in combine_57]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x256xf32) <- (-1x1x2x256xf32)
        transpose_14 = paddle._C_ops.transpose(reshape__56, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x256xf32) <- (-1x2x1x256xf32)
        softmax__14 = paddle._C_ops.softmax_(transpose_14, 1)

        # pd_op.full: (1xi32) <- ()
        full_116 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_117 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_118 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_58 = [slice_72, full_116, full_117, full_118]

        # pd_op.reshape_: (-1x512x1x1xf32, 0x-1x2x1x256xf32) <- (-1x2x1x256xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__58, reshape__59 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__14, [x.reshape([]) for x in combine_58]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_119 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x1x1xf32, -1x256x1x1xf32]) <- (-1x512x1x1xf32, 1xi32)
        split_with_num_29 = paddle._C_ops.split_with_num(reshape__58, 2, full_119)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_73 = split_with_num_29[0]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__28 = paddle._C_ops.multiply_(slice_70, slice_73)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_74 = split_with_num_29[1]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__29 = paddle._C_ops.multiply_(slice_71, slice_74)

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_59 = [multiply__28, multiply__29]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_29 = paddle._C_ops.add_n(combine_59)

        # pd_op.conv2d: (-1x1024x16x16xf32) <- (-1x256x16x16xf32, 1024x256x1x1xf32)
        conv2d_80 = paddle._C_ops.conv2d(add_n_29, parameter_355, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__390, batch_norm__391, batch_norm__392, batch_norm__393, batch_norm__394, batch_norm__395 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_80, parameter_356, parameter_357, parameter_358, parameter_359, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32, -1x1024x16x16xf32)
        add__29 = paddle._C_ops.add_(relu__58, batch_norm__390)

        # pd_op.relu_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32)
        relu__62 = paddle._C_ops.relu_(add__29)

        # pd_op.conv2d: (-1x256x16x16xf32) <- (-1x1024x16x16xf32, 256x1024x1x1xf32)
        conv2d_81 = paddle._C_ops.conv2d(relu__62, parameter_360, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__396, batch_norm__397, batch_norm__398, batch_norm__399, batch_norm__400, batch_norm__401 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_81, parameter_361, parameter_362, parameter_363, parameter_364, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x16x16xf32) <- (-1x256x16x16xf32)
        relu__63 = paddle._C_ops.relu_(batch_norm__396)

        # pd_op.conv2d: (-1x512x16x16xf32) <- (-1x256x16x16xf32, 512x128x3x3xf32)
        conv2d_82 = paddle._C_ops.conv2d(relu__63, parameter_365, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__402, batch_norm__403, batch_norm__404, batch_norm__405, batch_norm__406, batch_norm__407 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_82, parameter_366, parameter_367, parameter_368, parameter_369, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x16x16xf32) <- (-1x512x16x16xf32)
        relu__64 = paddle._C_ops.relu_(batch_norm__402)

        # pd_op.full: (1xi32) <- ()
        full_120 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x512x16x16xf32, 1xi32)
        split_with_num_30 = paddle._C_ops.split_with_num(relu__64, 2, full_120)

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_75 = split_with_num_30[0]

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_76 = split_with_num_30[1]

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_60 = [slice_75, slice_76]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_30 = paddle._C_ops.add_n(combine_60)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_66 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf32) <- (-1x256x16x16xf32, 2xi64)
        pool2d_21 = paddle._C_ops.pool2d(add_n_30, full_int_array_66, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x1x1xf32) <- (-1x256x1x1xf32, 128x256x1x1xf32)
        conv2d_83 = paddle._C_ops.conv2d(pool2d_21, parameter_370, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__408, batch_norm__409, batch_norm__410, batch_norm__411, batch_norm__412, batch_norm__413 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_83, parameter_371, parameter_372, parameter_373, parameter_374, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x1x1xf32) <- (-1x128x1x1xf32)
        relu__65 = paddle._C_ops.relu_(batch_norm__408)

        # pd_op.conv2d: (-1x512x1x1xf32) <- (-1x128x1x1xf32, 512x128x1x1xf32)
        conv2d_84 = paddle._C_ops.conv2d(relu__65, parameter_375, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_67 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32, 0x512xf32) <- (512xf32, 4xi64)
        reshape_30, reshape_31 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_376, full_int_array_67), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x1x1xf32) <- (-1x512x1x1xf32, 1x512x1x1xf32)
        add__30 = paddle._C_ops.add_(conv2d_84, reshape_30)

        # pd_op.shape: (4xi32) <- (-1x512x1x1xf32)
        shape_15 = paddle._C_ops.shape(add__30)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_68 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_69 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_77 = paddle._C_ops.slice(shape_15, [0], full_int_array_68, full_int_array_69, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_121 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_122 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_123 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_61 = [slice_77, full_121, full_122, full_123]

        # pd_op.reshape_: (-1x1x2x256xf32, 0x-1x512x1x1xf32) <- (-1x512x1x1xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__60, reshape__61 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__30, [x.reshape([]) for x in combine_61]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x256xf32) <- (-1x1x2x256xf32)
        transpose_15 = paddle._C_ops.transpose(reshape__60, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x256xf32) <- (-1x2x1x256xf32)
        softmax__15 = paddle._C_ops.softmax_(transpose_15, 1)

        # pd_op.full: (1xi32) <- ()
        full_124 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_125 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_126 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_62 = [slice_77, full_124, full_125, full_126]

        # pd_op.reshape_: (-1x512x1x1xf32, 0x-1x2x1x256xf32) <- (-1x2x1x256xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__62, reshape__63 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__15, [x.reshape([]) for x in combine_62]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_127 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x1x1xf32, -1x256x1x1xf32]) <- (-1x512x1x1xf32, 1xi32)
        split_with_num_31 = paddle._C_ops.split_with_num(reshape__62, 2, full_127)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_78 = split_with_num_31[0]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__30 = paddle._C_ops.multiply_(slice_75, slice_78)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_79 = split_with_num_31[1]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__31 = paddle._C_ops.multiply_(slice_76, slice_79)

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_63 = [multiply__30, multiply__31]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_31 = paddle._C_ops.add_n(combine_63)

        # pd_op.conv2d: (-1x1024x16x16xf32) <- (-1x256x16x16xf32, 1024x256x1x1xf32)
        conv2d_85 = paddle._C_ops.conv2d(add_n_31, parameter_377, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__414, batch_norm__415, batch_norm__416, batch_norm__417, batch_norm__418, batch_norm__419 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_85, parameter_378, parameter_379, parameter_380, parameter_381, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32, -1x1024x16x16xf32)
        add__31 = paddle._C_ops.add_(relu__62, batch_norm__414)

        # pd_op.relu_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32)
        relu__66 = paddle._C_ops.relu_(add__31)

        # pd_op.conv2d: (-1x256x16x16xf32) <- (-1x1024x16x16xf32, 256x1024x1x1xf32)
        conv2d_86 = paddle._C_ops.conv2d(relu__66, parameter_382, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__420, batch_norm__421, batch_norm__422, batch_norm__423, batch_norm__424, batch_norm__425 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_86, parameter_383, parameter_384, parameter_385, parameter_386, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x16x16xf32) <- (-1x256x16x16xf32)
        relu__67 = paddle._C_ops.relu_(batch_norm__420)

        # pd_op.conv2d: (-1x512x16x16xf32) <- (-1x256x16x16xf32, 512x128x3x3xf32)
        conv2d_87 = paddle._C_ops.conv2d(relu__67, parameter_387, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__426, batch_norm__427, batch_norm__428, batch_norm__429, batch_norm__430, batch_norm__431 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_87, parameter_388, parameter_389, parameter_390, parameter_391, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x16x16xf32) <- (-1x512x16x16xf32)
        relu__68 = paddle._C_ops.relu_(batch_norm__426)

        # pd_op.full: (1xi32) <- ()
        full_128 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x512x16x16xf32, 1xi32)
        split_with_num_32 = paddle._C_ops.split_with_num(relu__68, 2, full_128)

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_80 = split_with_num_32[0]

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_81 = split_with_num_32[1]

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_64 = [slice_80, slice_81]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_32 = paddle._C_ops.add_n(combine_64)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_70 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf32) <- (-1x256x16x16xf32, 2xi64)
        pool2d_22 = paddle._C_ops.pool2d(add_n_32, full_int_array_70, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x1x1xf32) <- (-1x256x1x1xf32, 128x256x1x1xf32)
        conv2d_88 = paddle._C_ops.conv2d(pool2d_22, parameter_392, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__432, batch_norm__433, batch_norm__434, batch_norm__435, batch_norm__436, batch_norm__437 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_88, parameter_393, parameter_394, parameter_395, parameter_396, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x1x1xf32) <- (-1x128x1x1xf32)
        relu__69 = paddle._C_ops.relu_(batch_norm__432)

        # pd_op.conv2d: (-1x512x1x1xf32) <- (-1x128x1x1xf32, 512x128x1x1xf32)
        conv2d_89 = paddle._C_ops.conv2d(relu__69, parameter_397, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_71 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32, 0x512xf32) <- (512xf32, 4xi64)
        reshape_32, reshape_33 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_398, full_int_array_71), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x1x1xf32) <- (-1x512x1x1xf32, 1x512x1x1xf32)
        add__32 = paddle._C_ops.add_(conv2d_89, reshape_32)

        # pd_op.shape: (4xi32) <- (-1x512x1x1xf32)
        shape_16 = paddle._C_ops.shape(add__32)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_72 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_73 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_82 = paddle._C_ops.slice(shape_16, [0], full_int_array_72, full_int_array_73, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_129 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_130 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_131 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_65 = [slice_82, full_129, full_130, full_131]

        # pd_op.reshape_: (-1x1x2x256xf32, 0x-1x512x1x1xf32) <- (-1x512x1x1xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__64, reshape__65 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__32, [x.reshape([]) for x in combine_65]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x256xf32) <- (-1x1x2x256xf32)
        transpose_16 = paddle._C_ops.transpose(reshape__64, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x256xf32) <- (-1x2x1x256xf32)
        softmax__16 = paddle._C_ops.softmax_(transpose_16, 1)

        # pd_op.full: (1xi32) <- ()
        full_132 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_133 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_134 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_66 = [slice_82, full_132, full_133, full_134]

        # pd_op.reshape_: (-1x512x1x1xf32, 0x-1x2x1x256xf32) <- (-1x2x1x256xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__66, reshape__67 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__16, [x.reshape([]) for x in combine_66]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_135 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x1x1xf32, -1x256x1x1xf32]) <- (-1x512x1x1xf32, 1xi32)
        split_with_num_33 = paddle._C_ops.split_with_num(reshape__66, 2, full_135)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_83 = split_with_num_33[0]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__32 = paddle._C_ops.multiply_(slice_80, slice_83)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_84 = split_with_num_33[1]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__33 = paddle._C_ops.multiply_(slice_81, slice_84)

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_67 = [multiply__32, multiply__33]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_33 = paddle._C_ops.add_n(combine_67)

        # pd_op.conv2d: (-1x1024x16x16xf32) <- (-1x256x16x16xf32, 1024x256x1x1xf32)
        conv2d_90 = paddle._C_ops.conv2d(add_n_33, parameter_399, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__438, batch_norm__439, batch_norm__440, batch_norm__441, batch_norm__442, batch_norm__443 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_90, parameter_400, parameter_401, parameter_402, parameter_403, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32, -1x1024x16x16xf32)
        add__33 = paddle._C_ops.add_(relu__66, batch_norm__438)

        # pd_op.relu_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32)
        relu__70 = paddle._C_ops.relu_(add__33)

        # pd_op.conv2d: (-1x256x16x16xf32) <- (-1x1024x16x16xf32, 256x1024x1x1xf32)
        conv2d_91 = paddle._C_ops.conv2d(relu__70, parameter_404, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__444, batch_norm__445, batch_norm__446, batch_norm__447, batch_norm__448, batch_norm__449 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_91, parameter_405, parameter_406, parameter_407, parameter_408, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x16x16xf32) <- (-1x256x16x16xf32)
        relu__71 = paddle._C_ops.relu_(batch_norm__444)

        # pd_op.conv2d: (-1x512x16x16xf32) <- (-1x256x16x16xf32, 512x128x3x3xf32)
        conv2d_92 = paddle._C_ops.conv2d(relu__71, parameter_409, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__450, batch_norm__451, batch_norm__452, batch_norm__453, batch_norm__454, batch_norm__455 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_92, parameter_410, parameter_411, parameter_412, parameter_413, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x16x16xf32) <- (-1x512x16x16xf32)
        relu__72 = paddle._C_ops.relu_(batch_norm__450)

        # pd_op.full: (1xi32) <- ()
        full_136 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x512x16x16xf32, 1xi32)
        split_with_num_34 = paddle._C_ops.split_with_num(relu__72, 2, full_136)

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_85 = split_with_num_34[0]

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_86 = split_with_num_34[1]

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_68 = [slice_85, slice_86]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_34 = paddle._C_ops.add_n(combine_68)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_74 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf32) <- (-1x256x16x16xf32, 2xi64)
        pool2d_23 = paddle._C_ops.pool2d(add_n_34, full_int_array_74, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x1x1xf32) <- (-1x256x1x1xf32, 128x256x1x1xf32)
        conv2d_93 = paddle._C_ops.conv2d(pool2d_23, parameter_414, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__456, batch_norm__457, batch_norm__458, batch_norm__459, batch_norm__460, batch_norm__461 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_93, parameter_415, parameter_416, parameter_417, parameter_418, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x1x1xf32) <- (-1x128x1x1xf32)
        relu__73 = paddle._C_ops.relu_(batch_norm__456)

        # pd_op.conv2d: (-1x512x1x1xf32) <- (-1x128x1x1xf32, 512x128x1x1xf32)
        conv2d_94 = paddle._C_ops.conv2d(relu__73, parameter_419, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_75 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32, 0x512xf32) <- (512xf32, 4xi64)
        reshape_34, reshape_35 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_420, full_int_array_75), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x1x1xf32) <- (-1x512x1x1xf32, 1x512x1x1xf32)
        add__34 = paddle._C_ops.add_(conv2d_94, reshape_34)

        # pd_op.shape: (4xi32) <- (-1x512x1x1xf32)
        shape_17 = paddle._C_ops.shape(add__34)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_76 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_77 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_87 = paddle._C_ops.slice(shape_17, [0], full_int_array_76, full_int_array_77, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_137 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_138 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_139 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_69 = [slice_87, full_137, full_138, full_139]

        # pd_op.reshape_: (-1x1x2x256xf32, 0x-1x512x1x1xf32) <- (-1x512x1x1xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__68, reshape__69 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__34, [x.reshape([]) for x in combine_69]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x256xf32) <- (-1x1x2x256xf32)
        transpose_17 = paddle._C_ops.transpose(reshape__68, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x256xf32) <- (-1x2x1x256xf32)
        softmax__17 = paddle._C_ops.softmax_(transpose_17, 1)

        # pd_op.full: (1xi32) <- ()
        full_140 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_141 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_142 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_70 = [slice_87, full_140, full_141, full_142]

        # pd_op.reshape_: (-1x512x1x1xf32, 0x-1x2x1x256xf32) <- (-1x2x1x256xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__70, reshape__71 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__17, [x.reshape([]) for x in combine_70]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_143 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x1x1xf32, -1x256x1x1xf32]) <- (-1x512x1x1xf32, 1xi32)
        split_with_num_35 = paddle._C_ops.split_with_num(reshape__70, 2, full_143)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_88 = split_with_num_35[0]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__34 = paddle._C_ops.multiply_(slice_85, slice_88)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_89 = split_with_num_35[1]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__35 = paddle._C_ops.multiply_(slice_86, slice_89)

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_71 = [multiply__34, multiply__35]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_35 = paddle._C_ops.add_n(combine_71)

        # pd_op.conv2d: (-1x1024x16x16xf32) <- (-1x256x16x16xf32, 1024x256x1x1xf32)
        conv2d_95 = paddle._C_ops.conv2d(add_n_35, parameter_421, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__462, batch_norm__463, batch_norm__464, batch_norm__465, batch_norm__466, batch_norm__467 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_95, parameter_422, parameter_423, parameter_424, parameter_425, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32, -1x1024x16x16xf32)
        add__35 = paddle._C_ops.add_(relu__70, batch_norm__462)

        # pd_op.relu_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32)
        relu__74 = paddle._C_ops.relu_(add__35)

        # pd_op.conv2d: (-1x256x16x16xf32) <- (-1x1024x16x16xf32, 256x1024x1x1xf32)
        conv2d_96 = paddle._C_ops.conv2d(relu__74, parameter_426, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__468, batch_norm__469, batch_norm__470, batch_norm__471, batch_norm__472, batch_norm__473 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_96, parameter_427, parameter_428, parameter_429, parameter_430, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x16x16xf32) <- (-1x256x16x16xf32)
        relu__75 = paddle._C_ops.relu_(batch_norm__468)

        # pd_op.conv2d: (-1x512x16x16xf32) <- (-1x256x16x16xf32, 512x128x3x3xf32)
        conv2d_97 = paddle._C_ops.conv2d(relu__75, parameter_431, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__474, batch_norm__475, batch_norm__476, batch_norm__477, batch_norm__478, batch_norm__479 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_97, parameter_432, parameter_433, parameter_434, parameter_435, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x16x16xf32) <- (-1x512x16x16xf32)
        relu__76 = paddle._C_ops.relu_(batch_norm__474)

        # pd_op.full: (1xi32) <- ()
        full_144 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x512x16x16xf32, 1xi32)
        split_with_num_36 = paddle._C_ops.split_with_num(relu__76, 2, full_144)

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_90 = split_with_num_36[0]

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_91 = split_with_num_36[1]

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_72 = [slice_90, slice_91]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_36 = paddle._C_ops.add_n(combine_72)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_78 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf32) <- (-1x256x16x16xf32, 2xi64)
        pool2d_24 = paddle._C_ops.pool2d(add_n_36, full_int_array_78, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x1x1xf32) <- (-1x256x1x1xf32, 128x256x1x1xf32)
        conv2d_98 = paddle._C_ops.conv2d(pool2d_24, parameter_436, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__480, batch_norm__481, batch_norm__482, batch_norm__483, batch_norm__484, batch_norm__485 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_98, parameter_437, parameter_438, parameter_439, parameter_440, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x1x1xf32) <- (-1x128x1x1xf32)
        relu__77 = paddle._C_ops.relu_(batch_norm__480)

        # pd_op.conv2d: (-1x512x1x1xf32) <- (-1x128x1x1xf32, 512x128x1x1xf32)
        conv2d_99 = paddle._C_ops.conv2d(relu__77, parameter_441, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_79 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32, 0x512xf32) <- (512xf32, 4xi64)
        reshape_36, reshape_37 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_442, full_int_array_79), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x1x1xf32) <- (-1x512x1x1xf32, 1x512x1x1xf32)
        add__36 = paddle._C_ops.add_(conv2d_99, reshape_36)

        # pd_op.shape: (4xi32) <- (-1x512x1x1xf32)
        shape_18 = paddle._C_ops.shape(add__36)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_80 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_81 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_92 = paddle._C_ops.slice(shape_18, [0], full_int_array_80, full_int_array_81, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_145 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_146 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_147 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_73 = [slice_92, full_145, full_146, full_147]

        # pd_op.reshape_: (-1x1x2x256xf32, 0x-1x512x1x1xf32) <- (-1x512x1x1xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__72, reshape__73 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__36, [x.reshape([]) for x in combine_73]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x256xf32) <- (-1x1x2x256xf32)
        transpose_18 = paddle._C_ops.transpose(reshape__72, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x256xf32) <- (-1x2x1x256xf32)
        softmax__18 = paddle._C_ops.softmax_(transpose_18, 1)

        # pd_op.full: (1xi32) <- ()
        full_148 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_149 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_150 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_74 = [slice_92, full_148, full_149, full_150]

        # pd_op.reshape_: (-1x512x1x1xf32, 0x-1x2x1x256xf32) <- (-1x2x1x256xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__74, reshape__75 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__18, [x.reshape([]) for x in combine_74]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_151 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x1x1xf32, -1x256x1x1xf32]) <- (-1x512x1x1xf32, 1xi32)
        split_with_num_37 = paddle._C_ops.split_with_num(reshape__74, 2, full_151)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_93 = split_with_num_37[0]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__36 = paddle._C_ops.multiply_(slice_90, slice_93)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_94 = split_with_num_37[1]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__37 = paddle._C_ops.multiply_(slice_91, slice_94)

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_75 = [multiply__36, multiply__37]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_37 = paddle._C_ops.add_n(combine_75)

        # pd_op.conv2d: (-1x1024x16x16xf32) <- (-1x256x16x16xf32, 1024x256x1x1xf32)
        conv2d_100 = paddle._C_ops.conv2d(add_n_37, parameter_443, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__486, batch_norm__487, batch_norm__488, batch_norm__489, batch_norm__490, batch_norm__491 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_100, parameter_444, parameter_445, parameter_446, parameter_447, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32, -1x1024x16x16xf32)
        add__37 = paddle._C_ops.add_(relu__74, batch_norm__486)

        # pd_op.relu_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32)
        relu__78 = paddle._C_ops.relu_(add__37)

        # pd_op.conv2d: (-1x256x16x16xf32) <- (-1x1024x16x16xf32, 256x1024x1x1xf32)
        conv2d_101 = paddle._C_ops.conv2d(relu__78, parameter_448, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__492, batch_norm__493, batch_norm__494, batch_norm__495, batch_norm__496, batch_norm__497 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_101, parameter_449, parameter_450, parameter_451, parameter_452, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x16x16xf32) <- (-1x256x16x16xf32)
        relu__79 = paddle._C_ops.relu_(batch_norm__492)

        # pd_op.conv2d: (-1x512x16x16xf32) <- (-1x256x16x16xf32, 512x128x3x3xf32)
        conv2d_102 = paddle._C_ops.conv2d(relu__79, parameter_453, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__498, batch_norm__499, batch_norm__500, batch_norm__501, batch_norm__502, batch_norm__503 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_102, parameter_454, parameter_455, parameter_456, parameter_457, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x16x16xf32) <- (-1x512x16x16xf32)
        relu__80 = paddle._C_ops.relu_(batch_norm__498)

        # pd_op.full: (1xi32) <- ()
        full_152 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x512x16x16xf32, 1xi32)
        split_with_num_38 = paddle._C_ops.split_with_num(relu__80, 2, full_152)

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_95 = split_with_num_38[0]

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_96 = split_with_num_38[1]

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_76 = [slice_95, slice_96]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_38 = paddle._C_ops.add_n(combine_76)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_82 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf32) <- (-1x256x16x16xf32, 2xi64)
        pool2d_25 = paddle._C_ops.pool2d(add_n_38, full_int_array_82, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x1x1xf32) <- (-1x256x1x1xf32, 128x256x1x1xf32)
        conv2d_103 = paddle._C_ops.conv2d(pool2d_25, parameter_458, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__504, batch_norm__505, batch_norm__506, batch_norm__507, batch_norm__508, batch_norm__509 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_103, parameter_459, parameter_460, parameter_461, parameter_462, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x1x1xf32) <- (-1x128x1x1xf32)
        relu__81 = paddle._C_ops.relu_(batch_norm__504)

        # pd_op.conv2d: (-1x512x1x1xf32) <- (-1x128x1x1xf32, 512x128x1x1xf32)
        conv2d_104 = paddle._C_ops.conv2d(relu__81, parameter_463, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_83 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32, 0x512xf32) <- (512xf32, 4xi64)
        reshape_38, reshape_39 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_464, full_int_array_83), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x1x1xf32) <- (-1x512x1x1xf32, 1x512x1x1xf32)
        add__38 = paddle._C_ops.add_(conv2d_104, reshape_38)

        # pd_op.shape: (4xi32) <- (-1x512x1x1xf32)
        shape_19 = paddle._C_ops.shape(add__38)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_84 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_85 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_97 = paddle._C_ops.slice(shape_19, [0], full_int_array_84, full_int_array_85, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_153 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_154 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_155 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_77 = [slice_97, full_153, full_154, full_155]

        # pd_op.reshape_: (-1x1x2x256xf32, 0x-1x512x1x1xf32) <- (-1x512x1x1xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__76, reshape__77 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__38, [x.reshape([]) for x in combine_77]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x256xf32) <- (-1x1x2x256xf32)
        transpose_19 = paddle._C_ops.transpose(reshape__76, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x256xf32) <- (-1x2x1x256xf32)
        softmax__19 = paddle._C_ops.softmax_(transpose_19, 1)

        # pd_op.full: (1xi32) <- ()
        full_156 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_157 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_158 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_78 = [slice_97, full_156, full_157, full_158]

        # pd_op.reshape_: (-1x512x1x1xf32, 0x-1x2x1x256xf32) <- (-1x2x1x256xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__78, reshape__79 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__19, [x.reshape([]) for x in combine_78]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_159 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x1x1xf32, -1x256x1x1xf32]) <- (-1x512x1x1xf32, 1xi32)
        split_with_num_39 = paddle._C_ops.split_with_num(reshape__78, 2, full_159)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_98 = split_with_num_39[0]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__38 = paddle._C_ops.multiply_(slice_95, slice_98)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_99 = split_with_num_39[1]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__39 = paddle._C_ops.multiply_(slice_96, slice_99)

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_79 = [multiply__38, multiply__39]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_39 = paddle._C_ops.add_n(combine_79)

        # pd_op.conv2d: (-1x1024x16x16xf32) <- (-1x256x16x16xf32, 1024x256x1x1xf32)
        conv2d_105 = paddle._C_ops.conv2d(add_n_39, parameter_465, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__510, batch_norm__511, batch_norm__512, batch_norm__513, batch_norm__514, batch_norm__515 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_105, parameter_466, parameter_467, parameter_468, parameter_469, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32, -1x1024x16x16xf32)
        add__39 = paddle._C_ops.add_(relu__78, batch_norm__510)

        # pd_op.relu_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32)
        relu__82 = paddle._C_ops.relu_(add__39)

        # pd_op.conv2d: (-1x256x16x16xf32) <- (-1x1024x16x16xf32, 256x1024x1x1xf32)
        conv2d_106 = paddle._C_ops.conv2d(relu__82, parameter_470, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__516, batch_norm__517, batch_norm__518, batch_norm__519, batch_norm__520, batch_norm__521 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_106, parameter_471, parameter_472, parameter_473, parameter_474, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x16x16xf32) <- (-1x256x16x16xf32)
        relu__83 = paddle._C_ops.relu_(batch_norm__516)

        # pd_op.conv2d: (-1x512x16x16xf32) <- (-1x256x16x16xf32, 512x128x3x3xf32)
        conv2d_107 = paddle._C_ops.conv2d(relu__83, parameter_475, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__522, batch_norm__523, batch_norm__524, batch_norm__525, batch_norm__526, batch_norm__527 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_107, parameter_476, parameter_477, parameter_478, parameter_479, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x16x16xf32) <- (-1x512x16x16xf32)
        relu__84 = paddle._C_ops.relu_(batch_norm__522)

        # pd_op.full: (1xi32) <- ()
        full_160 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x512x16x16xf32, 1xi32)
        split_with_num_40 = paddle._C_ops.split_with_num(relu__84, 2, full_160)

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_100 = split_with_num_40[0]

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_101 = split_with_num_40[1]

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_80 = [slice_100, slice_101]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_40 = paddle._C_ops.add_n(combine_80)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_86 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf32) <- (-1x256x16x16xf32, 2xi64)
        pool2d_26 = paddle._C_ops.pool2d(add_n_40, full_int_array_86, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x1x1xf32) <- (-1x256x1x1xf32, 128x256x1x1xf32)
        conv2d_108 = paddle._C_ops.conv2d(pool2d_26, parameter_480, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__528, batch_norm__529, batch_norm__530, batch_norm__531, batch_norm__532, batch_norm__533 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_108, parameter_481, parameter_482, parameter_483, parameter_484, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x1x1xf32) <- (-1x128x1x1xf32)
        relu__85 = paddle._C_ops.relu_(batch_norm__528)

        # pd_op.conv2d: (-1x512x1x1xf32) <- (-1x128x1x1xf32, 512x128x1x1xf32)
        conv2d_109 = paddle._C_ops.conv2d(relu__85, parameter_485, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_87 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32, 0x512xf32) <- (512xf32, 4xi64)
        reshape_40, reshape_41 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_486, full_int_array_87), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x1x1xf32) <- (-1x512x1x1xf32, 1x512x1x1xf32)
        add__40 = paddle._C_ops.add_(conv2d_109, reshape_40)

        # pd_op.shape: (4xi32) <- (-1x512x1x1xf32)
        shape_20 = paddle._C_ops.shape(add__40)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_88 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_89 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_102 = paddle._C_ops.slice(shape_20, [0], full_int_array_88, full_int_array_89, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_161 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_162 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_163 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_81 = [slice_102, full_161, full_162, full_163]

        # pd_op.reshape_: (-1x1x2x256xf32, 0x-1x512x1x1xf32) <- (-1x512x1x1xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__80, reshape__81 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__40, [x.reshape([]) for x in combine_81]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x256xf32) <- (-1x1x2x256xf32)
        transpose_20 = paddle._C_ops.transpose(reshape__80, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x256xf32) <- (-1x2x1x256xf32)
        softmax__20 = paddle._C_ops.softmax_(transpose_20, 1)

        # pd_op.full: (1xi32) <- ()
        full_164 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_165 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_166 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_82 = [slice_102, full_164, full_165, full_166]

        # pd_op.reshape_: (-1x512x1x1xf32, 0x-1x2x1x256xf32) <- (-1x2x1x256xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__82, reshape__83 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__20, [x.reshape([]) for x in combine_82]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_167 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x1x1xf32, -1x256x1x1xf32]) <- (-1x512x1x1xf32, 1xi32)
        split_with_num_41 = paddle._C_ops.split_with_num(reshape__82, 2, full_167)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_103 = split_with_num_41[0]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__40 = paddle._C_ops.multiply_(slice_100, slice_103)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_104 = split_with_num_41[1]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__41 = paddle._C_ops.multiply_(slice_101, slice_104)

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_83 = [multiply__40, multiply__41]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_41 = paddle._C_ops.add_n(combine_83)

        # pd_op.conv2d: (-1x1024x16x16xf32) <- (-1x256x16x16xf32, 1024x256x1x1xf32)
        conv2d_110 = paddle._C_ops.conv2d(add_n_41, parameter_487, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__534, batch_norm__535, batch_norm__536, batch_norm__537, batch_norm__538, batch_norm__539 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_110, parameter_488, parameter_489, parameter_490, parameter_491, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32, -1x1024x16x16xf32)
        add__41 = paddle._C_ops.add_(relu__82, batch_norm__534)

        # pd_op.relu_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32)
        relu__86 = paddle._C_ops.relu_(add__41)

        # pd_op.conv2d: (-1x256x16x16xf32) <- (-1x1024x16x16xf32, 256x1024x1x1xf32)
        conv2d_111 = paddle._C_ops.conv2d(relu__86, parameter_492, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__540, batch_norm__541, batch_norm__542, batch_norm__543, batch_norm__544, batch_norm__545 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_111, parameter_493, parameter_494, parameter_495, parameter_496, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x16x16xf32) <- (-1x256x16x16xf32)
        relu__87 = paddle._C_ops.relu_(batch_norm__540)

        # pd_op.conv2d: (-1x512x16x16xf32) <- (-1x256x16x16xf32, 512x128x3x3xf32)
        conv2d_112 = paddle._C_ops.conv2d(relu__87, parameter_497, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__546, batch_norm__547, batch_norm__548, batch_norm__549, batch_norm__550, batch_norm__551 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_112, parameter_498, parameter_499, parameter_500, parameter_501, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x16x16xf32) <- (-1x512x16x16xf32)
        relu__88 = paddle._C_ops.relu_(batch_norm__546)

        # pd_op.full: (1xi32) <- ()
        full_168 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x512x16x16xf32, 1xi32)
        split_with_num_42 = paddle._C_ops.split_with_num(relu__88, 2, full_168)

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_105 = split_with_num_42[0]

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_106 = split_with_num_42[1]

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_84 = [slice_105, slice_106]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_42 = paddle._C_ops.add_n(combine_84)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_90 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf32) <- (-1x256x16x16xf32, 2xi64)
        pool2d_27 = paddle._C_ops.pool2d(add_n_42, full_int_array_90, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x1x1xf32) <- (-1x256x1x1xf32, 128x256x1x1xf32)
        conv2d_113 = paddle._C_ops.conv2d(pool2d_27, parameter_502, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__552, batch_norm__553, batch_norm__554, batch_norm__555, batch_norm__556, batch_norm__557 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_113, parameter_503, parameter_504, parameter_505, parameter_506, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x1x1xf32) <- (-1x128x1x1xf32)
        relu__89 = paddle._C_ops.relu_(batch_norm__552)

        # pd_op.conv2d: (-1x512x1x1xf32) <- (-1x128x1x1xf32, 512x128x1x1xf32)
        conv2d_114 = paddle._C_ops.conv2d(relu__89, parameter_507, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_91 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32, 0x512xf32) <- (512xf32, 4xi64)
        reshape_42, reshape_43 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_508, full_int_array_91), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x1x1xf32) <- (-1x512x1x1xf32, 1x512x1x1xf32)
        add__42 = paddle._C_ops.add_(conv2d_114, reshape_42)

        # pd_op.shape: (4xi32) <- (-1x512x1x1xf32)
        shape_21 = paddle._C_ops.shape(add__42)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_92 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_93 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_107 = paddle._C_ops.slice(shape_21, [0], full_int_array_92, full_int_array_93, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_169 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_170 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_171 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_85 = [slice_107, full_169, full_170, full_171]

        # pd_op.reshape_: (-1x1x2x256xf32, 0x-1x512x1x1xf32) <- (-1x512x1x1xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__84, reshape__85 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__42, [x.reshape([]) for x in combine_85]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x256xf32) <- (-1x1x2x256xf32)
        transpose_21 = paddle._C_ops.transpose(reshape__84, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x256xf32) <- (-1x2x1x256xf32)
        softmax__21 = paddle._C_ops.softmax_(transpose_21, 1)

        # pd_op.full: (1xi32) <- ()
        full_172 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_173 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_174 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_86 = [slice_107, full_172, full_173, full_174]

        # pd_op.reshape_: (-1x512x1x1xf32, 0x-1x2x1x256xf32) <- (-1x2x1x256xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__86, reshape__87 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__21, [x.reshape([]) for x in combine_86]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_175 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x1x1xf32, -1x256x1x1xf32]) <- (-1x512x1x1xf32, 1xi32)
        split_with_num_43 = paddle._C_ops.split_with_num(reshape__86, 2, full_175)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_108 = split_with_num_43[0]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__42 = paddle._C_ops.multiply_(slice_105, slice_108)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_109 = split_with_num_43[1]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__43 = paddle._C_ops.multiply_(slice_106, slice_109)

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_87 = [multiply__42, multiply__43]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_43 = paddle._C_ops.add_n(combine_87)

        # pd_op.conv2d: (-1x1024x16x16xf32) <- (-1x256x16x16xf32, 1024x256x1x1xf32)
        conv2d_115 = paddle._C_ops.conv2d(add_n_43, parameter_509, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__558, batch_norm__559, batch_norm__560, batch_norm__561, batch_norm__562, batch_norm__563 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_115, parameter_510, parameter_511, parameter_512, parameter_513, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32, -1x1024x16x16xf32)
        add__43 = paddle._C_ops.add_(relu__86, batch_norm__558)

        # pd_op.relu_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32)
        relu__90 = paddle._C_ops.relu_(add__43)

        # pd_op.conv2d: (-1x256x16x16xf32) <- (-1x1024x16x16xf32, 256x1024x1x1xf32)
        conv2d_116 = paddle._C_ops.conv2d(relu__90, parameter_514, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__564, batch_norm__565, batch_norm__566, batch_norm__567, batch_norm__568, batch_norm__569 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_116, parameter_515, parameter_516, parameter_517, parameter_518, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x16x16xf32) <- (-1x256x16x16xf32)
        relu__91 = paddle._C_ops.relu_(batch_norm__564)

        # pd_op.conv2d: (-1x512x16x16xf32) <- (-1x256x16x16xf32, 512x128x3x3xf32)
        conv2d_117 = paddle._C_ops.conv2d(relu__91, parameter_519, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__570, batch_norm__571, batch_norm__572, batch_norm__573, batch_norm__574, batch_norm__575 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_117, parameter_520, parameter_521, parameter_522, parameter_523, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x16x16xf32) <- (-1x512x16x16xf32)
        relu__92 = paddle._C_ops.relu_(batch_norm__570)

        # pd_op.full: (1xi32) <- ()
        full_176 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x512x16x16xf32, 1xi32)
        split_with_num_44 = paddle._C_ops.split_with_num(relu__92, 2, full_176)

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_110 = split_with_num_44[0]

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_111 = split_with_num_44[1]

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_88 = [slice_110, slice_111]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_44 = paddle._C_ops.add_n(combine_88)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_94 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf32) <- (-1x256x16x16xf32, 2xi64)
        pool2d_28 = paddle._C_ops.pool2d(add_n_44, full_int_array_94, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x1x1xf32) <- (-1x256x1x1xf32, 128x256x1x1xf32)
        conv2d_118 = paddle._C_ops.conv2d(pool2d_28, parameter_524, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__576, batch_norm__577, batch_norm__578, batch_norm__579, batch_norm__580, batch_norm__581 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_118, parameter_525, parameter_526, parameter_527, parameter_528, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x1x1xf32) <- (-1x128x1x1xf32)
        relu__93 = paddle._C_ops.relu_(batch_norm__576)

        # pd_op.conv2d: (-1x512x1x1xf32) <- (-1x128x1x1xf32, 512x128x1x1xf32)
        conv2d_119 = paddle._C_ops.conv2d(relu__93, parameter_529, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_95 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32, 0x512xf32) <- (512xf32, 4xi64)
        reshape_44, reshape_45 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_530, full_int_array_95), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x1x1xf32) <- (-1x512x1x1xf32, 1x512x1x1xf32)
        add__44 = paddle._C_ops.add_(conv2d_119, reshape_44)

        # pd_op.shape: (4xi32) <- (-1x512x1x1xf32)
        shape_22 = paddle._C_ops.shape(add__44)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_96 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_97 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_112 = paddle._C_ops.slice(shape_22, [0], full_int_array_96, full_int_array_97, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_177 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_178 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_179 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_89 = [slice_112, full_177, full_178, full_179]

        # pd_op.reshape_: (-1x1x2x256xf32, 0x-1x512x1x1xf32) <- (-1x512x1x1xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__88, reshape__89 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__44, [x.reshape([]) for x in combine_89]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x256xf32) <- (-1x1x2x256xf32)
        transpose_22 = paddle._C_ops.transpose(reshape__88, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x256xf32) <- (-1x2x1x256xf32)
        softmax__22 = paddle._C_ops.softmax_(transpose_22, 1)

        # pd_op.full: (1xi32) <- ()
        full_180 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_181 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_182 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_90 = [slice_112, full_180, full_181, full_182]

        # pd_op.reshape_: (-1x512x1x1xf32, 0x-1x2x1x256xf32) <- (-1x2x1x256xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__90, reshape__91 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__22, [x.reshape([]) for x in combine_90]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_183 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x1x1xf32, -1x256x1x1xf32]) <- (-1x512x1x1xf32, 1xi32)
        split_with_num_45 = paddle._C_ops.split_with_num(reshape__90, 2, full_183)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_113 = split_with_num_45[0]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__44 = paddle._C_ops.multiply_(slice_110, slice_113)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_114 = split_with_num_45[1]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__45 = paddle._C_ops.multiply_(slice_111, slice_114)

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_91 = [multiply__44, multiply__45]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_45 = paddle._C_ops.add_n(combine_91)

        # pd_op.conv2d: (-1x1024x16x16xf32) <- (-1x256x16x16xf32, 1024x256x1x1xf32)
        conv2d_120 = paddle._C_ops.conv2d(add_n_45, parameter_531, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__582, batch_norm__583, batch_norm__584, batch_norm__585, batch_norm__586, batch_norm__587 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_120, parameter_532, parameter_533, parameter_534, parameter_535, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32, -1x1024x16x16xf32)
        add__45 = paddle._C_ops.add_(relu__90, batch_norm__582)

        # pd_op.relu_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32)
        relu__94 = paddle._C_ops.relu_(add__45)

        # pd_op.conv2d: (-1x256x16x16xf32) <- (-1x1024x16x16xf32, 256x1024x1x1xf32)
        conv2d_121 = paddle._C_ops.conv2d(relu__94, parameter_536, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__588, batch_norm__589, batch_norm__590, batch_norm__591, batch_norm__592, batch_norm__593 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_121, parameter_537, parameter_538, parameter_539, parameter_540, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x16x16xf32) <- (-1x256x16x16xf32)
        relu__95 = paddle._C_ops.relu_(batch_norm__588)

        # pd_op.conv2d: (-1x512x16x16xf32) <- (-1x256x16x16xf32, 512x128x3x3xf32)
        conv2d_122 = paddle._C_ops.conv2d(relu__95, parameter_541, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__594, batch_norm__595, batch_norm__596, batch_norm__597, batch_norm__598, batch_norm__599 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_122, parameter_542, parameter_543, parameter_544, parameter_545, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x16x16xf32) <- (-1x512x16x16xf32)
        relu__96 = paddle._C_ops.relu_(batch_norm__594)

        # pd_op.full: (1xi32) <- ()
        full_184 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x512x16x16xf32, 1xi32)
        split_with_num_46 = paddle._C_ops.split_with_num(relu__96, 2, full_184)

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_115 = split_with_num_46[0]

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_116 = split_with_num_46[1]

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_92 = [slice_115, slice_116]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_46 = paddle._C_ops.add_n(combine_92)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_98 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf32) <- (-1x256x16x16xf32, 2xi64)
        pool2d_29 = paddle._C_ops.pool2d(add_n_46, full_int_array_98, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x1x1xf32) <- (-1x256x1x1xf32, 128x256x1x1xf32)
        conv2d_123 = paddle._C_ops.conv2d(pool2d_29, parameter_546, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__600, batch_norm__601, batch_norm__602, batch_norm__603, batch_norm__604, batch_norm__605 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_123, parameter_547, parameter_548, parameter_549, parameter_550, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x1x1xf32) <- (-1x128x1x1xf32)
        relu__97 = paddle._C_ops.relu_(batch_norm__600)

        # pd_op.conv2d: (-1x512x1x1xf32) <- (-1x128x1x1xf32, 512x128x1x1xf32)
        conv2d_124 = paddle._C_ops.conv2d(relu__97, parameter_551, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_99 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32, 0x512xf32) <- (512xf32, 4xi64)
        reshape_46, reshape_47 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_552, full_int_array_99), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x1x1xf32) <- (-1x512x1x1xf32, 1x512x1x1xf32)
        add__46 = paddle._C_ops.add_(conv2d_124, reshape_46)

        # pd_op.shape: (4xi32) <- (-1x512x1x1xf32)
        shape_23 = paddle._C_ops.shape(add__46)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_100 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_101 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_117 = paddle._C_ops.slice(shape_23, [0], full_int_array_100, full_int_array_101, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_185 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_186 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_187 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_93 = [slice_117, full_185, full_186, full_187]

        # pd_op.reshape_: (-1x1x2x256xf32, 0x-1x512x1x1xf32) <- (-1x512x1x1xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__92, reshape__93 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__46, [x.reshape([]) for x in combine_93]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x256xf32) <- (-1x1x2x256xf32)
        transpose_23 = paddle._C_ops.transpose(reshape__92, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x256xf32) <- (-1x2x1x256xf32)
        softmax__23 = paddle._C_ops.softmax_(transpose_23, 1)

        # pd_op.full: (1xi32) <- ()
        full_188 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_189 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_190 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_94 = [slice_117, full_188, full_189, full_190]

        # pd_op.reshape_: (-1x512x1x1xf32, 0x-1x2x1x256xf32) <- (-1x2x1x256xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__94, reshape__95 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__23, [x.reshape([]) for x in combine_94]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_191 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x1x1xf32, -1x256x1x1xf32]) <- (-1x512x1x1xf32, 1xi32)
        split_with_num_47 = paddle._C_ops.split_with_num(reshape__94, 2, full_191)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_118 = split_with_num_47[0]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__46 = paddle._C_ops.multiply_(slice_115, slice_118)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_119 = split_with_num_47[1]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__47 = paddle._C_ops.multiply_(slice_116, slice_119)

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_95 = [multiply__46, multiply__47]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_47 = paddle._C_ops.add_n(combine_95)

        # pd_op.conv2d: (-1x1024x16x16xf32) <- (-1x256x16x16xf32, 1024x256x1x1xf32)
        conv2d_125 = paddle._C_ops.conv2d(add_n_47, parameter_553, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__606, batch_norm__607, batch_norm__608, batch_norm__609, batch_norm__610, batch_norm__611 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_125, parameter_554, parameter_555, parameter_556, parameter_557, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32, -1x1024x16x16xf32)
        add__47 = paddle._C_ops.add_(relu__94, batch_norm__606)

        # pd_op.relu_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32)
        relu__98 = paddle._C_ops.relu_(add__47)

        # pd_op.conv2d: (-1x256x16x16xf32) <- (-1x1024x16x16xf32, 256x1024x1x1xf32)
        conv2d_126 = paddle._C_ops.conv2d(relu__98, parameter_558, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__612, batch_norm__613, batch_norm__614, batch_norm__615, batch_norm__616, batch_norm__617 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_126, parameter_559, parameter_560, parameter_561, parameter_562, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x16x16xf32) <- (-1x256x16x16xf32)
        relu__99 = paddle._C_ops.relu_(batch_norm__612)

        # pd_op.conv2d: (-1x512x16x16xf32) <- (-1x256x16x16xf32, 512x128x3x3xf32)
        conv2d_127 = paddle._C_ops.conv2d(relu__99, parameter_563, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__618, batch_norm__619, batch_norm__620, batch_norm__621, batch_norm__622, batch_norm__623 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_127, parameter_564, parameter_565, parameter_566, parameter_567, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x16x16xf32) <- (-1x512x16x16xf32)
        relu__100 = paddle._C_ops.relu_(batch_norm__618)

        # pd_op.full: (1xi32) <- ()
        full_192 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x512x16x16xf32, 1xi32)
        split_with_num_48 = paddle._C_ops.split_with_num(relu__100, 2, full_192)

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_120 = split_with_num_48[0]

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_121 = split_with_num_48[1]

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_96 = [slice_120, slice_121]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_48 = paddle._C_ops.add_n(combine_96)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_102 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf32) <- (-1x256x16x16xf32, 2xi64)
        pool2d_30 = paddle._C_ops.pool2d(add_n_48, full_int_array_102, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x1x1xf32) <- (-1x256x1x1xf32, 128x256x1x1xf32)
        conv2d_128 = paddle._C_ops.conv2d(pool2d_30, parameter_568, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__624, batch_norm__625, batch_norm__626, batch_norm__627, batch_norm__628, batch_norm__629 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_128, parameter_569, parameter_570, parameter_571, parameter_572, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x1x1xf32) <- (-1x128x1x1xf32)
        relu__101 = paddle._C_ops.relu_(batch_norm__624)

        # pd_op.conv2d: (-1x512x1x1xf32) <- (-1x128x1x1xf32, 512x128x1x1xf32)
        conv2d_129 = paddle._C_ops.conv2d(relu__101, parameter_573, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_103 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32, 0x512xf32) <- (512xf32, 4xi64)
        reshape_48, reshape_49 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_574, full_int_array_103), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x1x1xf32) <- (-1x512x1x1xf32, 1x512x1x1xf32)
        add__48 = paddle._C_ops.add_(conv2d_129, reshape_48)

        # pd_op.shape: (4xi32) <- (-1x512x1x1xf32)
        shape_24 = paddle._C_ops.shape(add__48)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_104 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_105 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_122 = paddle._C_ops.slice(shape_24, [0], full_int_array_104, full_int_array_105, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_193 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_194 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_195 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_97 = [slice_122, full_193, full_194, full_195]

        # pd_op.reshape_: (-1x1x2x256xf32, 0x-1x512x1x1xf32) <- (-1x512x1x1xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__96, reshape__97 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__48, [x.reshape([]) for x in combine_97]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x256xf32) <- (-1x1x2x256xf32)
        transpose_24 = paddle._C_ops.transpose(reshape__96, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x256xf32) <- (-1x2x1x256xf32)
        softmax__24 = paddle._C_ops.softmax_(transpose_24, 1)

        # pd_op.full: (1xi32) <- ()
        full_196 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_197 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_198 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_98 = [slice_122, full_196, full_197, full_198]

        # pd_op.reshape_: (-1x512x1x1xf32, 0x-1x2x1x256xf32) <- (-1x2x1x256xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__98, reshape__99 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__24, [x.reshape([]) for x in combine_98]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_199 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x1x1xf32, -1x256x1x1xf32]) <- (-1x512x1x1xf32, 1xi32)
        split_with_num_49 = paddle._C_ops.split_with_num(reshape__98, 2, full_199)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_123 = split_with_num_49[0]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__48 = paddle._C_ops.multiply_(slice_120, slice_123)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_124 = split_with_num_49[1]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__49 = paddle._C_ops.multiply_(slice_121, slice_124)

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_99 = [multiply__48, multiply__49]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_49 = paddle._C_ops.add_n(combine_99)

        # pd_op.conv2d: (-1x1024x16x16xf32) <- (-1x256x16x16xf32, 1024x256x1x1xf32)
        conv2d_130 = paddle._C_ops.conv2d(add_n_49, parameter_575, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__630, batch_norm__631, batch_norm__632, batch_norm__633, batch_norm__634, batch_norm__635 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_130, parameter_576, parameter_577, parameter_578, parameter_579, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32, -1x1024x16x16xf32)
        add__49 = paddle._C_ops.add_(relu__98, batch_norm__630)

        # pd_op.relu_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32)
        relu__102 = paddle._C_ops.relu_(add__49)

        # pd_op.conv2d: (-1x256x16x16xf32) <- (-1x1024x16x16xf32, 256x1024x1x1xf32)
        conv2d_131 = paddle._C_ops.conv2d(relu__102, parameter_580, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__636, batch_norm__637, batch_norm__638, batch_norm__639, batch_norm__640, batch_norm__641 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_131, parameter_581, parameter_582, parameter_583, parameter_584, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x16x16xf32) <- (-1x256x16x16xf32)
        relu__103 = paddle._C_ops.relu_(batch_norm__636)

        # pd_op.conv2d: (-1x512x16x16xf32) <- (-1x256x16x16xf32, 512x128x3x3xf32)
        conv2d_132 = paddle._C_ops.conv2d(relu__103, parameter_585, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__642, batch_norm__643, batch_norm__644, batch_norm__645, batch_norm__646, batch_norm__647 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_132, parameter_586, parameter_587, parameter_588, parameter_589, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x16x16xf32) <- (-1x512x16x16xf32)
        relu__104 = paddle._C_ops.relu_(batch_norm__642)

        # pd_op.full: (1xi32) <- ()
        full_200 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x512x16x16xf32, 1xi32)
        split_with_num_50 = paddle._C_ops.split_with_num(relu__104, 2, full_200)

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_125 = split_with_num_50[0]

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_126 = split_with_num_50[1]

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_100 = [slice_125, slice_126]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_50 = paddle._C_ops.add_n(combine_100)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_106 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf32) <- (-1x256x16x16xf32, 2xi64)
        pool2d_31 = paddle._C_ops.pool2d(add_n_50, full_int_array_106, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x1x1xf32) <- (-1x256x1x1xf32, 128x256x1x1xf32)
        conv2d_133 = paddle._C_ops.conv2d(pool2d_31, parameter_590, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__648, batch_norm__649, batch_norm__650, batch_norm__651, batch_norm__652, batch_norm__653 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_133, parameter_591, parameter_592, parameter_593, parameter_594, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x1x1xf32) <- (-1x128x1x1xf32)
        relu__105 = paddle._C_ops.relu_(batch_norm__648)

        # pd_op.conv2d: (-1x512x1x1xf32) <- (-1x128x1x1xf32, 512x128x1x1xf32)
        conv2d_134 = paddle._C_ops.conv2d(relu__105, parameter_595, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_107 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32, 0x512xf32) <- (512xf32, 4xi64)
        reshape_50, reshape_51 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_596, full_int_array_107), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x1x1xf32) <- (-1x512x1x1xf32, 1x512x1x1xf32)
        add__50 = paddle._C_ops.add_(conv2d_134, reshape_50)

        # pd_op.shape: (4xi32) <- (-1x512x1x1xf32)
        shape_25 = paddle._C_ops.shape(add__50)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_108 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_109 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_127 = paddle._C_ops.slice(shape_25, [0], full_int_array_108, full_int_array_109, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_201 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_202 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_203 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_101 = [slice_127, full_201, full_202, full_203]

        # pd_op.reshape_: (-1x1x2x256xf32, 0x-1x512x1x1xf32) <- (-1x512x1x1xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__100, reshape__101 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__50, [x.reshape([]) for x in combine_101]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x256xf32) <- (-1x1x2x256xf32)
        transpose_25 = paddle._C_ops.transpose(reshape__100, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x256xf32) <- (-1x2x1x256xf32)
        softmax__25 = paddle._C_ops.softmax_(transpose_25, 1)

        # pd_op.full: (1xi32) <- ()
        full_204 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_205 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_206 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_102 = [slice_127, full_204, full_205, full_206]

        # pd_op.reshape_: (-1x512x1x1xf32, 0x-1x2x1x256xf32) <- (-1x2x1x256xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__102, reshape__103 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__25, [x.reshape([]) for x in combine_102]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_207 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x1x1xf32, -1x256x1x1xf32]) <- (-1x512x1x1xf32, 1xi32)
        split_with_num_51 = paddle._C_ops.split_with_num(reshape__102, 2, full_207)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_128 = split_with_num_51[0]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__50 = paddle._C_ops.multiply_(slice_125, slice_128)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_129 = split_with_num_51[1]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__51 = paddle._C_ops.multiply_(slice_126, slice_129)

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_103 = [multiply__50, multiply__51]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_51 = paddle._C_ops.add_n(combine_103)

        # pd_op.conv2d: (-1x1024x16x16xf32) <- (-1x256x16x16xf32, 1024x256x1x1xf32)
        conv2d_135 = paddle._C_ops.conv2d(add_n_51, parameter_597, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__654, batch_norm__655, batch_norm__656, batch_norm__657, batch_norm__658, batch_norm__659 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_135, parameter_598, parameter_599, parameter_600, parameter_601, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32, -1x1024x16x16xf32)
        add__51 = paddle._C_ops.add_(relu__102, batch_norm__654)

        # pd_op.relu_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32)
        relu__106 = paddle._C_ops.relu_(add__51)

        # pd_op.conv2d: (-1x256x16x16xf32) <- (-1x1024x16x16xf32, 256x1024x1x1xf32)
        conv2d_136 = paddle._C_ops.conv2d(relu__106, parameter_602, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__660, batch_norm__661, batch_norm__662, batch_norm__663, batch_norm__664, batch_norm__665 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_136, parameter_603, parameter_604, parameter_605, parameter_606, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x16x16xf32) <- (-1x256x16x16xf32)
        relu__107 = paddle._C_ops.relu_(batch_norm__660)

        # pd_op.conv2d: (-1x512x16x16xf32) <- (-1x256x16x16xf32, 512x128x3x3xf32)
        conv2d_137 = paddle._C_ops.conv2d(relu__107, parameter_607, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__666, batch_norm__667, batch_norm__668, batch_norm__669, batch_norm__670, batch_norm__671 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_137, parameter_608, parameter_609, parameter_610, parameter_611, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x16x16xf32) <- (-1x512x16x16xf32)
        relu__108 = paddle._C_ops.relu_(batch_norm__666)

        # pd_op.full: (1xi32) <- ()
        full_208 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x512x16x16xf32, 1xi32)
        split_with_num_52 = paddle._C_ops.split_with_num(relu__108, 2, full_208)

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_130 = split_with_num_52[0]

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_131 = split_with_num_52[1]

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_104 = [slice_130, slice_131]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_52 = paddle._C_ops.add_n(combine_104)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_110 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf32) <- (-1x256x16x16xf32, 2xi64)
        pool2d_32 = paddle._C_ops.pool2d(add_n_52, full_int_array_110, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x1x1xf32) <- (-1x256x1x1xf32, 128x256x1x1xf32)
        conv2d_138 = paddle._C_ops.conv2d(pool2d_32, parameter_612, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__672, batch_norm__673, batch_norm__674, batch_norm__675, batch_norm__676, batch_norm__677 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_138, parameter_613, parameter_614, parameter_615, parameter_616, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x1x1xf32) <- (-1x128x1x1xf32)
        relu__109 = paddle._C_ops.relu_(batch_norm__672)

        # pd_op.conv2d: (-1x512x1x1xf32) <- (-1x128x1x1xf32, 512x128x1x1xf32)
        conv2d_139 = paddle._C_ops.conv2d(relu__109, parameter_617, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_111 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32, 0x512xf32) <- (512xf32, 4xi64)
        reshape_52, reshape_53 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_618, full_int_array_111), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x1x1xf32) <- (-1x512x1x1xf32, 1x512x1x1xf32)
        add__52 = paddle._C_ops.add_(conv2d_139, reshape_52)

        # pd_op.shape: (4xi32) <- (-1x512x1x1xf32)
        shape_26 = paddle._C_ops.shape(add__52)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_112 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_113 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_132 = paddle._C_ops.slice(shape_26, [0], full_int_array_112, full_int_array_113, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_209 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_210 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_211 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_105 = [slice_132, full_209, full_210, full_211]

        # pd_op.reshape_: (-1x1x2x256xf32, 0x-1x512x1x1xf32) <- (-1x512x1x1xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__104, reshape__105 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__52, [x.reshape([]) for x in combine_105]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x256xf32) <- (-1x1x2x256xf32)
        transpose_26 = paddle._C_ops.transpose(reshape__104, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x256xf32) <- (-1x2x1x256xf32)
        softmax__26 = paddle._C_ops.softmax_(transpose_26, 1)

        # pd_op.full: (1xi32) <- ()
        full_212 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_213 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_214 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_106 = [slice_132, full_212, full_213, full_214]

        # pd_op.reshape_: (-1x512x1x1xf32, 0x-1x2x1x256xf32) <- (-1x2x1x256xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__106, reshape__107 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__26, [x.reshape([]) for x in combine_106]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_215 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x1x1xf32, -1x256x1x1xf32]) <- (-1x512x1x1xf32, 1xi32)
        split_with_num_53 = paddle._C_ops.split_with_num(reshape__106, 2, full_215)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_133 = split_with_num_53[0]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__52 = paddle._C_ops.multiply_(slice_130, slice_133)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_134 = split_with_num_53[1]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__53 = paddle._C_ops.multiply_(slice_131, slice_134)

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_107 = [multiply__52, multiply__53]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_53 = paddle._C_ops.add_n(combine_107)

        # pd_op.conv2d: (-1x1024x16x16xf32) <- (-1x256x16x16xf32, 1024x256x1x1xf32)
        conv2d_140 = paddle._C_ops.conv2d(add_n_53, parameter_619, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__678, batch_norm__679, batch_norm__680, batch_norm__681, batch_norm__682, batch_norm__683 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_140, parameter_620, parameter_621, parameter_622, parameter_623, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32, -1x1024x16x16xf32)
        add__53 = paddle._C_ops.add_(relu__106, batch_norm__678)

        # pd_op.relu_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32)
        relu__110 = paddle._C_ops.relu_(add__53)

        # pd_op.conv2d: (-1x256x16x16xf32) <- (-1x1024x16x16xf32, 256x1024x1x1xf32)
        conv2d_141 = paddle._C_ops.conv2d(relu__110, parameter_624, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__684, batch_norm__685, batch_norm__686, batch_norm__687, batch_norm__688, batch_norm__689 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_141, parameter_625, parameter_626, parameter_627, parameter_628, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x16x16xf32) <- (-1x256x16x16xf32)
        relu__111 = paddle._C_ops.relu_(batch_norm__684)

        # pd_op.conv2d: (-1x512x16x16xf32) <- (-1x256x16x16xf32, 512x128x3x3xf32)
        conv2d_142 = paddle._C_ops.conv2d(relu__111, parameter_629, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__690, batch_norm__691, batch_norm__692, batch_norm__693, batch_norm__694, batch_norm__695 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_142, parameter_630, parameter_631, parameter_632, parameter_633, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x16x16xf32) <- (-1x512x16x16xf32)
        relu__112 = paddle._C_ops.relu_(batch_norm__690)

        # pd_op.full: (1xi32) <- ()
        full_216 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x512x16x16xf32, 1xi32)
        split_with_num_54 = paddle._C_ops.split_with_num(relu__112, 2, full_216)

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_135 = split_with_num_54[0]

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_136 = split_with_num_54[1]

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_108 = [slice_135, slice_136]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_54 = paddle._C_ops.add_n(combine_108)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_114 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf32) <- (-1x256x16x16xf32, 2xi64)
        pool2d_33 = paddle._C_ops.pool2d(add_n_54, full_int_array_114, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x1x1xf32) <- (-1x256x1x1xf32, 128x256x1x1xf32)
        conv2d_143 = paddle._C_ops.conv2d(pool2d_33, parameter_634, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__696, batch_norm__697, batch_norm__698, batch_norm__699, batch_norm__700, batch_norm__701 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_143, parameter_635, parameter_636, parameter_637, parameter_638, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x1x1xf32) <- (-1x128x1x1xf32)
        relu__113 = paddle._C_ops.relu_(batch_norm__696)

        # pd_op.conv2d: (-1x512x1x1xf32) <- (-1x128x1x1xf32, 512x128x1x1xf32)
        conv2d_144 = paddle._C_ops.conv2d(relu__113, parameter_639, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_115 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32, 0x512xf32) <- (512xf32, 4xi64)
        reshape_54, reshape_55 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_640, full_int_array_115), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x1x1xf32) <- (-1x512x1x1xf32, 1x512x1x1xf32)
        add__54 = paddle._C_ops.add_(conv2d_144, reshape_54)

        # pd_op.shape: (4xi32) <- (-1x512x1x1xf32)
        shape_27 = paddle._C_ops.shape(add__54)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_116 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_117 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_137 = paddle._C_ops.slice(shape_27, [0], full_int_array_116, full_int_array_117, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_217 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_218 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_219 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_109 = [slice_137, full_217, full_218, full_219]

        # pd_op.reshape_: (-1x1x2x256xf32, 0x-1x512x1x1xf32) <- (-1x512x1x1xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__108, reshape__109 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__54, [x.reshape([]) for x in combine_109]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x256xf32) <- (-1x1x2x256xf32)
        transpose_27 = paddle._C_ops.transpose(reshape__108, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x256xf32) <- (-1x2x1x256xf32)
        softmax__27 = paddle._C_ops.softmax_(transpose_27, 1)

        # pd_op.full: (1xi32) <- ()
        full_220 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_221 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_222 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_110 = [slice_137, full_220, full_221, full_222]

        # pd_op.reshape_: (-1x512x1x1xf32, 0x-1x2x1x256xf32) <- (-1x2x1x256xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__110, reshape__111 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__27, [x.reshape([]) for x in combine_110]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_223 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x1x1xf32, -1x256x1x1xf32]) <- (-1x512x1x1xf32, 1xi32)
        split_with_num_55 = paddle._C_ops.split_with_num(reshape__110, 2, full_223)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_138 = split_with_num_55[0]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__54 = paddle._C_ops.multiply_(slice_135, slice_138)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_139 = split_with_num_55[1]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__55 = paddle._C_ops.multiply_(slice_136, slice_139)

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_111 = [multiply__54, multiply__55]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_55 = paddle._C_ops.add_n(combine_111)

        # pd_op.conv2d: (-1x1024x16x16xf32) <- (-1x256x16x16xf32, 1024x256x1x1xf32)
        conv2d_145 = paddle._C_ops.conv2d(add_n_55, parameter_641, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__702, batch_norm__703, batch_norm__704, batch_norm__705, batch_norm__706, batch_norm__707 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_145, parameter_642, parameter_643, parameter_644, parameter_645, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32, -1x1024x16x16xf32)
        add__55 = paddle._C_ops.add_(relu__110, batch_norm__702)

        # pd_op.relu_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32)
        relu__114 = paddle._C_ops.relu_(add__55)

        # pd_op.conv2d: (-1x256x16x16xf32) <- (-1x1024x16x16xf32, 256x1024x1x1xf32)
        conv2d_146 = paddle._C_ops.conv2d(relu__114, parameter_646, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__708, batch_norm__709, batch_norm__710, batch_norm__711, batch_norm__712, batch_norm__713 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_146, parameter_647, parameter_648, parameter_649, parameter_650, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x16x16xf32) <- (-1x256x16x16xf32)
        relu__115 = paddle._C_ops.relu_(batch_norm__708)

        # pd_op.conv2d: (-1x512x16x16xf32) <- (-1x256x16x16xf32, 512x128x3x3xf32)
        conv2d_147 = paddle._C_ops.conv2d(relu__115, parameter_651, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__714, batch_norm__715, batch_norm__716, batch_norm__717, batch_norm__718, batch_norm__719 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_147, parameter_652, parameter_653, parameter_654, parameter_655, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x16x16xf32) <- (-1x512x16x16xf32)
        relu__116 = paddle._C_ops.relu_(batch_norm__714)

        # pd_op.full: (1xi32) <- ()
        full_224 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x512x16x16xf32, 1xi32)
        split_with_num_56 = paddle._C_ops.split_with_num(relu__116, 2, full_224)

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_140 = split_with_num_56[0]

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_141 = split_with_num_56[1]

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_112 = [slice_140, slice_141]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_56 = paddle._C_ops.add_n(combine_112)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_118 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf32) <- (-1x256x16x16xf32, 2xi64)
        pool2d_34 = paddle._C_ops.pool2d(add_n_56, full_int_array_118, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x1x1xf32) <- (-1x256x1x1xf32, 128x256x1x1xf32)
        conv2d_148 = paddle._C_ops.conv2d(pool2d_34, parameter_656, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__720, batch_norm__721, batch_norm__722, batch_norm__723, batch_norm__724, batch_norm__725 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_148, parameter_657, parameter_658, parameter_659, parameter_660, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x1x1xf32) <- (-1x128x1x1xf32)
        relu__117 = paddle._C_ops.relu_(batch_norm__720)

        # pd_op.conv2d: (-1x512x1x1xf32) <- (-1x128x1x1xf32, 512x128x1x1xf32)
        conv2d_149 = paddle._C_ops.conv2d(relu__117, parameter_661, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_119 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32, 0x512xf32) <- (512xf32, 4xi64)
        reshape_56, reshape_57 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_662, full_int_array_119), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x1x1xf32) <- (-1x512x1x1xf32, 1x512x1x1xf32)
        add__56 = paddle._C_ops.add_(conv2d_149, reshape_56)

        # pd_op.shape: (4xi32) <- (-1x512x1x1xf32)
        shape_28 = paddle._C_ops.shape(add__56)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_120 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_121 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_142 = paddle._C_ops.slice(shape_28, [0], full_int_array_120, full_int_array_121, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_225 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_226 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_227 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_113 = [slice_142, full_225, full_226, full_227]

        # pd_op.reshape_: (-1x1x2x256xf32, 0x-1x512x1x1xf32) <- (-1x512x1x1xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__112, reshape__113 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__56, [x.reshape([]) for x in combine_113]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x256xf32) <- (-1x1x2x256xf32)
        transpose_28 = paddle._C_ops.transpose(reshape__112, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x256xf32) <- (-1x2x1x256xf32)
        softmax__28 = paddle._C_ops.softmax_(transpose_28, 1)

        # pd_op.full: (1xi32) <- ()
        full_228 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_229 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_230 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_114 = [slice_142, full_228, full_229, full_230]

        # pd_op.reshape_: (-1x512x1x1xf32, 0x-1x2x1x256xf32) <- (-1x2x1x256xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__114, reshape__115 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__28, [x.reshape([]) for x in combine_114]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_231 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x1x1xf32, -1x256x1x1xf32]) <- (-1x512x1x1xf32, 1xi32)
        split_with_num_57 = paddle._C_ops.split_with_num(reshape__114, 2, full_231)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_143 = split_with_num_57[0]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__56 = paddle._C_ops.multiply_(slice_140, slice_143)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_144 = split_with_num_57[1]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__57 = paddle._C_ops.multiply_(slice_141, slice_144)

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_115 = [multiply__56, multiply__57]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_57 = paddle._C_ops.add_n(combine_115)

        # pd_op.conv2d: (-1x1024x16x16xf32) <- (-1x256x16x16xf32, 1024x256x1x1xf32)
        conv2d_150 = paddle._C_ops.conv2d(add_n_57, parameter_663, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__726, batch_norm__727, batch_norm__728, batch_norm__729, batch_norm__730, batch_norm__731 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_150, parameter_664, parameter_665, parameter_666, parameter_667, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32, -1x1024x16x16xf32)
        add__57 = paddle._C_ops.add_(relu__114, batch_norm__726)

        # pd_op.relu_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32)
        relu__118 = paddle._C_ops.relu_(add__57)

        # pd_op.conv2d: (-1x256x16x16xf32) <- (-1x1024x16x16xf32, 256x1024x1x1xf32)
        conv2d_151 = paddle._C_ops.conv2d(relu__118, parameter_668, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x16x16xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__732, batch_norm__733, batch_norm__734, batch_norm__735, batch_norm__736, batch_norm__737 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_151, parameter_669, parameter_670, parameter_671, parameter_672, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x16x16xf32) <- (-1x256x16x16xf32)
        relu__119 = paddle._C_ops.relu_(batch_norm__732)

        # pd_op.conv2d: (-1x512x16x16xf32) <- (-1x256x16x16xf32, 512x128x3x3xf32)
        conv2d_152 = paddle._C_ops.conv2d(relu__119, parameter_673, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__738, batch_norm__739, batch_norm__740, batch_norm__741, batch_norm__742, batch_norm__743 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_152, parameter_674, parameter_675, parameter_676, parameter_677, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x16x16xf32) <- (-1x512x16x16xf32)
        relu__120 = paddle._C_ops.relu_(batch_norm__738)

        # pd_op.full: (1xi32) <- ()
        full_232 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x512x16x16xf32, 1xi32)
        split_with_num_58 = paddle._C_ops.split_with_num(relu__120, 2, full_232)

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_145 = split_with_num_58[0]

        # builtin.slice: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        slice_146 = split_with_num_58[1]

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_116 = [slice_145, slice_146]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_58 = paddle._C_ops.add_n(combine_116)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_122 = [1, 1]

        # pd_op.pool2d: (-1x256x1x1xf32) <- (-1x256x16x16xf32, 2xi64)
        pool2d_35 = paddle._C_ops.pool2d(add_n_58, full_int_array_122, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x1x1xf32) <- (-1x256x1x1xf32, 128x256x1x1xf32)
        conv2d_153 = paddle._C_ops.conv2d(pool2d_35, parameter_678, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x1x1xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__744, batch_norm__745, batch_norm__746, batch_norm__747, batch_norm__748, batch_norm__749 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_153, parameter_679, parameter_680, parameter_681, parameter_682, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x1x1xf32) <- (-1x128x1x1xf32)
        relu__121 = paddle._C_ops.relu_(batch_norm__744)

        # pd_op.conv2d: (-1x512x1x1xf32) <- (-1x128x1x1xf32, 512x128x1x1xf32)
        conv2d_154 = paddle._C_ops.conv2d(relu__121, parameter_683, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_123 = [1, 512, 1, 1]

        # pd_op.reshape: (1x512x1x1xf32, 0x512xf32) <- (512xf32, 4xi64)
        reshape_58, reshape_59 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_684, full_int_array_123), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x512x1x1xf32) <- (-1x512x1x1xf32, 1x512x1x1xf32)
        add__58 = paddle._C_ops.add_(conv2d_154, reshape_58)

        # pd_op.shape: (4xi32) <- (-1x512x1x1xf32)
        shape_29 = paddle._C_ops.shape(add__58)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_124 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_125 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_147 = paddle._C_ops.slice(shape_29, [0], full_int_array_124, full_int_array_125, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_233 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_234 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_235 = paddle._C_ops.full([1], float('256'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_117 = [slice_147, full_233, full_234, full_235]

        # pd_op.reshape_: (-1x1x2x256xf32, 0x-1x512x1x1xf32) <- (-1x512x1x1xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__116, reshape__117 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__58, [x.reshape([]) for x in combine_117]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x256xf32) <- (-1x1x2x256xf32)
        transpose_29 = paddle._C_ops.transpose(reshape__116, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x256xf32) <- (-1x2x1x256xf32)
        softmax__29 = paddle._C_ops.softmax_(transpose_29, 1)

        # pd_op.full: (1xi32) <- ()
        full_236 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_237 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_238 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_118 = [slice_147, full_236, full_237, full_238]

        # pd_op.reshape_: (-1x512x1x1xf32, 0x-1x2x1x256xf32) <- (-1x2x1x256xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__118, reshape__119 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__29, [x.reshape([]) for x in combine_118]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_239 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x256x1x1xf32, -1x256x1x1xf32]) <- (-1x512x1x1xf32, 1xi32)
        split_with_num_59 = paddle._C_ops.split_with_num(reshape__118, 2, full_239)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_148 = split_with_num_59[0]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__58 = paddle._C_ops.multiply_(slice_145, slice_148)

        # builtin.slice: (-1x256x1x1xf32) <- ([-1x256x1x1xf32, -1x256x1x1xf32])
        slice_149 = split_with_num_59[1]

        # pd_op.multiply_: (-1x256x16x16xf32) <- (-1x256x16x16xf32, -1x256x1x1xf32)
        multiply__59 = paddle._C_ops.multiply_(slice_146, slice_149)

        # builtin.combine: ([-1x256x16x16xf32, -1x256x16x16xf32]) <- (-1x256x16x16xf32, -1x256x16x16xf32)
        combine_119 = [multiply__58, multiply__59]

        # pd_op.add_n: (-1x256x16x16xf32) <- ([-1x256x16x16xf32, -1x256x16x16xf32])
        add_n_59 = paddle._C_ops.add_n(combine_119)

        # pd_op.conv2d: (-1x1024x16x16xf32) <- (-1x256x16x16xf32, 1024x256x1x1xf32)
        conv2d_155 = paddle._C_ops.conv2d(add_n_59, parameter_685, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__750, batch_norm__751, batch_norm__752, batch_norm__753, batch_norm__754, batch_norm__755 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_155, parameter_686, parameter_687, parameter_688, parameter_689, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32, -1x1024x16x16xf32)
        add__59 = paddle._C_ops.add_(relu__118, batch_norm__750)

        # pd_op.relu_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32)
        relu__122 = paddle._C_ops.relu_(add__59)

        # pd_op.conv2d: (-1x512x16x16xf32) <- (-1x1024x16x16xf32, 512x1024x1x1xf32)
        conv2d_156 = paddle._C_ops.conv2d(relu__122, parameter_690, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x16x16xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__756, batch_norm__757, batch_norm__758, batch_norm__759, batch_norm__760, batch_norm__761 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_156, parameter_691, parameter_692, parameter_693, parameter_694, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x16x16xf32) <- (-1x512x16x16xf32)
        relu__123 = paddle._C_ops.relu_(batch_norm__756)

        # pd_op.conv2d: (-1x1024x16x16xf32) <- (-1x512x16x16xf32, 1024x256x3x3xf32)
        conv2d_157 = paddle._C_ops.conv2d(relu__123, parameter_695, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x16x16xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__762, batch_norm__763, batch_norm__764, batch_norm__765, batch_norm__766, batch_norm__767 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_157, parameter_696, parameter_697, parameter_698, parameter_699, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1024x16x16xf32) <- (-1x1024x16x16xf32)
        relu__124 = paddle._C_ops.relu_(batch_norm__762)

        # pd_op.full: (1xi32) <- ()
        full_240 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512x16x16xf32, -1x512x16x16xf32]) <- (-1x1024x16x16xf32, 1xi32)
        split_with_num_60 = paddle._C_ops.split_with_num(relu__124, 2, full_240)

        # builtin.slice: (-1x512x16x16xf32) <- ([-1x512x16x16xf32, -1x512x16x16xf32])
        slice_150 = split_with_num_60[0]

        # builtin.slice: (-1x512x16x16xf32) <- ([-1x512x16x16xf32, -1x512x16x16xf32])
        slice_151 = split_with_num_60[1]

        # builtin.combine: ([-1x512x16x16xf32, -1x512x16x16xf32]) <- (-1x512x16x16xf32, -1x512x16x16xf32)
        combine_120 = [slice_150, slice_151]

        # pd_op.add_n: (-1x512x16x16xf32) <- ([-1x512x16x16xf32, -1x512x16x16xf32])
        add_n_60 = paddle._C_ops.add_n(combine_120)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_126 = [1, 1]

        # pd_op.pool2d: (-1x512x1x1xf32) <- (-1x512x16x16xf32, 2xi64)
        pool2d_36 = paddle._C_ops.pool2d(add_n_60, full_int_array_126, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x256x1x1xf32) <- (-1x512x1x1xf32, 256x512x1x1xf32)
        conv2d_158 = paddle._C_ops.conv2d(pool2d_36, parameter_700, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x1x1xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x1x1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__768, batch_norm__769, batch_norm__770, batch_norm__771, batch_norm__772, batch_norm__773 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_158, parameter_701, parameter_702, parameter_703, parameter_704, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x1x1xf32) <- (-1x256x1x1xf32)
        relu__125 = paddle._C_ops.relu_(batch_norm__768)

        # pd_op.conv2d: (-1x1024x1x1xf32) <- (-1x256x1x1xf32, 1024x256x1x1xf32)
        conv2d_159 = paddle._C_ops.conv2d(relu__125, parameter_705, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_127 = [1, 1024, 1, 1]

        # pd_op.reshape: (1x1024x1x1xf32, 0x1024xf32) <- (1024xf32, 4xi64)
        reshape_60, reshape_61 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_706, full_int_array_127), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1024x1x1xf32) <- (-1x1024x1x1xf32, 1x1024x1x1xf32)
        add__60 = paddle._C_ops.add_(conv2d_159, reshape_60)

        # pd_op.shape: (4xi32) <- (-1x1024x1x1xf32)
        shape_30 = paddle._C_ops.shape(add__60)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_128 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_129 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_152 = paddle._C_ops.slice(shape_30, [0], full_int_array_128, full_int_array_129, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_241 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_242 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_243 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_121 = [slice_152, full_241, full_242, full_243]

        # pd_op.reshape_: (-1x1x2x512xf32, 0x-1x1024x1x1xf32) <- (-1x1024x1x1xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__120, reshape__121 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__60, [x.reshape([]) for x in combine_121]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x512xf32) <- (-1x1x2x512xf32)
        transpose_30 = paddle._C_ops.transpose(reshape__120, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x512xf32) <- (-1x2x1x512xf32)
        softmax__30 = paddle._C_ops.softmax_(transpose_30, 1)

        # pd_op.full: (1xi32) <- ()
        full_244 = paddle._C_ops.full([1], float('1024'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_245 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_246 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_122 = [slice_152, full_244, full_245, full_246]

        # pd_op.reshape_: (-1x1024x1x1xf32, 0x-1x2x1x512xf32) <- (-1x2x1x512xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__122, reshape__123 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__30, [x.reshape([]) for x in combine_122]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_247 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512x1x1xf32, -1x512x1x1xf32]) <- (-1x1024x1x1xf32, 1xi32)
        split_with_num_61 = paddle._C_ops.split_with_num(reshape__122, 2, full_247)

        # builtin.slice: (-1x512x1x1xf32) <- ([-1x512x1x1xf32, -1x512x1x1xf32])
        slice_153 = split_with_num_61[0]

        # pd_op.multiply_: (-1x512x16x16xf32) <- (-1x512x16x16xf32, -1x512x1x1xf32)
        multiply__60 = paddle._C_ops.multiply_(slice_150, slice_153)

        # builtin.slice: (-1x512x1x1xf32) <- ([-1x512x1x1xf32, -1x512x1x1xf32])
        slice_154 = split_with_num_61[1]

        # pd_op.multiply_: (-1x512x16x16xf32) <- (-1x512x16x16xf32, -1x512x1x1xf32)
        multiply__61 = paddle._C_ops.multiply_(slice_151, slice_154)

        # builtin.combine: ([-1x512x16x16xf32, -1x512x16x16xf32]) <- (-1x512x16x16xf32, -1x512x16x16xf32)
        combine_123 = [multiply__60, multiply__61]

        # pd_op.add_n: (-1x512x16x16xf32) <- ([-1x512x16x16xf32, -1x512x16x16xf32])
        add_n_61 = paddle._C_ops.add_n(combine_123)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_130 = [3, 3]

        # pd_op.pool2d: (-1x512x8x8xf32) <- (-1x512x16x16xf32, 2xi64)
        pool2d_37 = paddle._C_ops.pool2d(add_n_61, full_int_array_130, [2, 2], [1, 1], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x2048x8x8xf32) <- (-1x512x8x8xf32, 2048x512x1x1xf32)
        conv2d_160 = paddle._C_ops.conv2d(pool2d_37, parameter_707, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x2048x8x8xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, None) <- (-1x2048x8x8xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        batch_norm__774, batch_norm__775, batch_norm__776, batch_norm__777, batch_norm__778, batch_norm__779 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_160, parameter_708, parameter_709, parameter_710, parameter_711, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_131 = [2, 2]

        # pd_op.pool2d: (-1x1024x8x8xf32) <- (-1x1024x16x16xf32, 2xi64)
        pool2d_38 = paddle._C_ops.pool2d(relu__122, full_int_array_131, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x2048x8x8xf32) <- (-1x1024x8x8xf32, 2048x1024x1x1xf32)
        conv2d_161 = paddle._C_ops.conv2d(pool2d_38, parameter_712, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x2048x8x8xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, None) <- (-1x2048x8x8xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        batch_norm__780, batch_norm__781, batch_norm__782, batch_norm__783, batch_norm__784, batch_norm__785 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_161, parameter_713, parameter_714, parameter_715, parameter_716, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x2048x8x8xf32) <- (-1x2048x8x8xf32, -1x2048x8x8xf32)
        add__61 = paddle._C_ops.add_(batch_norm__780, batch_norm__774)

        # pd_op.relu_: (-1x2048x8x8xf32) <- (-1x2048x8x8xf32)
        relu__126 = paddle._C_ops.relu_(add__61)

        # pd_op.conv2d: (-1x512x8x8xf32) <- (-1x2048x8x8xf32, 512x2048x1x1xf32)
        conv2d_162 = paddle._C_ops.conv2d(relu__126, parameter_717, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x8x8xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x8x8xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__786, batch_norm__787, batch_norm__788, batch_norm__789, batch_norm__790, batch_norm__791 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_162, parameter_718, parameter_719, parameter_720, parameter_721, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x8x8xf32) <- (-1x512x8x8xf32)
        relu__127 = paddle._C_ops.relu_(batch_norm__786)

        # pd_op.conv2d: (-1x1024x8x8xf32) <- (-1x512x8x8xf32, 1024x256x3x3xf32)
        conv2d_163 = paddle._C_ops.conv2d(relu__127, parameter_722, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x8x8xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x8x8xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__792, batch_norm__793, batch_norm__794, batch_norm__795, batch_norm__796, batch_norm__797 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_163, parameter_723, parameter_724, parameter_725, parameter_726, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1024x8x8xf32) <- (-1x1024x8x8xf32)
        relu__128 = paddle._C_ops.relu_(batch_norm__792)

        # pd_op.full: (1xi32) <- ()
        full_248 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512x8x8xf32, -1x512x8x8xf32]) <- (-1x1024x8x8xf32, 1xi32)
        split_with_num_62 = paddle._C_ops.split_with_num(relu__128, 2, full_248)

        # builtin.slice: (-1x512x8x8xf32) <- ([-1x512x8x8xf32, -1x512x8x8xf32])
        slice_155 = split_with_num_62[0]

        # builtin.slice: (-1x512x8x8xf32) <- ([-1x512x8x8xf32, -1x512x8x8xf32])
        slice_156 = split_with_num_62[1]

        # builtin.combine: ([-1x512x8x8xf32, -1x512x8x8xf32]) <- (-1x512x8x8xf32, -1x512x8x8xf32)
        combine_124 = [slice_155, slice_156]

        # pd_op.add_n: (-1x512x8x8xf32) <- ([-1x512x8x8xf32, -1x512x8x8xf32])
        add_n_62 = paddle._C_ops.add_n(combine_124)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_132 = [1, 1]

        # pd_op.pool2d: (-1x512x1x1xf32) <- (-1x512x8x8xf32, 2xi64)
        pool2d_39 = paddle._C_ops.pool2d(add_n_62, full_int_array_132, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x256x1x1xf32) <- (-1x512x1x1xf32, 256x512x1x1xf32)
        conv2d_164 = paddle._C_ops.conv2d(pool2d_39, parameter_727, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x1x1xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x1x1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__798, batch_norm__799, batch_norm__800, batch_norm__801, batch_norm__802, batch_norm__803 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_164, parameter_728, parameter_729, parameter_730, parameter_731, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x1x1xf32) <- (-1x256x1x1xf32)
        relu__129 = paddle._C_ops.relu_(batch_norm__798)

        # pd_op.conv2d: (-1x1024x1x1xf32) <- (-1x256x1x1xf32, 1024x256x1x1xf32)
        conv2d_165 = paddle._C_ops.conv2d(relu__129, parameter_732, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_133 = [1, 1024, 1, 1]

        # pd_op.reshape: (1x1024x1x1xf32, 0x1024xf32) <- (1024xf32, 4xi64)
        reshape_62, reshape_63 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_733, full_int_array_133), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1024x1x1xf32) <- (-1x1024x1x1xf32, 1x1024x1x1xf32)
        add__62 = paddle._C_ops.add_(conv2d_165, reshape_62)

        # pd_op.shape: (4xi32) <- (-1x1024x1x1xf32)
        shape_31 = paddle._C_ops.shape(add__62)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_134 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_135 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_157 = paddle._C_ops.slice(shape_31, [0], full_int_array_134, full_int_array_135, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_249 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_250 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_251 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_125 = [slice_157, full_249, full_250, full_251]

        # pd_op.reshape_: (-1x1x2x512xf32, 0x-1x1024x1x1xf32) <- (-1x1024x1x1xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__124, reshape__125 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__62, [x.reshape([]) for x in combine_125]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x512xf32) <- (-1x1x2x512xf32)
        transpose_31 = paddle._C_ops.transpose(reshape__124, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x512xf32) <- (-1x2x1x512xf32)
        softmax__31 = paddle._C_ops.softmax_(transpose_31, 1)

        # pd_op.full: (1xi32) <- ()
        full_252 = paddle._C_ops.full([1], float('1024'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_253 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_254 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_126 = [slice_157, full_252, full_253, full_254]

        # pd_op.reshape_: (-1x1024x1x1xf32, 0x-1x2x1x512xf32) <- (-1x2x1x512xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__126, reshape__127 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__31, [x.reshape([]) for x in combine_126]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_255 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512x1x1xf32, -1x512x1x1xf32]) <- (-1x1024x1x1xf32, 1xi32)
        split_with_num_63 = paddle._C_ops.split_with_num(reshape__126, 2, full_255)

        # builtin.slice: (-1x512x1x1xf32) <- ([-1x512x1x1xf32, -1x512x1x1xf32])
        slice_158 = split_with_num_63[0]

        # pd_op.multiply_: (-1x512x8x8xf32) <- (-1x512x8x8xf32, -1x512x1x1xf32)
        multiply__62 = paddle._C_ops.multiply_(slice_155, slice_158)

        # builtin.slice: (-1x512x1x1xf32) <- ([-1x512x1x1xf32, -1x512x1x1xf32])
        slice_159 = split_with_num_63[1]

        # pd_op.multiply_: (-1x512x8x8xf32) <- (-1x512x8x8xf32, -1x512x1x1xf32)
        multiply__63 = paddle._C_ops.multiply_(slice_156, slice_159)

        # builtin.combine: ([-1x512x8x8xf32, -1x512x8x8xf32]) <- (-1x512x8x8xf32, -1x512x8x8xf32)
        combine_127 = [multiply__62, multiply__63]

        # pd_op.add_n: (-1x512x8x8xf32) <- ([-1x512x8x8xf32, -1x512x8x8xf32])
        add_n_63 = paddle._C_ops.add_n(combine_127)

        # pd_op.conv2d: (-1x2048x8x8xf32) <- (-1x512x8x8xf32, 2048x512x1x1xf32)
        conv2d_166 = paddle._C_ops.conv2d(add_n_63, parameter_734, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x2048x8x8xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, None) <- (-1x2048x8x8xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        batch_norm__804, batch_norm__805, batch_norm__806, batch_norm__807, batch_norm__808, batch_norm__809 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_166, parameter_735, parameter_736, parameter_737, parameter_738, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x2048x8x8xf32) <- (-1x2048x8x8xf32, -1x2048x8x8xf32)
        add__63 = paddle._C_ops.add_(relu__126, batch_norm__804)

        # pd_op.relu_: (-1x2048x8x8xf32) <- (-1x2048x8x8xf32)
        relu__130 = paddle._C_ops.relu_(add__63)

        # pd_op.conv2d: (-1x512x8x8xf32) <- (-1x2048x8x8xf32, 512x2048x1x1xf32)
        conv2d_167 = paddle._C_ops.conv2d(relu__130, parameter_739, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x8x8xf32, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x8x8xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__810, batch_norm__811, batch_norm__812, batch_norm__813, batch_norm__814, batch_norm__815 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_167, parameter_740, parameter_741, parameter_742, parameter_743, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x8x8xf32) <- (-1x512x8x8xf32)
        relu__131 = paddle._C_ops.relu_(batch_norm__810)

        # pd_op.conv2d: (-1x1024x8x8xf32) <- (-1x512x8x8xf32, 1024x256x3x3xf32)
        conv2d_168 = paddle._C_ops.conv2d(relu__131, parameter_744, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 2, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x8x8xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x8x8xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__816, batch_norm__817, batch_norm__818, batch_norm__819, batch_norm__820, batch_norm__821 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_168, parameter_745, parameter_746, parameter_747, parameter_748, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1024x8x8xf32) <- (-1x1024x8x8xf32)
        relu__132 = paddle._C_ops.relu_(batch_norm__816)

        # pd_op.full: (1xi32) <- ()
        full_256 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512x8x8xf32, -1x512x8x8xf32]) <- (-1x1024x8x8xf32, 1xi32)
        split_with_num_64 = paddle._C_ops.split_with_num(relu__132, 2, full_256)

        # builtin.slice: (-1x512x8x8xf32) <- ([-1x512x8x8xf32, -1x512x8x8xf32])
        slice_160 = split_with_num_64[0]

        # builtin.slice: (-1x512x8x8xf32) <- ([-1x512x8x8xf32, -1x512x8x8xf32])
        slice_161 = split_with_num_64[1]

        # builtin.combine: ([-1x512x8x8xf32, -1x512x8x8xf32]) <- (-1x512x8x8xf32, -1x512x8x8xf32)
        combine_128 = [slice_160, slice_161]

        # pd_op.add_n: (-1x512x8x8xf32) <- ([-1x512x8x8xf32, -1x512x8x8xf32])
        add_n_64 = paddle._C_ops.add_n(combine_128)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_136 = [1, 1]

        # pd_op.pool2d: (-1x512x1x1xf32) <- (-1x512x8x8xf32, 2xi64)
        pool2d_40 = paddle._C_ops.pool2d(add_n_64, full_int_array_136, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x256x1x1xf32) <- (-1x512x1x1xf32, 256x512x1x1xf32)
        conv2d_169 = paddle._C_ops.conv2d(pool2d_40, parameter_749, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x1x1xf32, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x1x1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__822, batch_norm__823, batch_norm__824, batch_norm__825, batch_norm__826, batch_norm__827 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_169, parameter_750, parameter_751, parameter_752, parameter_753, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x1x1xf32) <- (-1x256x1x1xf32)
        relu__133 = paddle._C_ops.relu_(batch_norm__822)

        # pd_op.conv2d: (-1x1024x1x1xf32) <- (-1x256x1x1xf32, 1024x256x1x1xf32)
        conv2d_170 = paddle._C_ops.conv2d(relu__133, parameter_754, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_137 = [1, 1024, 1, 1]

        # pd_op.reshape: (1x1024x1x1xf32, 0x1024xf32) <- (1024xf32, 4xi64)
        reshape_64, reshape_65 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_755, full_int_array_137), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1024x1x1xf32) <- (-1x1024x1x1xf32, 1x1024x1x1xf32)
        add__64 = paddle._C_ops.add_(conv2d_170, reshape_64)

        # pd_op.shape: (4xi32) <- (-1x1024x1x1xf32)
        shape_32 = paddle._C_ops.shape(add__64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_138 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_139 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_162 = paddle._C_ops.slice(shape_32, [0], full_int_array_138, full_int_array_139, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_257 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_258 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_259 = paddle._C_ops.full([1], float('512'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_129 = [slice_162, full_257, full_258, full_259]

        # pd_op.reshape_: (-1x1x2x512xf32, 0x-1x1024x1x1xf32) <- (-1x1024x1x1xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__128, reshape__129 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__64, [x.reshape([]) for x in combine_129]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x2x1x512xf32) <- (-1x1x2x512xf32)
        transpose_32 = paddle._C_ops.transpose(reshape__128, [0, 2, 1, 3])

        # pd_op.softmax_: (-1x2x1x512xf32) <- (-1x2x1x512xf32)
        softmax__32 = paddle._C_ops.softmax_(transpose_32, 1)

        # pd_op.full: (1xi32) <- ()
        full_260 = paddle._C_ops.full([1], float('1024'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_261 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_262 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_130 = [slice_162, full_260, full_261, full_262]

        # pd_op.reshape_: (-1x1024x1x1xf32, 0x-1x2x1x512xf32) <- (-1x2x1x512xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__130, reshape__131 = (lambda x, f: f(x))(paddle._C_ops.reshape_(softmax__32, [x.reshape([]) for x in combine_130]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_263 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x512x1x1xf32, -1x512x1x1xf32]) <- (-1x1024x1x1xf32, 1xi32)
        split_with_num_65 = paddle._C_ops.split_with_num(reshape__130, 2, full_263)

        # builtin.slice: (-1x512x1x1xf32) <- ([-1x512x1x1xf32, -1x512x1x1xf32])
        slice_163 = split_with_num_65[0]

        # pd_op.multiply_: (-1x512x8x8xf32) <- (-1x512x8x8xf32, -1x512x1x1xf32)
        multiply__64 = paddle._C_ops.multiply_(slice_160, slice_163)

        # builtin.slice: (-1x512x1x1xf32) <- ([-1x512x1x1xf32, -1x512x1x1xf32])
        slice_164 = split_with_num_65[1]

        # pd_op.multiply_: (-1x512x8x8xf32) <- (-1x512x8x8xf32, -1x512x1x1xf32)
        multiply__65 = paddle._C_ops.multiply_(slice_161, slice_164)

        # builtin.combine: ([-1x512x8x8xf32, -1x512x8x8xf32]) <- (-1x512x8x8xf32, -1x512x8x8xf32)
        combine_131 = [multiply__64, multiply__65]

        # pd_op.add_n: (-1x512x8x8xf32) <- ([-1x512x8x8xf32, -1x512x8x8xf32])
        add_n_65 = paddle._C_ops.add_n(combine_131)

        # pd_op.conv2d: (-1x2048x8x8xf32) <- (-1x512x8x8xf32, 2048x512x1x1xf32)
        conv2d_171 = paddle._C_ops.conv2d(add_n_65, parameter_756, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x2048x8x8xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32, None) <- (-1x2048x8x8xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        batch_norm__828, batch_norm__829, batch_norm__830, batch_norm__831, batch_norm__832, batch_norm__833 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_171, parameter_757, parameter_758, parameter_759, parameter_760, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x2048x8x8xf32) <- (-1x2048x8x8xf32, -1x2048x8x8xf32)
        add__65 = paddle._C_ops.add_(relu__130, batch_norm__828)

        # pd_op.relu_: (-1x2048x8x8xf32) <- (-1x2048x8x8xf32)
        relu__134 = paddle._C_ops.relu_(add__65)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_140 = [1, 1]

        # pd_op.pool2d: (-1x2048x1x1xf32) <- (-1x2048x8x8xf32, 2xi64)
        pool2d_41 = paddle._C_ops.pool2d(relu__134, full_int_array_140, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_141 = [-1, 2048]

        # pd_op.reshape_: (-1x2048xf32, 0x-1x2048x1x1xf32) <- (-1x2048x1x1xf32, 2xi64)
        reshape__132, reshape__133 = (lambda x, f: f(x))(paddle._C_ops.reshape_(pool2d_41, full_int_array_141), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x1000xf32) <- (-1x2048xf32, 2048x1000xf32)
        matmul_0 = paddle._C_ops.matmul(reshape__132, parameter_761, False, False)

        # pd_op.add_: (-1x1000xf32) <- (-1x1000xf32, 1000xf32)
        add__66 = paddle._C_ops.add_(matmul_0, parameter_762)

        # pd_op.softmax_: (-1x1000xf32) <- (-1x1000xf32)
        softmax__33 = paddle._C_ops.softmax_(add__66, -1)
        return softmax__33



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

    def forward(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_31, parameter_32, parameter_36, parameter_33, parameter_35, parameter_34, parameter_37, parameter_41, parameter_38, parameter_40, parameter_39, parameter_42, parameter_46, parameter_43, parameter_45, parameter_44, parameter_47, parameter_51, parameter_48, parameter_50, parameter_49, parameter_52, parameter_56, parameter_53, parameter_55, parameter_54, parameter_57, parameter_58, parameter_59, parameter_63, parameter_60, parameter_62, parameter_61, parameter_64, parameter_68, parameter_65, parameter_67, parameter_66, parameter_69, parameter_73, parameter_70, parameter_72, parameter_71, parameter_74, parameter_78, parameter_75, parameter_77, parameter_76, parameter_79, parameter_80, parameter_81, parameter_85, parameter_82, parameter_84, parameter_83, parameter_86, parameter_90, parameter_87, parameter_89, parameter_88, parameter_91, parameter_95, parameter_92, parameter_94, parameter_93, parameter_96, parameter_100, parameter_97, parameter_99, parameter_98, parameter_101, parameter_102, parameter_103, parameter_107, parameter_104, parameter_106, parameter_105, parameter_108, parameter_112, parameter_109, parameter_111, parameter_110, parameter_113, parameter_117, parameter_114, parameter_116, parameter_115, parameter_118, parameter_122, parameter_119, parameter_121, parameter_120, parameter_123, parameter_127, parameter_124, parameter_126, parameter_125, parameter_128, parameter_129, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_151, parameter_152, parameter_156, parameter_153, parameter_155, parameter_154, parameter_157, parameter_161, parameter_158, parameter_160, parameter_159, parameter_162, parameter_166, parameter_163, parameter_165, parameter_164, parameter_167, parameter_171, parameter_168, parameter_170, parameter_169, parameter_172, parameter_173, parameter_174, parameter_178, parameter_175, parameter_177, parameter_176, parameter_179, parameter_183, parameter_180, parameter_182, parameter_181, parameter_184, parameter_188, parameter_185, parameter_187, parameter_186, parameter_189, parameter_193, parameter_190, parameter_192, parameter_191, parameter_194, parameter_195, parameter_196, parameter_200, parameter_197, parameter_199, parameter_198, parameter_201, parameter_205, parameter_202, parameter_204, parameter_203, parameter_206, parameter_210, parameter_207, parameter_209, parameter_208, parameter_211, parameter_215, parameter_212, parameter_214, parameter_213, parameter_216, parameter_220, parameter_217, parameter_219, parameter_218, parameter_221, parameter_222, parameter_223, parameter_227, parameter_224, parameter_226, parameter_225, parameter_228, parameter_232, parameter_229, parameter_231, parameter_230, parameter_233, parameter_237, parameter_234, parameter_236, parameter_235, parameter_238, parameter_242, parameter_239, parameter_241, parameter_240, parameter_243, parameter_244, parameter_245, parameter_249, parameter_246, parameter_248, parameter_247, parameter_250, parameter_254, parameter_251, parameter_253, parameter_252, parameter_255, parameter_259, parameter_256, parameter_258, parameter_257, parameter_260, parameter_264, parameter_261, parameter_263, parameter_262, parameter_265, parameter_266, parameter_267, parameter_271, parameter_268, parameter_270, parameter_269, parameter_272, parameter_276, parameter_273, parameter_275, parameter_274, parameter_277, parameter_281, parameter_278, parameter_280, parameter_279, parameter_282, parameter_286, parameter_283, parameter_285, parameter_284, parameter_287, parameter_288, parameter_289, parameter_293, parameter_290, parameter_292, parameter_291, parameter_294, parameter_298, parameter_295, parameter_297, parameter_296, parameter_299, parameter_303, parameter_300, parameter_302, parameter_301, parameter_304, parameter_308, parameter_305, parameter_307, parameter_306, parameter_309, parameter_310, parameter_311, parameter_315, parameter_312, parameter_314, parameter_313, parameter_316, parameter_320, parameter_317, parameter_319, parameter_318, parameter_321, parameter_325, parameter_322, parameter_324, parameter_323, parameter_326, parameter_330, parameter_327, parameter_329, parameter_328, parameter_331, parameter_332, parameter_333, parameter_337, parameter_334, parameter_336, parameter_335, parameter_338, parameter_342, parameter_339, parameter_341, parameter_340, parameter_343, parameter_347, parameter_344, parameter_346, parameter_345, parameter_348, parameter_352, parameter_349, parameter_351, parameter_350, parameter_353, parameter_354, parameter_355, parameter_359, parameter_356, parameter_358, parameter_357, parameter_360, parameter_364, parameter_361, parameter_363, parameter_362, parameter_365, parameter_369, parameter_366, parameter_368, parameter_367, parameter_370, parameter_374, parameter_371, parameter_373, parameter_372, parameter_375, parameter_376, parameter_377, parameter_381, parameter_378, parameter_380, parameter_379, parameter_382, parameter_386, parameter_383, parameter_385, parameter_384, parameter_387, parameter_391, parameter_388, parameter_390, parameter_389, parameter_392, parameter_396, parameter_393, parameter_395, parameter_394, parameter_397, parameter_398, parameter_399, parameter_403, parameter_400, parameter_402, parameter_401, parameter_404, parameter_408, parameter_405, parameter_407, parameter_406, parameter_409, parameter_413, parameter_410, parameter_412, parameter_411, parameter_414, parameter_418, parameter_415, parameter_417, parameter_416, parameter_419, parameter_420, parameter_421, parameter_425, parameter_422, parameter_424, parameter_423, parameter_426, parameter_430, parameter_427, parameter_429, parameter_428, parameter_431, parameter_435, parameter_432, parameter_434, parameter_433, parameter_436, parameter_440, parameter_437, parameter_439, parameter_438, parameter_441, parameter_442, parameter_443, parameter_447, parameter_444, parameter_446, parameter_445, parameter_448, parameter_452, parameter_449, parameter_451, parameter_450, parameter_453, parameter_457, parameter_454, parameter_456, parameter_455, parameter_458, parameter_462, parameter_459, parameter_461, parameter_460, parameter_463, parameter_464, parameter_465, parameter_469, parameter_466, parameter_468, parameter_467, parameter_470, parameter_474, parameter_471, parameter_473, parameter_472, parameter_475, parameter_479, parameter_476, parameter_478, parameter_477, parameter_480, parameter_484, parameter_481, parameter_483, parameter_482, parameter_485, parameter_486, parameter_487, parameter_491, parameter_488, parameter_490, parameter_489, parameter_492, parameter_496, parameter_493, parameter_495, parameter_494, parameter_497, parameter_501, parameter_498, parameter_500, parameter_499, parameter_502, parameter_506, parameter_503, parameter_505, parameter_504, parameter_507, parameter_508, parameter_509, parameter_513, parameter_510, parameter_512, parameter_511, parameter_514, parameter_518, parameter_515, parameter_517, parameter_516, parameter_519, parameter_523, parameter_520, parameter_522, parameter_521, parameter_524, parameter_528, parameter_525, parameter_527, parameter_526, parameter_529, parameter_530, parameter_531, parameter_535, parameter_532, parameter_534, parameter_533, parameter_536, parameter_540, parameter_537, parameter_539, parameter_538, parameter_541, parameter_545, parameter_542, parameter_544, parameter_543, parameter_546, parameter_550, parameter_547, parameter_549, parameter_548, parameter_551, parameter_552, parameter_553, parameter_557, parameter_554, parameter_556, parameter_555, parameter_558, parameter_562, parameter_559, parameter_561, parameter_560, parameter_563, parameter_567, parameter_564, parameter_566, parameter_565, parameter_568, parameter_572, parameter_569, parameter_571, parameter_570, parameter_573, parameter_574, parameter_575, parameter_579, parameter_576, parameter_578, parameter_577, parameter_580, parameter_584, parameter_581, parameter_583, parameter_582, parameter_585, parameter_589, parameter_586, parameter_588, parameter_587, parameter_590, parameter_594, parameter_591, parameter_593, parameter_592, parameter_595, parameter_596, parameter_597, parameter_601, parameter_598, parameter_600, parameter_599, parameter_602, parameter_606, parameter_603, parameter_605, parameter_604, parameter_607, parameter_611, parameter_608, parameter_610, parameter_609, parameter_612, parameter_616, parameter_613, parameter_615, parameter_614, parameter_617, parameter_618, parameter_619, parameter_623, parameter_620, parameter_622, parameter_621, parameter_624, parameter_628, parameter_625, parameter_627, parameter_626, parameter_629, parameter_633, parameter_630, parameter_632, parameter_631, parameter_634, parameter_638, parameter_635, parameter_637, parameter_636, parameter_639, parameter_640, parameter_641, parameter_645, parameter_642, parameter_644, parameter_643, parameter_646, parameter_650, parameter_647, parameter_649, parameter_648, parameter_651, parameter_655, parameter_652, parameter_654, parameter_653, parameter_656, parameter_660, parameter_657, parameter_659, parameter_658, parameter_661, parameter_662, parameter_663, parameter_667, parameter_664, parameter_666, parameter_665, parameter_668, parameter_672, parameter_669, parameter_671, parameter_670, parameter_673, parameter_677, parameter_674, parameter_676, parameter_675, parameter_678, parameter_682, parameter_679, parameter_681, parameter_680, parameter_683, parameter_684, parameter_685, parameter_689, parameter_686, parameter_688, parameter_687, parameter_690, parameter_694, parameter_691, parameter_693, parameter_692, parameter_695, parameter_699, parameter_696, parameter_698, parameter_697, parameter_700, parameter_704, parameter_701, parameter_703, parameter_702, parameter_705, parameter_706, parameter_707, parameter_711, parameter_708, parameter_710, parameter_709, parameter_712, parameter_716, parameter_713, parameter_715, parameter_714, parameter_717, parameter_721, parameter_718, parameter_720, parameter_719, parameter_722, parameter_726, parameter_723, parameter_725, parameter_724, parameter_727, parameter_731, parameter_728, parameter_730, parameter_729, parameter_732, parameter_733, parameter_734, parameter_738, parameter_735, parameter_737, parameter_736, parameter_739, parameter_743, parameter_740, parameter_742, parameter_741, parameter_744, parameter_748, parameter_745, parameter_747, parameter_746, parameter_749, parameter_753, parameter_750, parameter_752, parameter_751, parameter_754, parameter_755, parameter_756, parameter_760, parameter_757, parameter_759, parameter_758, parameter_761, parameter_762, feed_0):
        return self.builtin_module_2427_0_0(parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_31, parameter_32, parameter_36, parameter_33, parameter_35, parameter_34, parameter_37, parameter_41, parameter_38, parameter_40, parameter_39, parameter_42, parameter_46, parameter_43, parameter_45, parameter_44, parameter_47, parameter_51, parameter_48, parameter_50, parameter_49, parameter_52, parameter_56, parameter_53, parameter_55, parameter_54, parameter_57, parameter_58, parameter_59, parameter_63, parameter_60, parameter_62, parameter_61, parameter_64, parameter_68, parameter_65, parameter_67, parameter_66, parameter_69, parameter_73, parameter_70, parameter_72, parameter_71, parameter_74, parameter_78, parameter_75, parameter_77, parameter_76, parameter_79, parameter_80, parameter_81, parameter_85, parameter_82, parameter_84, parameter_83, parameter_86, parameter_90, parameter_87, parameter_89, parameter_88, parameter_91, parameter_95, parameter_92, parameter_94, parameter_93, parameter_96, parameter_100, parameter_97, parameter_99, parameter_98, parameter_101, parameter_102, parameter_103, parameter_107, parameter_104, parameter_106, parameter_105, parameter_108, parameter_112, parameter_109, parameter_111, parameter_110, parameter_113, parameter_117, parameter_114, parameter_116, parameter_115, parameter_118, parameter_122, parameter_119, parameter_121, parameter_120, parameter_123, parameter_127, parameter_124, parameter_126, parameter_125, parameter_128, parameter_129, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_151, parameter_152, parameter_156, parameter_153, parameter_155, parameter_154, parameter_157, parameter_161, parameter_158, parameter_160, parameter_159, parameter_162, parameter_166, parameter_163, parameter_165, parameter_164, parameter_167, parameter_171, parameter_168, parameter_170, parameter_169, parameter_172, parameter_173, parameter_174, parameter_178, parameter_175, parameter_177, parameter_176, parameter_179, parameter_183, parameter_180, parameter_182, parameter_181, parameter_184, parameter_188, parameter_185, parameter_187, parameter_186, parameter_189, parameter_193, parameter_190, parameter_192, parameter_191, parameter_194, parameter_195, parameter_196, parameter_200, parameter_197, parameter_199, parameter_198, parameter_201, parameter_205, parameter_202, parameter_204, parameter_203, parameter_206, parameter_210, parameter_207, parameter_209, parameter_208, parameter_211, parameter_215, parameter_212, parameter_214, parameter_213, parameter_216, parameter_220, parameter_217, parameter_219, parameter_218, parameter_221, parameter_222, parameter_223, parameter_227, parameter_224, parameter_226, parameter_225, parameter_228, parameter_232, parameter_229, parameter_231, parameter_230, parameter_233, parameter_237, parameter_234, parameter_236, parameter_235, parameter_238, parameter_242, parameter_239, parameter_241, parameter_240, parameter_243, parameter_244, parameter_245, parameter_249, parameter_246, parameter_248, parameter_247, parameter_250, parameter_254, parameter_251, parameter_253, parameter_252, parameter_255, parameter_259, parameter_256, parameter_258, parameter_257, parameter_260, parameter_264, parameter_261, parameter_263, parameter_262, parameter_265, parameter_266, parameter_267, parameter_271, parameter_268, parameter_270, parameter_269, parameter_272, parameter_276, parameter_273, parameter_275, parameter_274, parameter_277, parameter_281, parameter_278, parameter_280, parameter_279, parameter_282, parameter_286, parameter_283, parameter_285, parameter_284, parameter_287, parameter_288, parameter_289, parameter_293, parameter_290, parameter_292, parameter_291, parameter_294, parameter_298, parameter_295, parameter_297, parameter_296, parameter_299, parameter_303, parameter_300, parameter_302, parameter_301, parameter_304, parameter_308, parameter_305, parameter_307, parameter_306, parameter_309, parameter_310, parameter_311, parameter_315, parameter_312, parameter_314, parameter_313, parameter_316, parameter_320, parameter_317, parameter_319, parameter_318, parameter_321, parameter_325, parameter_322, parameter_324, parameter_323, parameter_326, parameter_330, parameter_327, parameter_329, parameter_328, parameter_331, parameter_332, parameter_333, parameter_337, parameter_334, parameter_336, parameter_335, parameter_338, parameter_342, parameter_339, parameter_341, parameter_340, parameter_343, parameter_347, parameter_344, parameter_346, parameter_345, parameter_348, parameter_352, parameter_349, parameter_351, parameter_350, parameter_353, parameter_354, parameter_355, parameter_359, parameter_356, parameter_358, parameter_357, parameter_360, parameter_364, parameter_361, parameter_363, parameter_362, parameter_365, parameter_369, parameter_366, parameter_368, parameter_367, parameter_370, parameter_374, parameter_371, parameter_373, parameter_372, parameter_375, parameter_376, parameter_377, parameter_381, parameter_378, parameter_380, parameter_379, parameter_382, parameter_386, parameter_383, parameter_385, parameter_384, parameter_387, parameter_391, parameter_388, parameter_390, parameter_389, parameter_392, parameter_396, parameter_393, parameter_395, parameter_394, parameter_397, parameter_398, parameter_399, parameter_403, parameter_400, parameter_402, parameter_401, parameter_404, parameter_408, parameter_405, parameter_407, parameter_406, parameter_409, parameter_413, parameter_410, parameter_412, parameter_411, parameter_414, parameter_418, parameter_415, parameter_417, parameter_416, parameter_419, parameter_420, parameter_421, parameter_425, parameter_422, parameter_424, parameter_423, parameter_426, parameter_430, parameter_427, parameter_429, parameter_428, parameter_431, parameter_435, parameter_432, parameter_434, parameter_433, parameter_436, parameter_440, parameter_437, parameter_439, parameter_438, parameter_441, parameter_442, parameter_443, parameter_447, parameter_444, parameter_446, parameter_445, parameter_448, parameter_452, parameter_449, parameter_451, parameter_450, parameter_453, parameter_457, parameter_454, parameter_456, parameter_455, parameter_458, parameter_462, parameter_459, parameter_461, parameter_460, parameter_463, parameter_464, parameter_465, parameter_469, parameter_466, parameter_468, parameter_467, parameter_470, parameter_474, parameter_471, parameter_473, parameter_472, parameter_475, parameter_479, parameter_476, parameter_478, parameter_477, parameter_480, parameter_484, parameter_481, parameter_483, parameter_482, parameter_485, parameter_486, parameter_487, parameter_491, parameter_488, parameter_490, parameter_489, parameter_492, parameter_496, parameter_493, parameter_495, parameter_494, parameter_497, parameter_501, parameter_498, parameter_500, parameter_499, parameter_502, parameter_506, parameter_503, parameter_505, parameter_504, parameter_507, parameter_508, parameter_509, parameter_513, parameter_510, parameter_512, parameter_511, parameter_514, parameter_518, parameter_515, parameter_517, parameter_516, parameter_519, parameter_523, parameter_520, parameter_522, parameter_521, parameter_524, parameter_528, parameter_525, parameter_527, parameter_526, parameter_529, parameter_530, parameter_531, parameter_535, parameter_532, parameter_534, parameter_533, parameter_536, parameter_540, parameter_537, parameter_539, parameter_538, parameter_541, parameter_545, parameter_542, parameter_544, parameter_543, parameter_546, parameter_550, parameter_547, parameter_549, parameter_548, parameter_551, parameter_552, parameter_553, parameter_557, parameter_554, parameter_556, parameter_555, parameter_558, parameter_562, parameter_559, parameter_561, parameter_560, parameter_563, parameter_567, parameter_564, parameter_566, parameter_565, parameter_568, parameter_572, parameter_569, parameter_571, parameter_570, parameter_573, parameter_574, parameter_575, parameter_579, parameter_576, parameter_578, parameter_577, parameter_580, parameter_584, parameter_581, parameter_583, parameter_582, parameter_585, parameter_589, parameter_586, parameter_588, parameter_587, parameter_590, parameter_594, parameter_591, parameter_593, parameter_592, parameter_595, parameter_596, parameter_597, parameter_601, parameter_598, parameter_600, parameter_599, parameter_602, parameter_606, parameter_603, parameter_605, parameter_604, parameter_607, parameter_611, parameter_608, parameter_610, parameter_609, parameter_612, parameter_616, parameter_613, parameter_615, parameter_614, parameter_617, parameter_618, parameter_619, parameter_623, parameter_620, parameter_622, parameter_621, parameter_624, parameter_628, parameter_625, parameter_627, parameter_626, parameter_629, parameter_633, parameter_630, parameter_632, parameter_631, parameter_634, parameter_638, parameter_635, parameter_637, parameter_636, parameter_639, parameter_640, parameter_641, parameter_645, parameter_642, parameter_644, parameter_643, parameter_646, parameter_650, parameter_647, parameter_649, parameter_648, parameter_651, parameter_655, parameter_652, parameter_654, parameter_653, parameter_656, parameter_660, parameter_657, parameter_659, parameter_658, parameter_661, parameter_662, parameter_663, parameter_667, parameter_664, parameter_666, parameter_665, parameter_668, parameter_672, parameter_669, parameter_671, parameter_670, parameter_673, parameter_677, parameter_674, parameter_676, parameter_675, parameter_678, parameter_682, parameter_679, parameter_681, parameter_680, parameter_683, parameter_684, parameter_685, parameter_689, parameter_686, parameter_688, parameter_687, parameter_690, parameter_694, parameter_691, parameter_693, parameter_692, parameter_695, parameter_699, parameter_696, parameter_698, parameter_697, parameter_700, parameter_704, parameter_701, parameter_703, parameter_702, parameter_705, parameter_706, parameter_707, parameter_711, parameter_708, parameter_710, parameter_709, parameter_712, parameter_716, parameter_713, parameter_715, parameter_714, parameter_717, parameter_721, parameter_718, parameter_720, parameter_719, parameter_722, parameter_726, parameter_723, parameter_725, parameter_724, parameter_727, parameter_731, parameter_728, parameter_730, parameter_729, parameter_732, parameter_733, parameter_734, parameter_738, parameter_735, parameter_737, parameter_736, parameter_739, parameter_743, parameter_740, parameter_742, parameter_741, parameter_744, parameter_748, parameter_745, parameter_747, parameter_746, parameter_749, parameter_753, parameter_750, parameter_752, parameter_751, parameter_754, parameter_755, parameter_756, parameter_760, parameter_757, parameter_759, parameter_758, parameter_761, parameter_762, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_2427_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([64, 3, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([128, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([64, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([128, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([32, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([256, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([128, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([32, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([128, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([32, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([128, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([256, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([64, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([512, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([512, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([128, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([256, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([64, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([512, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([128, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_139
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([256, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([64, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_149
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([512, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_154
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([128, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([256, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([64, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([512, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.uniform([512, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_189
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.uniform([512, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([1024, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_209
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([512, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_215
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_214
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_213
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_217
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_219
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_218
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_221
            paddle.uniform([512, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_223
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_227
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_224
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_225
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_229
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_233
            paddle.uniform([512, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_234
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_238
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_239
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_241
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_243
            paddle.uniform([512, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_244
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_245
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_249
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_246
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_248
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_250
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_254
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_251
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_253
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_252
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_255
            paddle.uniform([512, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_259
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_256
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_258
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_257
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_264
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_261
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_263
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_262
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_265
            paddle.uniform([512, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_266
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_267
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_271
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_268
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_270
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_269
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_272
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_276
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_273
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_275
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_274
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_277
            paddle.uniform([512, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_281
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_278
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_280
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_279
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_282
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_286
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_283
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_285
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_284
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_287
            paddle.uniform([512, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_288
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_289
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_293
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_290
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_292
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_291
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_294
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_298
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_295
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_297
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_296
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_299
            paddle.uniform([512, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_303
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_300
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_302
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_301
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_304
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_308
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_305
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_307
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_306
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_309
            paddle.uniform([512, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_310
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_311
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_315
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_312
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_314
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_313
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_316
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_320
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_317
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_319
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_318
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_321
            paddle.uniform([512, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_325
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_322
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_324
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_323
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_326
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_330
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_327
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_329
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_328
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_331
            paddle.uniform([512, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_332
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_333
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_337
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_334
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_336
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_335
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_338
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_342
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_339
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_341
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_340
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_343
            paddle.uniform([512, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_347
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_344
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_346
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_345
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_348
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_352
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_349
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_351
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_350
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_353
            paddle.uniform([512, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_354
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_355
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_359
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_356
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_358
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_357
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_360
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_364
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_361
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_363
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_362
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_365
            paddle.uniform([512, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_369
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_366
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_368
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_367
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_370
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_374
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_371
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_373
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_372
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_375
            paddle.uniform([512, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_376
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_377
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_381
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_378
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_380
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_379
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_382
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_386
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_383
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_385
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_384
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_387
            paddle.uniform([512, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_391
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_388
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_390
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_389
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_392
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_396
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_393
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_395
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_394
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_397
            paddle.uniform([512, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_398
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_399
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_403
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_400
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_402
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_401
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_404
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_408
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_405
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_407
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_406
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_409
            paddle.uniform([512, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_413
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_410
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_412
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_411
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_414
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_418
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_415
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_417
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_416
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_419
            paddle.uniform([512, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_420
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_421
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_425
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_422
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_424
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_423
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_426
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_430
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_427
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_429
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_428
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_431
            paddle.uniform([512, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_435
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_432
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_434
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_433
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_436
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_440
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_437
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_439
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_438
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_441
            paddle.uniform([512, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_442
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_443
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_447
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_444
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_446
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_445
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_448
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_452
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_449
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_451
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_450
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_453
            paddle.uniform([512, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_457
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_454
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_456
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_455
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_458
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_462
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_459
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_461
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_460
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_463
            paddle.uniform([512, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_464
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_465
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_469
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_466
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_468
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_467
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_470
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_474
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_471
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_473
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_472
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_475
            paddle.uniform([512, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_479
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_476
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_478
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_477
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_480
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_484
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_481
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_483
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_482
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_485
            paddle.uniform([512, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_486
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_487
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_491
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_488
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_490
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_489
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_492
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_496
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_493
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_495
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_494
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_497
            paddle.uniform([512, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_501
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_498
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_500
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_499
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_502
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_506
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_503
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_505
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_504
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_507
            paddle.uniform([512, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_508
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_509
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_513
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_510
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_512
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_511
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_514
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_518
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_515
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_517
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_516
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_519
            paddle.uniform([512, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_523
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_520
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_522
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_521
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_524
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_528
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_525
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_527
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_526
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_529
            paddle.uniform([512, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_530
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_531
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_535
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_532
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_534
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_533
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_536
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_540
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_537
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_539
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_538
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_541
            paddle.uniform([512, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_545
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_542
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_544
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_543
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_546
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_550
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_547
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_549
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_548
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_551
            paddle.uniform([512, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_552
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_553
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_557
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_554
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_556
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_555
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_558
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_562
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_559
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_561
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_560
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_563
            paddle.uniform([512, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_567
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_564
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_566
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_565
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_568
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_572
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_569
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_571
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_570
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_573
            paddle.uniform([512, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_574
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_575
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_579
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_576
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_578
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_577
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_580
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_584
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_581
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_583
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_582
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_585
            paddle.uniform([512, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_589
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_586
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_588
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_587
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_590
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_594
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_591
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_593
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_592
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_595
            paddle.uniform([512, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_596
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_597
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_601
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_598
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_600
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_599
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_602
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_606
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_603
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_605
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_604
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_607
            paddle.uniform([512, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_611
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_608
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_610
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_609
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_612
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_616
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_613
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_615
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_614
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_617
            paddle.uniform([512, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_618
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_619
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_623
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_620
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_622
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_621
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_624
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_628
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_625
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_627
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_626
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_629
            paddle.uniform([512, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_633
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_630
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_632
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_631
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_634
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_638
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_635
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_637
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_636
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_639
            paddle.uniform([512, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_640
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_641
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_645
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_642
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_644
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_643
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_646
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_650
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_647
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_649
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_648
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_651
            paddle.uniform([512, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_655
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_652
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_654
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_653
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_656
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_660
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_657
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_659
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_658
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_661
            paddle.uniform([512, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_662
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_663
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_667
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_664
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_666
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_665
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_668
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_672
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_669
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_671
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_670
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_673
            paddle.uniform([512, 128, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_677
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_674
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_676
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_675
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_678
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_682
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_679
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_681
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_680
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_683
            paddle.uniform([512, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_684
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_685
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_689
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_686
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_688
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_687
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_690
            paddle.uniform([512, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_694
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_691
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_693
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_692
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_695
            paddle.uniform([1024, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_699
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_696
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_698
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_697
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_700
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_704
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_701
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_703
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_702
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_705
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_706
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_707
            paddle.uniform([2048, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_711
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_708
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_710
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_709
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_712
            paddle.uniform([2048, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_716
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_713
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_715
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_714
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_717
            paddle.uniform([512, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_721
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_718
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_720
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_719
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_722
            paddle.uniform([1024, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_726
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_723
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_725
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_724
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_727
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_731
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_728
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_730
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_729
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_732
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_733
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_734
            paddle.uniform([2048, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_738
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_735
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_737
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_736
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_739
            paddle.uniform([512, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_743
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_740
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_742
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_741
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_744
            paddle.uniform([1024, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_748
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_745
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_747
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_746
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_749
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_753
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_750
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_752
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_751
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_754
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_755
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_756
            paddle.uniform([2048, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_760
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_757
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_759
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_758
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_761
            paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
            # parameter_762
            paddle.uniform([1000], dtype='float32', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 3, 224, 224], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_0
            paddle.static.InputSpec(shape=[64, 3, 3, 3], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[128, 64, 3, 3], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[64, 128, 1, 1], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[128, 32, 3, 3], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[32, 64, 1, 1], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[128, 32, 1, 1], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[128, 32, 3, 3], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[32, 64, 1, 1], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[128, 32, 1, 1], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[128, 32, 3, 3], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[32, 64, 1, 1], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[128, 32, 1, 1], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[256, 64, 3, 3], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[64, 128, 1, 1], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[512, 256, 1, 1], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[256, 64, 3, 3], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[64, 128, 1, 1], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float32'),
            # parameter_139
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[256, 64, 3, 3], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[64, 128, 1, 1], dtype='float32'),
            # parameter_149
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_154
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[256, 64, 3, 3], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[64, 128, 1, 1], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[256, 512, 1, 1], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[512, 128, 3, 3], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_189
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[1024, 512, 1, 1], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_209
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[512, 128, 3, 3], dtype='float32'),
            # parameter_215
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_214
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_213
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_217
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_219
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_218
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_221
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_223
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_227
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_224
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_225
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_229
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_233
            paddle.static.InputSpec(shape=[512, 128, 3, 3], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_234
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_238
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_239
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_241
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_243
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float32'),
            # parameter_244
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_245
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_249
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_246
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_248
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_250
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_254
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_251
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_253
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_252
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_255
            paddle.static.InputSpec(shape=[512, 128, 3, 3], dtype='float32'),
            # parameter_259
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_256
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_258
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_257
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float32'),
            # parameter_264
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_261
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_263
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_262
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_265
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float32'),
            # parameter_266
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_267
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_271
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_268
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_270
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_269
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_272
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_276
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_273
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_275
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_274
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_277
            paddle.static.InputSpec(shape=[512, 128, 3, 3], dtype='float32'),
            # parameter_281
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_278
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_280
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_279
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_282
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float32'),
            # parameter_286
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_283
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_285
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_284
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_287
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float32'),
            # parameter_288
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_289
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_293
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_290
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_292
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_291
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_294
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_298
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_295
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_297
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_296
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_299
            paddle.static.InputSpec(shape=[512, 128, 3, 3], dtype='float32'),
            # parameter_303
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_300
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_302
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_301
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_304
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float32'),
            # parameter_308
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_305
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_307
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_306
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_309
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float32'),
            # parameter_310
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_311
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_315
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_312
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_314
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_313
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_316
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_320
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_317
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_319
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_318
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_321
            paddle.static.InputSpec(shape=[512, 128, 3, 3], dtype='float32'),
            # parameter_325
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_322
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_324
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_323
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_326
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float32'),
            # parameter_330
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_327
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_329
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_328
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_331
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float32'),
            # parameter_332
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_333
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_337
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_334
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_336
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_335
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_338
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_342
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_339
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_341
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_340
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_343
            paddle.static.InputSpec(shape=[512, 128, 3, 3], dtype='float32'),
            # parameter_347
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_344
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_346
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_345
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_348
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float32'),
            # parameter_352
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_349
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_351
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_350
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_353
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float32'),
            # parameter_354
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_355
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_359
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_356
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_358
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_357
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_360
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_364
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_361
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_363
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_362
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_365
            paddle.static.InputSpec(shape=[512, 128, 3, 3], dtype='float32'),
            # parameter_369
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_366
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_368
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_367
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_370
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float32'),
            # parameter_374
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_371
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_373
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_372
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_375
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float32'),
            # parameter_376
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_377
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_381
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_378
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_380
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_379
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_382
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_386
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_383
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_385
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_384
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_387
            paddle.static.InputSpec(shape=[512, 128, 3, 3], dtype='float32'),
            # parameter_391
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_388
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_390
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_389
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_392
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float32'),
            # parameter_396
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_393
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_395
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_394
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_397
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float32'),
            # parameter_398
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_399
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_403
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_400
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_402
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_401
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_404
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_408
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_405
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_407
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_406
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_409
            paddle.static.InputSpec(shape=[512, 128, 3, 3], dtype='float32'),
            # parameter_413
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_410
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_412
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_411
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_414
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float32'),
            # parameter_418
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_415
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_417
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_416
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_419
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float32'),
            # parameter_420
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_421
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_425
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_422
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_424
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_423
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_426
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_430
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_427
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_429
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_428
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_431
            paddle.static.InputSpec(shape=[512, 128, 3, 3], dtype='float32'),
            # parameter_435
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_432
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_434
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_433
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_436
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float32'),
            # parameter_440
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_437
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_439
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_438
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_441
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float32'),
            # parameter_442
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_443
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_447
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_444
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_446
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_445
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_448
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_452
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_449
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_451
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_450
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_453
            paddle.static.InputSpec(shape=[512, 128, 3, 3], dtype='float32'),
            # parameter_457
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_454
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_456
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_455
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_458
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float32'),
            # parameter_462
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_459
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_461
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_460
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_463
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float32'),
            # parameter_464
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_465
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_469
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_466
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_468
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_467
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_470
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_474
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_471
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_473
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_472
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_475
            paddle.static.InputSpec(shape=[512, 128, 3, 3], dtype='float32'),
            # parameter_479
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_476
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_478
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_477
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_480
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float32'),
            # parameter_484
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_481
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_483
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_482
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_485
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float32'),
            # parameter_486
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_487
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_491
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_488
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_490
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_489
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_492
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_496
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_493
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_495
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_494
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_497
            paddle.static.InputSpec(shape=[512, 128, 3, 3], dtype='float32'),
            # parameter_501
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_498
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_500
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_499
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_502
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float32'),
            # parameter_506
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_503
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_505
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_504
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_507
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float32'),
            # parameter_508
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_509
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_513
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_510
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_512
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_511
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_514
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_518
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_515
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_517
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_516
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_519
            paddle.static.InputSpec(shape=[512, 128, 3, 3], dtype='float32'),
            # parameter_523
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_520
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_522
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_521
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_524
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float32'),
            # parameter_528
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_525
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_527
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_526
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_529
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float32'),
            # parameter_530
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_531
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_535
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_532
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_534
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_533
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_536
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_540
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_537
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_539
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_538
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_541
            paddle.static.InputSpec(shape=[512, 128, 3, 3], dtype='float32'),
            # parameter_545
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_542
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_544
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_543
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_546
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float32'),
            # parameter_550
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_547
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_549
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_548
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_551
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float32'),
            # parameter_552
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_553
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_557
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_554
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_556
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_555
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_558
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_562
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_559
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_561
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_560
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_563
            paddle.static.InputSpec(shape=[512, 128, 3, 3], dtype='float32'),
            # parameter_567
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_564
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_566
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_565
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_568
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float32'),
            # parameter_572
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_569
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_571
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_570
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_573
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float32'),
            # parameter_574
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_575
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_579
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_576
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_578
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_577
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_580
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_584
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_581
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_583
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_582
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_585
            paddle.static.InputSpec(shape=[512, 128, 3, 3], dtype='float32'),
            # parameter_589
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_586
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_588
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_587
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_590
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float32'),
            # parameter_594
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_591
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_593
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_592
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_595
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float32'),
            # parameter_596
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_597
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_601
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_598
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_600
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_599
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_602
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_606
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_603
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_605
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_604
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_607
            paddle.static.InputSpec(shape=[512, 128, 3, 3], dtype='float32'),
            # parameter_611
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_608
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_610
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_609
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_612
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float32'),
            # parameter_616
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_613
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_615
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_614
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_617
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float32'),
            # parameter_618
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_619
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_623
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_620
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_622
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_621
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_624
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_628
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_625
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_627
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_626
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_629
            paddle.static.InputSpec(shape=[512, 128, 3, 3], dtype='float32'),
            # parameter_633
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_630
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_632
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_631
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_634
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float32'),
            # parameter_638
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_635
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_637
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_636
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_639
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float32'),
            # parameter_640
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_641
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_645
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_642
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_644
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_643
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_646
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_650
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_647
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_649
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_648
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_651
            paddle.static.InputSpec(shape=[512, 128, 3, 3], dtype='float32'),
            # parameter_655
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_652
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_654
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_653
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_656
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float32'),
            # parameter_660
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_657
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_659
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_658
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_661
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float32'),
            # parameter_662
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_663
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_667
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_664
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_666
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_665
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_668
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_672
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_669
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_671
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_670
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_673
            paddle.static.InputSpec(shape=[512, 128, 3, 3], dtype='float32'),
            # parameter_677
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_674
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_676
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_675
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_678
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float32'),
            # parameter_682
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_679
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_681
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_680
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_683
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float32'),
            # parameter_684
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_685
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_689
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_686
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_688
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_687
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_690
            paddle.static.InputSpec(shape=[512, 1024, 1, 1], dtype='float32'),
            # parameter_694
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_691
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_693
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_692
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_695
            paddle.static.InputSpec(shape=[1024, 256, 3, 3], dtype='float32'),
            # parameter_699
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_696
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_698
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_697
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_700
            paddle.static.InputSpec(shape=[256, 512, 1, 1], dtype='float32'),
            # parameter_704
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_701
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_703
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_702
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_705
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_706
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_707
            paddle.static.InputSpec(shape=[2048, 512, 1, 1], dtype='float32'),
            # parameter_711
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_708
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_710
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_709
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_712
            paddle.static.InputSpec(shape=[2048, 1024, 1, 1], dtype='float32'),
            # parameter_716
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_713
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_715
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_714
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_717
            paddle.static.InputSpec(shape=[512, 2048, 1, 1], dtype='float32'),
            # parameter_721
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_718
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_720
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_719
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_722
            paddle.static.InputSpec(shape=[1024, 256, 3, 3], dtype='float32'),
            # parameter_726
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_723
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_725
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_724
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_727
            paddle.static.InputSpec(shape=[256, 512, 1, 1], dtype='float32'),
            # parameter_731
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_728
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_730
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_729
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_732
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_733
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_734
            paddle.static.InputSpec(shape=[2048, 512, 1, 1], dtype='float32'),
            # parameter_738
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_735
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_737
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_736
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_739
            paddle.static.InputSpec(shape=[512, 2048, 1, 1], dtype='float32'),
            # parameter_743
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_740
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_742
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_741
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_744
            paddle.static.InputSpec(shape=[1024, 256, 3, 3], dtype='float32'),
            # parameter_748
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_745
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_747
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_746
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_749
            paddle.static.InputSpec(shape=[256, 512, 1, 1], dtype='float32'),
            # parameter_753
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_750
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_752
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_751
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_754
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_755
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_756
            paddle.static.InputSpec(shape=[2048, 512, 1, 1], dtype='float32'),
            # parameter_760
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_757
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_759
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_758
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_761
            paddle.static.InputSpec(shape=[2048, 1000], dtype='float32'),
            # parameter_762
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