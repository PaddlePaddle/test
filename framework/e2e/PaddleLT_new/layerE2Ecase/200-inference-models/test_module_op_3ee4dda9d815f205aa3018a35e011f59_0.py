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
    return [925][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_1794_0_0(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_114, parameter_111, parameter_113, parameter_112, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_154, parameter_151, parameter_153, parameter_152, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_169, parameter_166, parameter_168, parameter_167, parameter_170, parameter_174, parameter_171, parameter_173, parameter_172, parameter_175, parameter_179, parameter_176, parameter_178, parameter_177, parameter_180, parameter_184, parameter_181, parameter_183, parameter_182, parameter_185, parameter_189, parameter_186, parameter_188, parameter_187, parameter_190, parameter_194, parameter_191, parameter_193, parameter_192, parameter_195, parameter_199, parameter_196, parameter_198, parameter_197, parameter_200, parameter_204, parameter_201, parameter_203, parameter_202, parameter_205, parameter_209, parameter_206, parameter_208, parameter_207, parameter_210, parameter_214, parameter_211, parameter_213, parameter_212, parameter_215, parameter_219, parameter_216, parameter_218, parameter_217, parameter_220, parameter_224, parameter_221, parameter_223, parameter_222, parameter_225, parameter_229, parameter_226, parameter_228, parameter_227, parameter_230, parameter_234, parameter_231, parameter_233, parameter_232, parameter_235, parameter_239, parameter_236, parameter_238, parameter_237, parameter_240, parameter_244, parameter_241, parameter_243, parameter_242, parameter_245, parameter_249, parameter_246, parameter_248, parameter_247, parameter_250, parameter_254, parameter_251, parameter_253, parameter_252, parameter_255, parameter_259, parameter_256, parameter_258, parameter_257, parameter_260, parameter_264, parameter_261, parameter_263, parameter_262, parameter_265, parameter_269, parameter_266, parameter_268, parameter_267, parameter_270, parameter_274, parameter_271, parameter_273, parameter_272, parameter_275, parameter_279, parameter_276, parameter_278, parameter_277, parameter_280, parameter_284, parameter_281, parameter_283, parameter_282, parameter_285, parameter_289, parameter_286, parameter_288, parameter_287, parameter_290, parameter_294, parameter_291, parameter_293, parameter_292, parameter_295, parameter_299, parameter_296, parameter_298, parameter_297, parameter_300, parameter_304, parameter_301, parameter_303, parameter_302, parameter_305, parameter_309, parameter_306, parameter_308, parameter_307, parameter_310, parameter_314, parameter_311, parameter_313, parameter_312, parameter_315, parameter_319, parameter_316, parameter_318, parameter_317, parameter_320, parameter_324, parameter_321, parameter_323, parameter_322, parameter_325, parameter_329, parameter_326, parameter_328, parameter_327, parameter_330, parameter_334, parameter_331, parameter_333, parameter_332, parameter_335, parameter_339, parameter_336, parameter_338, parameter_337, parameter_340, parameter_344, parameter_341, parameter_343, parameter_342, parameter_345, parameter_349, parameter_346, parameter_348, parameter_347, parameter_350, parameter_354, parameter_351, parameter_353, parameter_352, parameter_355, parameter_359, parameter_356, parameter_358, parameter_357, parameter_360, parameter_364, parameter_361, parameter_363, parameter_362, parameter_365, parameter_369, parameter_366, parameter_368, parameter_367, parameter_370, parameter_374, parameter_371, parameter_373, parameter_372, parameter_375, parameter_379, parameter_376, parameter_378, parameter_377, parameter_380, parameter_384, parameter_381, parameter_383, parameter_382, parameter_385, parameter_389, parameter_386, parameter_388, parameter_387, parameter_390, parameter_394, parameter_391, parameter_393, parameter_392, parameter_395, parameter_399, parameter_396, parameter_398, parameter_397, parameter_400, parameter_404, parameter_401, parameter_403, parameter_402, parameter_405, parameter_409, parameter_406, parameter_408, parameter_407, parameter_410, parameter_414, parameter_411, parameter_413, parameter_412, parameter_415, parameter_419, parameter_416, parameter_418, parameter_417, parameter_420, parameter_424, parameter_421, parameter_423, parameter_422, parameter_425, parameter_429, parameter_426, parameter_428, parameter_427, parameter_430, parameter_434, parameter_431, parameter_433, parameter_432, parameter_435, parameter_439, parameter_436, parameter_438, parameter_437, parameter_440, parameter_444, parameter_441, parameter_443, parameter_442, parameter_445, parameter_449, parameter_446, parameter_448, parameter_447, parameter_450, parameter_454, parameter_451, parameter_453, parameter_452, parameter_455, parameter_459, parameter_456, parameter_458, parameter_457, parameter_460, parameter_464, parameter_461, parameter_463, parameter_462, parameter_465, parameter_469, parameter_466, parameter_468, parameter_467, parameter_470, parameter_474, parameter_471, parameter_473, parameter_472, parameter_475, parameter_479, parameter_476, parameter_478, parameter_477, parameter_480, parameter_484, parameter_481, parameter_483, parameter_482, parameter_485, parameter_489, parameter_486, parameter_488, parameter_487, parameter_490, parameter_494, parameter_491, parameter_493, parameter_492, parameter_495, parameter_499, parameter_496, parameter_498, parameter_497, parameter_500, parameter_504, parameter_501, parameter_503, parameter_502, parameter_505, parameter_509, parameter_506, parameter_508, parameter_507, parameter_510, parameter_514, parameter_511, parameter_513, parameter_512, parameter_515, parameter_519, parameter_516, parameter_518, parameter_517, parameter_520, parameter_524, parameter_521, parameter_523, parameter_522, parameter_525, parameter_529, parameter_526, parameter_528, parameter_527, parameter_530, parameter_534, parameter_531, parameter_533, parameter_532, parameter_535, parameter_539, parameter_536, parameter_538, parameter_537, parameter_540, parameter_544, parameter_541, parameter_543, parameter_542, parameter_545, parameter_549, parameter_546, parameter_548, parameter_547, parameter_550, parameter_554, parameter_551, parameter_553, parameter_552, parameter_555, parameter_559, parameter_556, parameter_558, parameter_557, parameter_560, parameter_564, parameter_561, parameter_563, parameter_562, parameter_565, parameter_569, parameter_566, parameter_568, parameter_567, parameter_570, parameter_574, parameter_571, parameter_573, parameter_572, parameter_575, parameter_579, parameter_576, parameter_578, parameter_577, parameter_580, parameter_584, parameter_581, parameter_583, parameter_582, parameter_585, parameter_589, parameter_586, parameter_588, parameter_587, parameter_590, parameter_594, parameter_591, parameter_593, parameter_592, parameter_595, parameter_599, parameter_596, parameter_598, parameter_597, parameter_600, parameter_604, parameter_601, parameter_603, parameter_602, parameter_605, parameter_609, parameter_606, parameter_608, parameter_607, parameter_610, parameter_614, parameter_611, parameter_613, parameter_612, parameter_615, parameter_619, parameter_616, parameter_618, parameter_617, parameter_620, parameter_624, parameter_621, parameter_623, parameter_622, parameter_625, parameter_629, parameter_626, parameter_628, parameter_627, parameter_630, parameter_634, parameter_631, parameter_633, parameter_632, parameter_635, parameter_639, parameter_636, parameter_638, parameter_637, parameter_640, parameter_644, parameter_641, parameter_643, parameter_642, parameter_645, parameter_649, parameter_646, parameter_648, parameter_647, parameter_650, parameter_654, parameter_651, parameter_653, parameter_652, parameter_655, parameter_659, parameter_656, parameter_658, parameter_657, parameter_660, parameter_664, parameter_661, parameter_663, parameter_662, parameter_665, parameter_669, parameter_666, parameter_668, parameter_667, parameter_670, parameter_674, parameter_671, parameter_673, parameter_672, parameter_675, parameter_679, parameter_676, parameter_678, parameter_677, parameter_680, parameter_684, parameter_681, parameter_683, parameter_682, parameter_685, parameter_689, parameter_686, parameter_688, parameter_687, parameter_690, parameter_694, parameter_691, parameter_693, parameter_692, parameter_695, parameter_699, parameter_696, parameter_698, parameter_697, parameter_700, parameter_704, parameter_701, parameter_703, parameter_702, parameter_705, parameter_709, parameter_706, parameter_708, parameter_707, parameter_710, parameter_714, parameter_711, parameter_713, parameter_712, parameter_715, parameter_719, parameter_716, parameter_718, parameter_717, parameter_720, parameter_724, parameter_721, parameter_723, parameter_722, parameter_725, parameter_729, parameter_726, parameter_728, parameter_727, parameter_730, parameter_734, parameter_731, parameter_733, parameter_732, parameter_735, parameter_739, parameter_736, parameter_738, parameter_737, parameter_740, parameter_744, parameter_741, parameter_743, parameter_742, parameter_745, parameter_749, parameter_746, parameter_748, parameter_747, parameter_750, parameter_754, parameter_751, parameter_753, parameter_752, parameter_755, parameter_759, parameter_756, parameter_758, parameter_757, parameter_760, parameter_764, parameter_761, parameter_763, parameter_762, parameter_765, parameter_769, parameter_766, parameter_768, parameter_767, parameter_770, parameter_774, parameter_771, parameter_773, parameter_772, parameter_775, parameter_779, parameter_776, parameter_778, parameter_777, parameter_780, parameter_784, parameter_781, parameter_783, parameter_782, parameter_785, parameter_789, parameter_786, parameter_788, parameter_787, parameter_790, parameter_794, parameter_791, parameter_793, parameter_792, parameter_795, parameter_799, parameter_796, parameter_798, parameter_797, parameter_800, parameter_804, parameter_801, parameter_803, parameter_802, parameter_805, parameter_809, parameter_806, parameter_808, parameter_807, parameter_810, parameter_814, parameter_811, parameter_813, parameter_812, parameter_815, parameter_819, parameter_816, parameter_818, parameter_817, parameter_820, parameter_824, parameter_821, parameter_823, parameter_822, parameter_825, parameter_829, parameter_826, parameter_828, parameter_827, parameter_830, parameter_834, parameter_831, parameter_833, parameter_832, parameter_835, parameter_839, parameter_836, parameter_838, parameter_837, parameter_840, parameter_844, parameter_841, parameter_843, parameter_842, parameter_845, parameter_849, parameter_846, parameter_848, parameter_847, parameter_850, parameter_854, parameter_851, parameter_853, parameter_852, parameter_855, parameter_859, parameter_856, parameter_858, parameter_857, parameter_860, parameter_861, feed_0):

        # pd_op.cast: (-1x3x224x224xf16) <- (-1x3x224x224xf32)
        cast_0 = paddle._C_ops.cast(feed_0, paddle.float16)

        # pd_op.conv2d: (-1x32x112x112xf16) <- (-1x3x224x224xf16, 32x3x3x3xf16)
        conv2d_0 = paddle._C_ops.conv2d(cast_0, parameter_0, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x112x112xf16, 32xf32, 32xf32, 32xf32, 32xf32, None) <- (-1x32x112x112xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_0, parameter_1, parameter_2, parameter_3, parameter_4, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x32x112x112xf16) <- (-1x32x112x112xf16)
        relu__0 = paddle._C_ops.relu_(batch_norm__0)

        # pd_op.conv2d: (-1x32x112x112xf16) <- (-1x32x112x112xf16, 32x32x3x3xf16)
        conv2d_1 = paddle._C_ops.conv2d(relu__0, parameter_5, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x112x112xf16, 32xf32, 32xf32, 32xf32, 32xf32, None) <- (-1x32x112x112xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_1, parameter_6, parameter_7, parameter_8, parameter_9, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x32x112x112xf16) <- (-1x32x112x112xf16)
        relu__1 = paddle._C_ops.relu_(batch_norm__6)

        # pd_op.conv2d: (-1x64x112x112xf16) <- (-1x32x112x112xf16, 64x32x3x3xf16)
        conv2d_2 = paddle._C_ops.conv2d(relu__1, parameter_10, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x112x112xf16, 64xf32, 64xf32, 64xf32, 64xf32, None) <- (-1x64x112x112xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_2, parameter_11, parameter_12, parameter_13, parameter_14, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x112x112xf16) <- (-1x64x112x112xf16)
        relu__2 = paddle._C_ops.relu_(batch_norm__12)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [3, 3]

        # pd_op.pool2d: (-1x64x56x56xf16) <- (-1x64x112x112xf16, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(relu__2, full_int_array_0, [2, 2], [1, 1], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x104x56x56xf16) <- (-1x64x56x56xf16, 104x64x1x1xf16)
        conv2d_3 = paddle._C_ops.conv2d(pool2d_0, parameter_15, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x56x56xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x56x56xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_3, parameter_16, parameter_17, parameter_18, parameter_19, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x56x56xf16) <- (-1x104x56x56xf16)
        relu__3 = paddle._C_ops.relu_(batch_norm__18)

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16]) <- (-1x104x56x56xf16, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(relu__3, 4, full_0)

        # builtin.slice: (-1x26x56x56xf16) <- ([-1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16])
        slice_0 = split_with_num_0[0]

        # pd_op.conv2d: (-1x26x56x56xf16) <- (-1x26x56x56xf16, 26x26x3x3xf16)
        conv2d_4 = paddle._C_ops.conv2d(slice_0, parameter_20, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x26x56x56xf16, 26xf32, 26xf32, 26xf32, 26xf32, None) <- (-1x26x56x56xf16, 26xf32, 26xf32, 26xf32, 26xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_4, parameter_21, parameter_22, parameter_23, parameter_24, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x26x56x56xf16) <- (-1x26x56x56xf16)
        relu__4 = paddle._C_ops.relu_(batch_norm__24)

        # builtin.slice: (-1x26x56x56xf16) <- ([-1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16])
        slice_1 = split_with_num_0[1]

        # pd_op.add_: (-1x26x56x56xf16) <- (-1x26x56x56xf16, -1x26x56x56xf16)
        add__0 = paddle._C_ops.add_(slice_1, relu__4)

        # pd_op.conv2d: (-1x26x56x56xf16) <- (-1x26x56x56xf16, 26x26x3x3xf16)
        conv2d_5 = paddle._C_ops.conv2d(add__0, parameter_25, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x26x56x56xf16, 26xf32, 26xf32, 26xf32, 26xf32, None) <- (-1x26x56x56xf16, 26xf32, 26xf32, 26xf32, 26xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_5, parameter_26, parameter_27, parameter_28, parameter_29, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x26x56x56xf16) <- (-1x26x56x56xf16)
        relu__5 = paddle._C_ops.relu_(batch_norm__30)

        # builtin.slice: (-1x26x56x56xf16) <- ([-1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16])
        slice_2 = split_with_num_0[2]

        # pd_op.add_: (-1x26x56x56xf16) <- (-1x26x56x56xf16, -1x26x56x56xf16)
        add__1 = paddle._C_ops.add_(slice_2, relu__5)

        # pd_op.conv2d: (-1x26x56x56xf16) <- (-1x26x56x56xf16, 26x26x3x3xf16)
        conv2d_6 = paddle._C_ops.conv2d(add__1, parameter_30, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x26x56x56xf16, 26xf32, 26xf32, 26xf32, 26xf32, None) <- (-1x26x56x56xf16, 26xf32, 26xf32, 26xf32, 26xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_6, parameter_31, parameter_32, parameter_33, parameter_34, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x26x56x56xf16) <- (-1x26x56x56xf16)
        relu__6 = paddle._C_ops.relu_(batch_norm__36)

        # builtin.slice: (-1x26x56x56xf16) <- ([-1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16])
        slice_3 = split_with_num_0[3]

        # builtin.combine: ([-1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16]) <- (-1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16)
        combine_0 = [relu__4, relu__5, relu__6, slice_3]

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x104x56x56xf16) <- ([-1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_1)

        # pd_op.conv2d: (-1x256x56x56xf16) <- (-1x104x56x56xf16, 256x104x1x1xf16)
        conv2d_7 = paddle._C_ops.conv2d(concat_0, parameter_35, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x56x56xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x56x56xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_7, parameter_36, parameter_37, parameter_38, parameter_39, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x256x56x56xf16) <- (-1x64x56x56xf16, 256x64x1x1xf16)
        conv2d_8 = paddle._C_ops.conv2d(pool2d_0, parameter_40, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x56x56xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x56x56xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_8, parameter_41, parameter_42, parameter_43, parameter_44, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x256x56x56xf16) <- (-1x256x56x56xf16, -1x256x56x56xf16)
        add__2 = paddle._C_ops.add_(batch_norm__48, batch_norm__42)

        # pd_op.relu_: (-1x256x56x56xf16) <- (-1x256x56x56xf16)
        relu__7 = paddle._C_ops.relu_(add__2)

        # pd_op.conv2d: (-1x104x56x56xf16) <- (-1x256x56x56xf16, 104x256x1x1xf16)
        conv2d_9 = paddle._C_ops.conv2d(relu__7, parameter_45, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x56x56xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x56x56xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_9, parameter_46, parameter_47, parameter_48, parameter_49, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x56x56xf16) <- (-1x104x56x56xf16)
        relu__8 = paddle._C_ops.relu_(batch_norm__54)

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16]) <- (-1x104x56x56xf16, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(relu__8, 4, full_2)

        # builtin.slice: (-1x26x56x56xf16) <- ([-1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16])
        slice_4 = split_with_num_1[0]

        # pd_op.conv2d: (-1x26x56x56xf16) <- (-1x26x56x56xf16, 26x26x3x3xf16)
        conv2d_10 = paddle._C_ops.conv2d(slice_4, parameter_50, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x26x56x56xf16, 26xf32, 26xf32, 26xf32, 26xf32, None) <- (-1x26x56x56xf16, 26xf32, 26xf32, 26xf32, 26xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_10, parameter_51, parameter_52, parameter_53, parameter_54, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x26x56x56xf16) <- (-1x26x56x56xf16)
        relu__9 = paddle._C_ops.relu_(batch_norm__60)

        # builtin.slice: (-1x26x56x56xf16) <- ([-1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16])
        slice_5 = split_with_num_1[1]

        # pd_op.add_: (-1x26x56x56xf16) <- (-1x26x56x56xf16, -1x26x56x56xf16)
        add__3 = paddle._C_ops.add_(slice_5, relu__9)

        # pd_op.conv2d: (-1x26x56x56xf16) <- (-1x26x56x56xf16, 26x26x3x3xf16)
        conv2d_11 = paddle._C_ops.conv2d(add__3, parameter_55, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x26x56x56xf16, 26xf32, 26xf32, 26xf32, 26xf32, None) <- (-1x26x56x56xf16, 26xf32, 26xf32, 26xf32, 26xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_11, parameter_56, parameter_57, parameter_58, parameter_59, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x26x56x56xf16) <- (-1x26x56x56xf16)
        relu__10 = paddle._C_ops.relu_(batch_norm__66)

        # builtin.slice: (-1x26x56x56xf16) <- ([-1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16])
        slice_6 = split_with_num_1[2]

        # pd_op.add_: (-1x26x56x56xf16) <- (-1x26x56x56xf16, -1x26x56x56xf16)
        add__4 = paddle._C_ops.add_(slice_6, relu__10)

        # pd_op.conv2d: (-1x26x56x56xf16) <- (-1x26x56x56xf16, 26x26x3x3xf16)
        conv2d_12 = paddle._C_ops.conv2d(add__4, parameter_60, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x26x56x56xf16, 26xf32, 26xf32, 26xf32, 26xf32, None) <- (-1x26x56x56xf16, 26xf32, 26xf32, 26xf32, 26xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_12, parameter_61, parameter_62, parameter_63, parameter_64, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x26x56x56xf16) <- (-1x26x56x56xf16)
        relu__11 = paddle._C_ops.relu_(batch_norm__72)

        # builtin.slice: (-1x26x56x56xf16) <- ([-1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16])
        slice_7 = split_with_num_1[3]

        # builtin.combine: ([-1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16]) <- (-1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16)
        combine_1 = [relu__9, relu__10, relu__11, slice_7]

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x104x56x56xf16) <- ([-1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_3)

        # pd_op.conv2d: (-1x256x56x56xf16) <- (-1x104x56x56xf16, 256x104x1x1xf16)
        conv2d_13 = paddle._C_ops.conv2d(concat_1, parameter_65, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x56x56xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x56x56xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_13, parameter_66, parameter_67, parameter_68, parameter_69, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x256x56x56xf16) <- (-1x256x56x56xf16, -1x256x56x56xf16)
        add__5 = paddle._C_ops.add_(relu__7, batch_norm__78)

        # pd_op.relu_: (-1x256x56x56xf16) <- (-1x256x56x56xf16)
        relu__12 = paddle._C_ops.relu_(add__5)

        # pd_op.conv2d: (-1x104x56x56xf16) <- (-1x256x56x56xf16, 104x256x1x1xf16)
        conv2d_14 = paddle._C_ops.conv2d(relu__12, parameter_70, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x56x56xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x56x56xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_14, parameter_71, parameter_72, parameter_73, parameter_74, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x56x56xf16) <- (-1x104x56x56xf16)
        relu__13 = paddle._C_ops.relu_(batch_norm__84)

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16]) <- (-1x104x56x56xf16, 1xi32)
        split_with_num_2 = paddle._C_ops.split_with_num(relu__13, 4, full_4)

        # builtin.slice: (-1x26x56x56xf16) <- ([-1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16])
        slice_8 = split_with_num_2[0]

        # pd_op.conv2d: (-1x26x56x56xf16) <- (-1x26x56x56xf16, 26x26x3x3xf16)
        conv2d_15 = paddle._C_ops.conv2d(slice_8, parameter_75, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x26x56x56xf16, 26xf32, 26xf32, 26xf32, 26xf32, None) <- (-1x26x56x56xf16, 26xf32, 26xf32, 26xf32, 26xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_15, parameter_76, parameter_77, parameter_78, parameter_79, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x26x56x56xf16) <- (-1x26x56x56xf16)
        relu__14 = paddle._C_ops.relu_(batch_norm__90)

        # builtin.slice: (-1x26x56x56xf16) <- ([-1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16])
        slice_9 = split_with_num_2[1]

        # pd_op.add_: (-1x26x56x56xf16) <- (-1x26x56x56xf16, -1x26x56x56xf16)
        add__6 = paddle._C_ops.add_(slice_9, relu__14)

        # pd_op.conv2d: (-1x26x56x56xf16) <- (-1x26x56x56xf16, 26x26x3x3xf16)
        conv2d_16 = paddle._C_ops.conv2d(add__6, parameter_80, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x26x56x56xf16, 26xf32, 26xf32, 26xf32, 26xf32, None) <- (-1x26x56x56xf16, 26xf32, 26xf32, 26xf32, 26xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_16, parameter_81, parameter_82, parameter_83, parameter_84, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x26x56x56xf16) <- (-1x26x56x56xf16)
        relu__15 = paddle._C_ops.relu_(batch_norm__96)

        # builtin.slice: (-1x26x56x56xf16) <- ([-1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16])
        slice_10 = split_with_num_2[2]

        # pd_op.add_: (-1x26x56x56xf16) <- (-1x26x56x56xf16, -1x26x56x56xf16)
        add__7 = paddle._C_ops.add_(slice_10, relu__15)

        # pd_op.conv2d: (-1x26x56x56xf16) <- (-1x26x56x56xf16, 26x26x3x3xf16)
        conv2d_17 = paddle._C_ops.conv2d(add__7, parameter_85, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x26x56x56xf16, 26xf32, 26xf32, 26xf32, 26xf32, None) <- (-1x26x56x56xf16, 26xf32, 26xf32, 26xf32, 26xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_17, parameter_86, parameter_87, parameter_88, parameter_89, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x26x56x56xf16) <- (-1x26x56x56xf16)
        relu__16 = paddle._C_ops.relu_(batch_norm__102)

        # builtin.slice: (-1x26x56x56xf16) <- ([-1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16])
        slice_11 = split_with_num_2[3]

        # builtin.combine: ([-1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16]) <- (-1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16)
        combine_2 = [relu__14, relu__15, relu__16, slice_11]

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x104x56x56xf16) <- ([-1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16, -1x26x56x56xf16], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_5)

        # pd_op.conv2d: (-1x256x56x56xf16) <- (-1x104x56x56xf16, 256x104x1x1xf16)
        conv2d_18 = paddle._C_ops.conv2d(concat_2, parameter_90, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x56x56xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x56x56xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_18, parameter_91, parameter_92, parameter_93, parameter_94, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x256x56x56xf16) <- (-1x256x56x56xf16, -1x256x56x56xf16)
        add__8 = paddle._C_ops.add_(relu__12, batch_norm__108)

        # pd_op.relu_: (-1x256x56x56xf16) <- (-1x256x56x56xf16)
        relu__17 = paddle._C_ops.relu_(add__8)

        # pd_op.conv2d: (-1x208x56x56xf16) <- (-1x256x56x56xf16, 208x256x1x1xf16)
        conv2d_19 = paddle._C_ops.conv2d(relu__17, parameter_95, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x208x56x56xf16, 208xf32, 208xf32, 208xf32, 208xf32, None) <- (-1x208x56x56xf16, 208xf32, 208xf32, 208xf32, 208xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_19, parameter_96, parameter_97, parameter_98, parameter_99, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x208x56x56xf16) <- (-1x208x56x56xf16)
        relu__18 = paddle._C_ops.relu_(batch_norm__114)

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x52x56x56xf16, -1x52x56x56xf16, -1x52x56x56xf16, -1x52x56x56xf16]) <- (-1x208x56x56xf16, 1xi32)
        split_with_num_3 = paddle._C_ops.split_with_num(relu__18, 4, full_6)

        # builtin.slice: (-1x52x56x56xf16) <- ([-1x52x56x56xf16, -1x52x56x56xf16, -1x52x56x56xf16, -1x52x56x56xf16])
        slice_12 = split_with_num_3[0]

        # pd_op.conv2d: (-1x52x28x28xf16) <- (-1x52x56x56xf16, 52x52x3x3xf16)
        conv2d_20 = paddle._C_ops.conv2d(slice_12, parameter_100, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x52x28x28xf16, 52xf32, 52xf32, 52xf32, 52xf32, None) <- (-1x52x28x28xf16, 52xf32, 52xf32, 52xf32, 52xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_20, parameter_101, parameter_102, parameter_103, parameter_104, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x52x28x28xf16) <- (-1x52x28x28xf16)
        relu__19 = paddle._C_ops.relu_(batch_norm__120)

        # builtin.slice: (-1x52x56x56xf16) <- ([-1x52x56x56xf16, -1x52x56x56xf16, -1x52x56x56xf16, -1x52x56x56xf16])
        slice_13 = split_with_num_3[1]

        # pd_op.conv2d: (-1x52x28x28xf16) <- (-1x52x56x56xf16, 52x52x3x3xf16)
        conv2d_21 = paddle._C_ops.conv2d(slice_13, parameter_105, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x52x28x28xf16, 52xf32, 52xf32, 52xf32, 52xf32, None) <- (-1x52x28x28xf16, 52xf32, 52xf32, 52xf32, 52xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_21, parameter_106, parameter_107, parameter_108, parameter_109, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x52x28x28xf16) <- (-1x52x28x28xf16)
        relu__20 = paddle._C_ops.relu_(batch_norm__126)

        # builtin.slice: (-1x52x56x56xf16) <- ([-1x52x56x56xf16, -1x52x56x56xf16, -1x52x56x56xf16, -1x52x56x56xf16])
        slice_14 = split_with_num_3[2]

        # pd_op.conv2d: (-1x52x28x28xf16) <- (-1x52x56x56xf16, 52x52x3x3xf16)
        conv2d_22 = paddle._C_ops.conv2d(slice_14, parameter_110, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x52x28x28xf16, 52xf32, 52xf32, 52xf32, 52xf32, None) <- (-1x52x28x28xf16, 52xf32, 52xf32, 52xf32, 52xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_22, parameter_111, parameter_112, parameter_113, parameter_114, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x52x28x28xf16) <- (-1x52x28x28xf16)
        relu__21 = paddle._C_ops.relu_(batch_norm__132)

        # builtin.slice: (-1x52x56x56xf16) <- ([-1x52x56x56xf16, -1x52x56x56xf16, -1x52x56x56xf16, -1x52x56x56xf16])
        slice_15 = split_with_num_3[3]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [3, 3]

        # pd_op.pool2d: (-1x52x28x28xf16) <- (-1x52x56x56xf16, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(slice_15, full_int_array_1, [2, 2], [1, 1], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # builtin.combine: ([-1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16]) <- (-1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16)
        combine_3 = [relu__19, relu__20, relu__21, pool2d_1]

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x208x28x28xf16) <- ([-1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_3, full_7)

        # pd_op.conv2d: (-1x512x28x28xf16) <- (-1x208x28x28xf16, 512x208x1x1xf16)
        conv2d_23 = paddle._C_ops.conv2d(concat_3, parameter_115, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x28x28xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x28x28xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_23, parameter_116, parameter_117, parameter_118, parameter_119, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [2, 2]

        # pd_op.pool2d: (-1x256x28x28xf16) <- (-1x256x56x56xf16, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(relu__17, full_int_array_2, [2, 2], [0, 0], True, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x512x28x28xf16) <- (-1x256x28x28xf16, 512x256x1x1xf16)
        conv2d_24 = paddle._C_ops.conv2d(pool2d_2, parameter_120, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x28x28xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x28x28xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_24, parameter_121, parameter_122, parameter_123, parameter_124, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x28x28xf16) <- (-1x512x28x28xf16, -1x512x28x28xf16)
        add__9 = paddle._C_ops.add_(batch_norm__144, batch_norm__138)

        # pd_op.relu_: (-1x512x28x28xf16) <- (-1x512x28x28xf16)
        relu__22 = paddle._C_ops.relu_(add__9)

        # pd_op.conv2d: (-1x208x28x28xf16) <- (-1x512x28x28xf16, 208x512x1x1xf16)
        conv2d_25 = paddle._C_ops.conv2d(relu__22, parameter_125, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x208x28x28xf16, 208xf32, 208xf32, 208xf32, 208xf32, None) <- (-1x208x28x28xf16, 208xf32, 208xf32, 208xf32, 208xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_25, parameter_126, parameter_127, parameter_128, parameter_129, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x208x28x28xf16) <- (-1x208x28x28xf16)
        relu__23 = paddle._C_ops.relu_(batch_norm__150)

        # pd_op.full: (1xi32) <- ()
        full_8 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16]) <- (-1x208x28x28xf16, 1xi32)
        split_with_num_4 = paddle._C_ops.split_with_num(relu__23, 4, full_8)

        # builtin.slice: (-1x52x28x28xf16) <- ([-1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16])
        slice_16 = split_with_num_4[0]

        # pd_op.conv2d: (-1x52x28x28xf16) <- (-1x52x28x28xf16, 52x52x3x3xf16)
        conv2d_26 = paddle._C_ops.conv2d(slice_16, parameter_130, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x52x28x28xf16, 52xf32, 52xf32, 52xf32, 52xf32, None) <- (-1x52x28x28xf16, 52xf32, 52xf32, 52xf32, 52xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_26, parameter_131, parameter_132, parameter_133, parameter_134, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x52x28x28xf16) <- (-1x52x28x28xf16)
        relu__24 = paddle._C_ops.relu_(batch_norm__156)

        # builtin.slice: (-1x52x28x28xf16) <- ([-1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16])
        slice_17 = split_with_num_4[1]

        # pd_op.add_: (-1x52x28x28xf16) <- (-1x52x28x28xf16, -1x52x28x28xf16)
        add__10 = paddle._C_ops.add_(slice_17, relu__24)

        # pd_op.conv2d: (-1x52x28x28xf16) <- (-1x52x28x28xf16, 52x52x3x3xf16)
        conv2d_27 = paddle._C_ops.conv2d(add__10, parameter_135, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x52x28x28xf16, 52xf32, 52xf32, 52xf32, 52xf32, None) <- (-1x52x28x28xf16, 52xf32, 52xf32, 52xf32, 52xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_27, parameter_136, parameter_137, parameter_138, parameter_139, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x52x28x28xf16) <- (-1x52x28x28xf16)
        relu__25 = paddle._C_ops.relu_(batch_norm__162)

        # builtin.slice: (-1x52x28x28xf16) <- ([-1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16])
        slice_18 = split_with_num_4[2]

        # pd_op.add_: (-1x52x28x28xf16) <- (-1x52x28x28xf16, -1x52x28x28xf16)
        add__11 = paddle._C_ops.add_(slice_18, relu__25)

        # pd_op.conv2d: (-1x52x28x28xf16) <- (-1x52x28x28xf16, 52x52x3x3xf16)
        conv2d_28 = paddle._C_ops.conv2d(add__11, parameter_140, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x52x28x28xf16, 52xf32, 52xf32, 52xf32, 52xf32, None) <- (-1x52x28x28xf16, 52xf32, 52xf32, 52xf32, 52xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_28, parameter_141, parameter_142, parameter_143, parameter_144, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x52x28x28xf16) <- (-1x52x28x28xf16)
        relu__26 = paddle._C_ops.relu_(batch_norm__168)

        # builtin.slice: (-1x52x28x28xf16) <- ([-1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16])
        slice_19 = split_with_num_4[3]

        # builtin.combine: ([-1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16]) <- (-1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16)
        combine_4 = [relu__24, relu__25, relu__26, slice_19]

        # pd_op.full: (1xi32) <- ()
        full_9 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x208x28x28xf16) <- ([-1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_4, full_9)

        # pd_op.conv2d: (-1x512x28x28xf16) <- (-1x208x28x28xf16, 512x208x1x1xf16)
        conv2d_29 = paddle._C_ops.conv2d(concat_4, parameter_145, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x28x28xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x28x28xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_29, parameter_146, parameter_147, parameter_148, parameter_149, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x28x28xf16) <- (-1x512x28x28xf16, -1x512x28x28xf16)
        add__12 = paddle._C_ops.add_(relu__22, batch_norm__174)

        # pd_op.relu_: (-1x512x28x28xf16) <- (-1x512x28x28xf16)
        relu__27 = paddle._C_ops.relu_(add__12)

        # pd_op.conv2d: (-1x208x28x28xf16) <- (-1x512x28x28xf16, 208x512x1x1xf16)
        conv2d_30 = paddle._C_ops.conv2d(relu__27, parameter_150, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x208x28x28xf16, 208xf32, 208xf32, 208xf32, 208xf32, None) <- (-1x208x28x28xf16, 208xf32, 208xf32, 208xf32, 208xf32)
        batch_norm__180, batch_norm__181, batch_norm__182, batch_norm__183, batch_norm__184, batch_norm__185 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_30, parameter_151, parameter_152, parameter_153, parameter_154, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x208x28x28xf16) <- (-1x208x28x28xf16)
        relu__28 = paddle._C_ops.relu_(batch_norm__180)

        # pd_op.full: (1xi32) <- ()
        full_10 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16]) <- (-1x208x28x28xf16, 1xi32)
        split_with_num_5 = paddle._C_ops.split_with_num(relu__28, 4, full_10)

        # builtin.slice: (-1x52x28x28xf16) <- ([-1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16])
        slice_20 = split_with_num_5[0]

        # pd_op.conv2d: (-1x52x28x28xf16) <- (-1x52x28x28xf16, 52x52x3x3xf16)
        conv2d_31 = paddle._C_ops.conv2d(slice_20, parameter_155, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x52x28x28xf16, 52xf32, 52xf32, 52xf32, 52xf32, None) <- (-1x52x28x28xf16, 52xf32, 52xf32, 52xf32, 52xf32)
        batch_norm__186, batch_norm__187, batch_norm__188, batch_norm__189, batch_norm__190, batch_norm__191 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_31, parameter_156, parameter_157, parameter_158, parameter_159, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x52x28x28xf16) <- (-1x52x28x28xf16)
        relu__29 = paddle._C_ops.relu_(batch_norm__186)

        # builtin.slice: (-1x52x28x28xf16) <- ([-1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16])
        slice_21 = split_with_num_5[1]

        # pd_op.add_: (-1x52x28x28xf16) <- (-1x52x28x28xf16, -1x52x28x28xf16)
        add__13 = paddle._C_ops.add_(slice_21, relu__29)

        # pd_op.conv2d: (-1x52x28x28xf16) <- (-1x52x28x28xf16, 52x52x3x3xf16)
        conv2d_32 = paddle._C_ops.conv2d(add__13, parameter_160, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x52x28x28xf16, 52xf32, 52xf32, 52xf32, 52xf32, None) <- (-1x52x28x28xf16, 52xf32, 52xf32, 52xf32, 52xf32)
        batch_norm__192, batch_norm__193, batch_norm__194, batch_norm__195, batch_norm__196, batch_norm__197 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_32, parameter_161, parameter_162, parameter_163, parameter_164, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x52x28x28xf16) <- (-1x52x28x28xf16)
        relu__30 = paddle._C_ops.relu_(batch_norm__192)

        # builtin.slice: (-1x52x28x28xf16) <- ([-1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16])
        slice_22 = split_with_num_5[2]

        # pd_op.add_: (-1x52x28x28xf16) <- (-1x52x28x28xf16, -1x52x28x28xf16)
        add__14 = paddle._C_ops.add_(slice_22, relu__30)

        # pd_op.conv2d: (-1x52x28x28xf16) <- (-1x52x28x28xf16, 52x52x3x3xf16)
        conv2d_33 = paddle._C_ops.conv2d(add__14, parameter_165, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x52x28x28xf16, 52xf32, 52xf32, 52xf32, 52xf32, None) <- (-1x52x28x28xf16, 52xf32, 52xf32, 52xf32, 52xf32)
        batch_norm__198, batch_norm__199, batch_norm__200, batch_norm__201, batch_norm__202, batch_norm__203 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_33, parameter_166, parameter_167, parameter_168, parameter_169, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x52x28x28xf16) <- (-1x52x28x28xf16)
        relu__31 = paddle._C_ops.relu_(batch_norm__198)

        # builtin.slice: (-1x52x28x28xf16) <- ([-1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16])
        slice_23 = split_with_num_5[3]

        # builtin.combine: ([-1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16]) <- (-1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16)
        combine_5 = [relu__29, relu__30, relu__31, slice_23]

        # pd_op.full: (1xi32) <- ()
        full_11 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x208x28x28xf16) <- ([-1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_5, full_11)

        # pd_op.conv2d: (-1x512x28x28xf16) <- (-1x208x28x28xf16, 512x208x1x1xf16)
        conv2d_34 = paddle._C_ops.conv2d(concat_5, parameter_170, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x28x28xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x28x28xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__204, batch_norm__205, batch_norm__206, batch_norm__207, batch_norm__208, batch_norm__209 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_34, parameter_171, parameter_172, parameter_173, parameter_174, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x28x28xf16) <- (-1x512x28x28xf16, -1x512x28x28xf16)
        add__15 = paddle._C_ops.add_(relu__27, batch_norm__204)

        # pd_op.relu_: (-1x512x28x28xf16) <- (-1x512x28x28xf16)
        relu__32 = paddle._C_ops.relu_(add__15)

        # pd_op.conv2d: (-1x208x28x28xf16) <- (-1x512x28x28xf16, 208x512x1x1xf16)
        conv2d_35 = paddle._C_ops.conv2d(relu__32, parameter_175, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x208x28x28xf16, 208xf32, 208xf32, 208xf32, 208xf32, None) <- (-1x208x28x28xf16, 208xf32, 208xf32, 208xf32, 208xf32)
        batch_norm__210, batch_norm__211, batch_norm__212, batch_norm__213, batch_norm__214, batch_norm__215 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_35, parameter_176, parameter_177, parameter_178, parameter_179, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x208x28x28xf16) <- (-1x208x28x28xf16)
        relu__33 = paddle._C_ops.relu_(batch_norm__210)

        # pd_op.full: (1xi32) <- ()
        full_12 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16]) <- (-1x208x28x28xf16, 1xi32)
        split_with_num_6 = paddle._C_ops.split_with_num(relu__33, 4, full_12)

        # builtin.slice: (-1x52x28x28xf16) <- ([-1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16])
        slice_24 = split_with_num_6[0]

        # pd_op.conv2d: (-1x52x28x28xf16) <- (-1x52x28x28xf16, 52x52x3x3xf16)
        conv2d_36 = paddle._C_ops.conv2d(slice_24, parameter_180, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x52x28x28xf16, 52xf32, 52xf32, 52xf32, 52xf32, None) <- (-1x52x28x28xf16, 52xf32, 52xf32, 52xf32, 52xf32)
        batch_norm__216, batch_norm__217, batch_norm__218, batch_norm__219, batch_norm__220, batch_norm__221 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_36, parameter_181, parameter_182, parameter_183, parameter_184, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x52x28x28xf16) <- (-1x52x28x28xf16)
        relu__34 = paddle._C_ops.relu_(batch_norm__216)

        # builtin.slice: (-1x52x28x28xf16) <- ([-1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16])
        slice_25 = split_with_num_6[1]

        # pd_op.add_: (-1x52x28x28xf16) <- (-1x52x28x28xf16, -1x52x28x28xf16)
        add__16 = paddle._C_ops.add_(slice_25, relu__34)

        # pd_op.conv2d: (-1x52x28x28xf16) <- (-1x52x28x28xf16, 52x52x3x3xf16)
        conv2d_37 = paddle._C_ops.conv2d(add__16, parameter_185, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x52x28x28xf16, 52xf32, 52xf32, 52xf32, 52xf32, None) <- (-1x52x28x28xf16, 52xf32, 52xf32, 52xf32, 52xf32)
        batch_norm__222, batch_norm__223, batch_norm__224, batch_norm__225, batch_norm__226, batch_norm__227 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_37, parameter_186, parameter_187, parameter_188, parameter_189, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x52x28x28xf16) <- (-1x52x28x28xf16)
        relu__35 = paddle._C_ops.relu_(batch_norm__222)

        # builtin.slice: (-1x52x28x28xf16) <- ([-1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16])
        slice_26 = split_with_num_6[2]

        # pd_op.add_: (-1x52x28x28xf16) <- (-1x52x28x28xf16, -1x52x28x28xf16)
        add__17 = paddle._C_ops.add_(slice_26, relu__35)

        # pd_op.conv2d: (-1x52x28x28xf16) <- (-1x52x28x28xf16, 52x52x3x3xf16)
        conv2d_38 = paddle._C_ops.conv2d(add__17, parameter_190, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x52x28x28xf16, 52xf32, 52xf32, 52xf32, 52xf32, None) <- (-1x52x28x28xf16, 52xf32, 52xf32, 52xf32, 52xf32)
        batch_norm__228, batch_norm__229, batch_norm__230, batch_norm__231, batch_norm__232, batch_norm__233 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_38, parameter_191, parameter_192, parameter_193, parameter_194, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x52x28x28xf16) <- (-1x52x28x28xf16)
        relu__36 = paddle._C_ops.relu_(batch_norm__228)

        # builtin.slice: (-1x52x28x28xf16) <- ([-1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16])
        slice_27 = split_with_num_6[3]

        # builtin.combine: ([-1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16]) <- (-1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16)
        combine_6 = [relu__34, relu__35, relu__36, slice_27]

        # pd_op.full: (1xi32) <- ()
        full_13 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x208x28x28xf16) <- ([-1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16, -1x52x28x28xf16], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_6, full_13)

        # pd_op.conv2d: (-1x512x28x28xf16) <- (-1x208x28x28xf16, 512x208x1x1xf16)
        conv2d_39 = paddle._C_ops.conv2d(concat_6, parameter_195, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x28x28xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x28x28xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__234, batch_norm__235, batch_norm__236, batch_norm__237, batch_norm__238, batch_norm__239 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_39, parameter_196, parameter_197, parameter_198, parameter_199, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x28x28xf16) <- (-1x512x28x28xf16, -1x512x28x28xf16)
        add__18 = paddle._C_ops.add_(relu__32, batch_norm__234)

        # pd_op.relu_: (-1x512x28x28xf16) <- (-1x512x28x28xf16)
        relu__37 = paddle._C_ops.relu_(add__18)

        # pd_op.conv2d: (-1x416x28x28xf16) <- (-1x512x28x28xf16, 416x512x1x1xf16)
        conv2d_40 = paddle._C_ops.conv2d(relu__37, parameter_200, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x416x28x28xf16, 416xf32, 416xf32, 416xf32, 416xf32, None) <- (-1x416x28x28xf16, 416xf32, 416xf32, 416xf32, 416xf32)
        batch_norm__240, batch_norm__241, batch_norm__242, batch_norm__243, batch_norm__244, batch_norm__245 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_40, parameter_201, parameter_202, parameter_203, parameter_204, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x416x28x28xf16) <- (-1x416x28x28xf16)
        relu__38 = paddle._C_ops.relu_(batch_norm__240)

        # pd_op.full: (1xi32) <- ()
        full_14 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x104x28x28xf16, -1x104x28x28xf16, -1x104x28x28xf16, -1x104x28x28xf16]) <- (-1x416x28x28xf16, 1xi32)
        split_with_num_7 = paddle._C_ops.split_with_num(relu__38, 4, full_14)

        # builtin.slice: (-1x104x28x28xf16) <- ([-1x104x28x28xf16, -1x104x28x28xf16, -1x104x28x28xf16, -1x104x28x28xf16])
        slice_28 = split_with_num_7[0]

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x28x28xf16, 104x104x3x3xf16)
        conv2d_41 = paddle._C_ops.conv2d(slice_28, parameter_205, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__246, batch_norm__247, batch_norm__248, batch_norm__249, batch_norm__250, batch_norm__251 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_41, parameter_206, parameter_207, parameter_208, parameter_209, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__39 = paddle._C_ops.relu_(batch_norm__246)

        # builtin.slice: (-1x104x28x28xf16) <- ([-1x104x28x28xf16, -1x104x28x28xf16, -1x104x28x28xf16, -1x104x28x28xf16])
        slice_29 = split_with_num_7[1]

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x28x28xf16, 104x104x3x3xf16)
        conv2d_42 = paddle._C_ops.conv2d(slice_29, parameter_210, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__252, batch_norm__253, batch_norm__254, batch_norm__255, batch_norm__256, batch_norm__257 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_42, parameter_211, parameter_212, parameter_213, parameter_214, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__40 = paddle._C_ops.relu_(batch_norm__252)

        # builtin.slice: (-1x104x28x28xf16) <- ([-1x104x28x28xf16, -1x104x28x28xf16, -1x104x28x28xf16, -1x104x28x28xf16])
        slice_30 = split_with_num_7[2]

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x28x28xf16, 104x104x3x3xf16)
        conv2d_43 = paddle._C_ops.conv2d(slice_30, parameter_215, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__258, batch_norm__259, batch_norm__260, batch_norm__261, batch_norm__262, batch_norm__263 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_43, parameter_216, parameter_217, parameter_218, parameter_219, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__41 = paddle._C_ops.relu_(batch_norm__258)

        # builtin.slice: (-1x104x28x28xf16) <- ([-1x104x28x28xf16, -1x104x28x28xf16, -1x104x28x28xf16, -1x104x28x28xf16])
        slice_31 = split_with_num_7[3]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [3, 3]

        # pd_op.pool2d: (-1x104x14x14xf16) <- (-1x104x28x28xf16, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(slice_31, full_int_array_3, [2, 2], [1, 1], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # builtin.combine: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16)
        combine_7 = [relu__39, relu__40, relu__41, pool2d_3]

        # pd_op.full: (1xi32) <- ()
        full_15 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x416x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_7, full_15)

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x416x14x14xf16, 1024x416x1x1xf16)
        conv2d_44 = paddle._C_ops.conv2d(concat_7, parameter_220, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__264, batch_norm__265, batch_norm__266, batch_norm__267, batch_norm__268, batch_norm__269 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_44, parameter_221, parameter_222, parameter_223, parameter_224, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [2, 2]

        # pd_op.pool2d: (-1x512x14x14xf16) <- (-1x512x28x28xf16, 2xi64)
        pool2d_4 = paddle._C_ops.pool2d(relu__37, full_int_array_4, [2, 2], [0, 0], True, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x512x14x14xf16, 1024x512x1x1xf16)
        conv2d_45 = paddle._C_ops.conv2d(pool2d_4, parameter_225, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__270, batch_norm__271, batch_norm__272, batch_norm__273, batch_norm__274, batch_norm__275 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_45, parameter_226, parameter_227, parameter_228, parameter_229, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__19 = paddle._C_ops.add_(batch_norm__270, batch_norm__264)

        # pd_op.relu_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16)
        relu__42 = paddle._C_ops.relu_(add__19)

        # pd_op.conv2d: (-1x416x14x14xf16) <- (-1x1024x14x14xf16, 416x1024x1x1xf16)
        conv2d_46 = paddle._C_ops.conv2d(relu__42, parameter_230, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32, None) <- (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32)
        batch_norm__276, batch_norm__277, batch_norm__278, batch_norm__279, batch_norm__280, batch_norm__281 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_46, parameter_231, parameter_232, parameter_233, parameter_234, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x416x14x14xf16) <- (-1x416x14x14xf16)
        relu__43 = paddle._C_ops.relu_(batch_norm__276)

        # pd_op.full: (1xi32) <- ()
        full_16 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x416x14x14xf16, 1xi32)
        split_with_num_8 = paddle._C_ops.split_with_num(relu__43, 4, full_16)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_32 = split_with_num_8[0]

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_47 = paddle._C_ops.conv2d(slice_32, parameter_235, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__282, batch_norm__283, batch_norm__284, batch_norm__285, batch_norm__286, batch_norm__287 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_47, parameter_236, parameter_237, parameter_238, parameter_239, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__44 = paddle._C_ops.relu_(batch_norm__282)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_33 = split_with_num_8[1]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__20 = paddle._C_ops.add_(slice_33, relu__44)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_48 = paddle._C_ops.conv2d(add__20, parameter_240, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__288, batch_norm__289, batch_norm__290, batch_norm__291, batch_norm__292, batch_norm__293 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_48, parameter_241, parameter_242, parameter_243, parameter_244, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__45 = paddle._C_ops.relu_(batch_norm__288)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_34 = split_with_num_8[2]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__21 = paddle._C_ops.add_(slice_34, relu__45)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_49 = paddle._C_ops.conv2d(add__21, parameter_245, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__294, batch_norm__295, batch_norm__296, batch_norm__297, batch_norm__298, batch_norm__299 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_49, parameter_246, parameter_247, parameter_248, parameter_249, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__46 = paddle._C_ops.relu_(batch_norm__294)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_35 = split_with_num_8[3]

        # builtin.combine: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16)
        combine_8 = [relu__44, relu__45, relu__46, slice_35]

        # pd_op.full: (1xi32) <- ()
        full_17 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x416x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16], 1xi32)
        concat_8 = paddle._C_ops.concat(combine_8, full_17)

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x416x14x14xf16, 1024x416x1x1xf16)
        conv2d_50 = paddle._C_ops.conv2d(concat_8, parameter_250, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__300, batch_norm__301, batch_norm__302, batch_norm__303, batch_norm__304, batch_norm__305 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_50, parameter_251, parameter_252, parameter_253, parameter_254, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__22 = paddle._C_ops.add_(relu__42, batch_norm__300)

        # pd_op.relu_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16)
        relu__47 = paddle._C_ops.relu_(add__22)

        # pd_op.conv2d: (-1x416x14x14xf16) <- (-1x1024x14x14xf16, 416x1024x1x1xf16)
        conv2d_51 = paddle._C_ops.conv2d(relu__47, parameter_255, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32, None) <- (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32)
        batch_norm__306, batch_norm__307, batch_norm__308, batch_norm__309, batch_norm__310, batch_norm__311 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_51, parameter_256, parameter_257, parameter_258, parameter_259, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x416x14x14xf16) <- (-1x416x14x14xf16)
        relu__48 = paddle._C_ops.relu_(batch_norm__306)

        # pd_op.full: (1xi32) <- ()
        full_18 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x416x14x14xf16, 1xi32)
        split_with_num_9 = paddle._C_ops.split_with_num(relu__48, 4, full_18)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_36 = split_with_num_9[0]

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_52 = paddle._C_ops.conv2d(slice_36, parameter_260, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__312, batch_norm__313, batch_norm__314, batch_norm__315, batch_norm__316, batch_norm__317 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_52, parameter_261, parameter_262, parameter_263, parameter_264, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__49 = paddle._C_ops.relu_(batch_norm__312)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_37 = split_with_num_9[1]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__23 = paddle._C_ops.add_(slice_37, relu__49)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_53 = paddle._C_ops.conv2d(add__23, parameter_265, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__318, batch_norm__319, batch_norm__320, batch_norm__321, batch_norm__322, batch_norm__323 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_53, parameter_266, parameter_267, parameter_268, parameter_269, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__50 = paddle._C_ops.relu_(batch_norm__318)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_38 = split_with_num_9[2]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__24 = paddle._C_ops.add_(slice_38, relu__50)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_54 = paddle._C_ops.conv2d(add__24, parameter_270, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__324, batch_norm__325, batch_norm__326, batch_norm__327, batch_norm__328, batch_norm__329 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_54, parameter_271, parameter_272, parameter_273, parameter_274, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__51 = paddle._C_ops.relu_(batch_norm__324)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_39 = split_with_num_9[3]

        # builtin.combine: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16)
        combine_9 = [relu__49, relu__50, relu__51, slice_39]

        # pd_op.full: (1xi32) <- ()
        full_19 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x416x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16], 1xi32)
        concat_9 = paddle._C_ops.concat(combine_9, full_19)

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x416x14x14xf16, 1024x416x1x1xf16)
        conv2d_55 = paddle._C_ops.conv2d(concat_9, parameter_275, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__330, batch_norm__331, batch_norm__332, batch_norm__333, batch_norm__334, batch_norm__335 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_55, parameter_276, parameter_277, parameter_278, parameter_279, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__25 = paddle._C_ops.add_(relu__47, batch_norm__330)

        # pd_op.relu_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16)
        relu__52 = paddle._C_ops.relu_(add__25)

        # pd_op.conv2d: (-1x416x14x14xf16) <- (-1x1024x14x14xf16, 416x1024x1x1xf16)
        conv2d_56 = paddle._C_ops.conv2d(relu__52, parameter_280, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32, None) <- (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32)
        batch_norm__336, batch_norm__337, batch_norm__338, batch_norm__339, batch_norm__340, batch_norm__341 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_56, parameter_281, parameter_282, parameter_283, parameter_284, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x416x14x14xf16) <- (-1x416x14x14xf16)
        relu__53 = paddle._C_ops.relu_(batch_norm__336)

        # pd_op.full: (1xi32) <- ()
        full_20 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x416x14x14xf16, 1xi32)
        split_with_num_10 = paddle._C_ops.split_with_num(relu__53, 4, full_20)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_40 = split_with_num_10[0]

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_57 = paddle._C_ops.conv2d(slice_40, parameter_285, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__342, batch_norm__343, batch_norm__344, batch_norm__345, batch_norm__346, batch_norm__347 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_57, parameter_286, parameter_287, parameter_288, parameter_289, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__54 = paddle._C_ops.relu_(batch_norm__342)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_41 = split_with_num_10[1]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__26 = paddle._C_ops.add_(slice_41, relu__54)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_58 = paddle._C_ops.conv2d(add__26, parameter_290, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__348, batch_norm__349, batch_norm__350, batch_norm__351, batch_norm__352, batch_norm__353 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_58, parameter_291, parameter_292, parameter_293, parameter_294, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__55 = paddle._C_ops.relu_(batch_norm__348)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_42 = split_with_num_10[2]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__27 = paddle._C_ops.add_(slice_42, relu__55)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_59 = paddle._C_ops.conv2d(add__27, parameter_295, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__354, batch_norm__355, batch_norm__356, batch_norm__357, batch_norm__358, batch_norm__359 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_59, parameter_296, parameter_297, parameter_298, parameter_299, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__56 = paddle._C_ops.relu_(batch_norm__354)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_43 = split_with_num_10[3]

        # builtin.combine: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16)
        combine_10 = [relu__54, relu__55, relu__56, slice_43]

        # pd_op.full: (1xi32) <- ()
        full_21 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x416x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16], 1xi32)
        concat_10 = paddle._C_ops.concat(combine_10, full_21)

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x416x14x14xf16, 1024x416x1x1xf16)
        conv2d_60 = paddle._C_ops.conv2d(concat_10, parameter_300, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__360, batch_norm__361, batch_norm__362, batch_norm__363, batch_norm__364, batch_norm__365 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_60, parameter_301, parameter_302, parameter_303, parameter_304, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__28 = paddle._C_ops.add_(relu__52, batch_norm__360)

        # pd_op.relu_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16)
        relu__57 = paddle._C_ops.relu_(add__28)

        # pd_op.conv2d: (-1x416x14x14xf16) <- (-1x1024x14x14xf16, 416x1024x1x1xf16)
        conv2d_61 = paddle._C_ops.conv2d(relu__57, parameter_305, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32, None) <- (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32)
        batch_norm__366, batch_norm__367, batch_norm__368, batch_norm__369, batch_norm__370, batch_norm__371 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_61, parameter_306, parameter_307, parameter_308, parameter_309, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x416x14x14xf16) <- (-1x416x14x14xf16)
        relu__58 = paddle._C_ops.relu_(batch_norm__366)

        # pd_op.full: (1xi32) <- ()
        full_22 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x416x14x14xf16, 1xi32)
        split_with_num_11 = paddle._C_ops.split_with_num(relu__58, 4, full_22)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_44 = split_with_num_11[0]

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_62 = paddle._C_ops.conv2d(slice_44, parameter_310, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__372, batch_norm__373, batch_norm__374, batch_norm__375, batch_norm__376, batch_norm__377 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_62, parameter_311, parameter_312, parameter_313, parameter_314, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__59 = paddle._C_ops.relu_(batch_norm__372)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_45 = split_with_num_11[1]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__29 = paddle._C_ops.add_(slice_45, relu__59)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_63 = paddle._C_ops.conv2d(add__29, parameter_315, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__378, batch_norm__379, batch_norm__380, batch_norm__381, batch_norm__382, batch_norm__383 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_63, parameter_316, parameter_317, parameter_318, parameter_319, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__60 = paddle._C_ops.relu_(batch_norm__378)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_46 = split_with_num_11[2]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__30 = paddle._C_ops.add_(slice_46, relu__60)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_64 = paddle._C_ops.conv2d(add__30, parameter_320, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__384, batch_norm__385, batch_norm__386, batch_norm__387, batch_norm__388, batch_norm__389 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_64, parameter_321, parameter_322, parameter_323, parameter_324, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__61 = paddle._C_ops.relu_(batch_norm__384)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_47 = split_with_num_11[3]

        # builtin.combine: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16)
        combine_11 = [relu__59, relu__60, relu__61, slice_47]

        # pd_op.full: (1xi32) <- ()
        full_23 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x416x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16], 1xi32)
        concat_11 = paddle._C_ops.concat(combine_11, full_23)

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x416x14x14xf16, 1024x416x1x1xf16)
        conv2d_65 = paddle._C_ops.conv2d(concat_11, parameter_325, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__390, batch_norm__391, batch_norm__392, batch_norm__393, batch_norm__394, batch_norm__395 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_65, parameter_326, parameter_327, parameter_328, parameter_329, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__31 = paddle._C_ops.add_(relu__57, batch_norm__390)

        # pd_op.relu_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16)
        relu__62 = paddle._C_ops.relu_(add__31)

        # pd_op.conv2d: (-1x416x14x14xf16) <- (-1x1024x14x14xf16, 416x1024x1x1xf16)
        conv2d_66 = paddle._C_ops.conv2d(relu__62, parameter_330, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32, None) <- (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32)
        batch_norm__396, batch_norm__397, batch_norm__398, batch_norm__399, batch_norm__400, batch_norm__401 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_66, parameter_331, parameter_332, parameter_333, parameter_334, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x416x14x14xf16) <- (-1x416x14x14xf16)
        relu__63 = paddle._C_ops.relu_(batch_norm__396)

        # pd_op.full: (1xi32) <- ()
        full_24 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x416x14x14xf16, 1xi32)
        split_with_num_12 = paddle._C_ops.split_with_num(relu__63, 4, full_24)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_48 = split_with_num_12[0]

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_67 = paddle._C_ops.conv2d(slice_48, parameter_335, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__402, batch_norm__403, batch_norm__404, batch_norm__405, batch_norm__406, batch_norm__407 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_67, parameter_336, parameter_337, parameter_338, parameter_339, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__64 = paddle._C_ops.relu_(batch_norm__402)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_49 = split_with_num_12[1]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__32 = paddle._C_ops.add_(slice_49, relu__64)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_68 = paddle._C_ops.conv2d(add__32, parameter_340, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__408, batch_norm__409, batch_norm__410, batch_norm__411, batch_norm__412, batch_norm__413 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_68, parameter_341, parameter_342, parameter_343, parameter_344, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__65 = paddle._C_ops.relu_(batch_norm__408)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_50 = split_with_num_12[2]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__33 = paddle._C_ops.add_(slice_50, relu__65)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_69 = paddle._C_ops.conv2d(add__33, parameter_345, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__414, batch_norm__415, batch_norm__416, batch_norm__417, batch_norm__418, batch_norm__419 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_69, parameter_346, parameter_347, parameter_348, parameter_349, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__66 = paddle._C_ops.relu_(batch_norm__414)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_51 = split_with_num_12[3]

        # builtin.combine: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16)
        combine_12 = [relu__64, relu__65, relu__66, slice_51]

        # pd_op.full: (1xi32) <- ()
        full_25 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x416x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16], 1xi32)
        concat_12 = paddle._C_ops.concat(combine_12, full_25)

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x416x14x14xf16, 1024x416x1x1xf16)
        conv2d_70 = paddle._C_ops.conv2d(concat_12, parameter_350, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__420, batch_norm__421, batch_norm__422, batch_norm__423, batch_norm__424, batch_norm__425 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_70, parameter_351, parameter_352, parameter_353, parameter_354, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__34 = paddle._C_ops.add_(relu__62, batch_norm__420)

        # pd_op.relu_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16)
        relu__67 = paddle._C_ops.relu_(add__34)

        # pd_op.conv2d: (-1x416x14x14xf16) <- (-1x1024x14x14xf16, 416x1024x1x1xf16)
        conv2d_71 = paddle._C_ops.conv2d(relu__67, parameter_355, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32, None) <- (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32)
        batch_norm__426, batch_norm__427, batch_norm__428, batch_norm__429, batch_norm__430, batch_norm__431 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_71, parameter_356, parameter_357, parameter_358, parameter_359, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x416x14x14xf16) <- (-1x416x14x14xf16)
        relu__68 = paddle._C_ops.relu_(batch_norm__426)

        # pd_op.full: (1xi32) <- ()
        full_26 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x416x14x14xf16, 1xi32)
        split_with_num_13 = paddle._C_ops.split_with_num(relu__68, 4, full_26)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_52 = split_with_num_13[0]

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_72 = paddle._C_ops.conv2d(slice_52, parameter_360, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__432, batch_norm__433, batch_norm__434, batch_norm__435, batch_norm__436, batch_norm__437 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_72, parameter_361, parameter_362, parameter_363, parameter_364, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__69 = paddle._C_ops.relu_(batch_norm__432)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_53 = split_with_num_13[1]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__35 = paddle._C_ops.add_(slice_53, relu__69)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_73 = paddle._C_ops.conv2d(add__35, parameter_365, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__438, batch_norm__439, batch_norm__440, batch_norm__441, batch_norm__442, batch_norm__443 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_73, parameter_366, parameter_367, parameter_368, parameter_369, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__70 = paddle._C_ops.relu_(batch_norm__438)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_54 = split_with_num_13[2]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__36 = paddle._C_ops.add_(slice_54, relu__70)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_74 = paddle._C_ops.conv2d(add__36, parameter_370, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__444, batch_norm__445, batch_norm__446, batch_norm__447, batch_norm__448, batch_norm__449 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_74, parameter_371, parameter_372, parameter_373, parameter_374, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__71 = paddle._C_ops.relu_(batch_norm__444)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_55 = split_with_num_13[3]

        # builtin.combine: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16)
        combine_13 = [relu__69, relu__70, relu__71, slice_55]

        # pd_op.full: (1xi32) <- ()
        full_27 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x416x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16], 1xi32)
        concat_13 = paddle._C_ops.concat(combine_13, full_27)

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x416x14x14xf16, 1024x416x1x1xf16)
        conv2d_75 = paddle._C_ops.conv2d(concat_13, parameter_375, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__450, batch_norm__451, batch_norm__452, batch_norm__453, batch_norm__454, batch_norm__455 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_75, parameter_376, parameter_377, parameter_378, parameter_379, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__37 = paddle._C_ops.add_(relu__67, batch_norm__450)

        # pd_op.relu_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16)
        relu__72 = paddle._C_ops.relu_(add__37)

        # pd_op.conv2d: (-1x416x14x14xf16) <- (-1x1024x14x14xf16, 416x1024x1x1xf16)
        conv2d_76 = paddle._C_ops.conv2d(relu__72, parameter_380, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32, None) <- (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32)
        batch_norm__456, batch_norm__457, batch_norm__458, batch_norm__459, batch_norm__460, batch_norm__461 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_76, parameter_381, parameter_382, parameter_383, parameter_384, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x416x14x14xf16) <- (-1x416x14x14xf16)
        relu__73 = paddle._C_ops.relu_(batch_norm__456)

        # pd_op.full: (1xi32) <- ()
        full_28 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x416x14x14xf16, 1xi32)
        split_with_num_14 = paddle._C_ops.split_with_num(relu__73, 4, full_28)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_56 = split_with_num_14[0]

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_77 = paddle._C_ops.conv2d(slice_56, parameter_385, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__462, batch_norm__463, batch_norm__464, batch_norm__465, batch_norm__466, batch_norm__467 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_77, parameter_386, parameter_387, parameter_388, parameter_389, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__74 = paddle._C_ops.relu_(batch_norm__462)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_57 = split_with_num_14[1]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__38 = paddle._C_ops.add_(slice_57, relu__74)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_78 = paddle._C_ops.conv2d(add__38, parameter_390, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__468, batch_norm__469, batch_norm__470, batch_norm__471, batch_norm__472, batch_norm__473 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_78, parameter_391, parameter_392, parameter_393, parameter_394, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__75 = paddle._C_ops.relu_(batch_norm__468)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_58 = split_with_num_14[2]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__39 = paddle._C_ops.add_(slice_58, relu__75)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_79 = paddle._C_ops.conv2d(add__39, parameter_395, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__474, batch_norm__475, batch_norm__476, batch_norm__477, batch_norm__478, batch_norm__479 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_79, parameter_396, parameter_397, parameter_398, parameter_399, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__76 = paddle._C_ops.relu_(batch_norm__474)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_59 = split_with_num_14[3]

        # builtin.combine: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16)
        combine_14 = [relu__74, relu__75, relu__76, slice_59]

        # pd_op.full: (1xi32) <- ()
        full_29 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x416x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16], 1xi32)
        concat_14 = paddle._C_ops.concat(combine_14, full_29)

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x416x14x14xf16, 1024x416x1x1xf16)
        conv2d_80 = paddle._C_ops.conv2d(concat_14, parameter_400, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__480, batch_norm__481, batch_norm__482, batch_norm__483, batch_norm__484, batch_norm__485 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_80, parameter_401, parameter_402, parameter_403, parameter_404, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__40 = paddle._C_ops.add_(relu__72, batch_norm__480)

        # pd_op.relu_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16)
        relu__77 = paddle._C_ops.relu_(add__40)

        # pd_op.conv2d: (-1x416x14x14xf16) <- (-1x1024x14x14xf16, 416x1024x1x1xf16)
        conv2d_81 = paddle._C_ops.conv2d(relu__77, parameter_405, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32, None) <- (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32)
        batch_norm__486, batch_norm__487, batch_norm__488, batch_norm__489, batch_norm__490, batch_norm__491 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_81, parameter_406, parameter_407, parameter_408, parameter_409, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x416x14x14xf16) <- (-1x416x14x14xf16)
        relu__78 = paddle._C_ops.relu_(batch_norm__486)

        # pd_op.full: (1xi32) <- ()
        full_30 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x416x14x14xf16, 1xi32)
        split_with_num_15 = paddle._C_ops.split_with_num(relu__78, 4, full_30)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_60 = split_with_num_15[0]

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_82 = paddle._C_ops.conv2d(slice_60, parameter_410, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__492, batch_norm__493, batch_norm__494, batch_norm__495, batch_norm__496, batch_norm__497 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_82, parameter_411, parameter_412, parameter_413, parameter_414, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__79 = paddle._C_ops.relu_(batch_norm__492)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_61 = split_with_num_15[1]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__41 = paddle._C_ops.add_(slice_61, relu__79)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_83 = paddle._C_ops.conv2d(add__41, parameter_415, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__498, batch_norm__499, batch_norm__500, batch_norm__501, batch_norm__502, batch_norm__503 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_83, parameter_416, parameter_417, parameter_418, parameter_419, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__80 = paddle._C_ops.relu_(batch_norm__498)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_62 = split_with_num_15[2]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__42 = paddle._C_ops.add_(slice_62, relu__80)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_84 = paddle._C_ops.conv2d(add__42, parameter_420, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__504, batch_norm__505, batch_norm__506, batch_norm__507, batch_norm__508, batch_norm__509 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_84, parameter_421, parameter_422, parameter_423, parameter_424, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__81 = paddle._C_ops.relu_(batch_norm__504)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_63 = split_with_num_15[3]

        # builtin.combine: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16)
        combine_15 = [relu__79, relu__80, relu__81, slice_63]

        # pd_op.full: (1xi32) <- ()
        full_31 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x416x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16], 1xi32)
        concat_15 = paddle._C_ops.concat(combine_15, full_31)

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x416x14x14xf16, 1024x416x1x1xf16)
        conv2d_85 = paddle._C_ops.conv2d(concat_15, parameter_425, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__510, batch_norm__511, batch_norm__512, batch_norm__513, batch_norm__514, batch_norm__515 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_85, parameter_426, parameter_427, parameter_428, parameter_429, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__43 = paddle._C_ops.add_(relu__77, batch_norm__510)

        # pd_op.relu_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16)
        relu__82 = paddle._C_ops.relu_(add__43)

        # pd_op.conv2d: (-1x416x14x14xf16) <- (-1x1024x14x14xf16, 416x1024x1x1xf16)
        conv2d_86 = paddle._C_ops.conv2d(relu__82, parameter_430, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32, None) <- (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32)
        batch_norm__516, batch_norm__517, batch_norm__518, batch_norm__519, batch_norm__520, batch_norm__521 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_86, parameter_431, parameter_432, parameter_433, parameter_434, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x416x14x14xf16) <- (-1x416x14x14xf16)
        relu__83 = paddle._C_ops.relu_(batch_norm__516)

        # pd_op.full: (1xi32) <- ()
        full_32 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x416x14x14xf16, 1xi32)
        split_with_num_16 = paddle._C_ops.split_with_num(relu__83, 4, full_32)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_64 = split_with_num_16[0]

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_87 = paddle._C_ops.conv2d(slice_64, parameter_435, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__522, batch_norm__523, batch_norm__524, batch_norm__525, batch_norm__526, batch_norm__527 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_87, parameter_436, parameter_437, parameter_438, parameter_439, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__84 = paddle._C_ops.relu_(batch_norm__522)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_65 = split_with_num_16[1]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__44 = paddle._C_ops.add_(slice_65, relu__84)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_88 = paddle._C_ops.conv2d(add__44, parameter_440, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__528, batch_norm__529, batch_norm__530, batch_norm__531, batch_norm__532, batch_norm__533 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_88, parameter_441, parameter_442, parameter_443, parameter_444, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__85 = paddle._C_ops.relu_(batch_norm__528)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_66 = split_with_num_16[2]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__45 = paddle._C_ops.add_(slice_66, relu__85)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_89 = paddle._C_ops.conv2d(add__45, parameter_445, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__534, batch_norm__535, batch_norm__536, batch_norm__537, batch_norm__538, batch_norm__539 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_89, parameter_446, parameter_447, parameter_448, parameter_449, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__86 = paddle._C_ops.relu_(batch_norm__534)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_67 = split_with_num_16[3]

        # builtin.combine: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16)
        combine_16 = [relu__84, relu__85, relu__86, slice_67]

        # pd_op.full: (1xi32) <- ()
        full_33 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x416x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16], 1xi32)
        concat_16 = paddle._C_ops.concat(combine_16, full_33)

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x416x14x14xf16, 1024x416x1x1xf16)
        conv2d_90 = paddle._C_ops.conv2d(concat_16, parameter_450, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__540, batch_norm__541, batch_norm__542, batch_norm__543, batch_norm__544, batch_norm__545 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_90, parameter_451, parameter_452, parameter_453, parameter_454, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__46 = paddle._C_ops.add_(relu__82, batch_norm__540)

        # pd_op.relu_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16)
        relu__87 = paddle._C_ops.relu_(add__46)

        # pd_op.conv2d: (-1x416x14x14xf16) <- (-1x1024x14x14xf16, 416x1024x1x1xf16)
        conv2d_91 = paddle._C_ops.conv2d(relu__87, parameter_455, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32, None) <- (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32)
        batch_norm__546, batch_norm__547, batch_norm__548, batch_norm__549, batch_norm__550, batch_norm__551 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_91, parameter_456, parameter_457, parameter_458, parameter_459, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x416x14x14xf16) <- (-1x416x14x14xf16)
        relu__88 = paddle._C_ops.relu_(batch_norm__546)

        # pd_op.full: (1xi32) <- ()
        full_34 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x416x14x14xf16, 1xi32)
        split_with_num_17 = paddle._C_ops.split_with_num(relu__88, 4, full_34)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_68 = split_with_num_17[0]

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_92 = paddle._C_ops.conv2d(slice_68, parameter_460, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__552, batch_norm__553, batch_norm__554, batch_norm__555, batch_norm__556, batch_norm__557 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_92, parameter_461, parameter_462, parameter_463, parameter_464, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__89 = paddle._C_ops.relu_(batch_norm__552)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_69 = split_with_num_17[1]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__47 = paddle._C_ops.add_(slice_69, relu__89)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_93 = paddle._C_ops.conv2d(add__47, parameter_465, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__558, batch_norm__559, batch_norm__560, batch_norm__561, batch_norm__562, batch_norm__563 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_93, parameter_466, parameter_467, parameter_468, parameter_469, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__90 = paddle._C_ops.relu_(batch_norm__558)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_70 = split_with_num_17[2]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__48 = paddle._C_ops.add_(slice_70, relu__90)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_94 = paddle._C_ops.conv2d(add__48, parameter_470, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__564, batch_norm__565, batch_norm__566, batch_norm__567, batch_norm__568, batch_norm__569 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_94, parameter_471, parameter_472, parameter_473, parameter_474, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__91 = paddle._C_ops.relu_(batch_norm__564)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_71 = split_with_num_17[3]

        # builtin.combine: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16)
        combine_17 = [relu__89, relu__90, relu__91, slice_71]

        # pd_op.full: (1xi32) <- ()
        full_35 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x416x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16], 1xi32)
        concat_17 = paddle._C_ops.concat(combine_17, full_35)

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x416x14x14xf16, 1024x416x1x1xf16)
        conv2d_95 = paddle._C_ops.conv2d(concat_17, parameter_475, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__570, batch_norm__571, batch_norm__572, batch_norm__573, batch_norm__574, batch_norm__575 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_95, parameter_476, parameter_477, parameter_478, parameter_479, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__49 = paddle._C_ops.add_(relu__87, batch_norm__570)

        # pd_op.relu_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16)
        relu__92 = paddle._C_ops.relu_(add__49)

        # pd_op.conv2d: (-1x416x14x14xf16) <- (-1x1024x14x14xf16, 416x1024x1x1xf16)
        conv2d_96 = paddle._C_ops.conv2d(relu__92, parameter_480, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32, None) <- (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32)
        batch_norm__576, batch_norm__577, batch_norm__578, batch_norm__579, batch_norm__580, batch_norm__581 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_96, parameter_481, parameter_482, parameter_483, parameter_484, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x416x14x14xf16) <- (-1x416x14x14xf16)
        relu__93 = paddle._C_ops.relu_(batch_norm__576)

        # pd_op.full: (1xi32) <- ()
        full_36 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x416x14x14xf16, 1xi32)
        split_with_num_18 = paddle._C_ops.split_with_num(relu__93, 4, full_36)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_72 = split_with_num_18[0]

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_97 = paddle._C_ops.conv2d(slice_72, parameter_485, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__582, batch_norm__583, batch_norm__584, batch_norm__585, batch_norm__586, batch_norm__587 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_97, parameter_486, parameter_487, parameter_488, parameter_489, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__94 = paddle._C_ops.relu_(batch_norm__582)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_73 = split_with_num_18[1]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__50 = paddle._C_ops.add_(slice_73, relu__94)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_98 = paddle._C_ops.conv2d(add__50, parameter_490, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__588, batch_norm__589, batch_norm__590, batch_norm__591, batch_norm__592, batch_norm__593 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_98, parameter_491, parameter_492, parameter_493, parameter_494, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__95 = paddle._C_ops.relu_(batch_norm__588)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_74 = split_with_num_18[2]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__51 = paddle._C_ops.add_(slice_74, relu__95)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_99 = paddle._C_ops.conv2d(add__51, parameter_495, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__594, batch_norm__595, batch_norm__596, batch_norm__597, batch_norm__598, batch_norm__599 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_99, parameter_496, parameter_497, parameter_498, parameter_499, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__96 = paddle._C_ops.relu_(batch_norm__594)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_75 = split_with_num_18[3]

        # builtin.combine: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16)
        combine_18 = [relu__94, relu__95, relu__96, slice_75]

        # pd_op.full: (1xi32) <- ()
        full_37 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x416x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16], 1xi32)
        concat_18 = paddle._C_ops.concat(combine_18, full_37)

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x416x14x14xf16, 1024x416x1x1xf16)
        conv2d_100 = paddle._C_ops.conv2d(concat_18, parameter_500, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__600, batch_norm__601, batch_norm__602, batch_norm__603, batch_norm__604, batch_norm__605 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_100, parameter_501, parameter_502, parameter_503, parameter_504, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__52 = paddle._C_ops.add_(relu__92, batch_norm__600)

        # pd_op.relu_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16)
        relu__97 = paddle._C_ops.relu_(add__52)

        # pd_op.conv2d: (-1x416x14x14xf16) <- (-1x1024x14x14xf16, 416x1024x1x1xf16)
        conv2d_101 = paddle._C_ops.conv2d(relu__97, parameter_505, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32, None) <- (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32)
        batch_norm__606, batch_norm__607, batch_norm__608, batch_norm__609, batch_norm__610, batch_norm__611 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_101, parameter_506, parameter_507, parameter_508, parameter_509, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x416x14x14xf16) <- (-1x416x14x14xf16)
        relu__98 = paddle._C_ops.relu_(batch_norm__606)

        # pd_op.full: (1xi32) <- ()
        full_38 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x416x14x14xf16, 1xi32)
        split_with_num_19 = paddle._C_ops.split_with_num(relu__98, 4, full_38)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_76 = split_with_num_19[0]

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_102 = paddle._C_ops.conv2d(slice_76, parameter_510, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__612, batch_norm__613, batch_norm__614, batch_norm__615, batch_norm__616, batch_norm__617 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_102, parameter_511, parameter_512, parameter_513, parameter_514, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__99 = paddle._C_ops.relu_(batch_norm__612)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_77 = split_with_num_19[1]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__53 = paddle._C_ops.add_(slice_77, relu__99)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_103 = paddle._C_ops.conv2d(add__53, parameter_515, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__618, batch_norm__619, batch_norm__620, batch_norm__621, batch_norm__622, batch_norm__623 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_103, parameter_516, parameter_517, parameter_518, parameter_519, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__100 = paddle._C_ops.relu_(batch_norm__618)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_78 = split_with_num_19[2]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__54 = paddle._C_ops.add_(slice_78, relu__100)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_104 = paddle._C_ops.conv2d(add__54, parameter_520, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__624, batch_norm__625, batch_norm__626, batch_norm__627, batch_norm__628, batch_norm__629 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_104, parameter_521, parameter_522, parameter_523, parameter_524, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__101 = paddle._C_ops.relu_(batch_norm__624)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_79 = split_with_num_19[3]

        # builtin.combine: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16)
        combine_19 = [relu__99, relu__100, relu__101, slice_79]

        # pd_op.full: (1xi32) <- ()
        full_39 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x416x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16], 1xi32)
        concat_19 = paddle._C_ops.concat(combine_19, full_39)

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x416x14x14xf16, 1024x416x1x1xf16)
        conv2d_105 = paddle._C_ops.conv2d(concat_19, parameter_525, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__630, batch_norm__631, batch_norm__632, batch_norm__633, batch_norm__634, batch_norm__635 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_105, parameter_526, parameter_527, parameter_528, parameter_529, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__55 = paddle._C_ops.add_(relu__97, batch_norm__630)

        # pd_op.relu_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16)
        relu__102 = paddle._C_ops.relu_(add__55)

        # pd_op.conv2d: (-1x416x14x14xf16) <- (-1x1024x14x14xf16, 416x1024x1x1xf16)
        conv2d_106 = paddle._C_ops.conv2d(relu__102, parameter_530, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32, None) <- (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32)
        batch_norm__636, batch_norm__637, batch_norm__638, batch_norm__639, batch_norm__640, batch_norm__641 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_106, parameter_531, parameter_532, parameter_533, parameter_534, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x416x14x14xf16) <- (-1x416x14x14xf16)
        relu__103 = paddle._C_ops.relu_(batch_norm__636)

        # pd_op.full: (1xi32) <- ()
        full_40 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x416x14x14xf16, 1xi32)
        split_with_num_20 = paddle._C_ops.split_with_num(relu__103, 4, full_40)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_80 = split_with_num_20[0]

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_107 = paddle._C_ops.conv2d(slice_80, parameter_535, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__642, batch_norm__643, batch_norm__644, batch_norm__645, batch_norm__646, batch_norm__647 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_107, parameter_536, parameter_537, parameter_538, parameter_539, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__104 = paddle._C_ops.relu_(batch_norm__642)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_81 = split_with_num_20[1]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__56 = paddle._C_ops.add_(slice_81, relu__104)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_108 = paddle._C_ops.conv2d(add__56, parameter_540, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__648, batch_norm__649, batch_norm__650, batch_norm__651, batch_norm__652, batch_norm__653 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_108, parameter_541, parameter_542, parameter_543, parameter_544, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__105 = paddle._C_ops.relu_(batch_norm__648)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_82 = split_with_num_20[2]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__57 = paddle._C_ops.add_(slice_82, relu__105)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_109 = paddle._C_ops.conv2d(add__57, parameter_545, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__654, batch_norm__655, batch_norm__656, batch_norm__657, batch_norm__658, batch_norm__659 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_109, parameter_546, parameter_547, parameter_548, parameter_549, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__106 = paddle._C_ops.relu_(batch_norm__654)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_83 = split_with_num_20[3]

        # builtin.combine: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16)
        combine_20 = [relu__104, relu__105, relu__106, slice_83]

        # pd_op.full: (1xi32) <- ()
        full_41 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x416x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16], 1xi32)
        concat_20 = paddle._C_ops.concat(combine_20, full_41)

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x416x14x14xf16, 1024x416x1x1xf16)
        conv2d_110 = paddle._C_ops.conv2d(concat_20, parameter_550, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__660, batch_norm__661, batch_norm__662, batch_norm__663, batch_norm__664, batch_norm__665 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_110, parameter_551, parameter_552, parameter_553, parameter_554, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__58 = paddle._C_ops.add_(relu__102, batch_norm__660)

        # pd_op.relu_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16)
        relu__107 = paddle._C_ops.relu_(add__58)

        # pd_op.conv2d: (-1x416x14x14xf16) <- (-1x1024x14x14xf16, 416x1024x1x1xf16)
        conv2d_111 = paddle._C_ops.conv2d(relu__107, parameter_555, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32, None) <- (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32)
        batch_norm__666, batch_norm__667, batch_norm__668, batch_norm__669, batch_norm__670, batch_norm__671 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_111, parameter_556, parameter_557, parameter_558, parameter_559, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x416x14x14xf16) <- (-1x416x14x14xf16)
        relu__108 = paddle._C_ops.relu_(batch_norm__666)

        # pd_op.full: (1xi32) <- ()
        full_42 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x416x14x14xf16, 1xi32)
        split_with_num_21 = paddle._C_ops.split_with_num(relu__108, 4, full_42)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_84 = split_with_num_21[0]

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_112 = paddle._C_ops.conv2d(slice_84, parameter_560, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__672, batch_norm__673, batch_norm__674, batch_norm__675, batch_norm__676, batch_norm__677 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_112, parameter_561, parameter_562, parameter_563, parameter_564, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__109 = paddle._C_ops.relu_(batch_norm__672)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_85 = split_with_num_21[1]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__59 = paddle._C_ops.add_(slice_85, relu__109)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_113 = paddle._C_ops.conv2d(add__59, parameter_565, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__678, batch_norm__679, batch_norm__680, batch_norm__681, batch_norm__682, batch_norm__683 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_113, parameter_566, parameter_567, parameter_568, parameter_569, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__110 = paddle._C_ops.relu_(batch_norm__678)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_86 = split_with_num_21[2]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__60 = paddle._C_ops.add_(slice_86, relu__110)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_114 = paddle._C_ops.conv2d(add__60, parameter_570, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__684, batch_norm__685, batch_norm__686, batch_norm__687, batch_norm__688, batch_norm__689 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_114, parameter_571, parameter_572, parameter_573, parameter_574, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__111 = paddle._C_ops.relu_(batch_norm__684)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_87 = split_with_num_21[3]

        # builtin.combine: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16)
        combine_21 = [relu__109, relu__110, relu__111, slice_87]

        # pd_op.full: (1xi32) <- ()
        full_43 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x416x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16], 1xi32)
        concat_21 = paddle._C_ops.concat(combine_21, full_43)

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x416x14x14xf16, 1024x416x1x1xf16)
        conv2d_115 = paddle._C_ops.conv2d(concat_21, parameter_575, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__690, batch_norm__691, batch_norm__692, batch_norm__693, batch_norm__694, batch_norm__695 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_115, parameter_576, parameter_577, parameter_578, parameter_579, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__61 = paddle._C_ops.add_(relu__107, batch_norm__690)

        # pd_op.relu_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16)
        relu__112 = paddle._C_ops.relu_(add__61)

        # pd_op.conv2d: (-1x416x14x14xf16) <- (-1x1024x14x14xf16, 416x1024x1x1xf16)
        conv2d_116 = paddle._C_ops.conv2d(relu__112, parameter_580, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32, None) <- (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32)
        batch_norm__696, batch_norm__697, batch_norm__698, batch_norm__699, batch_norm__700, batch_norm__701 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_116, parameter_581, parameter_582, parameter_583, parameter_584, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x416x14x14xf16) <- (-1x416x14x14xf16)
        relu__113 = paddle._C_ops.relu_(batch_norm__696)

        # pd_op.full: (1xi32) <- ()
        full_44 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x416x14x14xf16, 1xi32)
        split_with_num_22 = paddle._C_ops.split_with_num(relu__113, 4, full_44)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_88 = split_with_num_22[0]

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_117 = paddle._C_ops.conv2d(slice_88, parameter_585, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__702, batch_norm__703, batch_norm__704, batch_norm__705, batch_norm__706, batch_norm__707 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_117, parameter_586, parameter_587, parameter_588, parameter_589, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__114 = paddle._C_ops.relu_(batch_norm__702)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_89 = split_with_num_22[1]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__62 = paddle._C_ops.add_(slice_89, relu__114)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_118 = paddle._C_ops.conv2d(add__62, parameter_590, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__708, batch_norm__709, batch_norm__710, batch_norm__711, batch_norm__712, batch_norm__713 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_118, parameter_591, parameter_592, parameter_593, parameter_594, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__115 = paddle._C_ops.relu_(batch_norm__708)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_90 = split_with_num_22[2]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__63 = paddle._C_ops.add_(slice_90, relu__115)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_119 = paddle._C_ops.conv2d(add__63, parameter_595, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__714, batch_norm__715, batch_norm__716, batch_norm__717, batch_norm__718, batch_norm__719 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_119, parameter_596, parameter_597, parameter_598, parameter_599, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__116 = paddle._C_ops.relu_(batch_norm__714)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_91 = split_with_num_22[3]

        # builtin.combine: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16)
        combine_22 = [relu__114, relu__115, relu__116, slice_91]

        # pd_op.full: (1xi32) <- ()
        full_45 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x416x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16], 1xi32)
        concat_22 = paddle._C_ops.concat(combine_22, full_45)

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x416x14x14xf16, 1024x416x1x1xf16)
        conv2d_120 = paddle._C_ops.conv2d(concat_22, parameter_600, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__720, batch_norm__721, batch_norm__722, batch_norm__723, batch_norm__724, batch_norm__725 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_120, parameter_601, parameter_602, parameter_603, parameter_604, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__64 = paddle._C_ops.add_(relu__112, batch_norm__720)

        # pd_op.relu_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16)
        relu__117 = paddle._C_ops.relu_(add__64)

        # pd_op.conv2d: (-1x416x14x14xf16) <- (-1x1024x14x14xf16, 416x1024x1x1xf16)
        conv2d_121 = paddle._C_ops.conv2d(relu__117, parameter_605, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32, None) <- (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32)
        batch_norm__726, batch_norm__727, batch_norm__728, batch_norm__729, batch_norm__730, batch_norm__731 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_121, parameter_606, parameter_607, parameter_608, parameter_609, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x416x14x14xf16) <- (-1x416x14x14xf16)
        relu__118 = paddle._C_ops.relu_(batch_norm__726)

        # pd_op.full: (1xi32) <- ()
        full_46 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x416x14x14xf16, 1xi32)
        split_with_num_23 = paddle._C_ops.split_with_num(relu__118, 4, full_46)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_92 = split_with_num_23[0]

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_122 = paddle._C_ops.conv2d(slice_92, parameter_610, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__732, batch_norm__733, batch_norm__734, batch_norm__735, batch_norm__736, batch_norm__737 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_122, parameter_611, parameter_612, parameter_613, parameter_614, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__119 = paddle._C_ops.relu_(batch_norm__732)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_93 = split_with_num_23[1]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__65 = paddle._C_ops.add_(slice_93, relu__119)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_123 = paddle._C_ops.conv2d(add__65, parameter_615, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__738, batch_norm__739, batch_norm__740, batch_norm__741, batch_norm__742, batch_norm__743 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_123, parameter_616, parameter_617, parameter_618, parameter_619, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__120 = paddle._C_ops.relu_(batch_norm__738)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_94 = split_with_num_23[2]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__66 = paddle._C_ops.add_(slice_94, relu__120)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_124 = paddle._C_ops.conv2d(add__66, parameter_620, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__744, batch_norm__745, batch_norm__746, batch_norm__747, batch_norm__748, batch_norm__749 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_124, parameter_621, parameter_622, parameter_623, parameter_624, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__121 = paddle._C_ops.relu_(batch_norm__744)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_95 = split_with_num_23[3]

        # builtin.combine: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16)
        combine_23 = [relu__119, relu__120, relu__121, slice_95]

        # pd_op.full: (1xi32) <- ()
        full_47 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x416x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16], 1xi32)
        concat_23 = paddle._C_ops.concat(combine_23, full_47)

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x416x14x14xf16, 1024x416x1x1xf16)
        conv2d_125 = paddle._C_ops.conv2d(concat_23, parameter_625, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__750, batch_norm__751, batch_norm__752, batch_norm__753, batch_norm__754, batch_norm__755 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_125, parameter_626, parameter_627, parameter_628, parameter_629, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__67 = paddle._C_ops.add_(relu__117, batch_norm__750)

        # pd_op.relu_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16)
        relu__122 = paddle._C_ops.relu_(add__67)

        # pd_op.conv2d: (-1x416x14x14xf16) <- (-1x1024x14x14xf16, 416x1024x1x1xf16)
        conv2d_126 = paddle._C_ops.conv2d(relu__122, parameter_630, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32, None) <- (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32)
        batch_norm__756, batch_norm__757, batch_norm__758, batch_norm__759, batch_norm__760, batch_norm__761 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_126, parameter_631, parameter_632, parameter_633, parameter_634, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x416x14x14xf16) <- (-1x416x14x14xf16)
        relu__123 = paddle._C_ops.relu_(batch_norm__756)

        # pd_op.full: (1xi32) <- ()
        full_48 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x416x14x14xf16, 1xi32)
        split_with_num_24 = paddle._C_ops.split_with_num(relu__123, 4, full_48)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_96 = split_with_num_24[0]

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_127 = paddle._C_ops.conv2d(slice_96, parameter_635, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__762, batch_norm__763, batch_norm__764, batch_norm__765, batch_norm__766, batch_norm__767 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_127, parameter_636, parameter_637, parameter_638, parameter_639, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__124 = paddle._C_ops.relu_(batch_norm__762)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_97 = split_with_num_24[1]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__68 = paddle._C_ops.add_(slice_97, relu__124)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_128 = paddle._C_ops.conv2d(add__68, parameter_640, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__768, batch_norm__769, batch_norm__770, batch_norm__771, batch_norm__772, batch_norm__773 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_128, parameter_641, parameter_642, parameter_643, parameter_644, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__125 = paddle._C_ops.relu_(batch_norm__768)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_98 = split_with_num_24[2]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__69 = paddle._C_ops.add_(slice_98, relu__125)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_129 = paddle._C_ops.conv2d(add__69, parameter_645, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__774, batch_norm__775, batch_norm__776, batch_norm__777, batch_norm__778, batch_norm__779 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_129, parameter_646, parameter_647, parameter_648, parameter_649, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__126 = paddle._C_ops.relu_(batch_norm__774)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_99 = split_with_num_24[3]

        # builtin.combine: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16)
        combine_24 = [relu__124, relu__125, relu__126, slice_99]

        # pd_op.full: (1xi32) <- ()
        full_49 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x416x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16], 1xi32)
        concat_24 = paddle._C_ops.concat(combine_24, full_49)

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x416x14x14xf16, 1024x416x1x1xf16)
        conv2d_130 = paddle._C_ops.conv2d(concat_24, parameter_650, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__780, batch_norm__781, batch_norm__782, batch_norm__783, batch_norm__784, batch_norm__785 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_130, parameter_651, parameter_652, parameter_653, parameter_654, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__70 = paddle._C_ops.add_(relu__122, batch_norm__780)

        # pd_op.relu_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16)
        relu__127 = paddle._C_ops.relu_(add__70)

        # pd_op.conv2d: (-1x416x14x14xf16) <- (-1x1024x14x14xf16, 416x1024x1x1xf16)
        conv2d_131 = paddle._C_ops.conv2d(relu__127, parameter_655, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32, None) <- (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32)
        batch_norm__786, batch_norm__787, batch_norm__788, batch_norm__789, batch_norm__790, batch_norm__791 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_131, parameter_656, parameter_657, parameter_658, parameter_659, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x416x14x14xf16) <- (-1x416x14x14xf16)
        relu__128 = paddle._C_ops.relu_(batch_norm__786)

        # pd_op.full: (1xi32) <- ()
        full_50 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x416x14x14xf16, 1xi32)
        split_with_num_25 = paddle._C_ops.split_with_num(relu__128, 4, full_50)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_100 = split_with_num_25[0]

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_132 = paddle._C_ops.conv2d(slice_100, parameter_660, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__792, batch_norm__793, batch_norm__794, batch_norm__795, batch_norm__796, batch_norm__797 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_132, parameter_661, parameter_662, parameter_663, parameter_664, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__129 = paddle._C_ops.relu_(batch_norm__792)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_101 = split_with_num_25[1]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__71 = paddle._C_ops.add_(slice_101, relu__129)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_133 = paddle._C_ops.conv2d(add__71, parameter_665, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__798, batch_norm__799, batch_norm__800, batch_norm__801, batch_norm__802, batch_norm__803 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_133, parameter_666, parameter_667, parameter_668, parameter_669, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__130 = paddle._C_ops.relu_(batch_norm__798)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_102 = split_with_num_25[2]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__72 = paddle._C_ops.add_(slice_102, relu__130)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_134 = paddle._C_ops.conv2d(add__72, parameter_670, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__804, batch_norm__805, batch_norm__806, batch_norm__807, batch_norm__808, batch_norm__809 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_134, parameter_671, parameter_672, parameter_673, parameter_674, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__131 = paddle._C_ops.relu_(batch_norm__804)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_103 = split_with_num_25[3]

        # builtin.combine: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16)
        combine_25 = [relu__129, relu__130, relu__131, slice_103]

        # pd_op.full: (1xi32) <- ()
        full_51 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x416x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16], 1xi32)
        concat_25 = paddle._C_ops.concat(combine_25, full_51)

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x416x14x14xf16, 1024x416x1x1xf16)
        conv2d_135 = paddle._C_ops.conv2d(concat_25, parameter_675, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__810, batch_norm__811, batch_norm__812, batch_norm__813, batch_norm__814, batch_norm__815 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_135, parameter_676, parameter_677, parameter_678, parameter_679, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__73 = paddle._C_ops.add_(relu__127, batch_norm__810)

        # pd_op.relu_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16)
        relu__132 = paddle._C_ops.relu_(add__73)

        # pd_op.conv2d: (-1x416x14x14xf16) <- (-1x1024x14x14xf16, 416x1024x1x1xf16)
        conv2d_136 = paddle._C_ops.conv2d(relu__132, parameter_680, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32, None) <- (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32)
        batch_norm__816, batch_norm__817, batch_norm__818, batch_norm__819, batch_norm__820, batch_norm__821 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_136, parameter_681, parameter_682, parameter_683, parameter_684, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x416x14x14xf16) <- (-1x416x14x14xf16)
        relu__133 = paddle._C_ops.relu_(batch_norm__816)

        # pd_op.full: (1xi32) <- ()
        full_52 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x416x14x14xf16, 1xi32)
        split_with_num_26 = paddle._C_ops.split_with_num(relu__133, 4, full_52)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_104 = split_with_num_26[0]

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_137 = paddle._C_ops.conv2d(slice_104, parameter_685, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__822, batch_norm__823, batch_norm__824, batch_norm__825, batch_norm__826, batch_norm__827 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_137, parameter_686, parameter_687, parameter_688, parameter_689, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__134 = paddle._C_ops.relu_(batch_norm__822)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_105 = split_with_num_26[1]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__74 = paddle._C_ops.add_(slice_105, relu__134)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_138 = paddle._C_ops.conv2d(add__74, parameter_690, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__828, batch_norm__829, batch_norm__830, batch_norm__831, batch_norm__832, batch_norm__833 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_138, parameter_691, parameter_692, parameter_693, parameter_694, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__135 = paddle._C_ops.relu_(batch_norm__828)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_106 = split_with_num_26[2]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__75 = paddle._C_ops.add_(slice_106, relu__135)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_139 = paddle._C_ops.conv2d(add__75, parameter_695, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__834, batch_norm__835, batch_norm__836, batch_norm__837, batch_norm__838, batch_norm__839 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_139, parameter_696, parameter_697, parameter_698, parameter_699, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__136 = paddle._C_ops.relu_(batch_norm__834)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_107 = split_with_num_26[3]

        # builtin.combine: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16)
        combine_26 = [relu__134, relu__135, relu__136, slice_107]

        # pd_op.full: (1xi32) <- ()
        full_53 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x416x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16], 1xi32)
        concat_26 = paddle._C_ops.concat(combine_26, full_53)

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x416x14x14xf16, 1024x416x1x1xf16)
        conv2d_140 = paddle._C_ops.conv2d(concat_26, parameter_700, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__840, batch_norm__841, batch_norm__842, batch_norm__843, batch_norm__844, batch_norm__845 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_140, parameter_701, parameter_702, parameter_703, parameter_704, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__76 = paddle._C_ops.add_(relu__132, batch_norm__840)

        # pd_op.relu_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16)
        relu__137 = paddle._C_ops.relu_(add__76)

        # pd_op.conv2d: (-1x416x14x14xf16) <- (-1x1024x14x14xf16, 416x1024x1x1xf16)
        conv2d_141 = paddle._C_ops.conv2d(relu__137, parameter_705, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32, None) <- (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32)
        batch_norm__846, batch_norm__847, batch_norm__848, batch_norm__849, batch_norm__850, batch_norm__851 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_141, parameter_706, parameter_707, parameter_708, parameter_709, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x416x14x14xf16) <- (-1x416x14x14xf16)
        relu__138 = paddle._C_ops.relu_(batch_norm__846)

        # pd_op.full: (1xi32) <- ()
        full_54 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x416x14x14xf16, 1xi32)
        split_with_num_27 = paddle._C_ops.split_with_num(relu__138, 4, full_54)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_108 = split_with_num_27[0]

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_142 = paddle._C_ops.conv2d(slice_108, parameter_710, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__852, batch_norm__853, batch_norm__854, batch_norm__855, batch_norm__856, batch_norm__857 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_142, parameter_711, parameter_712, parameter_713, parameter_714, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__139 = paddle._C_ops.relu_(batch_norm__852)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_109 = split_with_num_27[1]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__77 = paddle._C_ops.add_(slice_109, relu__139)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_143 = paddle._C_ops.conv2d(add__77, parameter_715, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__858, batch_norm__859, batch_norm__860, batch_norm__861, batch_norm__862, batch_norm__863 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_143, parameter_716, parameter_717, parameter_718, parameter_719, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__140 = paddle._C_ops.relu_(batch_norm__858)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_110 = split_with_num_27[2]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__78 = paddle._C_ops.add_(slice_110, relu__140)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_144 = paddle._C_ops.conv2d(add__78, parameter_720, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__864, batch_norm__865, batch_norm__866, batch_norm__867, batch_norm__868, batch_norm__869 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_144, parameter_721, parameter_722, parameter_723, parameter_724, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__141 = paddle._C_ops.relu_(batch_norm__864)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_111 = split_with_num_27[3]

        # builtin.combine: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16)
        combine_27 = [relu__139, relu__140, relu__141, slice_111]

        # pd_op.full: (1xi32) <- ()
        full_55 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x416x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16], 1xi32)
        concat_27 = paddle._C_ops.concat(combine_27, full_55)

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x416x14x14xf16, 1024x416x1x1xf16)
        conv2d_145 = paddle._C_ops.conv2d(concat_27, parameter_725, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__870, batch_norm__871, batch_norm__872, batch_norm__873, batch_norm__874, batch_norm__875 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_145, parameter_726, parameter_727, parameter_728, parameter_729, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__79 = paddle._C_ops.add_(relu__137, batch_norm__870)

        # pd_op.relu_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16)
        relu__142 = paddle._C_ops.relu_(add__79)

        # pd_op.conv2d: (-1x416x14x14xf16) <- (-1x1024x14x14xf16, 416x1024x1x1xf16)
        conv2d_146 = paddle._C_ops.conv2d(relu__142, parameter_730, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32, None) <- (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32)
        batch_norm__876, batch_norm__877, batch_norm__878, batch_norm__879, batch_norm__880, batch_norm__881 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_146, parameter_731, parameter_732, parameter_733, parameter_734, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x416x14x14xf16) <- (-1x416x14x14xf16)
        relu__143 = paddle._C_ops.relu_(batch_norm__876)

        # pd_op.full: (1xi32) <- ()
        full_56 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x416x14x14xf16, 1xi32)
        split_with_num_28 = paddle._C_ops.split_with_num(relu__143, 4, full_56)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_112 = split_with_num_28[0]

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_147 = paddle._C_ops.conv2d(slice_112, parameter_735, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__882, batch_norm__883, batch_norm__884, batch_norm__885, batch_norm__886, batch_norm__887 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_147, parameter_736, parameter_737, parameter_738, parameter_739, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__144 = paddle._C_ops.relu_(batch_norm__882)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_113 = split_with_num_28[1]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__80 = paddle._C_ops.add_(slice_113, relu__144)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_148 = paddle._C_ops.conv2d(add__80, parameter_740, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__888, batch_norm__889, batch_norm__890, batch_norm__891, batch_norm__892, batch_norm__893 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_148, parameter_741, parameter_742, parameter_743, parameter_744, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__145 = paddle._C_ops.relu_(batch_norm__888)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_114 = split_with_num_28[2]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__81 = paddle._C_ops.add_(slice_114, relu__145)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_149 = paddle._C_ops.conv2d(add__81, parameter_745, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__894, batch_norm__895, batch_norm__896, batch_norm__897, batch_norm__898, batch_norm__899 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_149, parameter_746, parameter_747, parameter_748, parameter_749, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__146 = paddle._C_ops.relu_(batch_norm__894)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_115 = split_with_num_28[3]

        # builtin.combine: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16)
        combine_28 = [relu__144, relu__145, relu__146, slice_115]

        # pd_op.full: (1xi32) <- ()
        full_57 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x416x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16], 1xi32)
        concat_28 = paddle._C_ops.concat(combine_28, full_57)

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x416x14x14xf16, 1024x416x1x1xf16)
        conv2d_150 = paddle._C_ops.conv2d(concat_28, parameter_750, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__900, batch_norm__901, batch_norm__902, batch_norm__903, batch_norm__904, batch_norm__905 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_150, parameter_751, parameter_752, parameter_753, parameter_754, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__82 = paddle._C_ops.add_(relu__142, batch_norm__900)

        # pd_op.relu_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16)
        relu__147 = paddle._C_ops.relu_(add__82)

        # pd_op.conv2d: (-1x416x14x14xf16) <- (-1x1024x14x14xf16, 416x1024x1x1xf16)
        conv2d_151 = paddle._C_ops.conv2d(relu__147, parameter_755, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32, None) <- (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32)
        batch_norm__906, batch_norm__907, batch_norm__908, batch_norm__909, batch_norm__910, batch_norm__911 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_151, parameter_756, parameter_757, parameter_758, parameter_759, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x416x14x14xf16) <- (-1x416x14x14xf16)
        relu__148 = paddle._C_ops.relu_(batch_norm__906)

        # pd_op.full: (1xi32) <- ()
        full_58 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x416x14x14xf16, 1xi32)
        split_with_num_29 = paddle._C_ops.split_with_num(relu__148, 4, full_58)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_116 = split_with_num_29[0]

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_152 = paddle._C_ops.conv2d(slice_116, parameter_760, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__912, batch_norm__913, batch_norm__914, batch_norm__915, batch_norm__916, batch_norm__917 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_152, parameter_761, parameter_762, parameter_763, parameter_764, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__149 = paddle._C_ops.relu_(batch_norm__912)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_117 = split_with_num_29[1]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__83 = paddle._C_ops.add_(slice_117, relu__149)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_153 = paddle._C_ops.conv2d(add__83, parameter_765, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__918, batch_norm__919, batch_norm__920, batch_norm__921, batch_norm__922, batch_norm__923 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_153, parameter_766, parameter_767, parameter_768, parameter_769, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__150 = paddle._C_ops.relu_(batch_norm__918)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_118 = split_with_num_29[2]

        # pd_op.add_: (-1x104x14x14xf16) <- (-1x104x14x14xf16, -1x104x14x14xf16)
        add__84 = paddle._C_ops.add_(slice_118, relu__150)

        # pd_op.conv2d: (-1x104x14x14xf16) <- (-1x104x14x14xf16, 104x104x3x3xf16)
        conv2d_154 = paddle._C_ops.conv2d(add__84, parameter_770, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32, None) <- (-1x104x14x14xf16, 104xf32, 104xf32, 104xf32, 104xf32)
        batch_norm__924, batch_norm__925, batch_norm__926, batch_norm__927, batch_norm__928, batch_norm__929 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_154, parameter_771, parameter_772, parameter_773, parameter_774, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x104x14x14xf16) <- (-1x104x14x14xf16)
        relu__151 = paddle._C_ops.relu_(batch_norm__924)

        # builtin.slice: (-1x104x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16])
        slice_119 = split_with_num_29[3]

        # builtin.combine: ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16]) <- (-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16)
        combine_29 = [relu__149, relu__150, relu__151, slice_119]

        # pd_op.full: (1xi32) <- ()
        full_59 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x416x14x14xf16) <- ([-1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16, -1x104x14x14xf16], 1xi32)
        concat_29 = paddle._C_ops.concat(combine_29, full_59)

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x416x14x14xf16, 1024x416x1x1xf16)
        conv2d_155 = paddle._C_ops.conv2d(concat_29, parameter_775, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__930, batch_norm__931, batch_norm__932, batch_norm__933, batch_norm__934, batch_norm__935 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_155, parameter_776, parameter_777, parameter_778, parameter_779, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16, -1x1024x14x14xf16)
        add__85 = paddle._C_ops.add_(relu__147, batch_norm__930)

        # pd_op.relu_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16)
        relu__152 = paddle._C_ops.relu_(add__85)

        # pd_op.conv2d: (-1x832x14x14xf16) <- (-1x1024x14x14xf16, 832x1024x1x1xf16)
        conv2d_156 = paddle._C_ops.conv2d(relu__152, parameter_780, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x832x14x14xf16, 832xf32, 832xf32, 832xf32, 832xf32, None) <- (-1x832x14x14xf16, 832xf32, 832xf32, 832xf32, 832xf32)
        batch_norm__936, batch_norm__937, batch_norm__938, batch_norm__939, batch_norm__940, batch_norm__941 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_156, parameter_781, parameter_782, parameter_783, parameter_784, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x832x14x14xf16) <- (-1x832x14x14xf16)
        relu__153 = paddle._C_ops.relu_(batch_norm__936)

        # pd_op.full: (1xi32) <- ()
        full_60 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x208x14x14xf16, -1x208x14x14xf16, -1x208x14x14xf16, -1x208x14x14xf16]) <- (-1x832x14x14xf16, 1xi32)
        split_with_num_30 = paddle._C_ops.split_with_num(relu__153, 4, full_60)

        # builtin.slice: (-1x208x14x14xf16) <- ([-1x208x14x14xf16, -1x208x14x14xf16, -1x208x14x14xf16, -1x208x14x14xf16])
        slice_120 = split_with_num_30[0]

        # pd_op.conv2d: (-1x208x7x7xf16) <- (-1x208x14x14xf16, 208x208x3x3xf16)
        conv2d_157 = paddle._C_ops.conv2d(slice_120, parameter_785, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x208x7x7xf16, 208xf32, 208xf32, 208xf32, 208xf32, None) <- (-1x208x7x7xf16, 208xf32, 208xf32, 208xf32, 208xf32)
        batch_norm__942, batch_norm__943, batch_norm__944, batch_norm__945, batch_norm__946, batch_norm__947 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_157, parameter_786, parameter_787, parameter_788, parameter_789, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x208x7x7xf16) <- (-1x208x7x7xf16)
        relu__154 = paddle._C_ops.relu_(batch_norm__942)

        # builtin.slice: (-1x208x14x14xf16) <- ([-1x208x14x14xf16, -1x208x14x14xf16, -1x208x14x14xf16, -1x208x14x14xf16])
        slice_121 = split_with_num_30[1]

        # pd_op.conv2d: (-1x208x7x7xf16) <- (-1x208x14x14xf16, 208x208x3x3xf16)
        conv2d_158 = paddle._C_ops.conv2d(slice_121, parameter_790, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x208x7x7xf16, 208xf32, 208xf32, 208xf32, 208xf32, None) <- (-1x208x7x7xf16, 208xf32, 208xf32, 208xf32, 208xf32)
        batch_norm__948, batch_norm__949, batch_norm__950, batch_norm__951, batch_norm__952, batch_norm__953 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_158, parameter_791, parameter_792, parameter_793, parameter_794, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x208x7x7xf16) <- (-1x208x7x7xf16)
        relu__155 = paddle._C_ops.relu_(batch_norm__948)

        # builtin.slice: (-1x208x14x14xf16) <- ([-1x208x14x14xf16, -1x208x14x14xf16, -1x208x14x14xf16, -1x208x14x14xf16])
        slice_122 = split_with_num_30[2]

        # pd_op.conv2d: (-1x208x7x7xf16) <- (-1x208x14x14xf16, 208x208x3x3xf16)
        conv2d_159 = paddle._C_ops.conv2d(slice_122, parameter_795, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x208x7x7xf16, 208xf32, 208xf32, 208xf32, 208xf32, None) <- (-1x208x7x7xf16, 208xf32, 208xf32, 208xf32, 208xf32)
        batch_norm__954, batch_norm__955, batch_norm__956, batch_norm__957, batch_norm__958, batch_norm__959 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_159, parameter_796, parameter_797, parameter_798, parameter_799, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x208x7x7xf16) <- (-1x208x7x7xf16)
        relu__156 = paddle._C_ops.relu_(batch_norm__954)

        # builtin.slice: (-1x208x14x14xf16) <- ([-1x208x14x14xf16, -1x208x14x14xf16, -1x208x14x14xf16, -1x208x14x14xf16])
        slice_123 = split_with_num_30[3]

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [3, 3]

        # pd_op.pool2d: (-1x208x7x7xf16) <- (-1x208x14x14xf16, 2xi64)
        pool2d_5 = paddle._C_ops.pool2d(slice_123, full_int_array_5, [2, 2], [1, 1], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # builtin.combine: ([-1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16]) <- (-1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16)
        combine_30 = [relu__154, relu__155, relu__156, pool2d_5]

        # pd_op.full: (1xi32) <- ()
        full_61 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x832x7x7xf16) <- ([-1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16], 1xi32)
        concat_30 = paddle._C_ops.concat(combine_30, full_61)

        # pd_op.conv2d: (-1x2048x7x7xf16) <- (-1x832x7x7xf16, 2048x832x1x1xf16)
        conv2d_160 = paddle._C_ops.conv2d(concat_30, parameter_800, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x2048x7x7xf16, 2048xf32, 2048xf32, 2048xf32, 2048xf32, None) <- (-1x2048x7x7xf16, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        batch_norm__960, batch_norm__961, batch_norm__962, batch_norm__963, batch_norm__964, batch_norm__965 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_160, parameter_801, parameter_802, parameter_803, parameter_804, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_6 = [2, 2]

        # pd_op.pool2d: (-1x1024x7x7xf16) <- (-1x1024x14x14xf16, 2xi64)
        pool2d_6 = paddle._C_ops.pool2d(relu__152, full_int_array_6, [2, 2], [0, 0], True, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x2048x7x7xf16) <- (-1x1024x7x7xf16, 2048x1024x1x1xf16)
        conv2d_161 = paddle._C_ops.conv2d(pool2d_6, parameter_805, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x2048x7x7xf16, 2048xf32, 2048xf32, 2048xf32, 2048xf32, None) <- (-1x2048x7x7xf16, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        batch_norm__966, batch_norm__967, batch_norm__968, batch_norm__969, batch_norm__970, batch_norm__971 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_161, parameter_806, parameter_807, parameter_808, parameter_809, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x2048x7x7xf16) <- (-1x2048x7x7xf16, -1x2048x7x7xf16)
        add__86 = paddle._C_ops.add_(batch_norm__966, batch_norm__960)

        # pd_op.relu_: (-1x2048x7x7xf16) <- (-1x2048x7x7xf16)
        relu__157 = paddle._C_ops.relu_(add__86)

        # pd_op.conv2d: (-1x832x7x7xf16) <- (-1x2048x7x7xf16, 832x2048x1x1xf16)
        conv2d_162 = paddle._C_ops.conv2d(relu__157, parameter_810, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x832x7x7xf16, 832xf32, 832xf32, 832xf32, 832xf32, None) <- (-1x832x7x7xf16, 832xf32, 832xf32, 832xf32, 832xf32)
        batch_norm__972, batch_norm__973, batch_norm__974, batch_norm__975, batch_norm__976, batch_norm__977 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_162, parameter_811, parameter_812, parameter_813, parameter_814, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x832x7x7xf16) <- (-1x832x7x7xf16)
        relu__158 = paddle._C_ops.relu_(batch_norm__972)

        # pd_op.full: (1xi32) <- ()
        full_62 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16]) <- (-1x832x7x7xf16, 1xi32)
        split_with_num_31 = paddle._C_ops.split_with_num(relu__158, 4, full_62)

        # builtin.slice: (-1x208x7x7xf16) <- ([-1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16])
        slice_124 = split_with_num_31[0]

        # pd_op.conv2d: (-1x208x7x7xf16) <- (-1x208x7x7xf16, 208x208x3x3xf16)
        conv2d_163 = paddle._C_ops.conv2d(slice_124, parameter_815, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x208x7x7xf16, 208xf32, 208xf32, 208xf32, 208xf32, None) <- (-1x208x7x7xf16, 208xf32, 208xf32, 208xf32, 208xf32)
        batch_norm__978, batch_norm__979, batch_norm__980, batch_norm__981, batch_norm__982, batch_norm__983 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_163, parameter_816, parameter_817, parameter_818, parameter_819, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x208x7x7xf16) <- (-1x208x7x7xf16)
        relu__159 = paddle._C_ops.relu_(batch_norm__978)

        # builtin.slice: (-1x208x7x7xf16) <- ([-1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16])
        slice_125 = split_with_num_31[1]

        # pd_op.add_: (-1x208x7x7xf16) <- (-1x208x7x7xf16, -1x208x7x7xf16)
        add__87 = paddle._C_ops.add_(slice_125, relu__159)

        # pd_op.conv2d: (-1x208x7x7xf16) <- (-1x208x7x7xf16, 208x208x3x3xf16)
        conv2d_164 = paddle._C_ops.conv2d(add__87, parameter_820, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x208x7x7xf16, 208xf32, 208xf32, 208xf32, 208xf32, None) <- (-1x208x7x7xf16, 208xf32, 208xf32, 208xf32, 208xf32)
        batch_norm__984, batch_norm__985, batch_norm__986, batch_norm__987, batch_norm__988, batch_norm__989 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_164, parameter_821, parameter_822, parameter_823, parameter_824, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x208x7x7xf16) <- (-1x208x7x7xf16)
        relu__160 = paddle._C_ops.relu_(batch_norm__984)

        # builtin.slice: (-1x208x7x7xf16) <- ([-1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16])
        slice_126 = split_with_num_31[2]

        # pd_op.add_: (-1x208x7x7xf16) <- (-1x208x7x7xf16, -1x208x7x7xf16)
        add__88 = paddle._C_ops.add_(slice_126, relu__160)

        # pd_op.conv2d: (-1x208x7x7xf16) <- (-1x208x7x7xf16, 208x208x3x3xf16)
        conv2d_165 = paddle._C_ops.conv2d(add__88, parameter_825, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x208x7x7xf16, 208xf32, 208xf32, 208xf32, 208xf32, None) <- (-1x208x7x7xf16, 208xf32, 208xf32, 208xf32, 208xf32)
        batch_norm__990, batch_norm__991, batch_norm__992, batch_norm__993, batch_norm__994, batch_norm__995 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_165, parameter_826, parameter_827, parameter_828, parameter_829, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x208x7x7xf16) <- (-1x208x7x7xf16)
        relu__161 = paddle._C_ops.relu_(batch_norm__990)

        # builtin.slice: (-1x208x7x7xf16) <- ([-1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16])
        slice_127 = split_with_num_31[3]

        # builtin.combine: ([-1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16]) <- (-1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16)
        combine_31 = [relu__159, relu__160, relu__161, slice_127]

        # pd_op.full: (1xi32) <- ()
        full_63 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x832x7x7xf16) <- ([-1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16], 1xi32)
        concat_31 = paddle._C_ops.concat(combine_31, full_63)

        # pd_op.conv2d: (-1x2048x7x7xf16) <- (-1x832x7x7xf16, 2048x832x1x1xf16)
        conv2d_166 = paddle._C_ops.conv2d(concat_31, parameter_830, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x2048x7x7xf16, 2048xf32, 2048xf32, 2048xf32, 2048xf32, None) <- (-1x2048x7x7xf16, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        batch_norm__996, batch_norm__997, batch_norm__998, batch_norm__999, batch_norm__1000, batch_norm__1001 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_166, parameter_831, parameter_832, parameter_833, parameter_834, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x2048x7x7xf16) <- (-1x2048x7x7xf16, -1x2048x7x7xf16)
        add__89 = paddle._C_ops.add_(relu__157, batch_norm__996)

        # pd_op.relu_: (-1x2048x7x7xf16) <- (-1x2048x7x7xf16)
        relu__162 = paddle._C_ops.relu_(add__89)

        # pd_op.conv2d: (-1x832x7x7xf16) <- (-1x2048x7x7xf16, 832x2048x1x1xf16)
        conv2d_167 = paddle._C_ops.conv2d(relu__162, parameter_835, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x832x7x7xf16, 832xf32, 832xf32, 832xf32, 832xf32, None) <- (-1x832x7x7xf16, 832xf32, 832xf32, 832xf32, 832xf32)
        batch_norm__1002, batch_norm__1003, batch_norm__1004, batch_norm__1005, batch_norm__1006, batch_norm__1007 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_167, parameter_836, parameter_837, parameter_838, parameter_839, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x832x7x7xf16) <- (-1x832x7x7xf16)
        relu__163 = paddle._C_ops.relu_(batch_norm__1002)

        # pd_op.full: (1xi32) <- ()
        full_64 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16]) <- (-1x832x7x7xf16, 1xi32)
        split_with_num_32 = paddle._C_ops.split_with_num(relu__163, 4, full_64)

        # builtin.slice: (-1x208x7x7xf16) <- ([-1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16])
        slice_128 = split_with_num_32[0]

        # pd_op.conv2d: (-1x208x7x7xf16) <- (-1x208x7x7xf16, 208x208x3x3xf16)
        conv2d_168 = paddle._C_ops.conv2d(slice_128, parameter_840, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x208x7x7xf16, 208xf32, 208xf32, 208xf32, 208xf32, None) <- (-1x208x7x7xf16, 208xf32, 208xf32, 208xf32, 208xf32)
        batch_norm__1008, batch_norm__1009, batch_norm__1010, batch_norm__1011, batch_norm__1012, batch_norm__1013 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_168, parameter_841, parameter_842, parameter_843, parameter_844, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x208x7x7xf16) <- (-1x208x7x7xf16)
        relu__164 = paddle._C_ops.relu_(batch_norm__1008)

        # builtin.slice: (-1x208x7x7xf16) <- ([-1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16])
        slice_129 = split_with_num_32[1]

        # pd_op.add_: (-1x208x7x7xf16) <- (-1x208x7x7xf16, -1x208x7x7xf16)
        add__90 = paddle._C_ops.add_(slice_129, relu__164)

        # pd_op.conv2d: (-1x208x7x7xf16) <- (-1x208x7x7xf16, 208x208x3x3xf16)
        conv2d_169 = paddle._C_ops.conv2d(add__90, parameter_845, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x208x7x7xf16, 208xf32, 208xf32, 208xf32, 208xf32, None) <- (-1x208x7x7xf16, 208xf32, 208xf32, 208xf32, 208xf32)
        batch_norm__1014, batch_norm__1015, batch_norm__1016, batch_norm__1017, batch_norm__1018, batch_norm__1019 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_169, parameter_846, parameter_847, parameter_848, parameter_849, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x208x7x7xf16) <- (-1x208x7x7xf16)
        relu__165 = paddle._C_ops.relu_(batch_norm__1014)

        # builtin.slice: (-1x208x7x7xf16) <- ([-1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16])
        slice_130 = split_with_num_32[2]

        # pd_op.add_: (-1x208x7x7xf16) <- (-1x208x7x7xf16, -1x208x7x7xf16)
        add__91 = paddle._C_ops.add_(slice_130, relu__165)

        # pd_op.conv2d: (-1x208x7x7xf16) <- (-1x208x7x7xf16, 208x208x3x3xf16)
        conv2d_170 = paddle._C_ops.conv2d(add__91, parameter_850, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x208x7x7xf16, 208xf32, 208xf32, 208xf32, 208xf32, None) <- (-1x208x7x7xf16, 208xf32, 208xf32, 208xf32, 208xf32)
        batch_norm__1020, batch_norm__1021, batch_norm__1022, batch_norm__1023, batch_norm__1024, batch_norm__1025 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_170, parameter_851, parameter_852, parameter_853, parameter_854, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x208x7x7xf16) <- (-1x208x7x7xf16)
        relu__166 = paddle._C_ops.relu_(batch_norm__1020)

        # builtin.slice: (-1x208x7x7xf16) <- ([-1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16])
        slice_131 = split_with_num_32[3]

        # builtin.combine: ([-1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16]) <- (-1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16)
        combine_32 = [relu__164, relu__165, relu__166, slice_131]

        # pd_op.full: (1xi32) <- ()
        full_65 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x832x7x7xf16) <- ([-1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16, -1x208x7x7xf16], 1xi32)
        concat_32 = paddle._C_ops.concat(combine_32, full_65)

        # pd_op.conv2d: (-1x2048x7x7xf16) <- (-1x832x7x7xf16, 2048x832x1x1xf16)
        conv2d_171 = paddle._C_ops.conv2d(concat_32, parameter_855, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x2048x7x7xf16, 2048xf32, 2048xf32, 2048xf32, 2048xf32, None) <- (-1x2048x7x7xf16, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        batch_norm__1026, batch_norm__1027, batch_norm__1028, batch_norm__1029, batch_norm__1030, batch_norm__1031 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_171, parameter_856, parameter_857, parameter_858, parameter_859, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x2048x7x7xf16) <- (-1x2048x7x7xf16, -1x2048x7x7xf16)
        add__92 = paddle._C_ops.add_(relu__162, batch_norm__1026)

        # pd_op.relu_: (-1x2048x7x7xf16) <- (-1x2048x7x7xf16)
        relu__167 = paddle._C_ops.relu_(add__92)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_7 = [1, 1]

        # pd_op.pool2d: (-1x2048x1x1xf16) <- (-1x2048x7x7xf16, 2xi64)
        pool2d_7 = paddle._C_ops.pool2d(relu__167, full_int_array_7, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_8 = [-1, 2048]

        # pd_op.reshape_: (-1x2048xf16, 0x-1x2048x1x1xf16) <- (-1x2048x1x1xf16, 2xi64)
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(pool2d_7, full_int_array_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x1000xf16) <- (-1x2048xf16, 2048x1000xf16)
        matmul_0 = paddle._C_ops.matmul(reshape__0, parameter_860, False, False)

        # pd_op.add_: (-1x1000xf16) <- (-1x1000xf16, 1000xf16)
        add__93 = paddle._C_ops.add_(matmul_0, parameter_861)

        # pd_op.softmax_: (-1x1000xf16) <- (-1x1000xf16)
        softmax__0 = paddle._C_ops.softmax_(add__93, -1)

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

    def forward(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_114, parameter_111, parameter_113, parameter_112, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_154, parameter_151, parameter_153, parameter_152, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_169, parameter_166, parameter_168, parameter_167, parameter_170, parameter_174, parameter_171, parameter_173, parameter_172, parameter_175, parameter_179, parameter_176, parameter_178, parameter_177, parameter_180, parameter_184, parameter_181, parameter_183, parameter_182, parameter_185, parameter_189, parameter_186, parameter_188, parameter_187, parameter_190, parameter_194, parameter_191, parameter_193, parameter_192, parameter_195, parameter_199, parameter_196, parameter_198, parameter_197, parameter_200, parameter_204, parameter_201, parameter_203, parameter_202, parameter_205, parameter_209, parameter_206, parameter_208, parameter_207, parameter_210, parameter_214, parameter_211, parameter_213, parameter_212, parameter_215, parameter_219, parameter_216, parameter_218, parameter_217, parameter_220, parameter_224, parameter_221, parameter_223, parameter_222, parameter_225, parameter_229, parameter_226, parameter_228, parameter_227, parameter_230, parameter_234, parameter_231, parameter_233, parameter_232, parameter_235, parameter_239, parameter_236, parameter_238, parameter_237, parameter_240, parameter_244, parameter_241, parameter_243, parameter_242, parameter_245, parameter_249, parameter_246, parameter_248, parameter_247, parameter_250, parameter_254, parameter_251, parameter_253, parameter_252, parameter_255, parameter_259, parameter_256, parameter_258, parameter_257, parameter_260, parameter_264, parameter_261, parameter_263, parameter_262, parameter_265, parameter_269, parameter_266, parameter_268, parameter_267, parameter_270, parameter_274, parameter_271, parameter_273, parameter_272, parameter_275, parameter_279, parameter_276, parameter_278, parameter_277, parameter_280, parameter_284, parameter_281, parameter_283, parameter_282, parameter_285, parameter_289, parameter_286, parameter_288, parameter_287, parameter_290, parameter_294, parameter_291, parameter_293, parameter_292, parameter_295, parameter_299, parameter_296, parameter_298, parameter_297, parameter_300, parameter_304, parameter_301, parameter_303, parameter_302, parameter_305, parameter_309, parameter_306, parameter_308, parameter_307, parameter_310, parameter_314, parameter_311, parameter_313, parameter_312, parameter_315, parameter_319, parameter_316, parameter_318, parameter_317, parameter_320, parameter_324, parameter_321, parameter_323, parameter_322, parameter_325, parameter_329, parameter_326, parameter_328, parameter_327, parameter_330, parameter_334, parameter_331, parameter_333, parameter_332, parameter_335, parameter_339, parameter_336, parameter_338, parameter_337, parameter_340, parameter_344, parameter_341, parameter_343, parameter_342, parameter_345, parameter_349, parameter_346, parameter_348, parameter_347, parameter_350, parameter_354, parameter_351, parameter_353, parameter_352, parameter_355, parameter_359, parameter_356, parameter_358, parameter_357, parameter_360, parameter_364, parameter_361, parameter_363, parameter_362, parameter_365, parameter_369, parameter_366, parameter_368, parameter_367, parameter_370, parameter_374, parameter_371, parameter_373, parameter_372, parameter_375, parameter_379, parameter_376, parameter_378, parameter_377, parameter_380, parameter_384, parameter_381, parameter_383, parameter_382, parameter_385, parameter_389, parameter_386, parameter_388, parameter_387, parameter_390, parameter_394, parameter_391, parameter_393, parameter_392, parameter_395, parameter_399, parameter_396, parameter_398, parameter_397, parameter_400, parameter_404, parameter_401, parameter_403, parameter_402, parameter_405, parameter_409, parameter_406, parameter_408, parameter_407, parameter_410, parameter_414, parameter_411, parameter_413, parameter_412, parameter_415, parameter_419, parameter_416, parameter_418, parameter_417, parameter_420, parameter_424, parameter_421, parameter_423, parameter_422, parameter_425, parameter_429, parameter_426, parameter_428, parameter_427, parameter_430, parameter_434, parameter_431, parameter_433, parameter_432, parameter_435, parameter_439, parameter_436, parameter_438, parameter_437, parameter_440, parameter_444, parameter_441, parameter_443, parameter_442, parameter_445, parameter_449, parameter_446, parameter_448, parameter_447, parameter_450, parameter_454, parameter_451, parameter_453, parameter_452, parameter_455, parameter_459, parameter_456, parameter_458, parameter_457, parameter_460, parameter_464, parameter_461, parameter_463, parameter_462, parameter_465, parameter_469, parameter_466, parameter_468, parameter_467, parameter_470, parameter_474, parameter_471, parameter_473, parameter_472, parameter_475, parameter_479, parameter_476, parameter_478, parameter_477, parameter_480, parameter_484, parameter_481, parameter_483, parameter_482, parameter_485, parameter_489, parameter_486, parameter_488, parameter_487, parameter_490, parameter_494, parameter_491, parameter_493, parameter_492, parameter_495, parameter_499, parameter_496, parameter_498, parameter_497, parameter_500, parameter_504, parameter_501, parameter_503, parameter_502, parameter_505, parameter_509, parameter_506, parameter_508, parameter_507, parameter_510, parameter_514, parameter_511, parameter_513, parameter_512, parameter_515, parameter_519, parameter_516, parameter_518, parameter_517, parameter_520, parameter_524, parameter_521, parameter_523, parameter_522, parameter_525, parameter_529, parameter_526, parameter_528, parameter_527, parameter_530, parameter_534, parameter_531, parameter_533, parameter_532, parameter_535, parameter_539, parameter_536, parameter_538, parameter_537, parameter_540, parameter_544, parameter_541, parameter_543, parameter_542, parameter_545, parameter_549, parameter_546, parameter_548, parameter_547, parameter_550, parameter_554, parameter_551, parameter_553, parameter_552, parameter_555, parameter_559, parameter_556, parameter_558, parameter_557, parameter_560, parameter_564, parameter_561, parameter_563, parameter_562, parameter_565, parameter_569, parameter_566, parameter_568, parameter_567, parameter_570, parameter_574, parameter_571, parameter_573, parameter_572, parameter_575, parameter_579, parameter_576, parameter_578, parameter_577, parameter_580, parameter_584, parameter_581, parameter_583, parameter_582, parameter_585, parameter_589, parameter_586, parameter_588, parameter_587, parameter_590, parameter_594, parameter_591, parameter_593, parameter_592, parameter_595, parameter_599, parameter_596, parameter_598, parameter_597, parameter_600, parameter_604, parameter_601, parameter_603, parameter_602, parameter_605, parameter_609, parameter_606, parameter_608, parameter_607, parameter_610, parameter_614, parameter_611, parameter_613, parameter_612, parameter_615, parameter_619, parameter_616, parameter_618, parameter_617, parameter_620, parameter_624, parameter_621, parameter_623, parameter_622, parameter_625, parameter_629, parameter_626, parameter_628, parameter_627, parameter_630, parameter_634, parameter_631, parameter_633, parameter_632, parameter_635, parameter_639, parameter_636, parameter_638, parameter_637, parameter_640, parameter_644, parameter_641, parameter_643, parameter_642, parameter_645, parameter_649, parameter_646, parameter_648, parameter_647, parameter_650, parameter_654, parameter_651, parameter_653, parameter_652, parameter_655, parameter_659, parameter_656, parameter_658, parameter_657, parameter_660, parameter_664, parameter_661, parameter_663, parameter_662, parameter_665, parameter_669, parameter_666, parameter_668, parameter_667, parameter_670, parameter_674, parameter_671, parameter_673, parameter_672, parameter_675, parameter_679, parameter_676, parameter_678, parameter_677, parameter_680, parameter_684, parameter_681, parameter_683, parameter_682, parameter_685, parameter_689, parameter_686, parameter_688, parameter_687, parameter_690, parameter_694, parameter_691, parameter_693, parameter_692, parameter_695, parameter_699, parameter_696, parameter_698, parameter_697, parameter_700, parameter_704, parameter_701, parameter_703, parameter_702, parameter_705, parameter_709, parameter_706, parameter_708, parameter_707, parameter_710, parameter_714, parameter_711, parameter_713, parameter_712, parameter_715, parameter_719, parameter_716, parameter_718, parameter_717, parameter_720, parameter_724, parameter_721, parameter_723, parameter_722, parameter_725, parameter_729, parameter_726, parameter_728, parameter_727, parameter_730, parameter_734, parameter_731, parameter_733, parameter_732, parameter_735, parameter_739, parameter_736, parameter_738, parameter_737, parameter_740, parameter_744, parameter_741, parameter_743, parameter_742, parameter_745, parameter_749, parameter_746, parameter_748, parameter_747, parameter_750, parameter_754, parameter_751, parameter_753, parameter_752, parameter_755, parameter_759, parameter_756, parameter_758, parameter_757, parameter_760, parameter_764, parameter_761, parameter_763, parameter_762, parameter_765, parameter_769, parameter_766, parameter_768, parameter_767, parameter_770, parameter_774, parameter_771, parameter_773, parameter_772, parameter_775, parameter_779, parameter_776, parameter_778, parameter_777, parameter_780, parameter_784, parameter_781, parameter_783, parameter_782, parameter_785, parameter_789, parameter_786, parameter_788, parameter_787, parameter_790, parameter_794, parameter_791, parameter_793, parameter_792, parameter_795, parameter_799, parameter_796, parameter_798, parameter_797, parameter_800, parameter_804, parameter_801, parameter_803, parameter_802, parameter_805, parameter_809, parameter_806, parameter_808, parameter_807, parameter_810, parameter_814, parameter_811, parameter_813, parameter_812, parameter_815, parameter_819, parameter_816, parameter_818, parameter_817, parameter_820, parameter_824, parameter_821, parameter_823, parameter_822, parameter_825, parameter_829, parameter_826, parameter_828, parameter_827, parameter_830, parameter_834, parameter_831, parameter_833, parameter_832, parameter_835, parameter_839, parameter_836, parameter_838, parameter_837, parameter_840, parameter_844, parameter_841, parameter_843, parameter_842, parameter_845, parameter_849, parameter_846, parameter_848, parameter_847, parameter_850, parameter_854, parameter_851, parameter_853, parameter_852, parameter_855, parameter_859, parameter_856, parameter_858, parameter_857, parameter_860, parameter_861, feed_0):
        return self.builtin_module_1794_0_0(parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_114, parameter_111, parameter_113, parameter_112, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_154, parameter_151, parameter_153, parameter_152, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_169, parameter_166, parameter_168, parameter_167, parameter_170, parameter_174, parameter_171, parameter_173, parameter_172, parameter_175, parameter_179, parameter_176, parameter_178, parameter_177, parameter_180, parameter_184, parameter_181, parameter_183, parameter_182, parameter_185, parameter_189, parameter_186, parameter_188, parameter_187, parameter_190, parameter_194, parameter_191, parameter_193, parameter_192, parameter_195, parameter_199, parameter_196, parameter_198, parameter_197, parameter_200, parameter_204, parameter_201, parameter_203, parameter_202, parameter_205, parameter_209, parameter_206, parameter_208, parameter_207, parameter_210, parameter_214, parameter_211, parameter_213, parameter_212, parameter_215, parameter_219, parameter_216, parameter_218, parameter_217, parameter_220, parameter_224, parameter_221, parameter_223, parameter_222, parameter_225, parameter_229, parameter_226, parameter_228, parameter_227, parameter_230, parameter_234, parameter_231, parameter_233, parameter_232, parameter_235, parameter_239, parameter_236, parameter_238, parameter_237, parameter_240, parameter_244, parameter_241, parameter_243, parameter_242, parameter_245, parameter_249, parameter_246, parameter_248, parameter_247, parameter_250, parameter_254, parameter_251, parameter_253, parameter_252, parameter_255, parameter_259, parameter_256, parameter_258, parameter_257, parameter_260, parameter_264, parameter_261, parameter_263, parameter_262, parameter_265, parameter_269, parameter_266, parameter_268, parameter_267, parameter_270, parameter_274, parameter_271, parameter_273, parameter_272, parameter_275, parameter_279, parameter_276, parameter_278, parameter_277, parameter_280, parameter_284, parameter_281, parameter_283, parameter_282, parameter_285, parameter_289, parameter_286, parameter_288, parameter_287, parameter_290, parameter_294, parameter_291, parameter_293, parameter_292, parameter_295, parameter_299, parameter_296, parameter_298, parameter_297, parameter_300, parameter_304, parameter_301, parameter_303, parameter_302, parameter_305, parameter_309, parameter_306, parameter_308, parameter_307, parameter_310, parameter_314, parameter_311, parameter_313, parameter_312, parameter_315, parameter_319, parameter_316, parameter_318, parameter_317, parameter_320, parameter_324, parameter_321, parameter_323, parameter_322, parameter_325, parameter_329, parameter_326, parameter_328, parameter_327, parameter_330, parameter_334, parameter_331, parameter_333, parameter_332, parameter_335, parameter_339, parameter_336, parameter_338, parameter_337, parameter_340, parameter_344, parameter_341, parameter_343, parameter_342, parameter_345, parameter_349, parameter_346, parameter_348, parameter_347, parameter_350, parameter_354, parameter_351, parameter_353, parameter_352, parameter_355, parameter_359, parameter_356, parameter_358, parameter_357, parameter_360, parameter_364, parameter_361, parameter_363, parameter_362, parameter_365, parameter_369, parameter_366, parameter_368, parameter_367, parameter_370, parameter_374, parameter_371, parameter_373, parameter_372, parameter_375, parameter_379, parameter_376, parameter_378, parameter_377, parameter_380, parameter_384, parameter_381, parameter_383, parameter_382, parameter_385, parameter_389, parameter_386, parameter_388, parameter_387, parameter_390, parameter_394, parameter_391, parameter_393, parameter_392, parameter_395, parameter_399, parameter_396, parameter_398, parameter_397, parameter_400, parameter_404, parameter_401, parameter_403, parameter_402, parameter_405, parameter_409, parameter_406, parameter_408, parameter_407, parameter_410, parameter_414, parameter_411, parameter_413, parameter_412, parameter_415, parameter_419, parameter_416, parameter_418, parameter_417, parameter_420, parameter_424, parameter_421, parameter_423, parameter_422, parameter_425, parameter_429, parameter_426, parameter_428, parameter_427, parameter_430, parameter_434, parameter_431, parameter_433, parameter_432, parameter_435, parameter_439, parameter_436, parameter_438, parameter_437, parameter_440, parameter_444, parameter_441, parameter_443, parameter_442, parameter_445, parameter_449, parameter_446, parameter_448, parameter_447, parameter_450, parameter_454, parameter_451, parameter_453, parameter_452, parameter_455, parameter_459, parameter_456, parameter_458, parameter_457, parameter_460, parameter_464, parameter_461, parameter_463, parameter_462, parameter_465, parameter_469, parameter_466, parameter_468, parameter_467, parameter_470, parameter_474, parameter_471, parameter_473, parameter_472, parameter_475, parameter_479, parameter_476, parameter_478, parameter_477, parameter_480, parameter_484, parameter_481, parameter_483, parameter_482, parameter_485, parameter_489, parameter_486, parameter_488, parameter_487, parameter_490, parameter_494, parameter_491, parameter_493, parameter_492, parameter_495, parameter_499, parameter_496, parameter_498, parameter_497, parameter_500, parameter_504, parameter_501, parameter_503, parameter_502, parameter_505, parameter_509, parameter_506, parameter_508, parameter_507, parameter_510, parameter_514, parameter_511, parameter_513, parameter_512, parameter_515, parameter_519, parameter_516, parameter_518, parameter_517, parameter_520, parameter_524, parameter_521, parameter_523, parameter_522, parameter_525, parameter_529, parameter_526, parameter_528, parameter_527, parameter_530, parameter_534, parameter_531, parameter_533, parameter_532, parameter_535, parameter_539, parameter_536, parameter_538, parameter_537, parameter_540, parameter_544, parameter_541, parameter_543, parameter_542, parameter_545, parameter_549, parameter_546, parameter_548, parameter_547, parameter_550, parameter_554, parameter_551, parameter_553, parameter_552, parameter_555, parameter_559, parameter_556, parameter_558, parameter_557, parameter_560, parameter_564, parameter_561, parameter_563, parameter_562, parameter_565, parameter_569, parameter_566, parameter_568, parameter_567, parameter_570, parameter_574, parameter_571, parameter_573, parameter_572, parameter_575, parameter_579, parameter_576, parameter_578, parameter_577, parameter_580, parameter_584, parameter_581, parameter_583, parameter_582, parameter_585, parameter_589, parameter_586, parameter_588, parameter_587, parameter_590, parameter_594, parameter_591, parameter_593, parameter_592, parameter_595, parameter_599, parameter_596, parameter_598, parameter_597, parameter_600, parameter_604, parameter_601, parameter_603, parameter_602, parameter_605, parameter_609, parameter_606, parameter_608, parameter_607, parameter_610, parameter_614, parameter_611, parameter_613, parameter_612, parameter_615, parameter_619, parameter_616, parameter_618, parameter_617, parameter_620, parameter_624, parameter_621, parameter_623, parameter_622, parameter_625, parameter_629, parameter_626, parameter_628, parameter_627, parameter_630, parameter_634, parameter_631, parameter_633, parameter_632, parameter_635, parameter_639, parameter_636, parameter_638, parameter_637, parameter_640, parameter_644, parameter_641, parameter_643, parameter_642, parameter_645, parameter_649, parameter_646, parameter_648, parameter_647, parameter_650, parameter_654, parameter_651, parameter_653, parameter_652, parameter_655, parameter_659, parameter_656, parameter_658, parameter_657, parameter_660, parameter_664, parameter_661, parameter_663, parameter_662, parameter_665, parameter_669, parameter_666, parameter_668, parameter_667, parameter_670, parameter_674, parameter_671, parameter_673, parameter_672, parameter_675, parameter_679, parameter_676, parameter_678, parameter_677, parameter_680, parameter_684, parameter_681, parameter_683, parameter_682, parameter_685, parameter_689, parameter_686, parameter_688, parameter_687, parameter_690, parameter_694, parameter_691, parameter_693, parameter_692, parameter_695, parameter_699, parameter_696, parameter_698, parameter_697, parameter_700, parameter_704, parameter_701, parameter_703, parameter_702, parameter_705, parameter_709, parameter_706, parameter_708, parameter_707, parameter_710, parameter_714, parameter_711, parameter_713, parameter_712, parameter_715, parameter_719, parameter_716, parameter_718, parameter_717, parameter_720, parameter_724, parameter_721, parameter_723, parameter_722, parameter_725, parameter_729, parameter_726, parameter_728, parameter_727, parameter_730, parameter_734, parameter_731, parameter_733, parameter_732, parameter_735, parameter_739, parameter_736, parameter_738, parameter_737, parameter_740, parameter_744, parameter_741, parameter_743, parameter_742, parameter_745, parameter_749, parameter_746, parameter_748, parameter_747, parameter_750, parameter_754, parameter_751, parameter_753, parameter_752, parameter_755, parameter_759, parameter_756, parameter_758, parameter_757, parameter_760, parameter_764, parameter_761, parameter_763, parameter_762, parameter_765, parameter_769, parameter_766, parameter_768, parameter_767, parameter_770, parameter_774, parameter_771, parameter_773, parameter_772, parameter_775, parameter_779, parameter_776, parameter_778, parameter_777, parameter_780, parameter_784, parameter_781, parameter_783, parameter_782, parameter_785, parameter_789, parameter_786, parameter_788, parameter_787, parameter_790, parameter_794, parameter_791, parameter_793, parameter_792, parameter_795, parameter_799, parameter_796, parameter_798, parameter_797, parameter_800, parameter_804, parameter_801, parameter_803, parameter_802, parameter_805, parameter_809, parameter_806, parameter_808, parameter_807, parameter_810, parameter_814, parameter_811, parameter_813, parameter_812, parameter_815, parameter_819, parameter_816, parameter_818, parameter_817, parameter_820, parameter_824, parameter_821, parameter_823, parameter_822, parameter_825, parameter_829, parameter_826, parameter_828, parameter_827, parameter_830, parameter_834, parameter_831, parameter_833, parameter_832, parameter_835, parameter_839, parameter_836, parameter_838, parameter_837, parameter_840, parameter_844, parameter_841, parameter_843, parameter_842, parameter_845, parameter_849, parameter_846, parameter_848, parameter_847, parameter_850, parameter_854, parameter_851, parameter_853, parameter_852, parameter_855, parameter_859, parameter_856, parameter_858, parameter_857, parameter_860, parameter_861, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_1794_0_0(CinnTestBase, unittest.TestCase):
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
            paddle.uniform([104, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_19
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([26, 26, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_24
            paddle.uniform([26], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([26], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([26], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([26], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([26, 26, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_29
            paddle.uniform([26], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([26], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([26], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([26], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([26, 26, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_34
            paddle.uniform([26], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([26], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([26], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([26], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([256, 104, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_39
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([256, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_44
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([104, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_49
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([26, 26, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_54
            paddle.uniform([26], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([26], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([26], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([26], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([26, 26, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_59
            paddle.uniform([26], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([26], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([26], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([26], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([26, 26, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_64
            paddle.uniform([26], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([26], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([26], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([26], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([256, 104, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_69
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([104, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_74
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([26, 26, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_79
            paddle.uniform([26], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([26], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([26], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([26], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([26, 26, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_84
            paddle.uniform([26], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([26], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([26], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([26], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([26, 26, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_89
            paddle.uniform([26], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([26], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([26], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([26], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([256, 104, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_94
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([208, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_99
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([52, 52, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_104
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([52, 52, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_109
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([52, 52, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_114
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([512, 208, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_119
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([512, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_124
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([208, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_129
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([52, 52, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_134
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([52, 52, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_139
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([52, 52, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_144
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([512, 208, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_149
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([208, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_154
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([52, 52, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_159
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([52, 52, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_164
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([52, 52, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_169
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([512, 208, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_174
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([208, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_179
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([52, 52, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_184
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([52, 52, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_189
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([52, 52, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_194
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([52], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([512, 208, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_199
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([416, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_204
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_209
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_214
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_213
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_215
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_219
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_218
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_217
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([1024, 416, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_224
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_221
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_223
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_225
            paddle.uniform([1024, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_229
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_227
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([416, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_234
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_233
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_239
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_238
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_244
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_241
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_243
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_245
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_249
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_246
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_248
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_250
            paddle.uniform([1024, 416, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_254
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_251
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_253
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_252
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_255
            paddle.uniform([416, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_259
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_256
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_258
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_257
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_264
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_261
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_263
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_262
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_265
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_269
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_266
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_268
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_267
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_270
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_274
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_271
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_273
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_272
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_275
            paddle.uniform([1024, 416, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_279
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_276
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_278
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_277
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_280
            paddle.uniform([416, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_284
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_281
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_283
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_282
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_285
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_289
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_286
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_288
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_287
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_290
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_294
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_291
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_293
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_292
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_295
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_299
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_296
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_298
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_297
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_300
            paddle.uniform([1024, 416, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_304
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_301
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_303
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_302
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_305
            paddle.uniform([416, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_309
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_306
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_308
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_307
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_310
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_314
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_311
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_313
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_312
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_315
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_319
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_316
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_318
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_317
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_320
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_324
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_321
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_323
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_322
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_325
            paddle.uniform([1024, 416, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_329
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_326
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_328
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_327
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_330
            paddle.uniform([416, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_334
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_331
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_333
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_332
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_335
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_339
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_336
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_338
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_337
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_340
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_344
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_341
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_343
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_342
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_345
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_349
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_346
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_348
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_347
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_350
            paddle.uniform([1024, 416, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_354
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_351
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_353
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_352
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_355
            paddle.uniform([416, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_359
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_356
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_358
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_357
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_360
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_364
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_361
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_363
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_362
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_365
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_369
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_366
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_368
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_367
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_370
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_374
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_371
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_373
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_372
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_375
            paddle.uniform([1024, 416, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_379
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_376
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_378
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_377
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_380
            paddle.uniform([416, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_384
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_381
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_383
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_382
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_385
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_389
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_386
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_388
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_387
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_390
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_394
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_391
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_393
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_392
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_395
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_399
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_396
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_398
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_397
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_400
            paddle.uniform([1024, 416, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_404
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_401
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_403
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_402
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_405
            paddle.uniform([416, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_409
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_406
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_408
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_407
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_410
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_414
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_411
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_413
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_412
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_415
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_419
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_416
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_418
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_417
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_420
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_424
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_421
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_423
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_422
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_425
            paddle.uniform([1024, 416, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_429
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_426
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_428
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_427
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_430
            paddle.uniform([416, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_434
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_431
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_433
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_432
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_435
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_439
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_436
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_438
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_437
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_440
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_444
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_441
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_443
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_442
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_445
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_449
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_446
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_448
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_447
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_450
            paddle.uniform([1024, 416, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_454
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_451
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_453
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_452
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_455
            paddle.uniform([416, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_459
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_456
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_458
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_457
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_460
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_464
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_461
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_463
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_462
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_465
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_469
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_466
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_468
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_467
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_470
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_474
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_471
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_473
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_472
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_475
            paddle.uniform([1024, 416, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_479
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_476
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_478
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_477
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_480
            paddle.uniform([416, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_484
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_481
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_483
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_482
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_485
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_489
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_486
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_488
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_487
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_490
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_494
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_491
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_493
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_492
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_495
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_499
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_496
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_498
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_497
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_500
            paddle.uniform([1024, 416, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_504
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_501
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_503
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_502
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_505
            paddle.uniform([416, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_509
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_506
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_508
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_507
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_510
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_514
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_511
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_513
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_512
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_515
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_519
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_516
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_518
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_517
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_520
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_524
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_521
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_523
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_522
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_525
            paddle.uniform([1024, 416, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_529
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_526
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_528
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_527
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_530
            paddle.uniform([416, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_534
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_531
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_533
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_532
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_535
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_539
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_536
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_538
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_537
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_540
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_544
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_541
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_543
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_542
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_545
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_549
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_546
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_548
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_547
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_550
            paddle.uniform([1024, 416, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_554
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_551
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_553
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_552
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_555
            paddle.uniform([416, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_559
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_556
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_558
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_557
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_560
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_564
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_561
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_563
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_562
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_565
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_569
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_566
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_568
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_567
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_570
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_574
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_571
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_573
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_572
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_575
            paddle.uniform([1024, 416, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_579
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_576
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_578
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_577
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_580
            paddle.uniform([416, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_584
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_581
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_583
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_582
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_585
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_589
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_586
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_588
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_587
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_590
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_594
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_591
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_593
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_592
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_595
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_599
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_596
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_598
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_597
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_600
            paddle.uniform([1024, 416, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_604
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_601
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_603
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_602
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_605
            paddle.uniform([416, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_609
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_606
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_608
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_607
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_610
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_614
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_611
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_613
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_612
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_615
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_619
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_616
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_618
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_617
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_620
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_624
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_621
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_623
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_622
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_625
            paddle.uniform([1024, 416, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_629
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_626
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_628
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_627
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_630
            paddle.uniform([416, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_634
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_631
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_633
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_632
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_635
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_639
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_636
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_638
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_637
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_640
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_644
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_641
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_643
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_642
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_645
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_649
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_646
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_648
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_647
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_650
            paddle.uniform([1024, 416, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_654
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_651
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_653
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_652
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_655
            paddle.uniform([416, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_659
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_656
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_658
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_657
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_660
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_664
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_661
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_663
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_662
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_665
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_669
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_666
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_668
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_667
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_670
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_674
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_671
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_673
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_672
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_675
            paddle.uniform([1024, 416, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_679
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_676
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_678
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_677
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_680
            paddle.uniform([416, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_684
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_681
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_683
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_682
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_685
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_689
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_686
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_688
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_687
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_690
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_694
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_691
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_693
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_692
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_695
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_699
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_696
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_698
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_697
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_700
            paddle.uniform([1024, 416, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_704
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_701
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_703
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_702
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_705
            paddle.uniform([416, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_709
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_706
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_708
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_707
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_710
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_714
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_711
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_713
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_712
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_715
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_719
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_716
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_718
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_717
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_720
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_724
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_721
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_723
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_722
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_725
            paddle.uniform([1024, 416, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_729
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_726
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_728
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_727
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_730
            paddle.uniform([416, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_734
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_731
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_733
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_732
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_735
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_739
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_736
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_738
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_737
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_740
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_744
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_741
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_743
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_742
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_745
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_749
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_746
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_748
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_747
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_750
            paddle.uniform([1024, 416, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_754
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_751
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_753
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_752
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_755
            paddle.uniform([416, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_759
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_756
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_758
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_757
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_760
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_764
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_761
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_763
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_762
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_765
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_769
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_766
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_768
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_767
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_770
            paddle.uniform([104, 104, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_774
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_771
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_773
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_772
            paddle.uniform([104], dtype='float32', min=0, max=0.5),
            # parameter_775
            paddle.uniform([1024, 416, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_779
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_776
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_778
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_777
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_780
            paddle.uniform([832, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_784
            paddle.uniform([832], dtype='float32', min=0, max=0.5),
            # parameter_781
            paddle.uniform([832], dtype='float32', min=0, max=0.5),
            # parameter_783
            paddle.uniform([832], dtype='float32', min=0, max=0.5),
            # parameter_782
            paddle.uniform([832], dtype='float32', min=0, max=0.5),
            # parameter_785
            paddle.uniform([208, 208, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_789
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_786
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_788
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_787
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_790
            paddle.uniform([208, 208, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_794
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_791
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_793
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_792
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_795
            paddle.uniform([208, 208, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_799
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_796
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_798
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_797
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_800
            paddle.uniform([2048, 832, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_804
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_801
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_803
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_802
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_805
            paddle.uniform([2048, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_809
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_806
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_808
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_807
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_810
            paddle.uniform([832, 2048, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_814
            paddle.uniform([832], dtype='float32', min=0, max=0.5),
            # parameter_811
            paddle.uniform([832], dtype='float32', min=0, max=0.5),
            # parameter_813
            paddle.uniform([832], dtype='float32', min=0, max=0.5),
            # parameter_812
            paddle.uniform([832], dtype='float32', min=0, max=0.5),
            # parameter_815
            paddle.uniform([208, 208, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_819
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_816
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_818
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_817
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_820
            paddle.uniform([208, 208, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_824
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_821
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_823
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_822
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_825
            paddle.uniform([208, 208, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_829
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_826
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_828
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_827
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_830
            paddle.uniform([2048, 832, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_834
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_831
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_833
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_832
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_835
            paddle.uniform([832, 2048, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_839
            paddle.uniform([832], dtype='float32', min=0, max=0.5),
            # parameter_836
            paddle.uniform([832], dtype='float32', min=0, max=0.5),
            # parameter_838
            paddle.uniform([832], dtype='float32', min=0, max=0.5),
            # parameter_837
            paddle.uniform([832], dtype='float32', min=0, max=0.5),
            # parameter_840
            paddle.uniform([208, 208, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_844
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_841
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_843
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_842
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_845
            paddle.uniform([208, 208, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_849
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_846
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_848
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_847
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_850
            paddle.uniform([208, 208, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_854
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_851
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_853
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_852
            paddle.uniform([208], dtype='float32', min=0, max=0.5),
            # parameter_855
            paddle.uniform([2048, 832, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_859
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_856
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_858
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_857
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_860
            paddle.uniform([2048, 1000], dtype='float16', min=0, max=0.5),
            # parameter_861
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
            paddle.static.InputSpec(shape=[104, 64, 1, 1], dtype='float16'),
            # parameter_19
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[26, 26, 3, 3], dtype='float16'),
            # parameter_24
            paddle.static.InputSpec(shape=[26], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[26], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[26], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[26], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[26, 26, 3, 3], dtype='float16'),
            # parameter_29
            paddle.static.InputSpec(shape=[26], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[26], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[26], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[26], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[26, 26, 3, 3], dtype='float16'),
            # parameter_34
            paddle.static.InputSpec(shape=[26], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[26], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[26], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[26], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[256, 104, 1, 1], dtype='float16'),
            # parameter_39
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float16'),
            # parameter_44
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[104, 256, 1, 1], dtype='float16'),
            # parameter_49
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[26, 26, 3, 3], dtype='float16'),
            # parameter_54
            paddle.static.InputSpec(shape=[26], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[26], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[26], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[26], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[26, 26, 3, 3], dtype='float16'),
            # parameter_59
            paddle.static.InputSpec(shape=[26], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[26], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[26], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[26], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[26, 26, 3, 3], dtype='float16'),
            # parameter_64
            paddle.static.InputSpec(shape=[26], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[26], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[26], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[26], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[256, 104, 1, 1], dtype='float16'),
            # parameter_69
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[104, 256, 1, 1], dtype='float16'),
            # parameter_74
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[26, 26, 3, 3], dtype='float16'),
            # parameter_79
            paddle.static.InputSpec(shape=[26], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[26], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[26], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[26], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[26, 26, 3, 3], dtype='float16'),
            # parameter_84
            paddle.static.InputSpec(shape=[26], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[26], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[26], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[26], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[26, 26, 3, 3], dtype='float16'),
            # parameter_89
            paddle.static.InputSpec(shape=[26], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[26], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[26], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[26], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[256, 104, 1, 1], dtype='float16'),
            # parameter_94
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[208, 256, 1, 1], dtype='float16'),
            # parameter_99
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[52, 52, 3, 3], dtype='float16'),
            # parameter_104
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[52, 52, 3, 3], dtype='float16'),
            # parameter_109
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[52, 52, 3, 3], dtype='float16'),
            # parameter_114
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[512, 208, 1, 1], dtype='float16'),
            # parameter_119
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[512, 256, 1, 1], dtype='float16'),
            # parameter_124
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[208, 512, 1, 1], dtype='float16'),
            # parameter_129
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[52, 52, 3, 3], dtype='float16'),
            # parameter_134
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[52, 52, 3, 3], dtype='float16'),
            # parameter_139
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[52, 52, 3, 3], dtype='float16'),
            # parameter_144
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[512, 208, 1, 1], dtype='float16'),
            # parameter_149
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[208, 512, 1, 1], dtype='float16'),
            # parameter_154
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[52, 52, 3, 3], dtype='float16'),
            # parameter_159
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[52, 52, 3, 3], dtype='float16'),
            # parameter_164
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[52, 52, 3, 3], dtype='float16'),
            # parameter_169
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[512, 208, 1, 1], dtype='float16'),
            # parameter_174
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[208, 512, 1, 1], dtype='float16'),
            # parameter_179
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[52, 52, 3, 3], dtype='float16'),
            # parameter_184
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[52, 52, 3, 3], dtype='float16'),
            # parameter_189
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[52, 52, 3, 3], dtype='float16'),
            # parameter_194
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[52], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[512, 208, 1, 1], dtype='float16'),
            # parameter_199
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[416, 512, 1, 1], dtype='float16'),
            # parameter_204
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_209
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_214
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_213
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_215
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_219
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_218
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_217
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[1024, 416, 1, 1], dtype='float16'),
            # parameter_224
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_221
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_223
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_225
            paddle.static.InputSpec(shape=[1024, 512, 1, 1], dtype='float16'),
            # parameter_229
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_227
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[416, 1024, 1, 1], dtype='float16'),
            # parameter_234
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_233
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_239
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_238
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_244
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_241
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_243
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_245
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_249
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_246
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_248
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_250
            paddle.static.InputSpec(shape=[1024, 416, 1, 1], dtype='float16'),
            # parameter_254
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_251
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_253
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_252
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_255
            paddle.static.InputSpec(shape=[416, 1024, 1, 1], dtype='float16'),
            # parameter_259
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_256
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_258
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_257
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_264
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_261
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_263
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_262
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_265
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_269
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_266
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_268
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_267
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_270
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_274
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_271
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_273
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_272
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_275
            paddle.static.InputSpec(shape=[1024, 416, 1, 1], dtype='float16'),
            # parameter_279
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_276
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_278
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_277
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_280
            paddle.static.InputSpec(shape=[416, 1024, 1, 1], dtype='float16'),
            # parameter_284
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_281
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_283
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_282
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_285
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_289
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_286
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_288
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_287
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_290
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_294
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_291
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_293
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_292
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_295
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_299
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_296
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_298
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_297
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_300
            paddle.static.InputSpec(shape=[1024, 416, 1, 1], dtype='float16'),
            # parameter_304
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_301
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_303
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_302
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_305
            paddle.static.InputSpec(shape=[416, 1024, 1, 1], dtype='float16'),
            # parameter_309
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_306
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_308
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_307
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_310
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_314
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_311
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_313
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_312
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_315
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_319
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_316
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_318
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_317
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_320
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_324
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_321
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_323
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_322
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_325
            paddle.static.InputSpec(shape=[1024, 416, 1, 1], dtype='float16'),
            # parameter_329
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_326
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_328
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_327
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_330
            paddle.static.InputSpec(shape=[416, 1024, 1, 1], dtype='float16'),
            # parameter_334
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_331
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_333
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_332
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_335
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_339
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_336
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_338
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_337
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_340
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_344
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_341
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_343
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_342
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_345
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_349
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_346
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_348
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_347
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_350
            paddle.static.InputSpec(shape=[1024, 416, 1, 1], dtype='float16'),
            # parameter_354
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_351
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_353
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_352
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_355
            paddle.static.InputSpec(shape=[416, 1024, 1, 1], dtype='float16'),
            # parameter_359
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_356
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_358
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_357
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_360
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_364
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_361
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_363
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_362
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_365
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_369
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_366
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_368
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_367
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_370
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_374
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_371
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_373
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_372
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_375
            paddle.static.InputSpec(shape=[1024, 416, 1, 1], dtype='float16'),
            # parameter_379
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_376
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_378
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_377
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_380
            paddle.static.InputSpec(shape=[416, 1024, 1, 1], dtype='float16'),
            # parameter_384
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_381
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_383
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_382
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_385
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_389
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_386
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_388
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_387
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_390
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_394
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_391
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_393
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_392
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_395
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_399
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_396
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_398
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_397
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_400
            paddle.static.InputSpec(shape=[1024, 416, 1, 1], dtype='float16'),
            # parameter_404
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_401
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_403
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_402
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_405
            paddle.static.InputSpec(shape=[416, 1024, 1, 1], dtype='float16'),
            # parameter_409
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_406
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_408
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_407
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_410
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_414
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_411
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_413
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_412
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_415
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_419
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_416
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_418
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_417
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_420
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_424
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_421
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_423
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_422
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_425
            paddle.static.InputSpec(shape=[1024, 416, 1, 1], dtype='float16'),
            # parameter_429
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_426
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_428
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_427
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_430
            paddle.static.InputSpec(shape=[416, 1024, 1, 1], dtype='float16'),
            # parameter_434
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_431
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_433
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_432
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_435
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_439
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_436
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_438
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_437
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_440
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_444
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_441
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_443
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_442
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_445
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_449
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_446
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_448
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_447
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_450
            paddle.static.InputSpec(shape=[1024, 416, 1, 1], dtype='float16'),
            # parameter_454
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_451
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_453
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_452
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_455
            paddle.static.InputSpec(shape=[416, 1024, 1, 1], dtype='float16'),
            # parameter_459
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_456
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_458
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_457
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_460
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_464
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_461
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_463
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_462
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_465
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_469
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_466
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_468
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_467
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_470
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_474
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_471
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_473
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_472
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_475
            paddle.static.InputSpec(shape=[1024, 416, 1, 1], dtype='float16'),
            # parameter_479
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_476
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_478
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_477
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_480
            paddle.static.InputSpec(shape=[416, 1024, 1, 1], dtype='float16'),
            # parameter_484
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_481
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_483
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_482
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_485
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_489
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_486
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_488
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_487
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_490
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_494
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_491
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_493
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_492
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_495
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_499
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_496
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_498
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_497
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_500
            paddle.static.InputSpec(shape=[1024, 416, 1, 1], dtype='float16'),
            # parameter_504
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_501
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_503
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_502
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_505
            paddle.static.InputSpec(shape=[416, 1024, 1, 1], dtype='float16'),
            # parameter_509
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_506
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_508
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_507
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_510
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_514
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_511
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_513
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_512
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_515
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_519
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_516
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_518
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_517
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_520
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_524
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_521
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_523
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_522
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_525
            paddle.static.InputSpec(shape=[1024, 416, 1, 1], dtype='float16'),
            # parameter_529
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_526
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_528
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_527
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_530
            paddle.static.InputSpec(shape=[416, 1024, 1, 1], dtype='float16'),
            # parameter_534
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_531
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_533
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_532
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_535
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_539
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_536
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_538
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_537
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_540
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_544
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_541
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_543
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_542
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_545
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_549
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_546
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_548
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_547
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_550
            paddle.static.InputSpec(shape=[1024, 416, 1, 1], dtype='float16'),
            # parameter_554
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_551
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_553
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_552
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_555
            paddle.static.InputSpec(shape=[416, 1024, 1, 1], dtype='float16'),
            # parameter_559
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_556
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_558
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_557
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_560
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_564
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_561
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_563
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_562
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_565
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_569
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_566
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_568
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_567
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_570
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_574
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_571
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_573
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_572
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_575
            paddle.static.InputSpec(shape=[1024, 416, 1, 1], dtype='float16'),
            # parameter_579
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_576
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_578
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_577
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_580
            paddle.static.InputSpec(shape=[416, 1024, 1, 1], dtype='float16'),
            # parameter_584
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_581
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_583
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_582
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_585
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_589
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_586
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_588
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_587
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_590
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_594
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_591
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_593
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_592
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_595
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_599
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_596
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_598
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_597
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_600
            paddle.static.InputSpec(shape=[1024, 416, 1, 1], dtype='float16'),
            # parameter_604
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_601
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_603
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_602
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_605
            paddle.static.InputSpec(shape=[416, 1024, 1, 1], dtype='float16'),
            # parameter_609
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_606
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_608
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_607
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_610
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_614
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_611
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_613
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_612
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_615
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_619
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_616
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_618
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_617
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_620
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_624
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_621
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_623
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_622
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_625
            paddle.static.InputSpec(shape=[1024, 416, 1, 1], dtype='float16'),
            # parameter_629
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_626
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_628
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_627
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_630
            paddle.static.InputSpec(shape=[416, 1024, 1, 1], dtype='float16'),
            # parameter_634
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_631
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_633
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_632
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_635
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_639
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_636
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_638
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_637
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_640
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_644
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_641
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_643
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_642
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_645
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_649
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_646
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_648
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_647
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_650
            paddle.static.InputSpec(shape=[1024, 416, 1, 1], dtype='float16'),
            # parameter_654
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_651
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_653
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_652
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_655
            paddle.static.InputSpec(shape=[416, 1024, 1, 1], dtype='float16'),
            # parameter_659
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_656
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_658
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_657
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_660
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_664
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_661
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_663
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_662
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_665
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_669
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_666
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_668
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_667
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_670
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_674
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_671
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_673
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_672
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_675
            paddle.static.InputSpec(shape=[1024, 416, 1, 1], dtype='float16'),
            # parameter_679
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_676
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_678
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_677
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_680
            paddle.static.InputSpec(shape=[416, 1024, 1, 1], dtype='float16'),
            # parameter_684
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_681
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_683
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_682
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_685
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_689
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_686
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_688
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_687
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_690
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_694
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_691
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_693
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_692
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_695
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_699
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_696
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_698
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_697
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_700
            paddle.static.InputSpec(shape=[1024, 416, 1, 1], dtype='float16'),
            # parameter_704
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_701
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_703
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_702
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_705
            paddle.static.InputSpec(shape=[416, 1024, 1, 1], dtype='float16'),
            # parameter_709
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_706
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_708
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_707
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_710
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_714
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_711
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_713
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_712
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_715
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_719
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_716
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_718
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_717
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_720
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_724
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_721
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_723
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_722
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_725
            paddle.static.InputSpec(shape=[1024, 416, 1, 1], dtype='float16'),
            # parameter_729
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_726
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_728
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_727
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_730
            paddle.static.InputSpec(shape=[416, 1024, 1, 1], dtype='float16'),
            # parameter_734
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_731
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_733
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_732
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_735
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_739
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_736
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_738
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_737
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_740
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_744
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_741
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_743
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_742
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_745
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_749
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_746
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_748
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_747
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_750
            paddle.static.InputSpec(shape=[1024, 416, 1, 1], dtype='float16'),
            # parameter_754
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_751
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_753
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_752
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_755
            paddle.static.InputSpec(shape=[416, 1024, 1, 1], dtype='float16'),
            # parameter_759
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_756
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_758
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_757
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_760
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_764
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_761
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_763
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_762
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_765
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_769
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_766
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_768
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_767
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_770
            paddle.static.InputSpec(shape=[104, 104, 3, 3], dtype='float16'),
            # parameter_774
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_771
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_773
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_772
            paddle.static.InputSpec(shape=[104], dtype='float32'),
            # parameter_775
            paddle.static.InputSpec(shape=[1024, 416, 1, 1], dtype='float16'),
            # parameter_779
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_776
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_778
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_777
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_780
            paddle.static.InputSpec(shape=[832, 1024, 1, 1], dtype='float16'),
            # parameter_784
            paddle.static.InputSpec(shape=[832], dtype='float32'),
            # parameter_781
            paddle.static.InputSpec(shape=[832], dtype='float32'),
            # parameter_783
            paddle.static.InputSpec(shape=[832], dtype='float32'),
            # parameter_782
            paddle.static.InputSpec(shape=[832], dtype='float32'),
            # parameter_785
            paddle.static.InputSpec(shape=[208, 208, 3, 3], dtype='float16'),
            # parameter_789
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_786
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_788
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_787
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_790
            paddle.static.InputSpec(shape=[208, 208, 3, 3], dtype='float16'),
            # parameter_794
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_791
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_793
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_792
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_795
            paddle.static.InputSpec(shape=[208, 208, 3, 3], dtype='float16'),
            # parameter_799
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_796
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_798
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_797
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_800
            paddle.static.InputSpec(shape=[2048, 832, 1, 1], dtype='float16'),
            # parameter_804
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_801
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_803
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_802
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_805
            paddle.static.InputSpec(shape=[2048, 1024, 1, 1], dtype='float16'),
            # parameter_809
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_806
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_808
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_807
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_810
            paddle.static.InputSpec(shape=[832, 2048, 1, 1], dtype='float16'),
            # parameter_814
            paddle.static.InputSpec(shape=[832], dtype='float32'),
            # parameter_811
            paddle.static.InputSpec(shape=[832], dtype='float32'),
            # parameter_813
            paddle.static.InputSpec(shape=[832], dtype='float32'),
            # parameter_812
            paddle.static.InputSpec(shape=[832], dtype='float32'),
            # parameter_815
            paddle.static.InputSpec(shape=[208, 208, 3, 3], dtype='float16'),
            # parameter_819
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_816
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_818
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_817
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_820
            paddle.static.InputSpec(shape=[208, 208, 3, 3], dtype='float16'),
            # parameter_824
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_821
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_823
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_822
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_825
            paddle.static.InputSpec(shape=[208, 208, 3, 3], dtype='float16'),
            # parameter_829
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_826
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_828
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_827
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_830
            paddle.static.InputSpec(shape=[2048, 832, 1, 1], dtype='float16'),
            # parameter_834
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_831
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_833
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_832
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_835
            paddle.static.InputSpec(shape=[832, 2048, 1, 1], dtype='float16'),
            # parameter_839
            paddle.static.InputSpec(shape=[832], dtype='float32'),
            # parameter_836
            paddle.static.InputSpec(shape=[832], dtype='float32'),
            # parameter_838
            paddle.static.InputSpec(shape=[832], dtype='float32'),
            # parameter_837
            paddle.static.InputSpec(shape=[832], dtype='float32'),
            # parameter_840
            paddle.static.InputSpec(shape=[208, 208, 3, 3], dtype='float16'),
            # parameter_844
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_841
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_843
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_842
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_845
            paddle.static.InputSpec(shape=[208, 208, 3, 3], dtype='float16'),
            # parameter_849
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_846
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_848
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_847
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_850
            paddle.static.InputSpec(shape=[208, 208, 3, 3], dtype='float16'),
            # parameter_854
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_851
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_853
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_852
            paddle.static.InputSpec(shape=[208], dtype='float32'),
            # parameter_855
            paddle.static.InputSpec(shape=[2048, 832, 1, 1], dtype='float16'),
            # parameter_859
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_856
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_858
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_857
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_860
            paddle.static.InputSpec(shape=[2048, 1000], dtype='float16'),
            # parameter_861
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