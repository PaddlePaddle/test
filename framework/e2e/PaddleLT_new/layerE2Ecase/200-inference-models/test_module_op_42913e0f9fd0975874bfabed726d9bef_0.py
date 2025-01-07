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
    return [989][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_1732_0_0(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_11, parameter_15, parameter_12, parameter_14, parameter_13, parameter_16, parameter_20, parameter_17, parameter_19, parameter_18, parameter_21, parameter_25, parameter_22, parameter_24, parameter_23, parameter_26, parameter_30, parameter_27, parameter_29, parameter_28, parameter_31, parameter_32, parameter_36, parameter_33, parameter_35, parameter_34, parameter_37, parameter_41, parameter_38, parameter_40, parameter_39, parameter_42, parameter_46, parameter_43, parameter_45, parameter_44, parameter_47, parameter_51, parameter_48, parameter_50, parameter_49, parameter_52, parameter_56, parameter_53, parameter_55, parameter_54, parameter_57, parameter_58, parameter_62, parameter_59, parameter_61, parameter_60, parameter_63, parameter_67, parameter_64, parameter_66, parameter_65, parameter_68, parameter_72, parameter_69, parameter_71, parameter_70, parameter_73, parameter_77, parameter_74, parameter_76, parameter_75, parameter_78, parameter_79, parameter_83, parameter_80, parameter_82, parameter_81, parameter_84, parameter_88, parameter_85, parameter_87, parameter_86, parameter_89, parameter_93, parameter_90, parameter_92, parameter_91, parameter_94, parameter_98, parameter_95, parameter_97, parameter_96, parameter_99, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_114, parameter_111, parameter_113, parameter_112, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_126, parameter_130, parameter_127, parameter_129, parameter_128, parameter_131, parameter_135, parameter_132, parameter_134, parameter_133, parameter_136, parameter_140, parameter_137, parameter_139, parameter_138, parameter_141, parameter_145, parameter_142, parameter_144, parameter_143, parameter_146, parameter_147, parameter_151, parameter_148, parameter_150, parameter_149, parameter_152, parameter_156, parameter_153, parameter_155, parameter_154, parameter_157, parameter_161, parameter_158, parameter_160, parameter_159, parameter_162, parameter_166, parameter_163, parameter_165, parameter_164, parameter_167, parameter_168, parameter_172, parameter_169, parameter_171, parameter_170, parameter_173, parameter_177, parameter_174, parameter_176, parameter_175, parameter_178, parameter_182, parameter_179, parameter_181, parameter_180, parameter_183, parameter_187, parameter_184, parameter_186, parameter_185, parameter_188, parameter_189, parameter_193, parameter_190, parameter_192, parameter_191, parameter_194, parameter_198, parameter_195, parameter_197, parameter_196, parameter_199, parameter_203, parameter_200, parameter_202, parameter_201, parameter_204, parameter_208, parameter_205, parameter_207, parameter_206, parameter_209, parameter_213, parameter_210, parameter_212, parameter_211, parameter_214, parameter_215, parameter_219, parameter_216, parameter_218, parameter_217, parameter_220, parameter_224, parameter_221, parameter_223, parameter_222, parameter_225, parameter_229, parameter_226, parameter_228, parameter_227, parameter_230, parameter_234, parameter_231, parameter_233, parameter_232, parameter_235, parameter_236, parameter_240, parameter_237, parameter_239, parameter_238, parameter_241, parameter_245, parameter_242, parameter_244, parameter_243, parameter_246, parameter_250, parameter_247, parameter_249, parameter_248, parameter_251, parameter_255, parameter_252, parameter_254, parameter_253, parameter_256, parameter_257, parameter_261, parameter_258, parameter_260, parameter_259, parameter_262, parameter_266, parameter_263, parameter_265, parameter_264, parameter_267, parameter_271, parameter_268, parameter_270, parameter_269, parameter_272, parameter_276, parameter_273, parameter_275, parameter_274, parameter_277, parameter_278, parameter_282, parameter_279, parameter_281, parameter_280, parameter_283, parameter_287, parameter_284, parameter_286, parameter_285, parameter_288, parameter_292, parameter_289, parameter_291, parameter_290, parameter_293, parameter_297, parameter_294, parameter_296, parameter_295, parameter_298, parameter_299, parameter_303, parameter_300, parameter_302, parameter_301, parameter_304, parameter_308, parameter_305, parameter_307, parameter_306, parameter_309, parameter_313, parameter_310, parameter_312, parameter_311, parameter_314, parameter_318, parameter_315, parameter_317, parameter_316, parameter_319, parameter_320, parameter_324, parameter_321, parameter_323, parameter_322, parameter_325, parameter_329, parameter_326, parameter_328, parameter_327, parameter_330, parameter_334, parameter_331, parameter_333, parameter_332, parameter_335, parameter_339, parameter_336, parameter_338, parameter_337, parameter_340, parameter_341, parameter_345, parameter_342, parameter_344, parameter_343, parameter_346, parameter_350, parameter_347, parameter_349, parameter_348, parameter_351, parameter_355, parameter_352, parameter_354, parameter_353, parameter_356, parameter_360, parameter_357, parameter_359, parameter_358, parameter_361, parameter_362, parameter_366, parameter_363, parameter_365, parameter_364, parameter_367, parameter_371, parameter_368, parameter_370, parameter_369, parameter_372, parameter_376, parameter_373, parameter_375, parameter_374, parameter_377, parameter_381, parameter_378, parameter_380, parameter_379, parameter_382, parameter_383, parameter_387, parameter_384, parameter_386, parameter_385, parameter_388, parameter_392, parameter_389, parameter_391, parameter_390, parameter_393, parameter_397, parameter_394, parameter_396, parameter_395, parameter_398, parameter_402, parameter_399, parameter_401, parameter_400, parameter_403, parameter_404, parameter_408, parameter_405, parameter_407, parameter_406, parameter_409, parameter_413, parameter_410, parameter_412, parameter_411, parameter_414, parameter_418, parameter_415, parameter_417, parameter_416, parameter_419, parameter_423, parameter_420, parameter_422, parameter_421, parameter_424, parameter_425, parameter_429, parameter_426, parameter_428, parameter_427, parameter_430, parameter_434, parameter_431, parameter_433, parameter_432, parameter_435, parameter_439, parameter_436, parameter_438, parameter_437, parameter_440, parameter_444, parameter_441, parameter_443, parameter_442, parameter_445, parameter_446, parameter_450, parameter_447, parameter_449, parameter_448, parameter_451, parameter_455, parameter_452, parameter_454, parameter_453, parameter_456, parameter_460, parameter_457, parameter_459, parameter_458, parameter_461, parameter_465, parameter_462, parameter_464, parameter_463, parameter_466, parameter_467, parameter_471, parameter_468, parameter_470, parameter_469, parameter_472, parameter_476, parameter_473, parameter_475, parameter_474, parameter_477, parameter_481, parameter_478, parameter_480, parameter_479, parameter_482, parameter_486, parameter_483, parameter_485, parameter_484, parameter_487, parameter_488, parameter_492, parameter_489, parameter_491, parameter_490, parameter_493, parameter_497, parameter_494, parameter_496, parameter_495, parameter_498, parameter_502, parameter_499, parameter_501, parameter_500, parameter_503, parameter_507, parameter_504, parameter_506, parameter_505, parameter_508, parameter_509, parameter_513, parameter_510, parameter_512, parameter_511, parameter_514, parameter_518, parameter_515, parameter_517, parameter_516, parameter_519, parameter_523, parameter_520, parameter_522, parameter_521, parameter_524, parameter_528, parameter_525, parameter_527, parameter_526, parameter_529, parameter_530, parameter_534, parameter_531, parameter_533, parameter_532, parameter_535, parameter_539, parameter_536, parameter_538, parameter_537, parameter_540, parameter_544, parameter_541, parameter_543, parameter_542, parameter_545, parameter_549, parameter_546, parameter_548, parameter_547, parameter_550, parameter_551, parameter_555, parameter_552, parameter_554, parameter_553, parameter_556, parameter_560, parameter_557, parameter_559, parameter_558, parameter_561, parameter_565, parameter_562, parameter_564, parameter_563, parameter_566, parameter_570, parameter_567, parameter_569, parameter_568, parameter_571, parameter_572, parameter_576, parameter_573, parameter_575, parameter_574, parameter_577, parameter_581, parameter_578, parameter_580, parameter_579, parameter_582, parameter_586, parameter_583, parameter_585, parameter_584, parameter_587, parameter_591, parameter_588, parameter_590, parameter_589, parameter_592, parameter_593, parameter_597, parameter_594, parameter_596, parameter_595, parameter_598, parameter_602, parameter_599, parameter_601, parameter_600, parameter_603, parameter_607, parameter_604, parameter_606, parameter_605, parameter_608, parameter_612, parameter_609, parameter_611, parameter_610, parameter_613, parameter_614, parameter_618, parameter_615, parameter_617, parameter_616, parameter_619, parameter_623, parameter_620, parameter_622, parameter_621, parameter_624, parameter_628, parameter_625, parameter_627, parameter_626, parameter_629, parameter_633, parameter_630, parameter_632, parameter_631, parameter_634, parameter_635, parameter_639, parameter_636, parameter_638, parameter_637, parameter_640, parameter_644, parameter_641, parameter_643, parameter_642, parameter_645, parameter_649, parameter_646, parameter_648, parameter_647, parameter_650, parameter_654, parameter_651, parameter_653, parameter_652, parameter_655, parameter_656, parameter_660, parameter_657, parameter_659, parameter_658, parameter_661, parameter_665, parameter_662, parameter_664, parameter_663, parameter_666, parameter_670, parameter_667, parameter_669, parameter_668, parameter_671, parameter_675, parameter_672, parameter_674, parameter_673, parameter_676, parameter_677, parameter_681, parameter_678, parameter_680, parameter_679, parameter_682, parameter_686, parameter_683, parameter_685, parameter_684, parameter_687, parameter_691, parameter_688, parameter_690, parameter_689, parameter_692, parameter_696, parameter_693, parameter_695, parameter_694, parameter_697, parameter_701, parameter_698, parameter_700, parameter_699, parameter_702, parameter_703, parameter_707, parameter_704, parameter_706, parameter_705, parameter_708, parameter_712, parameter_709, parameter_711, parameter_710, parameter_713, parameter_717, parameter_714, parameter_716, parameter_715, parameter_718, parameter_722, parameter_719, parameter_721, parameter_720, parameter_723, parameter_724, parameter_728, parameter_725, parameter_727, parameter_726, parameter_729, parameter_733, parameter_730, parameter_732, parameter_731, parameter_734, parameter_735, feed_0):

        # pd_op.conv2d: (-1x32x112x112xf32) <- (-1x3x224x224xf32, 32x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(feed_0, parameter_0, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x112x112xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_0, parameter_1, parameter_2, parameter_3, parameter_4, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x32x112x112xf32) <- (-1x32x112x112xf32)
        relu__0 = paddle._C_ops.relu_(batch_norm__0)

        # pd_op.conv2d: (-1x8x112x112xf32) <- (-1x32x112x112xf32, 8x32x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(relu__0, parameter_5, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x8x112x112xf32, 8xf32, 8xf32, xf32, xf32, None) <- (-1x8x112x112xf32, 8xf32, 8xf32, 8xf32, 8xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_1, parameter_6, parameter_7, parameter_8, parameter_9, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x8x112x112xf32) <- (-1x8x112x112xf32)
        relu__1 = paddle._C_ops.relu_(batch_norm__6)

        # pd_op.conv2d: (-1x18x112x112xf32) <- (-1x8x112x112xf32, 18x8x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(relu__1, parameter_10, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, 18, 1, 1]

        # pd_op.reshape: (1x18x1x1xf32, 0x18xf32) <- (18xf32, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_11, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x18x112x112xf32) <- (-1x18x112x112xf32, 1x18x1x1xf32)
        add__0 = paddle._C_ops.add_(conv2d_2, reshape_0)

        # pd_op.shape: (4xi32) <- (-1x18x112x112xf32)
        shape_0 = paddle._C_ops.shape(add__0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], full_int_array_1, full_int_array_2, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('9'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], float('112'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full([1], float('112'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_0 = [slice_0, full_0, full_1, full_2, full_3]

        # pd_op.reshape_: (-1x2x9x112x112xf32, 0x-1x18x112x112xf32) <- (-1x18x112x112xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__0, [x.reshape([]) for x in combine_0]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [2]

        # pd_op.unsqueeze_: (-1x2x1x9x112x112xf32, None) <- (-1x2x9x112x112xf32, 1xi64)
        unsqueeze__0, unsqueeze__1 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(reshape__0, full_int_array_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unfold: (-1x288x12544xf32) <- (-1x32x112x112xf32)
        unfold_0 = paddle._C_ops.unfold(relu__0, [3, 3], [1, 1], [1, 1, 1, 1], [1, 1])

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full([1], float('9'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full([1], float('112'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_8 = paddle._C_ops.full([1], float('112'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_1 = [slice_0, full_4, full_5, full_6, full_7, full_8]

        # pd_op.reshape_: (-1x2x16x9x112x112xf32, 0x-1x288x12544xf32) <- (-1x288x12544xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_0, [x.reshape([]) for x in combine_1]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x2x16x9x112x112xf32) <- (-1x2x1x9x112x112xf32, -1x2x16x9x112x112xf32)
        multiply_0 = paddle._C_ops.multiply(unsqueeze__0, reshape__2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [3]

        # pd_op.sum: (-1x2x16x112x112xf32) <- (-1x2x16x9x112x112xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(multiply_0, full_int_array_4, None, False)

        # pd_op.full: (1xi32) <- ()
        full_9 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_10 = paddle._C_ops.full([1], float('112'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_11 = paddle._C_ops.full([1], float('112'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_2 = [slice_0, full_9, full_10, full_11]

        # pd_op.reshape_: (-1x32x112x112xf32, 0x-1x2x16x112x112xf32) <- (-1x2x16x112x112xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sum_0, [x.reshape([]) for x in combine_2]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x32x112x112xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x112x112xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__4, parameter_12, parameter_13, parameter_14, parameter_15, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x32x112x112xf32) <- (-1x32x112x112xf32)
        relu__2 = paddle._C_ops.relu_(batch_norm__12)

        # pd_op.conv2d: (-1x64x112x112xf32) <- (-1x32x112x112xf32, 64x32x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(relu__2, parameter_16, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x112x112xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x112x112xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_3, parameter_17, parameter_18, parameter_19, parameter_20, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x112x112xf32) <- (-1x64x112x112xf32)
        relu__3 = paddle._C_ops.relu_(batch_norm__18)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [3, 3]

        # pd_op.pool2d: (-1x64x56x56xf32) <- (-1x64x112x112xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(relu__3, full_int_array_5, [2, 2], [1, 1], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x64x56x56xf32, 64x64x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(pool2d_0, parameter_21, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_4, parameter_22, parameter_23, parameter_24, parameter_25, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        relu__4 = paddle._C_ops.relu_(batch_norm__24)

        # pd_op.conv2d: (-1x16x56x56xf32) <- (-1x64x56x56xf32, 16x64x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(relu__4, parameter_26, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x16x56x56xf32, 16xf32, 16xf32, xf32, xf32, None) <- (-1x16x56x56xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_5, parameter_27, parameter_28, parameter_29, parameter_30, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x16x56x56xf32) <- (-1x16x56x56xf32)
        relu__5 = paddle._C_ops.relu_(batch_norm__30)

        # pd_op.conv2d: (-1x196x56x56xf32) <- (-1x16x56x56xf32, 196x16x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(relu__5, parameter_31, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_6 = [1, 196, 1, 1]

        # pd_op.reshape: (1x196x1x1xf32, 0x196xf32) <- (196xf32, 4xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_32, full_int_array_6), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x196x56x56xf32) <- (-1x196x56x56xf32, 1x196x1x1xf32)
        add__1 = paddle._C_ops.add_(conv2d_6, reshape_2)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_7 = [-1, 4, 49, 56, 56]

        # pd_op.reshape_: (-1x4x49x56x56xf32, 0x-1x196x56x56xf32) <- (-1x196x56x56xf32, 5xi64)
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__1, full_int_array_7), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [2]

        # pd_op.unsqueeze_: (-1x4x1x49x56x56xf32, None) <- (-1x4x49x56x56xf32, 1xi64)
        unsqueeze__2, unsqueeze__3 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(reshape__6, full_int_array_8), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unfold: (-1x3136x3136xf32) <- (-1x64x56x56xf32)
        unfold_1 = paddle._C_ops.unfold(relu__4, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_9 = [-1, 4, 16, 49, 56, 56]

        # pd_op.reshape_: (-1x4x16x49x56x56xf32, 0x-1x3136x3136xf32) <- (-1x3136x3136xf32, 6xi64)
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_1, full_int_array_9), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x4x16x49x56x56xf32) <- (-1x4x1x49x56x56xf32, -1x4x16x49x56x56xf32)
        multiply_1 = paddle._C_ops.multiply(unsqueeze__2, reshape__8)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_10 = [3]

        # pd_op.sum: (-1x4x16x56x56xf32) <- (-1x4x16x49x56x56xf32, 1xi64)
        sum_1 = paddle._C_ops.sum(multiply_1, full_int_array_10, None, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_11 = [-1, 64, 56, 56]

        # pd_op.reshape_: (-1x64x56x56xf32, 0x-1x4x16x56x56xf32) <- (-1x4x16x56x56xf32, 4xi64)
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sum_1, full_int_array_11), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__10, parameter_33, parameter_34, parameter_35, parameter_36, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        relu__6 = paddle._C_ops.relu_(batch_norm__36)

        # pd_op.conv2d: (-1x256x56x56xf32) <- (-1x64x56x56xf32, 256x64x1x1xf32)
        conv2d_7 = paddle._C_ops.conv2d(relu__6, parameter_37, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x56x56xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_7, parameter_38, parameter_39, parameter_40, parameter_41, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x256x56x56xf32) <- (-1x64x56x56xf32, 256x64x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(pool2d_0, parameter_42, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x56x56xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_8, parameter_43, parameter_44, parameter_45, parameter_46, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x256x56x56xf32) <- (-1x256x56x56xf32, -1x256x56x56xf32)
        add__2 = paddle._C_ops.add_(batch_norm__42, batch_norm__48)

        # pd_op.relu_: (-1x256x56x56xf32) <- (-1x256x56x56xf32)
        relu__7 = paddle._C_ops.relu_(add__2)

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x256x56x56xf32, 64x256x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(relu__7, parameter_47, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_9, parameter_48, parameter_49, parameter_50, parameter_51, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        relu__8 = paddle._C_ops.relu_(batch_norm__54)

        # pd_op.conv2d: (-1x16x56x56xf32) <- (-1x64x56x56xf32, 16x64x1x1xf32)
        conv2d_10 = paddle._C_ops.conv2d(relu__8, parameter_52, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x16x56x56xf32, 16xf32, 16xf32, xf32, xf32, None) <- (-1x16x56x56xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_10, parameter_53, parameter_54, parameter_55, parameter_56, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x16x56x56xf32) <- (-1x16x56x56xf32)
        relu__9 = paddle._C_ops.relu_(batch_norm__60)

        # pd_op.conv2d: (-1x196x56x56xf32) <- (-1x16x56x56xf32, 196x16x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(relu__9, parameter_57, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_12 = [1, 196, 1, 1]

        # pd_op.reshape: (1x196x1x1xf32, 0x196xf32) <- (196xf32, 4xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_58, full_int_array_12), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x196x56x56xf32) <- (-1x196x56x56xf32, 1x196x1x1xf32)
        add__3 = paddle._C_ops.add_(conv2d_11, reshape_4)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_13 = [-1, 4, 49, 56, 56]

        # pd_op.reshape_: (-1x4x49x56x56xf32, 0x-1x196x56x56xf32) <- (-1x196x56x56xf32, 5xi64)
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__3, full_int_array_13), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_14 = [2]

        # pd_op.unsqueeze_: (-1x4x1x49x56x56xf32, None) <- (-1x4x49x56x56xf32, 1xi64)
        unsqueeze__4, unsqueeze__5 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(reshape__12, full_int_array_14), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unfold: (-1x3136x3136xf32) <- (-1x64x56x56xf32)
        unfold_2 = paddle._C_ops.unfold(relu__8, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_15 = [-1, 4, 16, 49, 56, 56]

        # pd_op.reshape_: (-1x4x16x49x56x56xf32, 0x-1x3136x3136xf32) <- (-1x3136x3136xf32, 6xi64)
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_2, full_int_array_15), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x4x16x49x56x56xf32) <- (-1x4x1x49x56x56xf32, -1x4x16x49x56x56xf32)
        multiply_2 = paddle._C_ops.multiply(unsqueeze__4, reshape__14)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_16 = [3]

        # pd_op.sum: (-1x4x16x56x56xf32) <- (-1x4x16x49x56x56xf32, 1xi64)
        sum_2 = paddle._C_ops.sum(multiply_2, full_int_array_16, None, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_17 = [-1, 64, 56, 56]

        # pd_op.reshape_: (-1x64x56x56xf32, 0x-1x4x16x56x56xf32) <- (-1x4x16x56x56xf32, 4xi64)
        reshape__16, reshape__17 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sum_2, full_int_array_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__16, parameter_59, parameter_60, parameter_61, parameter_62, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        relu__10 = paddle._C_ops.relu_(batch_norm__66)

        # pd_op.conv2d: (-1x256x56x56xf32) <- (-1x64x56x56xf32, 256x64x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(relu__10, parameter_63, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x56x56xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_12, parameter_64, parameter_65, parameter_66, parameter_67, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x256x56x56xf32) <- (-1x256x56x56xf32, -1x256x56x56xf32)
        add__4 = paddle._C_ops.add_(batch_norm__72, relu__7)

        # pd_op.relu_: (-1x256x56x56xf32) <- (-1x256x56x56xf32)
        relu__11 = paddle._C_ops.relu_(add__4)

        # pd_op.conv2d: (-1x64x56x56xf32) <- (-1x256x56x56xf32, 64x256x1x1xf32)
        conv2d_13 = paddle._C_ops.conv2d(relu__11, parameter_68, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_13, parameter_69, parameter_70, parameter_71, parameter_72, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        relu__12 = paddle._C_ops.relu_(batch_norm__78)

        # pd_op.conv2d: (-1x16x56x56xf32) <- (-1x64x56x56xf32, 16x64x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(relu__12, parameter_73, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x16x56x56xf32, 16xf32, 16xf32, xf32, xf32, None) <- (-1x16x56x56xf32, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_14, parameter_74, parameter_75, parameter_76, parameter_77, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x16x56x56xf32) <- (-1x16x56x56xf32)
        relu__13 = paddle._C_ops.relu_(batch_norm__84)

        # pd_op.conv2d: (-1x196x56x56xf32) <- (-1x16x56x56xf32, 196x16x1x1xf32)
        conv2d_15 = paddle._C_ops.conv2d(relu__13, parameter_78, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_18 = [1, 196, 1, 1]

        # pd_op.reshape: (1x196x1x1xf32, 0x196xf32) <- (196xf32, 4xi64)
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_79, full_int_array_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x196x56x56xf32) <- (-1x196x56x56xf32, 1x196x1x1xf32)
        add__5 = paddle._C_ops.add_(conv2d_15, reshape_6)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_19 = [-1, 4, 49, 56, 56]

        # pd_op.reshape_: (-1x4x49x56x56xf32, 0x-1x196x56x56xf32) <- (-1x196x56x56xf32, 5xi64)
        reshape__18, reshape__19 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__5, full_int_array_19), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_20 = [2]

        # pd_op.unsqueeze_: (-1x4x1x49x56x56xf32, None) <- (-1x4x49x56x56xf32, 1xi64)
        unsqueeze__6, unsqueeze__7 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(reshape__18, full_int_array_20), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unfold: (-1x3136x3136xf32) <- (-1x64x56x56xf32)
        unfold_3 = paddle._C_ops.unfold(relu__12, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_21 = [-1, 4, 16, 49, 56, 56]

        # pd_op.reshape_: (-1x4x16x49x56x56xf32, 0x-1x3136x3136xf32) <- (-1x3136x3136xf32, 6xi64)
        reshape__20, reshape__21 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_3, full_int_array_21), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x4x16x49x56x56xf32) <- (-1x4x1x49x56x56xf32, -1x4x16x49x56x56xf32)
        multiply_3 = paddle._C_ops.multiply(unsqueeze__6, reshape__20)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_22 = [3]

        # pd_op.sum: (-1x4x16x56x56xf32) <- (-1x4x16x49x56x56xf32, 1xi64)
        sum_3 = paddle._C_ops.sum(multiply_3, full_int_array_22, None, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_23 = [-1, 64, 56, 56]

        # pd_op.reshape_: (-1x64x56x56xf32, 0x-1x4x16x56x56xf32) <- (-1x4x16x56x56xf32, 4xi64)
        reshape__22, reshape__23 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sum_3, full_int_array_23), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x64x56x56xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x56x56xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__22, parameter_80, parameter_81, parameter_82, parameter_83, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x56x56xf32) <- (-1x64x56x56xf32)
        relu__14 = paddle._C_ops.relu_(batch_norm__90)

        # pd_op.conv2d: (-1x256x56x56xf32) <- (-1x64x56x56xf32, 256x64x1x1xf32)
        conv2d_16 = paddle._C_ops.conv2d(relu__14, parameter_84, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x56x56xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x56x56xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_16, parameter_85, parameter_86, parameter_87, parameter_88, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x256x56x56xf32) <- (-1x256x56x56xf32, -1x256x56x56xf32)
        add__6 = paddle._C_ops.add_(batch_norm__96, relu__11)

        # pd_op.relu_: (-1x256x56x56xf32) <- (-1x256x56x56xf32)
        relu__15 = paddle._C_ops.relu_(add__6)

        # pd_op.conv2d: (-1x128x56x56xf32) <- (-1x256x56x56xf32, 128x256x1x1xf32)
        conv2d_17 = paddle._C_ops.conv2d(relu__15, parameter_89, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x56x56xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x56x56xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_17, parameter_90, parameter_91, parameter_92, parameter_93, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x56x56xf32) <- (-1x128x56x56xf32)
        relu__16 = paddle._C_ops.relu_(batch_norm__102)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_24 = [2, 2]

        # pd_op.pool2d: (-1x128x28x28xf32) <- (-1x128x56x56xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(relu__16, full_int_array_24, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x32x28x28xf32) <- (-1x128x28x28xf32, 32x128x1x1xf32)
        conv2d_18 = paddle._C_ops.conv2d(pool2d_1, parameter_94, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x28x28xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x28x28xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_18, parameter_95, parameter_96, parameter_97, parameter_98, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x32x28x28xf32) <- (-1x32x28x28xf32)
        relu__17 = paddle._C_ops.relu_(batch_norm__108)

        # pd_op.conv2d: (-1x392x28x28xf32) <- (-1x32x28x28xf32, 392x32x1x1xf32)
        conv2d_19 = paddle._C_ops.conv2d(relu__17, parameter_99, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_25 = [1, 392, 1, 1]

        # pd_op.reshape: (1x392x1x1xf32, 0x392xf32) <- (392xf32, 4xi64)
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_100, full_int_array_25), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x392x28x28xf32) <- (-1x392x28x28xf32, 1x392x1x1xf32)
        add__7 = paddle._C_ops.add_(conv2d_19, reshape_8)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_26 = [-1, 8, 49, 28, 28]

        # pd_op.reshape_: (-1x8x49x28x28xf32, 0x-1x392x28x28xf32) <- (-1x392x28x28xf32, 5xi64)
        reshape__24, reshape__25 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__7, full_int_array_26), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_27 = [2]

        # pd_op.unsqueeze_: (-1x8x1x49x28x28xf32, None) <- (-1x8x49x28x28xf32, 1xi64)
        unsqueeze__8, unsqueeze__9 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(reshape__24, full_int_array_27), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unfold: (-1x6272x784xf32) <- (-1x128x56x56xf32)
        unfold_4 = paddle._C_ops.unfold(relu__16, [7, 7], [2, 2], [3, 3, 3, 3], [1, 1])

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_28 = [-1, 8, 16, 49, 28, 28]

        # pd_op.reshape_: (-1x8x16x49x28x28xf32, 0x-1x6272x784xf32) <- (-1x6272x784xf32, 6xi64)
        reshape__26, reshape__27 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_4, full_int_array_28), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x8x16x49x28x28xf32) <- (-1x8x1x49x28x28xf32, -1x8x16x49x28x28xf32)
        multiply_4 = paddle._C_ops.multiply(unsqueeze__8, reshape__26)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_29 = [3]

        # pd_op.sum: (-1x8x16x28x28xf32) <- (-1x8x16x49x28x28xf32, 1xi64)
        sum_4 = paddle._C_ops.sum(multiply_4, full_int_array_29, None, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_30 = [-1, 128, 28, 28]

        # pd_op.reshape_: (-1x128x28x28xf32, 0x-1x8x16x28x28xf32) <- (-1x8x16x28x28xf32, 4xi64)
        reshape__28, reshape__29 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sum_4, full_int_array_30), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__28, parameter_101, parameter_102, parameter_103, parameter_104, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        relu__18 = paddle._C_ops.relu_(batch_norm__114)

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x128x28x28xf32, 512x128x1x1xf32)
        conv2d_20 = paddle._C_ops.conv2d(relu__18, parameter_105, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_20, parameter_106, parameter_107, parameter_108, parameter_109, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x256x56x56xf32, 512x256x1x1xf32)
        conv2d_21 = paddle._C_ops.conv2d(relu__15, parameter_110, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_21, parameter_111, parameter_112, parameter_113, parameter_114, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x28x28xf32) <- (-1x512x28x28xf32, -1x512x28x28xf32)
        add__8 = paddle._C_ops.add_(batch_norm__120, batch_norm__126)

        # pd_op.relu_: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        relu__19 = paddle._C_ops.relu_(add__8)

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x512x28x28xf32, 128x512x1x1xf32)
        conv2d_22 = paddle._C_ops.conv2d(relu__19, parameter_115, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_22, parameter_116, parameter_117, parameter_118, parameter_119, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        relu__20 = paddle._C_ops.relu_(batch_norm__132)

        # pd_op.conv2d: (-1x32x28x28xf32) <- (-1x128x28x28xf32, 32x128x1x1xf32)
        conv2d_23 = paddle._C_ops.conv2d(relu__20, parameter_120, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x28x28xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x28x28xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_23, parameter_121, parameter_122, parameter_123, parameter_124, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x32x28x28xf32) <- (-1x32x28x28xf32)
        relu__21 = paddle._C_ops.relu_(batch_norm__138)

        # pd_op.conv2d: (-1x392x28x28xf32) <- (-1x32x28x28xf32, 392x32x1x1xf32)
        conv2d_24 = paddle._C_ops.conv2d(relu__21, parameter_125, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_31 = [1, 392, 1, 1]

        # pd_op.reshape: (1x392x1x1xf32, 0x392xf32) <- (392xf32, 4xi64)
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_126, full_int_array_31), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x392x28x28xf32) <- (-1x392x28x28xf32, 1x392x1x1xf32)
        add__9 = paddle._C_ops.add_(conv2d_24, reshape_10)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_32 = [-1, 8, 49, 28, 28]

        # pd_op.reshape_: (-1x8x49x28x28xf32, 0x-1x392x28x28xf32) <- (-1x392x28x28xf32, 5xi64)
        reshape__30, reshape__31 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__9, full_int_array_32), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_33 = [2]

        # pd_op.unsqueeze_: (-1x8x1x49x28x28xf32, None) <- (-1x8x49x28x28xf32, 1xi64)
        unsqueeze__10, unsqueeze__11 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(reshape__30, full_int_array_33), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unfold: (-1x6272x784xf32) <- (-1x128x28x28xf32)
        unfold_5 = paddle._C_ops.unfold(relu__20, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_34 = [-1, 8, 16, 49, 28, 28]

        # pd_op.reshape_: (-1x8x16x49x28x28xf32, 0x-1x6272x784xf32) <- (-1x6272x784xf32, 6xi64)
        reshape__32, reshape__33 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_5, full_int_array_34), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x8x16x49x28x28xf32) <- (-1x8x1x49x28x28xf32, -1x8x16x49x28x28xf32)
        multiply_5 = paddle._C_ops.multiply(unsqueeze__10, reshape__32)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_35 = [3]

        # pd_op.sum: (-1x8x16x28x28xf32) <- (-1x8x16x49x28x28xf32, 1xi64)
        sum_5 = paddle._C_ops.sum(multiply_5, full_int_array_35, None, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_36 = [-1, 128, 28, 28]

        # pd_op.reshape_: (-1x128x28x28xf32, 0x-1x8x16x28x28xf32) <- (-1x8x16x28x28xf32, 4xi64)
        reshape__34, reshape__35 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sum_5, full_int_array_36), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__34, parameter_127, parameter_128, parameter_129, parameter_130, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        relu__22 = paddle._C_ops.relu_(batch_norm__144)

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x128x28x28xf32, 512x128x1x1xf32)
        conv2d_25 = paddle._C_ops.conv2d(relu__22, parameter_131, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_25, parameter_132, parameter_133, parameter_134, parameter_135, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x28x28xf32) <- (-1x512x28x28xf32, -1x512x28x28xf32)
        add__10 = paddle._C_ops.add_(batch_norm__150, relu__19)

        # pd_op.relu_: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        relu__23 = paddle._C_ops.relu_(add__10)

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x512x28x28xf32, 128x512x1x1xf32)
        conv2d_26 = paddle._C_ops.conv2d(relu__23, parameter_136, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_26, parameter_137, parameter_138, parameter_139, parameter_140, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        relu__24 = paddle._C_ops.relu_(batch_norm__156)

        # pd_op.conv2d: (-1x32x28x28xf32) <- (-1x128x28x28xf32, 32x128x1x1xf32)
        conv2d_27 = paddle._C_ops.conv2d(relu__24, parameter_141, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x28x28xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x28x28xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_27, parameter_142, parameter_143, parameter_144, parameter_145, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x32x28x28xf32) <- (-1x32x28x28xf32)
        relu__25 = paddle._C_ops.relu_(batch_norm__162)

        # pd_op.conv2d: (-1x392x28x28xf32) <- (-1x32x28x28xf32, 392x32x1x1xf32)
        conv2d_28 = paddle._C_ops.conv2d(relu__25, parameter_146, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_37 = [1, 392, 1, 1]

        # pd_op.reshape: (1x392x1x1xf32, 0x392xf32) <- (392xf32, 4xi64)
        reshape_12, reshape_13 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_147, full_int_array_37), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x392x28x28xf32) <- (-1x392x28x28xf32, 1x392x1x1xf32)
        add__11 = paddle._C_ops.add_(conv2d_28, reshape_12)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_38 = [-1, 8, 49, 28, 28]

        # pd_op.reshape_: (-1x8x49x28x28xf32, 0x-1x392x28x28xf32) <- (-1x392x28x28xf32, 5xi64)
        reshape__36, reshape__37 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__11, full_int_array_38), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_39 = [2]

        # pd_op.unsqueeze_: (-1x8x1x49x28x28xf32, None) <- (-1x8x49x28x28xf32, 1xi64)
        unsqueeze__12, unsqueeze__13 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(reshape__36, full_int_array_39), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unfold: (-1x6272x784xf32) <- (-1x128x28x28xf32)
        unfold_6 = paddle._C_ops.unfold(relu__24, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_40 = [-1, 8, 16, 49, 28, 28]

        # pd_op.reshape_: (-1x8x16x49x28x28xf32, 0x-1x6272x784xf32) <- (-1x6272x784xf32, 6xi64)
        reshape__38, reshape__39 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_6, full_int_array_40), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x8x16x49x28x28xf32) <- (-1x8x1x49x28x28xf32, -1x8x16x49x28x28xf32)
        multiply_6 = paddle._C_ops.multiply(unsqueeze__12, reshape__38)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_41 = [3]

        # pd_op.sum: (-1x8x16x28x28xf32) <- (-1x8x16x49x28x28xf32, 1xi64)
        sum_6 = paddle._C_ops.sum(multiply_6, full_int_array_41, None, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_42 = [-1, 128, 28, 28]

        # pd_op.reshape_: (-1x128x28x28xf32, 0x-1x8x16x28x28xf32) <- (-1x8x16x28x28xf32, 4xi64)
        reshape__40, reshape__41 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sum_6, full_int_array_42), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__40, parameter_148, parameter_149, parameter_150, parameter_151, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        relu__26 = paddle._C_ops.relu_(batch_norm__168)

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x128x28x28xf32, 512x128x1x1xf32)
        conv2d_29 = paddle._C_ops.conv2d(relu__26, parameter_152, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_29, parameter_153, parameter_154, parameter_155, parameter_156, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x28x28xf32) <- (-1x512x28x28xf32, -1x512x28x28xf32)
        add__12 = paddle._C_ops.add_(batch_norm__174, relu__23)

        # pd_op.relu_: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        relu__27 = paddle._C_ops.relu_(add__12)

        # pd_op.conv2d: (-1x128x28x28xf32) <- (-1x512x28x28xf32, 128x512x1x1xf32)
        conv2d_30 = paddle._C_ops.conv2d(relu__27, parameter_157, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__180, batch_norm__181, batch_norm__182, batch_norm__183, batch_norm__184, batch_norm__185 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_30, parameter_158, parameter_159, parameter_160, parameter_161, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        relu__28 = paddle._C_ops.relu_(batch_norm__180)

        # pd_op.conv2d: (-1x32x28x28xf32) <- (-1x128x28x28xf32, 32x128x1x1xf32)
        conv2d_31 = paddle._C_ops.conv2d(relu__28, parameter_162, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x28x28xf32, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x28x28xf32, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__186, batch_norm__187, batch_norm__188, batch_norm__189, batch_norm__190, batch_norm__191 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_31, parameter_163, parameter_164, parameter_165, parameter_166, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x32x28x28xf32) <- (-1x32x28x28xf32)
        relu__29 = paddle._C_ops.relu_(batch_norm__186)

        # pd_op.conv2d: (-1x392x28x28xf32) <- (-1x32x28x28xf32, 392x32x1x1xf32)
        conv2d_32 = paddle._C_ops.conv2d(relu__29, parameter_167, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_43 = [1, 392, 1, 1]

        # pd_op.reshape: (1x392x1x1xf32, 0x392xf32) <- (392xf32, 4xi64)
        reshape_14, reshape_15 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_168, full_int_array_43), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x392x28x28xf32) <- (-1x392x28x28xf32, 1x392x1x1xf32)
        add__13 = paddle._C_ops.add_(conv2d_32, reshape_14)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_44 = [-1, 8, 49, 28, 28]

        # pd_op.reshape_: (-1x8x49x28x28xf32, 0x-1x392x28x28xf32) <- (-1x392x28x28xf32, 5xi64)
        reshape__42, reshape__43 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__13, full_int_array_44), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_45 = [2]

        # pd_op.unsqueeze_: (-1x8x1x49x28x28xf32, None) <- (-1x8x49x28x28xf32, 1xi64)
        unsqueeze__14, unsqueeze__15 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(reshape__42, full_int_array_45), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unfold: (-1x6272x784xf32) <- (-1x128x28x28xf32)
        unfold_7 = paddle._C_ops.unfold(relu__28, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_46 = [-1, 8, 16, 49, 28, 28]

        # pd_op.reshape_: (-1x8x16x49x28x28xf32, 0x-1x6272x784xf32) <- (-1x6272x784xf32, 6xi64)
        reshape__44, reshape__45 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_7, full_int_array_46), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x8x16x49x28x28xf32) <- (-1x8x1x49x28x28xf32, -1x8x16x49x28x28xf32)
        multiply_7 = paddle._C_ops.multiply(unsqueeze__14, reshape__44)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_47 = [3]

        # pd_op.sum: (-1x8x16x28x28xf32) <- (-1x8x16x49x28x28xf32, 1xi64)
        sum_7 = paddle._C_ops.sum(multiply_7, full_int_array_47, None, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_48 = [-1, 128, 28, 28]

        # pd_op.reshape_: (-1x128x28x28xf32, 0x-1x8x16x28x28xf32) <- (-1x8x16x28x28xf32, 4xi64)
        reshape__46, reshape__47 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sum_7, full_int_array_48), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x128x28x28xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x28x28xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__192, batch_norm__193, batch_norm__194, batch_norm__195, batch_norm__196, batch_norm__197 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__46, parameter_169, parameter_170, parameter_171, parameter_172, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x28x28xf32) <- (-1x128x28x28xf32)
        relu__30 = paddle._C_ops.relu_(batch_norm__192)

        # pd_op.conv2d: (-1x512x28x28xf32) <- (-1x128x28x28xf32, 512x128x1x1xf32)
        conv2d_33 = paddle._C_ops.conv2d(relu__30, parameter_173, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x28x28xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x28x28xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__198, batch_norm__199, batch_norm__200, batch_norm__201, batch_norm__202, batch_norm__203 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_33, parameter_174, parameter_175, parameter_176, parameter_177, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x512x28x28xf32) <- (-1x512x28x28xf32, -1x512x28x28xf32)
        add__14 = paddle._C_ops.add_(batch_norm__198, relu__27)

        # pd_op.relu_: (-1x512x28x28xf32) <- (-1x512x28x28xf32)
        relu__31 = paddle._C_ops.relu_(add__14)

        # pd_op.conv2d: (-1x256x28x28xf32) <- (-1x512x28x28xf32, 256x512x1x1xf32)
        conv2d_34 = paddle._C_ops.conv2d(relu__31, parameter_178, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x28x28xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x28x28xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__204, batch_norm__205, batch_norm__206, batch_norm__207, batch_norm__208, batch_norm__209 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_34, parameter_179, parameter_180, parameter_181, parameter_182, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x28x28xf32) <- (-1x256x28x28xf32)
        relu__32 = paddle._C_ops.relu_(batch_norm__204)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_49 = [2, 2]

        # pd_op.pool2d: (-1x256x14x14xf32) <- (-1x256x28x28xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(relu__32, full_int_array_49, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x256x14x14xf32, 64x256x1x1xf32)
        conv2d_35 = paddle._C_ops.conv2d(pool2d_2, parameter_183, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__210, batch_norm__211, batch_norm__212, batch_norm__213, batch_norm__214, batch_norm__215 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_35, parameter_184, parameter_185, parameter_186, parameter_187, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x14x14xf32) <- (-1x64x14x14xf32)
        relu__33 = paddle._C_ops.relu_(batch_norm__210)

        # pd_op.conv2d: (-1x784x14x14xf32) <- (-1x64x14x14xf32, 784x64x1x1xf32)
        conv2d_36 = paddle._C_ops.conv2d(relu__33, parameter_188, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_50 = [1, 784, 1, 1]

        # pd_op.reshape: (1x784x1x1xf32, 0x784xf32) <- (784xf32, 4xi64)
        reshape_16, reshape_17 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_189, full_int_array_50), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x784x14x14xf32) <- (-1x784x14x14xf32, 1x784x1x1xf32)
        add__15 = paddle._C_ops.add_(conv2d_36, reshape_16)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_51 = [-1, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x49x14x14xf32, 0x-1x784x14x14xf32) <- (-1x784x14x14xf32, 5xi64)
        reshape__48, reshape__49 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__15, full_int_array_51), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_52 = [2]

        # pd_op.unsqueeze_: (-1x16x1x49x14x14xf32, None) <- (-1x16x49x14x14xf32, 1xi64)
        unsqueeze__16, unsqueeze__17 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(reshape__48, full_int_array_52), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unfold: (-1x12544x196xf32) <- (-1x256x28x28xf32)
        unfold_8 = paddle._C_ops.unfold(relu__32, [7, 7], [2, 2], [3, 3, 3, 3], [1, 1])

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_53 = [-1, 16, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x16x49x14x14xf32, 0x-1x12544x196xf32) <- (-1x12544x196xf32, 6xi64)
        reshape__50, reshape__51 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_8, full_int_array_53), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x16x16x49x14x14xf32) <- (-1x16x1x49x14x14xf32, -1x16x16x49x14x14xf32)
        multiply_8 = paddle._C_ops.multiply(unsqueeze__16, reshape__50)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_54 = [3]

        # pd_op.sum: (-1x16x16x14x14xf32) <- (-1x16x16x49x14x14xf32, 1xi64)
        sum_8 = paddle._C_ops.sum(multiply_8, full_int_array_54, None, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_55 = [-1, 256, 14, 14]

        # pd_op.reshape_: (-1x256x14x14xf32, 0x-1x16x16x14x14xf32) <- (-1x16x16x14x14xf32, 4xi64)
        reshape__52, reshape__53 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sum_8, full_int_array_55), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__216, batch_norm__217, batch_norm__218, batch_norm__219, batch_norm__220, batch_norm__221 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__52, parameter_190, parameter_191, parameter_192, parameter_193, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__34 = paddle._C_ops.relu_(batch_norm__216)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        conv2d_37 = paddle._C_ops.conv2d(relu__34, parameter_194, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__222, batch_norm__223, batch_norm__224, batch_norm__225, batch_norm__226, batch_norm__227 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_37, parameter_195, parameter_196, parameter_197, parameter_198, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x512x28x28xf32, 1024x512x1x1xf32)
        conv2d_38 = paddle._C_ops.conv2d(relu__31, parameter_199, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__228, batch_norm__229, batch_norm__230, batch_norm__231, batch_norm__232, batch_norm__233 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_38, parameter_200, parameter_201, parameter_202, parameter_203, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__16 = paddle._C_ops.add_(batch_norm__222, batch_norm__228)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__35 = paddle._C_ops.relu_(add__16)

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        conv2d_39 = paddle._C_ops.conv2d(relu__35, parameter_204, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__234, batch_norm__235, batch_norm__236, batch_norm__237, batch_norm__238, batch_norm__239 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_39, parameter_205, parameter_206, parameter_207, parameter_208, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__36 = paddle._C_ops.relu_(batch_norm__234)

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x256x14x14xf32, 64x256x1x1xf32)
        conv2d_40 = paddle._C_ops.conv2d(relu__36, parameter_209, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__240, batch_norm__241, batch_norm__242, batch_norm__243, batch_norm__244, batch_norm__245 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_40, parameter_210, parameter_211, parameter_212, parameter_213, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x14x14xf32) <- (-1x64x14x14xf32)
        relu__37 = paddle._C_ops.relu_(batch_norm__240)

        # pd_op.conv2d: (-1x784x14x14xf32) <- (-1x64x14x14xf32, 784x64x1x1xf32)
        conv2d_41 = paddle._C_ops.conv2d(relu__37, parameter_214, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_56 = [1, 784, 1, 1]

        # pd_op.reshape: (1x784x1x1xf32, 0x784xf32) <- (784xf32, 4xi64)
        reshape_18, reshape_19 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_215, full_int_array_56), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x784x14x14xf32) <- (-1x784x14x14xf32, 1x784x1x1xf32)
        add__17 = paddle._C_ops.add_(conv2d_41, reshape_18)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_57 = [-1, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x49x14x14xf32, 0x-1x784x14x14xf32) <- (-1x784x14x14xf32, 5xi64)
        reshape__54, reshape__55 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__17, full_int_array_57), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_58 = [2]

        # pd_op.unsqueeze_: (-1x16x1x49x14x14xf32, None) <- (-1x16x49x14x14xf32, 1xi64)
        unsqueeze__18, unsqueeze__19 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(reshape__54, full_int_array_58), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unfold: (-1x12544x196xf32) <- (-1x256x14x14xf32)
        unfold_9 = paddle._C_ops.unfold(relu__36, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_59 = [-1, 16, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x16x49x14x14xf32, 0x-1x12544x196xf32) <- (-1x12544x196xf32, 6xi64)
        reshape__56, reshape__57 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_9, full_int_array_59), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x16x16x49x14x14xf32) <- (-1x16x1x49x14x14xf32, -1x16x16x49x14x14xf32)
        multiply_9 = paddle._C_ops.multiply(unsqueeze__18, reshape__56)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_60 = [3]

        # pd_op.sum: (-1x16x16x14x14xf32) <- (-1x16x16x49x14x14xf32, 1xi64)
        sum_9 = paddle._C_ops.sum(multiply_9, full_int_array_60, None, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_61 = [-1, 256, 14, 14]

        # pd_op.reshape_: (-1x256x14x14xf32, 0x-1x16x16x14x14xf32) <- (-1x16x16x14x14xf32, 4xi64)
        reshape__58, reshape__59 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sum_9, full_int_array_61), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__246, batch_norm__247, batch_norm__248, batch_norm__249, batch_norm__250, batch_norm__251 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__58, parameter_216, parameter_217, parameter_218, parameter_219, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__38 = paddle._C_ops.relu_(batch_norm__246)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        conv2d_42 = paddle._C_ops.conv2d(relu__38, parameter_220, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__252, batch_norm__253, batch_norm__254, batch_norm__255, batch_norm__256, batch_norm__257 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_42, parameter_221, parameter_222, parameter_223, parameter_224, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__18 = paddle._C_ops.add_(batch_norm__252, relu__35)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__39 = paddle._C_ops.relu_(add__18)

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        conv2d_43 = paddle._C_ops.conv2d(relu__39, parameter_225, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__258, batch_norm__259, batch_norm__260, batch_norm__261, batch_norm__262, batch_norm__263 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_43, parameter_226, parameter_227, parameter_228, parameter_229, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__40 = paddle._C_ops.relu_(batch_norm__258)

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x256x14x14xf32, 64x256x1x1xf32)
        conv2d_44 = paddle._C_ops.conv2d(relu__40, parameter_230, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__264, batch_norm__265, batch_norm__266, batch_norm__267, batch_norm__268, batch_norm__269 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_44, parameter_231, parameter_232, parameter_233, parameter_234, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x14x14xf32) <- (-1x64x14x14xf32)
        relu__41 = paddle._C_ops.relu_(batch_norm__264)

        # pd_op.conv2d: (-1x784x14x14xf32) <- (-1x64x14x14xf32, 784x64x1x1xf32)
        conv2d_45 = paddle._C_ops.conv2d(relu__41, parameter_235, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_62 = [1, 784, 1, 1]

        # pd_op.reshape: (1x784x1x1xf32, 0x784xf32) <- (784xf32, 4xi64)
        reshape_20, reshape_21 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_236, full_int_array_62), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x784x14x14xf32) <- (-1x784x14x14xf32, 1x784x1x1xf32)
        add__19 = paddle._C_ops.add_(conv2d_45, reshape_20)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_63 = [-1, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x49x14x14xf32, 0x-1x784x14x14xf32) <- (-1x784x14x14xf32, 5xi64)
        reshape__60, reshape__61 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__19, full_int_array_63), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_64 = [2]

        # pd_op.unsqueeze_: (-1x16x1x49x14x14xf32, None) <- (-1x16x49x14x14xf32, 1xi64)
        unsqueeze__20, unsqueeze__21 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(reshape__60, full_int_array_64), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unfold: (-1x12544x196xf32) <- (-1x256x14x14xf32)
        unfold_10 = paddle._C_ops.unfold(relu__40, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_65 = [-1, 16, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x16x49x14x14xf32, 0x-1x12544x196xf32) <- (-1x12544x196xf32, 6xi64)
        reshape__62, reshape__63 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_10, full_int_array_65), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x16x16x49x14x14xf32) <- (-1x16x1x49x14x14xf32, -1x16x16x49x14x14xf32)
        multiply_10 = paddle._C_ops.multiply(unsqueeze__20, reshape__62)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_66 = [3]

        # pd_op.sum: (-1x16x16x14x14xf32) <- (-1x16x16x49x14x14xf32, 1xi64)
        sum_10 = paddle._C_ops.sum(multiply_10, full_int_array_66, None, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_67 = [-1, 256, 14, 14]

        # pd_op.reshape_: (-1x256x14x14xf32, 0x-1x16x16x14x14xf32) <- (-1x16x16x14x14xf32, 4xi64)
        reshape__64, reshape__65 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sum_10, full_int_array_67), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__270, batch_norm__271, batch_norm__272, batch_norm__273, batch_norm__274, batch_norm__275 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__64, parameter_237, parameter_238, parameter_239, parameter_240, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__42 = paddle._C_ops.relu_(batch_norm__270)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        conv2d_46 = paddle._C_ops.conv2d(relu__42, parameter_241, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__276, batch_norm__277, batch_norm__278, batch_norm__279, batch_norm__280, batch_norm__281 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_46, parameter_242, parameter_243, parameter_244, parameter_245, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__20 = paddle._C_ops.add_(batch_norm__276, relu__39)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__43 = paddle._C_ops.relu_(add__20)

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        conv2d_47 = paddle._C_ops.conv2d(relu__43, parameter_246, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__282, batch_norm__283, batch_norm__284, batch_norm__285, batch_norm__286, batch_norm__287 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_47, parameter_247, parameter_248, parameter_249, parameter_250, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__44 = paddle._C_ops.relu_(batch_norm__282)

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x256x14x14xf32, 64x256x1x1xf32)
        conv2d_48 = paddle._C_ops.conv2d(relu__44, parameter_251, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__288, batch_norm__289, batch_norm__290, batch_norm__291, batch_norm__292, batch_norm__293 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_48, parameter_252, parameter_253, parameter_254, parameter_255, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x14x14xf32) <- (-1x64x14x14xf32)
        relu__45 = paddle._C_ops.relu_(batch_norm__288)

        # pd_op.conv2d: (-1x784x14x14xf32) <- (-1x64x14x14xf32, 784x64x1x1xf32)
        conv2d_49 = paddle._C_ops.conv2d(relu__45, parameter_256, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_68 = [1, 784, 1, 1]

        # pd_op.reshape: (1x784x1x1xf32, 0x784xf32) <- (784xf32, 4xi64)
        reshape_22, reshape_23 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_257, full_int_array_68), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x784x14x14xf32) <- (-1x784x14x14xf32, 1x784x1x1xf32)
        add__21 = paddle._C_ops.add_(conv2d_49, reshape_22)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_69 = [-1, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x49x14x14xf32, 0x-1x784x14x14xf32) <- (-1x784x14x14xf32, 5xi64)
        reshape__66, reshape__67 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__21, full_int_array_69), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_70 = [2]

        # pd_op.unsqueeze_: (-1x16x1x49x14x14xf32, None) <- (-1x16x49x14x14xf32, 1xi64)
        unsqueeze__22, unsqueeze__23 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(reshape__66, full_int_array_70), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unfold: (-1x12544x196xf32) <- (-1x256x14x14xf32)
        unfold_11 = paddle._C_ops.unfold(relu__44, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_71 = [-1, 16, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x16x49x14x14xf32, 0x-1x12544x196xf32) <- (-1x12544x196xf32, 6xi64)
        reshape__68, reshape__69 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_11, full_int_array_71), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x16x16x49x14x14xf32) <- (-1x16x1x49x14x14xf32, -1x16x16x49x14x14xf32)
        multiply_11 = paddle._C_ops.multiply(unsqueeze__22, reshape__68)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_72 = [3]

        # pd_op.sum: (-1x16x16x14x14xf32) <- (-1x16x16x49x14x14xf32, 1xi64)
        sum_11 = paddle._C_ops.sum(multiply_11, full_int_array_72, None, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_73 = [-1, 256, 14, 14]

        # pd_op.reshape_: (-1x256x14x14xf32, 0x-1x16x16x14x14xf32) <- (-1x16x16x14x14xf32, 4xi64)
        reshape__70, reshape__71 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sum_11, full_int_array_73), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__294, batch_norm__295, batch_norm__296, batch_norm__297, batch_norm__298, batch_norm__299 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__70, parameter_258, parameter_259, parameter_260, parameter_261, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__46 = paddle._C_ops.relu_(batch_norm__294)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        conv2d_50 = paddle._C_ops.conv2d(relu__46, parameter_262, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__300, batch_norm__301, batch_norm__302, batch_norm__303, batch_norm__304, batch_norm__305 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_50, parameter_263, parameter_264, parameter_265, parameter_266, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__22 = paddle._C_ops.add_(batch_norm__300, relu__43)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__47 = paddle._C_ops.relu_(add__22)

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        conv2d_51 = paddle._C_ops.conv2d(relu__47, parameter_267, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__306, batch_norm__307, batch_norm__308, batch_norm__309, batch_norm__310, batch_norm__311 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_51, parameter_268, parameter_269, parameter_270, parameter_271, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__48 = paddle._C_ops.relu_(batch_norm__306)

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x256x14x14xf32, 64x256x1x1xf32)
        conv2d_52 = paddle._C_ops.conv2d(relu__48, parameter_272, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__312, batch_norm__313, batch_norm__314, batch_norm__315, batch_norm__316, batch_norm__317 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_52, parameter_273, parameter_274, parameter_275, parameter_276, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x14x14xf32) <- (-1x64x14x14xf32)
        relu__49 = paddle._C_ops.relu_(batch_norm__312)

        # pd_op.conv2d: (-1x784x14x14xf32) <- (-1x64x14x14xf32, 784x64x1x1xf32)
        conv2d_53 = paddle._C_ops.conv2d(relu__49, parameter_277, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_74 = [1, 784, 1, 1]

        # pd_op.reshape: (1x784x1x1xf32, 0x784xf32) <- (784xf32, 4xi64)
        reshape_24, reshape_25 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_278, full_int_array_74), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x784x14x14xf32) <- (-1x784x14x14xf32, 1x784x1x1xf32)
        add__23 = paddle._C_ops.add_(conv2d_53, reshape_24)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_75 = [-1, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x49x14x14xf32, 0x-1x784x14x14xf32) <- (-1x784x14x14xf32, 5xi64)
        reshape__72, reshape__73 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__23, full_int_array_75), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_76 = [2]

        # pd_op.unsqueeze_: (-1x16x1x49x14x14xf32, None) <- (-1x16x49x14x14xf32, 1xi64)
        unsqueeze__24, unsqueeze__25 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(reshape__72, full_int_array_76), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unfold: (-1x12544x196xf32) <- (-1x256x14x14xf32)
        unfold_12 = paddle._C_ops.unfold(relu__48, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_77 = [-1, 16, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x16x49x14x14xf32, 0x-1x12544x196xf32) <- (-1x12544x196xf32, 6xi64)
        reshape__74, reshape__75 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_12, full_int_array_77), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x16x16x49x14x14xf32) <- (-1x16x1x49x14x14xf32, -1x16x16x49x14x14xf32)
        multiply_12 = paddle._C_ops.multiply(unsqueeze__24, reshape__74)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_78 = [3]

        # pd_op.sum: (-1x16x16x14x14xf32) <- (-1x16x16x49x14x14xf32, 1xi64)
        sum_12 = paddle._C_ops.sum(multiply_12, full_int_array_78, None, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_79 = [-1, 256, 14, 14]

        # pd_op.reshape_: (-1x256x14x14xf32, 0x-1x16x16x14x14xf32) <- (-1x16x16x14x14xf32, 4xi64)
        reshape__76, reshape__77 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sum_12, full_int_array_79), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__318, batch_norm__319, batch_norm__320, batch_norm__321, batch_norm__322, batch_norm__323 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__76, parameter_279, parameter_280, parameter_281, parameter_282, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__50 = paddle._C_ops.relu_(batch_norm__318)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        conv2d_54 = paddle._C_ops.conv2d(relu__50, parameter_283, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__324, batch_norm__325, batch_norm__326, batch_norm__327, batch_norm__328, batch_norm__329 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_54, parameter_284, parameter_285, parameter_286, parameter_287, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__24 = paddle._C_ops.add_(batch_norm__324, relu__47)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__51 = paddle._C_ops.relu_(add__24)

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        conv2d_55 = paddle._C_ops.conv2d(relu__51, parameter_288, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__330, batch_norm__331, batch_norm__332, batch_norm__333, batch_norm__334, batch_norm__335 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_55, parameter_289, parameter_290, parameter_291, parameter_292, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__52 = paddle._C_ops.relu_(batch_norm__330)

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x256x14x14xf32, 64x256x1x1xf32)
        conv2d_56 = paddle._C_ops.conv2d(relu__52, parameter_293, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__336, batch_norm__337, batch_norm__338, batch_norm__339, batch_norm__340, batch_norm__341 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_56, parameter_294, parameter_295, parameter_296, parameter_297, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x14x14xf32) <- (-1x64x14x14xf32)
        relu__53 = paddle._C_ops.relu_(batch_norm__336)

        # pd_op.conv2d: (-1x784x14x14xf32) <- (-1x64x14x14xf32, 784x64x1x1xf32)
        conv2d_57 = paddle._C_ops.conv2d(relu__53, parameter_298, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_80 = [1, 784, 1, 1]

        # pd_op.reshape: (1x784x1x1xf32, 0x784xf32) <- (784xf32, 4xi64)
        reshape_26, reshape_27 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_299, full_int_array_80), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x784x14x14xf32) <- (-1x784x14x14xf32, 1x784x1x1xf32)
        add__25 = paddle._C_ops.add_(conv2d_57, reshape_26)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_81 = [-1, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x49x14x14xf32, 0x-1x784x14x14xf32) <- (-1x784x14x14xf32, 5xi64)
        reshape__78, reshape__79 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__25, full_int_array_81), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_82 = [2]

        # pd_op.unsqueeze_: (-1x16x1x49x14x14xf32, None) <- (-1x16x49x14x14xf32, 1xi64)
        unsqueeze__26, unsqueeze__27 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(reshape__78, full_int_array_82), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unfold: (-1x12544x196xf32) <- (-1x256x14x14xf32)
        unfold_13 = paddle._C_ops.unfold(relu__52, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_83 = [-1, 16, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x16x49x14x14xf32, 0x-1x12544x196xf32) <- (-1x12544x196xf32, 6xi64)
        reshape__80, reshape__81 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_13, full_int_array_83), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x16x16x49x14x14xf32) <- (-1x16x1x49x14x14xf32, -1x16x16x49x14x14xf32)
        multiply_13 = paddle._C_ops.multiply(unsqueeze__26, reshape__80)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_84 = [3]

        # pd_op.sum: (-1x16x16x14x14xf32) <- (-1x16x16x49x14x14xf32, 1xi64)
        sum_13 = paddle._C_ops.sum(multiply_13, full_int_array_84, None, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_85 = [-1, 256, 14, 14]

        # pd_op.reshape_: (-1x256x14x14xf32, 0x-1x16x16x14x14xf32) <- (-1x16x16x14x14xf32, 4xi64)
        reshape__82, reshape__83 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sum_13, full_int_array_85), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__342, batch_norm__343, batch_norm__344, batch_norm__345, batch_norm__346, batch_norm__347 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__82, parameter_300, parameter_301, parameter_302, parameter_303, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__54 = paddle._C_ops.relu_(batch_norm__342)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        conv2d_58 = paddle._C_ops.conv2d(relu__54, parameter_304, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__348, batch_norm__349, batch_norm__350, batch_norm__351, batch_norm__352, batch_norm__353 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_58, parameter_305, parameter_306, parameter_307, parameter_308, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__26 = paddle._C_ops.add_(batch_norm__348, relu__51)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__55 = paddle._C_ops.relu_(add__26)

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        conv2d_59 = paddle._C_ops.conv2d(relu__55, parameter_309, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__354, batch_norm__355, batch_norm__356, batch_norm__357, batch_norm__358, batch_norm__359 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_59, parameter_310, parameter_311, parameter_312, parameter_313, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__56 = paddle._C_ops.relu_(batch_norm__354)

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x256x14x14xf32, 64x256x1x1xf32)
        conv2d_60 = paddle._C_ops.conv2d(relu__56, parameter_314, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__360, batch_norm__361, batch_norm__362, batch_norm__363, batch_norm__364, batch_norm__365 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_60, parameter_315, parameter_316, parameter_317, parameter_318, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x14x14xf32) <- (-1x64x14x14xf32)
        relu__57 = paddle._C_ops.relu_(batch_norm__360)

        # pd_op.conv2d: (-1x784x14x14xf32) <- (-1x64x14x14xf32, 784x64x1x1xf32)
        conv2d_61 = paddle._C_ops.conv2d(relu__57, parameter_319, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_86 = [1, 784, 1, 1]

        # pd_op.reshape: (1x784x1x1xf32, 0x784xf32) <- (784xf32, 4xi64)
        reshape_28, reshape_29 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_320, full_int_array_86), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x784x14x14xf32) <- (-1x784x14x14xf32, 1x784x1x1xf32)
        add__27 = paddle._C_ops.add_(conv2d_61, reshape_28)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_87 = [-1, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x49x14x14xf32, 0x-1x784x14x14xf32) <- (-1x784x14x14xf32, 5xi64)
        reshape__84, reshape__85 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__27, full_int_array_87), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_88 = [2]

        # pd_op.unsqueeze_: (-1x16x1x49x14x14xf32, None) <- (-1x16x49x14x14xf32, 1xi64)
        unsqueeze__28, unsqueeze__29 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(reshape__84, full_int_array_88), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unfold: (-1x12544x196xf32) <- (-1x256x14x14xf32)
        unfold_14 = paddle._C_ops.unfold(relu__56, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_89 = [-1, 16, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x16x49x14x14xf32, 0x-1x12544x196xf32) <- (-1x12544x196xf32, 6xi64)
        reshape__86, reshape__87 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_14, full_int_array_89), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x16x16x49x14x14xf32) <- (-1x16x1x49x14x14xf32, -1x16x16x49x14x14xf32)
        multiply_14 = paddle._C_ops.multiply(unsqueeze__28, reshape__86)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_90 = [3]

        # pd_op.sum: (-1x16x16x14x14xf32) <- (-1x16x16x49x14x14xf32, 1xi64)
        sum_14 = paddle._C_ops.sum(multiply_14, full_int_array_90, None, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_91 = [-1, 256, 14, 14]

        # pd_op.reshape_: (-1x256x14x14xf32, 0x-1x16x16x14x14xf32) <- (-1x16x16x14x14xf32, 4xi64)
        reshape__88, reshape__89 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sum_14, full_int_array_91), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__366, batch_norm__367, batch_norm__368, batch_norm__369, batch_norm__370, batch_norm__371 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__88, parameter_321, parameter_322, parameter_323, parameter_324, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__58 = paddle._C_ops.relu_(batch_norm__366)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        conv2d_62 = paddle._C_ops.conv2d(relu__58, parameter_325, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__372, batch_norm__373, batch_norm__374, batch_norm__375, batch_norm__376, batch_norm__377 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_62, parameter_326, parameter_327, parameter_328, parameter_329, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__28 = paddle._C_ops.add_(batch_norm__372, relu__55)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__59 = paddle._C_ops.relu_(add__28)

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        conv2d_63 = paddle._C_ops.conv2d(relu__59, parameter_330, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__378, batch_norm__379, batch_norm__380, batch_norm__381, batch_norm__382, batch_norm__383 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_63, parameter_331, parameter_332, parameter_333, parameter_334, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__60 = paddle._C_ops.relu_(batch_norm__378)

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x256x14x14xf32, 64x256x1x1xf32)
        conv2d_64 = paddle._C_ops.conv2d(relu__60, parameter_335, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__384, batch_norm__385, batch_norm__386, batch_norm__387, batch_norm__388, batch_norm__389 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_64, parameter_336, parameter_337, parameter_338, parameter_339, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x14x14xf32) <- (-1x64x14x14xf32)
        relu__61 = paddle._C_ops.relu_(batch_norm__384)

        # pd_op.conv2d: (-1x784x14x14xf32) <- (-1x64x14x14xf32, 784x64x1x1xf32)
        conv2d_65 = paddle._C_ops.conv2d(relu__61, parameter_340, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_92 = [1, 784, 1, 1]

        # pd_op.reshape: (1x784x1x1xf32, 0x784xf32) <- (784xf32, 4xi64)
        reshape_30, reshape_31 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_341, full_int_array_92), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x784x14x14xf32) <- (-1x784x14x14xf32, 1x784x1x1xf32)
        add__29 = paddle._C_ops.add_(conv2d_65, reshape_30)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_93 = [-1, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x49x14x14xf32, 0x-1x784x14x14xf32) <- (-1x784x14x14xf32, 5xi64)
        reshape__90, reshape__91 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__29, full_int_array_93), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_94 = [2]

        # pd_op.unsqueeze_: (-1x16x1x49x14x14xf32, None) <- (-1x16x49x14x14xf32, 1xi64)
        unsqueeze__30, unsqueeze__31 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(reshape__90, full_int_array_94), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unfold: (-1x12544x196xf32) <- (-1x256x14x14xf32)
        unfold_15 = paddle._C_ops.unfold(relu__60, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_95 = [-1, 16, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x16x49x14x14xf32, 0x-1x12544x196xf32) <- (-1x12544x196xf32, 6xi64)
        reshape__92, reshape__93 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_15, full_int_array_95), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x16x16x49x14x14xf32) <- (-1x16x1x49x14x14xf32, -1x16x16x49x14x14xf32)
        multiply_15 = paddle._C_ops.multiply(unsqueeze__30, reshape__92)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_96 = [3]

        # pd_op.sum: (-1x16x16x14x14xf32) <- (-1x16x16x49x14x14xf32, 1xi64)
        sum_15 = paddle._C_ops.sum(multiply_15, full_int_array_96, None, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_97 = [-1, 256, 14, 14]

        # pd_op.reshape_: (-1x256x14x14xf32, 0x-1x16x16x14x14xf32) <- (-1x16x16x14x14xf32, 4xi64)
        reshape__94, reshape__95 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sum_15, full_int_array_97), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__390, batch_norm__391, batch_norm__392, batch_norm__393, batch_norm__394, batch_norm__395 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__94, parameter_342, parameter_343, parameter_344, parameter_345, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__62 = paddle._C_ops.relu_(batch_norm__390)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        conv2d_66 = paddle._C_ops.conv2d(relu__62, parameter_346, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__396, batch_norm__397, batch_norm__398, batch_norm__399, batch_norm__400, batch_norm__401 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_66, parameter_347, parameter_348, parameter_349, parameter_350, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__30 = paddle._C_ops.add_(batch_norm__396, relu__59)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__63 = paddle._C_ops.relu_(add__30)

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        conv2d_67 = paddle._C_ops.conv2d(relu__63, parameter_351, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__402, batch_norm__403, batch_norm__404, batch_norm__405, batch_norm__406, batch_norm__407 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_67, parameter_352, parameter_353, parameter_354, parameter_355, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__64 = paddle._C_ops.relu_(batch_norm__402)

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x256x14x14xf32, 64x256x1x1xf32)
        conv2d_68 = paddle._C_ops.conv2d(relu__64, parameter_356, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__408, batch_norm__409, batch_norm__410, batch_norm__411, batch_norm__412, batch_norm__413 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_68, parameter_357, parameter_358, parameter_359, parameter_360, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x14x14xf32) <- (-1x64x14x14xf32)
        relu__65 = paddle._C_ops.relu_(batch_norm__408)

        # pd_op.conv2d: (-1x784x14x14xf32) <- (-1x64x14x14xf32, 784x64x1x1xf32)
        conv2d_69 = paddle._C_ops.conv2d(relu__65, parameter_361, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_98 = [1, 784, 1, 1]

        # pd_op.reshape: (1x784x1x1xf32, 0x784xf32) <- (784xf32, 4xi64)
        reshape_32, reshape_33 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_362, full_int_array_98), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x784x14x14xf32) <- (-1x784x14x14xf32, 1x784x1x1xf32)
        add__31 = paddle._C_ops.add_(conv2d_69, reshape_32)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_99 = [-1, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x49x14x14xf32, 0x-1x784x14x14xf32) <- (-1x784x14x14xf32, 5xi64)
        reshape__96, reshape__97 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__31, full_int_array_99), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_100 = [2]

        # pd_op.unsqueeze_: (-1x16x1x49x14x14xf32, None) <- (-1x16x49x14x14xf32, 1xi64)
        unsqueeze__32, unsqueeze__33 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(reshape__96, full_int_array_100), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unfold: (-1x12544x196xf32) <- (-1x256x14x14xf32)
        unfold_16 = paddle._C_ops.unfold(relu__64, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_101 = [-1, 16, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x16x49x14x14xf32, 0x-1x12544x196xf32) <- (-1x12544x196xf32, 6xi64)
        reshape__98, reshape__99 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_16, full_int_array_101), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x16x16x49x14x14xf32) <- (-1x16x1x49x14x14xf32, -1x16x16x49x14x14xf32)
        multiply_16 = paddle._C_ops.multiply(unsqueeze__32, reshape__98)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_102 = [3]

        # pd_op.sum: (-1x16x16x14x14xf32) <- (-1x16x16x49x14x14xf32, 1xi64)
        sum_16 = paddle._C_ops.sum(multiply_16, full_int_array_102, None, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_103 = [-1, 256, 14, 14]

        # pd_op.reshape_: (-1x256x14x14xf32, 0x-1x16x16x14x14xf32) <- (-1x16x16x14x14xf32, 4xi64)
        reshape__100, reshape__101 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sum_16, full_int_array_103), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__414, batch_norm__415, batch_norm__416, batch_norm__417, batch_norm__418, batch_norm__419 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__100, parameter_363, parameter_364, parameter_365, parameter_366, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__66 = paddle._C_ops.relu_(batch_norm__414)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        conv2d_70 = paddle._C_ops.conv2d(relu__66, parameter_367, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__420, batch_norm__421, batch_norm__422, batch_norm__423, batch_norm__424, batch_norm__425 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_70, parameter_368, parameter_369, parameter_370, parameter_371, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__32 = paddle._C_ops.add_(batch_norm__420, relu__63)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__67 = paddle._C_ops.relu_(add__32)

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        conv2d_71 = paddle._C_ops.conv2d(relu__67, parameter_372, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__426, batch_norm__427, batch_norm__428, batch_norm__429, batch_norm__430, batch_norm__431 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_71, parameter_373, parameter_374, parameter_375, parameter_376, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__68 = paddle._C_ops.relu_(batch_norm__426)

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x256x14x14xf32, 64x256x1x1xf32)
        conv2d_72 = paddle._C_ops.conv2d(relu__68, parameter_377, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__432, batch_norm__433, batch_norm__434, batch_norm__435, batch_norm__436, batch_norm__437 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_72, parameter_378, parameter_379, parameter_380, parameter_381, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x14x14xf32) <- (-1x64x14x14xf32)
        relu__69 = paddle._C_ops.relu_(batch_norm__432)

        # pd_op.conv2d: (-1x784x14x14xf32) <- (-1x64x14x14xf32, 784x64x1x1xf32)
        conv2d_73 = paddle._C_ops.conv2d(relu__69, parameter_382, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_104 = [1, 784, 1, 1]

        # pd_op.reshape: (1x784x1x1xf32, 0x784xf32) <- (784xf32, 4xi64)
        reshape_34, reshape_35 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_383, full_int_array_104), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x784x14x14xf32) <- (-1x784x14x14xf32, 1x784x1x1xf32)
        add__33 = paddle._C_ops.add_(conv2d_73, reshape_34)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_105 = [-1, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x49x14x14xf32, 0x-1x784x14x14xf32) <- (-1x784x14x14xf32, 5xi64)
        reshape__102, reshape__103 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__33, full_int_array_105), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_106 = [2]

        # pd_op.unsqueeze_: (-1x16x1x49x14x14xf32, None) <- (-1x16x49x14x14xf32, 1xi64)
        unsqueeze__34, unsqueeze__35 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(reshape__102, full_int_array_106), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unfold: (-1x12544x196xf32) <- (-1x256x14x14xf32)
        unfold_17 = paddle._C_ops.unfold(relu__68, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_107 = [-1, 16, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x16x49x14x14xf32, 0x-1x12544x196xf32) <- (-1x12544x196xf32, 6xi64)
        reshape__104, reshape__105 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_17, full_int_array_107), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x16x16x49x14x14xf32) <- (-1x16x1x49x14x14xf32, -1x16x16x49x14x14xf32)
        multiply_17 = paddle._C_ops.multiply(unsqueeze__34, reshape__104)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_108 = [3]

        # pd_op.sum: (-1x16x16x14x14xf32) <- (-1x16x16x49x14x14xf32, 1xi64)
        sum_17 = paddle._C_ops.sum(multiply_17, full_int_array_108, None, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_109 = [-1, 256, 14, 14]

        # pd_op.reshape_: (-1x256x14x14xf32, 0x-1x16x16x14x14xf32) <- (-1x16x16x14x14xf32, 4xi64)
        reshape__106, reshape__107 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sum_17, full_int_array_109), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__438, batch_norm__439, batch_norm__440, batch_norm__441, batch_norm__442, batch_norm__443 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__106, parameter_384, parameter_385, parameter_386, parameter_387, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__70 = paddle._C_ops.relu_(batch_norm__438)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        conv2d_74 = paddle._C_ops.conv2d(relu__70, parameter_388, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__444, batch_norm__445, batch_norm__446, batch_norm__447, batch_norm__448, batch_norm__449 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_74, parameter_389, parameter_390, parameter_391, parameter_392, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__34 = paddle._C_ops.add_(batch_norm__444, relu__67)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__71 = paddle._C_ops.relu_(add__34)

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        conv2d_75 = paddle._C_ops.conv2d(relu__71, parameter_393, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__450, batch_norm__451, batch_norm__452, batch_norm__453, batch_norm__454, batch_norm__455 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_75, parameter_394, parameter_395, parameter_396, parameter_397, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__72 = paddle._C_ops.relu_(batch_norm__450)

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x256x14x14xf32, 64x256x1x1xf32)
        conv2d_76 = paddle._C_ops.conv2d(relu__72, parameter_398, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__456, batch_norm__457, batch_norm__458, batch_norm__459, batch_norm__460, batch_norm__461 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_76, parameter_399, parameter_400, parameter_401, parameter_402, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x14x14xf32) <- (-1x64x14x14xf32)
        relu__73 = paddle._C_ops.relu_(batch_norm__456)

        # pd_op.conv2d: (-1x784x14x14xf32) <- (-1x64x14x14xf32, 784x64x1x1xf32)
        conv2d_77 = paddle._C_ops.conv2d(relu__73, parameter_403, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_110 = [1, 784, 1, 1]

        # pd_op.reshape: (1x784x1x1xf32, 0x784xf32) <- (784xf32, 4xi64)
        reshape_36, reshape_37 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_404, full_int_array_110), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x784x14x14xf32) <- (-1x784x14x14xf32, 1x784x1x1xf32)
        add__35 = paddle._C_ops.add_(conv2d_77, reshape_36)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_111 = [-1, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x49x14x14xf32, 0x-1x784x14x14xf32) <- (-1x784x14x14xf32, 5xi64)
        reshape__108, reshape__109 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__35, full_int_array_111), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_112 = [2]

        # pd_op.unsqueeze_: (-1x16x1x49x14x14xf32, None) <- (-1x16x49x14x14xf32, 1xi64)
        unsqueeze__36, unsqueeze__37 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(reshape__108, full_int_array_112), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unfold: (-1x12544x196xf32) <- (-1x256x14x14xf32)
        unfold_18 = paddle._C_ops.unfold(relu__72, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_113 = [-1, 16, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x16x49x14x14xf32, 0x-1x12544x196xf32) <- (-1x12544x196xf32, 6xi64)
        reshape__110, reshape__111 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_18, full_int_array_113), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x16x16x49x14x14xf32) <- (-1x16x1x49x14x14xf32, -1x16x16x49x14x14xf32)
        multiply_18 = paddle._C_ops.multiply(unsqueeze__36, reshape__110)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_114 = [3]

        # pd_op.sum: (-1x16x16x14x14xf32) <- (-1x16x16x49x14x14xf32, 1xi64)
        sum_18 = paddle._C_ops.sum(multiply_18, full_int_array_114, None, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_115 = [-1, 256, 14, 14]

        # pd_op.reshape_: (-1x256x14x14xf32, 0x-1x16x16x14x14xf32) <- (-1x16x16x14x14xf32, 4xi64)
        reshape__112, reshape__113 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sum_18, full_int_array_115), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__462, batch_norm__463, batch_norm__464, batch_norm__465, batch_norm__466, batch_norm__467 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__112, parameter_405, parameter_406, parameter_407, parameter_408, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__74 = paddle._C_ops.relu_(batch_norm__462)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        conv2d_78 = paddle._C_ops.conv2d(relu__74, parameter_409, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__468, batch_norm__469, batch_norm__470, batch_norm__471, batch_norm__472, batch_norm__473 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_78, parameter_410, parameter_411, parameter_412, parameter_413, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__36 = paddle._C_ops.add_(batch_norm__468, relu__71)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__75 = paddle._C_ops.relu_(add__36)

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        conv2d_79 = paddle._C_ops.conv2d(relu__75, parameter_414, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__474, batch_norm__475, batch_norm__476, batch_norm__477, batch_norm__478, batch_norm__479 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_79, parameter_415, parameter_416, parameter_417, parameter_418, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__76 = paddle._C_ops.relu_(batch_norm__474)

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x256x14x14xf32, 64x256x1x1xf32)
        conv2d_80 = paddle._C_ops.conv2d(relu__76, parameter_419, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__480, batch_norm__481, batch_norm__482, batch_norm__483, batch_norm__484, batch_norm__485 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_80, parameter_420, parameter_421, parameter_422, parameter_423, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x14x14xf32) <- (-1x64x14x14xf32)
        relu__77 = paddle._C_ops.relu_(batch_norm__480)

        # pd_op.conv2d: (-1x784x14x14xf32) <- (-1x64x14x14xf32, 784x64x1x1xf32)
        conv2d_81 = paddle._C_ops.conv2d(relu__77, parameter_424, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_116 = [1, 784, 1, 1]

        # pd_op.reshape: (1x784x1x1xf32, 0x784xf32) <- (784xf32, 4xi64)
        reshape_38, reshape_39 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_425, full_int_array_116), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x784x14x14xf32) <- (-1x784x14x14xf32, 1x784x1x1xf32)
        add__37 = paddle._C_ops.add_(conv2d_81, reshape_38)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_117 = [-1, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x49x14x14xf32, 0x-1x784x14x14xf32) <- (-1x784x14x14xf32, 5xi64)
        reshape__114, reshape__115 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__37, full_int_array_117), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_118 = [2]

        # pd_op.unsqueeze_: (-1x16x1x49x14x14xf32, None) <- (-1x16x49x14x14xf32, 1xi64)
        unsqueeze__38, unsqueeze__39 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(reshape__114, full_int_array_118), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unfold: (-1x12544x196xf32) <- (-1x256x14x14xf32)
        unfold_19 = paddle._C_ops.unfold(relu__76, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_119 = [-1, 16, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x16x49x14x14xf32, 0x-1x12544x196xf32) <- (-1x12544x196xf32, 6xi64)
        reshape__116, reshape__117 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_19, full_int_array_119), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x16x16x49x14x14xf32) <- (-1x16x1x49x14x14xf32, -1x16x16x49x14x14xf32)
        multiply_19 = paddle._C_ops.multiply(unsqueeze__38, reshape__116)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_120 = [3]

        # pd_op.sum: (-1x16x16x14x14xf32) <- (-1x16x16x49x14x14xf32, 1xi64)
        sum_19 = paddle._C_ops.sum(multiply_19, full_int_array_120, None, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_121 = [-1, 256, 14, 14]

        # pd_op.reshape_: (-1x256x14x14xf32, 0x-1x16x16x14x14xf32) <- (-1x16x16x14x14xf32, 4xi64)
        reshape__118, reshape__119 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sum_19, full_int_array_121), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__486, batch_norm__487, batch_norm__488, batch_norm__489, batch_norm__490, batch_norm__491 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__118, parameter_426, parameter_427, parameter_428, parameter_429, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__78 = paddle._C_ops.relu_(batch_norm__486)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        conv2d_82 = paddle._C_ops.conv2d(relu__78, parameter_430, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__492, batch_norm__493, batch_norm__494, batch_norm__495, batch_norm__496, batch_norm__497 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_82, parameter_431, parameter_432, parameter_433, parameter_434, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__38 = paddle._C_ops.add_(batch_norm__492, relu__75)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__79 = paddle._C_ops.relu_(add__38)

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        conv2d_83 = paddle._C_ops.conv2d(relu__79, parameter_435, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__498, batch_norm__499, batch_norm__500, batch_norm__501, batch_norm__502, batch_norm__503 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_83, parameter_436, parameter_437, parameter_438, parameter_439, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__80 = paddle._C_ops.relu_(batch_norm__498)

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x256x14x14xf32, 64x256x1x1xf32)
        conv2d_84 = paddle._C_ops.conv2d(relu__80, parameter_440, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__504, batch_norm__505, batch_norm__506, batch_norm__507, batch_norm__508, batch_norm__509 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_84, parameter_441, parameter_442, parameter_443, parameter_444, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x14x14xf32) <- (-1x64x14x14xf32)
        relu__81 = paddle._C_ops.relu_(batch_norm__504)

        # pd_op.conv2d: (-1x784x14x14xf32) <- (-1x64x14x14xf32, 784x64x1x1xf32)
        conv2d_85 = paddle._C_ops.conv2d(relu__81, parameter_445, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_122 = [1, 784, 1, 1]

        # pd_op.reshape: (1x784x1x1xf32, 0x784xf32) <- (784xf32, 4xi64)
        reshape_40, reshape_41 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_446, full_int_array_122), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x784x14x14xf32) <- (-1x784x14x14xf32, 1x784x1x1xf32)
        add__39 = paddle._C_ops.add_(conv2d_85, reshape_40)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_123 = [-1, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x49x14x14xf32, 0x-1x784x14x14xf32) <- (-1x784x14x14xf32, 5xi64)
        reshape__120, reshape__121 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__39, full_int_array_123), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_124 = [2]

        # pd_op.unsqueeze_: (-1x16x1x49x14x14xf32, None) <- (-1x16x49x14x14xf32, 1xi64)
        unsqueeze__40, unsqueeze__41 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(reshape__120, full_int_array_124), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unfold: (-1x12544x196xf32) <- (-1x256x14x14xf32)
        unfold_20 = paddle._C_ops.unfold(relu__80, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_125 = [-1, 16, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x16x49x14x14xf32, 0x-1x12544x196xf32) <- (-1x12544x196xf32, 6xi64)
        reshape__122, reshape__123 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_20, full_int_array_125), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x16x16x49x14x14xf32) <- (-1x16x1x49x14x14xf32, -1x16x16x49x14x14xf32)
        multiply_20 = paddle._C_ops.multiply(unsqueeze__40, reshape__122)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_126 = [3]

        # pd_op.sum: (-1x16x16x14x14xf32) <- (-1x16x16x49x14x14xf32, 1xi64)
        sum_20 = paddle._C_ops.sum(multiply_20, full_int_array_126, None, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_127 = [-1, 256, 14, 14]

        # pd_op.reshape_: (-1x256x14x14xf32, 0x-1x16x16x14x14xf32) <- (-1x16x16x14x14xf32, 4xi64)
        reshape__124, reshape__125 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sum_20, full_int_array_127), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__510, batch_norm__511, batch_norm__512, batch_norm__513, batch_norm__514, batch_norm__515 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__124, parameter_447, parameter_448, parameter_449, parameter_450, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__82 = paddle._C_ops.relu_(batch_norm__510)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        conv2d_86 = paddle._C_ops.conv2d(relu__82, parameter_451, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__516, batch_norm__517, batch_norm__518, batch_norm__519, batch_norm__520, batch_norm__521 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_86, parameter_452, parameter_453, parameter_454, parameter_455, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__40 = paddle._C_ops.add_(batch_norm__516, relu__79)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__83 = paddle._C_ops.relu_(add__40)

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        conv2d_87 = paddle._C_ops.conv2d(relu__83, parameter_456, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__522, batch_norm__523, batch_norm__524, batch_norm__525, batch_norm__526, batch_norm__527 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_87, parameter_457, parameter_458, parameter_459, parameter_460, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__84 = paddle._C_ops.relu_(batch_norm__522)

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x256x14x14xf32, 64x256x1x1xf32)
        conv2d_88 = paddle._C_ops.conv2d(relu__84, parameter_461, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__528, batch_norm__529, batch_norm__530, batch_norm__531, batch_norm__532, batch_norm__533 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_88, parameter_462, parameter_463, parameter_464, parameter_465, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x14x14xf32) <- (-1x64x14x14xf32)
        relu__85 = paddle._C_ops.relu_(batch_norm__528)

        # pd_op.conv2d: (-1x784x14x14xf32) <- (-1x64x14x14xf32, 784x64x1x1xf32)
        conv2d_89 = paddle._C_ops.conv2d(relu__85, parameter_466, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_128 = [1, 784, 1, 1]

        # pd_op.reshape: (1x784x1x1xf32, 0x784xf32) <- (784xf32, 4xi64)
        reshape_42, reshape_43 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_467, full_int_array_128), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x784x14x14xf32) <- (-1x784x14x14xf32, 1x784x1x1xf32)
        add__41 = paddle._C_ops.add_(conv2d_89, reshape_42)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_129 = [-1, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x49x14x14xf32, 0x-1x784x14x14xf32) <- (-1x784x14x14xf32, 5xi64)
        reshape__126, reshape__127 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__41, full_int_array_129), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_130 = [2]

        # pd_op.unsqueeze_: (-1x16x1x49x14x14xf32, None) <- (-1x16x49x14x14xf32, 1xi64)
        unsqueeze__42, unsqueeze__43 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(reshape__126, full_int_array_130), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unfold: (-1x12544x196xf32) <- (-1x256x14x14xf32)
        unfold_21 = paddle._C_ops.unfold(relu__84, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_131 = [-1, 16, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x16x49x14x14xf32, 0x-1x12544x196xf32) <- (-1x12544x196xf32, 6xi64)
        reshape__128, reshape__129 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_21, full_int_array_131), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x16x16x49x14x14xf32) <- (-1x16x1x49x14x14xf32, -1x16x16x49x14x14xf32)
        multiply_21 = paddle._C_ops.multiply(unsqueeze__42, reshape__128)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_132 = [3]

        # pd_op.sum: (-1x16x16x14x14xf32) <- (-1x16x16x49x14x14xf32, 1xi64)
        sum_21 = paddle._C_ops.sum(multiply_21, full_int_array_132, None, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_133 = [-1, 256, 14, 14]

        # pd_op.reshape_: (-1x256x14x14xf32, 0x-1x16x16x14x14xf32) <- (-1x16x16x14x14xf32, 4xi64)
        reshape__130, reshape__131 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sum_21, full_int_array_133), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__534, batch_norm__535, batch_norm__536, batch_norm__537, batch_norm__538, batch_norm__539 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__130, parameter_468, parameter_469, parameter_470, parameter_471, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__86 = paddle._C_ops.relu_(batch_norm__534)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        conv2d_90 = paddle._C_ops.conv2d(relu__86, parameter_472, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__540, batch_norm__541, batch_norm__542, batch_norm__543, batch_norm__544, batch_norm__545 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_90, parameter_473, parameter_474, parameter_475, parameter_476, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__42 = paddle._C_ops.add_(batch_norm__540, relu__83)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__87 = paddle._C_ops.relu_(add__42)

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        conv2d_91 = paddle._C_ops.conv2d(relu__87, parameter_477, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__546, batch_norm__547, batch_norm__548, batch_norm__549, batch_norm__550, batch_norm__551 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_91, parameter_478, parameter_479, parameter_480, parameter_481, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__88 = paddle._C_ops.relu_(batch_norm__546)

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x256x14x14xf32, 64x256x1x1xf32)
        conv2d_92 = paddle._C_ops.conv2d(relu__88, parameter_482, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__552, batch_norm__553, batch_norm__554, batch_norm__555, batch_norm__556, batch_norm__557 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_92, parameter_483, parameter_484, parameter_485, parameter_486, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x14x14xf32) <- (-1x64x14x14xf32)
        relu__89 = paddle._C_ops.relu_(batch_norm__552)

        # pd_op.conv2d: (-1x784x14x14xf32) <- (-1x64x14x14xf32, 784x64x1x1xf32)
        conv2d_93 = paddle._C_ops.conv2d(relu__89, parameter_487, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_134 = [1, 784, 1, 1]

        # pd_op.reshape: (1x784x1x1xf32, 0x784xf32) <- (784xf32, 4xi64)
        reshape_44, reshape_45 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_488, full_int_array_134), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x784x14x14xf32) <- (-1x784x14x14xf32, 1x784x1x1xf32)
        add__43 = paddle._C_ops.add_(conv2d_93, reshape_44)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_135 = [-1, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x49x14x14xf32, 0x-1x784x14x14xf32) <- (-1x784x14x14xf32, 5xi64)
        reshape__132, reshape__133 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__43, full_int_array_135), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_136 = [2]

        # pd_op.unsqueeze_: (-1x16x1x49x14x14xf32, None) <- (-1x16x49x14x14xf32, 1xi64)
        unsqueeze__44, unsqueeze__45 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(reshape__132, full_int_array_136), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unfold: (-1x12544x196xf32) <- (-1x256x14x14xf32)
        unfold_22 = paddle._C_ops.unfold(relu__88, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_137 = [-1, 16, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x16x49x14x14xf32, 0x-1x12544x196xf32) <- (-1x12544x196xf32, 6xi64)
        reshape__134, reshape__135 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_22, full_int_array_137), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x16x16x49x14x14xf32) <- (-1x16x1x49x14x14xf32, -1x16x16x49x14x14xf32)
        multiply_22 = paddle._C_ops.multiply(unsqueeze__44, reshape__134)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_138 = [3]

        # pd_op.sum: (-1x16x16x14x14xf32) <- (-1x16x16x49x14x14xf32, 1xi64)
        sum_22 = paddle._C_ops.sum(multiply_22, full_int_array_138, None, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_139 = [-1, 256, 14, 14]

        # pd_op.reshape_: (-1x256x14x14xf32, 0x-1x16x16x14x14xf32) <- (-1x16x16x14x14xf32, 4xi64)
        reshape__136, reshape__137 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sum_22, full_int_array_139), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__558, batch_norm__559, batch_norm__560, batch_norm__561, batch_norm__562, batch_norm__563 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__136, parameter_489, parameter_490, parameter_491, parameter_492, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__90 = paddle._C_ops.relu_(batch_norm__558)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        conv2d_94 = paddle._C_ops.conv2d(relu__90, parameter_493, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__564, batch_norm__565, batch_norm__566, batch_norm__567, batch_norm__568, batch_norm__569 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_94, parameter_494, parameter_495, parameter_496, parameter_497, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__44 = paddle._C_ops.add_(batch_norm__564, relu__87)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__91 = paddle._C_ops.relu_(add__44)

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        conv2d_95 = paddle._C_ops.conv2d(relu__91, parameter_498, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__570, batch_norm__571, batch_norm__572, batch_norm__573, batch_norm__574, batch_norm__575 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_95, parameter_499, parameter_500, parameter_501, parameter_502, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__92 = paddle._C_ops.relu_(batch_norm__570)

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x256x14x14xf32, 64x256x1x1xf32)
        conv2d_96 = paddle._C_ops.conv2d(relu__92, parameter_503, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__576, batch_norm__577, batch_norm__578, batch_norm__579, batch_norm__580, batch_norm__581 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_96, parameter_504, parameter_505, parameter_506, parameter_507, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x14x14xf32) <- (-1x64x14x14xf32)
        relu__93 = paddle._C_ops.relu_(batch_norm__576)

        # pd_op.conv2d: (-1x784x14x14xf32) <- (-1x64x14x14xf32, 784x64x1x1xf32)
        conv2d_97 = paddle._C_ops.conv2d(relu__93, parameter_508, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_140 = [1, 784, 1, 1]

        # pd_op.reshape: (1x784x1x1xf32, 0x784xf32) <- (784xf32, 4xi64)
        reshape_46, reshape_47 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_509, full_int_array_140), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x784x14x14xf32) <- (-1x784x14x14xf32, 1x784x1x1xf32)
        add__45 = paddle._C_ops.add_(conv2d_97, reshape_46)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_141 = [-1, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x49x14x14xf32, 0x-1x784x14x14xf32) <- (-1x784x14x14xf32, 5xi64)
        reshape__138, reshape__139 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__45, full_int_array_141), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_142 = [2]

        # pd_op.unsqueeze_: (-1x16x1x49x14x14xf32, None) <- (-1x16x49x14x14xf32, 1xi64)
        unsqueeze__46, unsqueeze__47 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(reshape__138, full_int_array_142), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unfold: (-1x12544x196xf32) <- (-1x256x14x14xf32)
        unfold_23 = paddle._C_ops.unfold(relu__92, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_143 = [-1, 16, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x16x49x14x14xf32, 0x-1x12544x196xf32) <- (-1x12544x196xf32, 6xi64)
        reshape__140, reshape__141 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_23, full_int_array_143), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x16x16x49x14x14xf32) <- (-1x16x1x49x14x14xf32, -1x16x16x49x14x14xf32)
        multiply_23 = paddle._C_ops.multiply(unsqueeze__46, reshape__140)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_144 = [3]

        # pd_op.sum: (-1x16x16x14x14xf32) <- (-1x16x16x49x14x14xf32, 1xi64)
        sum_23 = paddle._C_ops.sum(multiply_23, full_int_array_144, None, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_145 = [-1, 256, 14, 14]

        # pd_op.reshape_: (-1x256x14x14xf32, 0x-1x16x16x14x14xf32) <- (-1x16x16x14x14xf32, 4xi64)
        reshape__142, reshape__143 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sum_23, full_int_array_145), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__582, batch_norm__583, batch_norm__584, batch_norm__585, batch_norm__586, batch_norm__587 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__142, parameter_510, parameter_511, parameter_512, parameter_513, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__94 = paddle._C_ops.relu_(batch_norm__582)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        conv2d_98 = paddle._C_ops.conv2d(relu__94, parameter_514, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__588, batch_norm__589, batch_norm__590, batch_norm__591, batch_norm__592, batch_norm__593 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_98, parameter_515, parameter_516, parameter_517, parameter_518, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__46 = paddle._C_ops.add_(batch_norm__588, relu__91)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__95 = paddle._C_ops.relu_(add__46)

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        conv2d_99 = paddle._C_ops.conv2d(relu__95, parameter_519, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__594, batch_norm__595, batch_norm__596, batch_norm__597, batch_norm__598, batch_norm__599 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_99, parameter_520, parameter_521, parameter_522, parameter_523, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__96 = paddle._C_ops.relu_(batch_norm__594)

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x256x14x14xf32, 64x256x1x1xf32)
        conv2d_100 = paddle._C_ops.conv2d(relu__96, parameter_524, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__600, batch_norm__601, batch_norm__602, batch_norm__603, batch_norm__604, batch_norm__605 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_100, parameter_525, parameter_526, parameter_527, parameter_528, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x14x14xf32) <- (-1x64x14x14xf32)
        relu__97 = paddle._C_ops.relu_(batch_norm__600)

        # pd_op.conv2d: (-1x784x14x14xf32) <- (-1x64x14x14xf32, 784x64x1x1xf32)
        conv2d_101 = paddle._C_ops.conv2d(relu__97, parameter_529, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_146 = [1, 784, 1, 1]

        # pd_op.reshape: (1x784x1x1xf32, 0x784xf32) <- (784xf32, 4xi64)
        reshape_48, reshape_49 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_530, full_int_array_146), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x784x14x14xf32) <- (-1x784x14x14xf32, 1x784x1x1xf32)
        add__47 = paddle._C_ops.add_(conv2d_101, reshape_48)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_147 = [-1, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x49x14x14xf32, 0x-1x784x14x14xf32) <- (-1x784x14x14xf32, 5xi64)
        reshape__144, reshape__145 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__47, full_int_array_147), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_148 = [2]

        # pd_op.unsqueeze_: (-1x16x1x49x14x14xf32, None) <- (-1x16x49x14x14xf32, 1xi64)
        unsqueeze__48, unsqueeze__49 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(reshape__144, full_int_array_148), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unfold: (-1x12544x196xf32) <- (-1x256x14x14xf32)
        unfold_24 = paddle._C_ops.unfold(relu__96, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_149 = [-1, 16, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x16x49x14x14xf32, 0x-1x12544x196xf32) <- (-1x12544x196xf32, 6xi64)
        reshape__146, reshape__147 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_24, full_int_array_149), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x16x16x49x14x14xf32) <- (-1x16x1x49x14x14xf32, -1x16x16x49x14x14xf32)
        multiply_24 = paddle._C_ops.multiply(unsqueeze__48, reshape__146)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_150 = [3]

        # pd_op.sum: (-1x16x16x14x14xf32) <- (-1x16x16x49x14x14xf32, 1xi64)
        sum_24 = paddle._C_ops.sum(multiply_24, full_int_array_150, None, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_151 = [-1, 256, 14, 14]

        # pd_op.reshape_: (-1x256x14x14xf32, 0x-1x16x16x14x14xf32) <- (-1x16x16x14x14xf32, 4xi64)
        reshape__148, reshape__149 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sum_24, full_int_array_151), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__606, batch_norm__607, batch_norm__608, batch_norm__609, batch_norm__610, batch_norm__611 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__148, parameter_531, parameter_532, parameter_533, parameter_534, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__98 = paddle._C_ops.relu_(batch_norm__606)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        conv2d_102 = paddle._C_ops.conv2d(relu__98, parameter_535, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__612, batch_norm__613, batch_norm__614, batch_norm__615, batch_norm__616, batch_norm__617 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_102, parameter_536, parameter_537, parameter_538, parameter_539, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__48 = paddle._C_ops.add_(batch_norm__612, relu__95)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__99 = paddle._C_ops.relu_(add__48)

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        conv2d_103 = paddle._C_ops.conv2d(relu__99, parameter_540, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__618, batch_norm__619, batch_norm__620, batch_norm__621, batch_norm__622, batch_norm__623 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_103, parameter_541, parameter_542, parameter_543, parameter_544, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__100 = paddle._C_ops.relu_(batch_norm__618)

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x256x14x14xf32, 64x256x1x1xf32)
        conv2d_104 = paddle._C_ops.conv2d(relu__100, parameter_545, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__624, batch_norm__625, batch_norm__626, batch_norm__627, batch_norm__628, batch_norm__629 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_104, parameter_546, parameter_547, parameter_548, parameter_549, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x14x14xf32) <- (-1x64x14x14xf32)
        relu__101 = paddle._C_ops.relu_(batch_norm__624)

        # pd_op.conv2d: (-1x784x14x14xf32) <- (-1x64x14x14xf32, 784x64x1x1xf32)
        conv2d_105 = paddle._C_ops.conv2d(relu__101, parameter_550, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_152 = [1, 784, 1, 1]

        # pd_op.reshape: (1x784x1x1xf32, 0x784xf32) <- (784xf32, 4xi64)
        reshape_50, reshape_51 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_551, full_int_array_152), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x784x14x14xf32) <- (-1x784x14x14xf32, 1x784x1x1xf32)
        add__49 = paddle._C_ops.add_(conv2d_105, reshape_50)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_153 = [-1, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x49x14x14xf32, 0x-1x784x14x14xf32) <- (-1x784x14x14xf32, 5xi64)
        reshape__150, reshape__151 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__49, full_int_array_153), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_154 = [2]

        # pd_op.unsqueeze_: (-1x16x1x49x14x14xf32, None) <- (-1x16x49x14x14xf32, 1xi64)
        unsqueeze__50, unsqueeze__51 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(reshape__150, full_int_array_154), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unfold: (-1x12544x196xf32) <- (-1x256x14x14xf32)
        unfold_25 = paddle._C_ops.unfold(relu__100, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_155 = [-1, 16, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x16x49x14x14xf32, 0x-1x12544x196xf32) <- (-1x12544x196xf32, 6xi64)
        reshape__152, reshape__153 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_25, full_int_array_155), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x16x16x49x14x14xf32) <- (-1x16x1x49x14x14xf32, -1x16x16x49x14x14xf32)
        multiply_25 = paddle._C_ops.multiply(unsqueeze__50, reshape__152)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_156 = [3]

        # pd_op.sum: (-1x16x16x14x14xf32) <- (-1x16x16x49x14x14xf32, 1xi64)
        sum_25 = paddle._C_ops.sum(multiply_25, full_int_array_156, None, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_157 = [-1, 256, 14, 14]

        # pd_op.reshape_: (-1x256x14x14xf32, 0x-1x16x16x14x14xf32) <- (-1x16x16x14x14xf32, 4xi64)
        reshape__154, reshape__155 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sum_25, full_int_array_157), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__630, batch_norm__631, batch_norm__632, batch_norm__633, batch_norm__634, batch_norm__635 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__154, parameter_552, parameter_553, parameter_554, parameter_555, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__102 = paddle._C_ops.relu_(batch_norm__630)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        conv2d_106 = paddle._C_ops.conv2d(relu__102, parameter_556, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__636, batch_norm__637, batch_norm__638, batch_norm__639, batch_norm__640, batch_norm__641 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_106, parameter_557, parameter_558, parameter_559, parameter_560, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__50 = paddle._C_ops.add_(batch_norm__636, relu__99)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__103 = paddle._C_ops.relu_(add__50)

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        conv2d_107 = paddle._C_ops.conv2d(relu__103, parameter_561, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__642, batch_norm__643, batch_norm__644, batch_norm__645, batch_norm__646, batch_norm__647 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_107, parameter_562, parameter_563, parameter_564, parameter_565, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__104 = paddle._C_ops.relu_(batch_norm__642)

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x256x14x14xf32, 64x256x1x1xf32)
        conv2d_108 = paddle._C_ops.conv2d(relu__104, parameter_566, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__648, batch_norm__649, batch_norm__650, batch_norm__651, batch_norm__652, batch_norm__653 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_108, parameter_567, parameter_568, parameter_569, parameter_570, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x14x14xf32) <- (-1x64x14x14xf32)
        relu__105 = paddle._C_ops.relu_(batch_norm__648)

        # pd_op.conv2d: (-1x784x14x14xf32) <- (-1x64x14x14xf32, 784x64x1x1xf32)
        conv2d_109 = paddle._C_ops.conv2d(relu__105, parameter_571, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_158 = [1, 784, 1, 1]

        # pd_op.reshape: (1x784x1x1xf32, 0x784xf32) <- (784xf32, 4xi64)
        reshape_52, reshape_53 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_572, full_int_array_158), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x784x14x14xf32) <- (-1x784x14x14xf32, 1x784x1x1xf32)
        add__51 = paddle._C_ops.add_(conv2d_109, reshape_52)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_159 = [-1, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x49x14x14xf32, 0x-1x784x14x14xf32) <- (-1x784x14x14xf32, 5xi64)
        reshape__156, reshape__157 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__51, full_int_array_159), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_160 = [2]

        # pd_op.unsqueeze_: (-1x16x1x49x14x14xf32, None) <- (-1x16x49x14x14xf32, 1xi64)
        unsqueeze__52, unsqueeze__53 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(reshape__156, full_int_array_160), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unfold: (-1x12544x196xf32) <- (-1x256x14x14xf32)
        unfold_26 = paddle._C_ops.unfold(relu__104, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_161 = [-1, 16, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x16x49x14x14xf32, 0x-1x12544x196xf32) <- (-1x12544x196xf32, 6xi64)
        reshape__158, reshape__159 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_26, full_int_array_161), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x16x16x49x14x14xf32) <- (-1x16x1x49x14x14xf32, -1x16x16x49x14x14xf32)
        multiply_26 = paddle._C_ops.multiply(unsqueeze__52, reshape__158)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_162 = [3]

        # pd_op.sum: (-1x16x16x14x14xf32) <- (-1x16x16x49x14x14xf32, 1xi64)
        sum_26 = paddle._C_ops.sum(multiply_26, full_int_array_162, None, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_163 = [-1, 256, 14, 14]

        # pd_op.reshape_: (-1x256x14x14xf32, 0x-1x16x16x14x14xf32) <- (-1x16x16x14x14xf32, 4xi64)
        reshape__160, reshape__161 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sum_26, full_int_array_163), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__654, batch_norm__655, batch_norm__656, batch_norm__657, batch_norm__658, batch_norm__659 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__160, parameter_573, parameter_574, parameter_575, parameter_576, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__106 = paddle._C_ops.relu_(batch_norm__654)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        conv2d_110 = paddle._C_ops.conv2d(relu__106, parameter_577, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__660, batch_norm__661, batch_norm__662, batch_norm__663, batch_norm__664, batch_norm__665 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_110, parameter_578, parameter_579, parameter_580, parameter_581, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__52 = paddle._C_ops.add_(batch_norm__660, relu__103)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__107 = paddle._C_ops.relu_(add__52)

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        conv2d_111 = paddle._C_ops.conv2d(relu__107, parameter_582, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__666, batch_norm__667, batch_norm__668, batch_norm__669, batch_norm__670, batch_norm__671 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_111, parameter_583, parameter_584, parameter_585, parameter_586, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__108 = paddle._C_ops.relu_(batch_norm__666)

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x256x14x14xf32, 64x256x1x1xf32)
        conv2d_112 = paddle._C_ops.conv2d(relu__108, parameter_587, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__672, batch_norm__673, batch_norm__674, batch_norm__675, batch_norm__676, batch_norm__677 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_112, parameter_588, parameter_589, parameter_590, parameter_591, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x14x14xf32) <- (-1x64x14x14xf32)
        relu__109 = paddle._C_ops.relu_(batch_norm__672)

        # pd_op.conv2d: (-1x784x14x14xf32) <- (-1x64x14x14xf32, 784x64x1x1xf32)
        conv2d_113 = paddle._C_ops.conv2d(relu__109, parameter_592, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_164 = [1, 784, 1, 1]

        # pd_op.reshape: (1x784x1x1xf32, 0x784xf32) <- (784xf32, 4xi64)
        reshape_54, reshape_55 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_593, full_int_array_164), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x784x14x14xf32) <- (-1x784x14x14xf32, 1x784x1x1xf32)
        add__53 = paddle._C_ops.add_(conv2d_113, reshape_54)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_165 = [-1, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x49x14x14xf32, 0x-1x784x14x14xf32) <- (-1x784x14x14xf32, 5xi64)
        reshape__162, reshape__163 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__53, full_int_array_165), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_166 = [2]

        # pd_op.unsqueeze_: (-1x16x1x49x14x14xf32, None) <- (-1x16x49x14x14xf32, 1xi64)
        unsqueeze__54, unsqueeze__55 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(reshape__162, full_int_array_166), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unfold: (-1x12544x196xf32) <- (-1x256x14x14xf32)
        unfold_27 = paddle._C_ops.unfold(relu__108, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_167 = [-1, 16, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x16x49x14x14xf32, 0x-1x12544x196xf32) <- (-1x12544x196xf32, 6xi64)
        reshape__164, reshape__165 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_27, full_int_array_167), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x16x16x49x14x14xf32) <- (-1x16x1x49x14x14xf32, -1x16x16x49x14x14xf32)
        multiply_27 = paddle._C_ops.multiply(unsqueeze__54, reshape__164)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_168 = [3]

        # pd_op.sum: (-1x16x16x14x14xf32) <- (-1x16x16x49x14x14xf32, 1xi64)
        sum_27 = paddle._C_ops.sum(multiply_27, full_int_array_168, None, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_169 = [-1, 256, 14, 14]

        # pd_op.reshape_: (-1x256x14x14xf32, 0x-1x16x16x14x14xf32) <- (-1x16x16x14x14xf32, 4xi64)
        reshape__166, reshape__167 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sum_27, full_int_array_169), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__678, batch_norm__679, batch_norm__680, batch_norm__681, batch_norm__682, batch_norm__683 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__166, parameter_594, parameter_595, parameter_596, parameter_597, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__110 = paddle._C_ops.relu_(batch_norm__678)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        conv2d_114 = paddle._C_ops.conv2d(relu__110, parameter_598, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__684, batch_norm__685, batch_norm__686, batch_norm__687, batch_norm__688, batch_norm__689 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_114, parameter_599, parameter_600, parameter_601, parameter_602, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__54 = paddle._C_ops.add_(batch_norm__684, relu__107)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__111 = paddle._C_ops.relu_(add__54)

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        conv2d_115 = paddle._C_ops.conv2d(relu__111, parameter_603, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__690, batch_norm__691, batch_norm__692, batch_norm__693, batch_norm__694, batch_norm__695 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_115, parameter_604, parameter_605, parameter_606, parameter_607, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__112 = paddle._C_ops.relu_(batch_norm__690)

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x256x14x14xf32, 64x256x1x1xf32)
        conv2d_116 = paddle._C_ops.conv2d(relu__112, parameter_608, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__696, batch_norm__697, batch_norm__698, batch_norm__699, batch_norm__700, batch_norm__701 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_116, parameter_609, parameter_610, parameter_611, parameter_612, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x14x14xf32) <- (-1x64x14x14xf32)
        relu__113 = paddle._C_ops.relu_(batch_norm__696)

        # pd_op.conv2d: (-1x784x14x14xf32) <- (-1x64x14x14xf32, 784x64x1x1xf32)
        conv2d_117 = paddle._C_ops.conv2d(relu__113, parameter_613, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_170 = [1, 784, 1, 1]

        # pd_op.reshape: (1x784x1x1xf32, 0x784xf32) <- (784xf32, 4xi64)
        reshape_56, reshape_57 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_614, full_int_array_170), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x784x14x14xf32) <- (-1x784x14x14xf32, 1x784x1x1xf32)
        add__55 = paddle._C_ops.add_(conv2d_117, reshape_56)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_171 = [-1, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x49x14x14xf32, 0x-1x784x14x14xf32) <- (-1x784x14x14xf32, 5xi64)
        reshape__168, reshape__169 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__55, full_int_array_171), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_172 = [2]

        # pd_op.unsqueeze_: (-1x16x1x49x14x14xf32, None) <- (-1x16x49x14x14xf32, 1xi64)
        unsqueeze__56, unsqueeze__57 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(reshape__168, full_int_array_172), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unfold: (-1x12544x196xf32) <- (-1x256x14x14xf32)
        unfold_28 = paddle._C_ops.unfold(relu__112, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_173 = [-1, 16, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x16x49x14x14xf32, 0x-1x12544x196xf32) <- (-1x12544x196xf32, 6xi64)
        reshape__170, reshape__171 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_28, full_int_array_173), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x16x16x49x14x14xf32) <- (-1x16x1x49x14x14xf32, -1x16x16x49x14x14xf32)
        multiply_28 = paddle._C_ops.multiply(unsqueeze__56, reshape__170)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_174 = [3]

        # pd_op.sum: (-1x16x16x14x14xf32) <- (-1x16x16x49x14x14xf32, 1xi64)
        sum_28 = paddle._C_ops.sum(multiply_28, full_int_array_174, None, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_175 = [-1, 256, 14, 14]

        # pd_op.reshape_: (-1x256x14x14xf32, 0x-1x16x16x14x14xf32) <- (-1x16x16x14x14xf32, 4xi64)
        reshape__172, reshape__173 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sum_28, full_int_array_175), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__702, batch_norm__703, batch_norm__704, batch_norm__705, batch_norm__706, batch_norm__707 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__172, parameter_615, parameter_616, parameter_617, parameter_618, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__114 = paddle._C_ops.relu_(batch_norm__702)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        conv2d_118 = paddle._C_ops.conv2d(relu__114, parameter_619, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__708, batch_norm__709, batch_norm__710, batch_norm__711, batch_norm__712, batch_norm__713 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_118, parameter_620, parameter_621, parameter_622, parameter_623, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__56 = paddle._C_ops.add_(batch_norm__708, relu__111)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__115 = paddle._C_ops.relu_(add__56)

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        conv2d_119 = paddle._C_ops.conv2d(relu__115, parameter_624, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__714, batch_norm__715, batch_norm__716, batch_norm__717, batch_norm__718, batch_norm__719 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_119, parameter_625, parameter_626, parameter_627, parameter_628, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__116 = paddle._C_ops.relu_(batch_norm__714)

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x256x14x14xf32, 64x256x1x1xf32)
        conv2d_120 = paddle._C_ops.conv2d(relu__116, parameter_629, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__720, batch_norm__721, batch_norm__722, batch_norm__723, batch_norm__724, batch_norm__725 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_120, parameter_630, parameter_631, parameter_632, parameter_633, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x14x14xf32) <- (-1x64x14x14xf32)
        relu__117 = paddle._C_ops.relu_(batch_norm__720)

        # pd_op.conv2d: (-1x784x14x14xf32) <- (-1x64x14x14xf32, 784x64x1x1xf32)
        conv2d_121 = paddle._C_ops.conv2d(relu__117, parameter_634, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_176 = [1, 784, 1, 1]

        # pd_op.reshape: (1x784x1x1xf32, 0x784xf32) <- (784xf32, 4xi64)
        reshape_58, reshape_59 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_635, full_int_array_176), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x784x14x14xf32) <- (-1x784x14x14xf32, 1x784x1x1xf32)
        add__57 = paddle._C_ops.add_(conv2d_121, reshape_58)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_177 = [-1, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x49x14x14xf32, 0x-1x784x14x14xf32) <- (-1x784x14x14xf32, 5xi64)
        reshape__174, reshape__175 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__57, full_int_array_177), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_178 = [2]

        # pd_op.unsqueeze_: (-1x16x1x49x14x14xf32, None) <- (-1x16x49x14x14xf32, 1xi64)
        unsqueeze__58, unsqueeze__59 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(reshape__174, full_int_array_178), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unfold: (-1x12544x196xf32) <- (-1x256x14x14xf32)
        unfold_29 = paddle._C_ops.unfold(relu__116, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_179 = [-1, 16, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x16x49x14x14xf32, 0x-1x12544x196xf32) <- (-1x12544x196xf32, 6xi64)
        reshape__176, reshape__177 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_29, full_int_array_179), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x16x16x49x14x14xf32) <- (-1x16x1x49x14x14xf32, -1x16x16x49x14x14xf32)
        multiply_29 = paddle._C_ops.multiply(unsqueeze__58, reshape__176)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_180 = [3]

        # pd_op.sum: (-1x16x16x14x14xf32) <- (-1x16x16x49x14x14xf32, 1xi64)
        sum_29 = paddle._C_ops.sum(multiply_29, full_int_array_180, None, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_181 = [-1, 256, 14, 14]

        # pd_op.reshape_: (-1x256x14x14xf32, 0x-1x16x16x14x14xf32) <- (-1x16x16x14x14xf32, 4xi64)
        reshape__178, reshape__179 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sum_29, full_int_array_181), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__726, batch_norm__727, batch_norm__728, batch_norm__729, batch_norm__730, batch_norm__731 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__178, parameter_636, parameter_637, parameter_638, parameter_639, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__118 = paddle._C_ops.relu_(batch_norm__726)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        conv2d_122 = paddle._C_ops.conv2d(relu__118, parameter_640, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__732, batch_norm__733, batch_norm__734, batch_norm__735, batch_norm__736, batch_norm__737 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_122, parameter_641, parameter_642, parameter_643, parameter_644, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__58 = paddle._C_ops.add_(batch_norm__732, relu__115)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__119 = paddle._C_ops.relu_(add__58)

        # pd_op.conv2d: (-1x256x14x14xf32) <- (-1x1024x14x14xf32, 256x1024x1x1xf32)
        conv2d_123 = paddle._C_ops.conv2d(relu__119, parameter_645, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__738, batch_norm__739, batch_norm__740, batch_norm__741, batch_norm__742, batch_norm__743 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_123, parameter_646, parameter_647, parameter_648, parameter_649, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__120 = paddle._C_ops.relu_(batch_norm__738)

        # pd_op.conv2d: (-1x64x14x14xf32) <- (-1x256x14x14xf32, 64x256x1x1xf32)
        conv2d_124 = paddle._C_ops.conv2d(relu__120, parameter_650, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x14x14xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x14x14xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__744, batch_norm__745, batch_norm__746, batch_norm__747, batch_norm__748, batch_norm__749 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_124, parameter_651, parameter_652, parameter_653, parameter_654, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x64x14x14xf32) <- (-1x64x14x14xf32)
        relu__121 = paddle._C_ops.relu_(batch_norm__744)

        # pd_op.conv2d: (-1x784x14x14xf32) <- (-1x64x14x14xf32, 784x64x1x1xf32)
        conv2d_125 = paddle._C_ops.conv2d(relu__121, parameter_655, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_182 = [1, 784, 1, 1]

        # pd_op.reshape: (1x784x1x1xf32, 0x784xf32) <- (784xf32, 4xi64)
        reshape_60, reshape_61 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_656, full_int_array_182), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x784x14x14xf32) <- (-1x784x14x14xf32, 1x784x1x1xf32)
        add__59 = paddle._C_ops.add_(conv2d_125, reshape_60)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_183 = [-1, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x49x14x14xf32, 0x-1x784x14x14xf32) <- (-1x784x14x14xf32, 5xi64)
        reshape__180, reshape__181 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__59, full_int_array_183), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_184 = [2]

        # pd_op.unsqueeze_: (-1x16x1x49x14x14xf32, None) <- (-1x16x49x14x14xf32, 1xi64)
        unsqueeze__60, unsqueeze__61 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(reshape__180, full_int_array_184), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unfold: (-1x12544x196xf32) <- (-1x256x14x14xf32)
        unfold_30 = paddle._C_ops.unfold(relu__120, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_185 = [-1, 16, 16, 49, 14, 14]

        # pd_op.reshape_: (-1x16x16x49x14x14xf32, 0x-1x12544x196xf32) <- (-1x12544x196xf32, 6xi64)
        reshape__182, reshape__183 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_30, full_int_array_185), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x16x16x49x14x14xf32) <- (-1x16x1x49x14x14xf32, -1x16x16x49x14x14xf32)
        multiply_30 = paddle._C_ops.multiply(unsqueeze__60, reshape__182)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_186 = [3]

        # pd_op.sum: (-1x16x16x14x14xf32) <- (-1x16x16x49x14x14xf32, 1xi64)
        sum_30 = paddle._C_ops.sum(multiply_30, full_int_array_186, None, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_187 = [-1, 256, 14, 14]

        # pd_op.reshape_: (-1x256x14x14xf32, 0x-1x16x16x14x14xf32) <- (-1x16x16x14x14xf32, 4xi64)
        reshape__184, reshape__185 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sum_30, full_int_array_187), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x256x14x14xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x14x14xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__750, batch_norm__751, batch_norm__752, batch_norm__753, batch_norm__754, batch_norm__755 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__184, parameter_657, parameter_658, parameter_659, parameter_660, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x14x14xf32) <- (-1x256x14x14xf32)
        relu__122 = paddle._C_ops.relu_(batch_norm__750)

        # pd_op.conv2d: (-1x1024x14x14xf32) <- (-1x256x14x14xf32, 1024x256x1x1xf32)
        conv2d_126 = paddle._C_ops.conv2d(relu__122, parameter_661, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf32, 1024xf32, 1024xf32, xf32, xf32, None) <- (-1x1024x14x14xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__756, batch_norm__757, batch_norm__758, batch_norm__759, batch_norm__760, batch_norm__761 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_126, parameter_662, parameter_663, parameter_664, parameter_665, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32, -1x1024x14x14xf32)
        add__60 = paddle._C_ops.add_(batch_norm__756, relu__119)

        # pd_op.relu_: (-1x1024x14x14xf32) <- (-1x1024x14x14xf32)
        relu__123 = paddle._C_ops.relu_(add__60)

        # pd_op.conv2d: (-1x512x14x14xf32) <- (-1x1024x14x14xf32, 512x1024x1x1xf32)
        conv2d_127 = paddle._C_ops.conv2d(relu__123, parameter_666, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x14x14xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__762, batch_norm__763, batch_norm__764, batch_norm__765, batch_norm__766, batch_norm__767 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_127, parameter_667, parameter_668, parameter_669, parameter_670, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf32) <- (-1x512x14x14xf32)
        relu__124 = paddle._C_ops.relu_(batch_norm__762)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_188 = [2, 2]

        # pd_op.pool2d: (-1x512x7x7xf32) <- (-1x512x14x14xf32, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(relu__124, full_int_array_188, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x7x7xf32) <- (-1x512x7x7xf32, 128x512x1x1xf32)
        conv2d_128 = paddle._C_ops.conv2d(pool2d_3, parameter_671, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x7x7xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__768, batch_norm__769, batch_norm__770, batch_norm__771, batch_norm__772, batch_norm__773 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_128, parameter_672, parameter_673, parameter_674, parameter_675, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x7x7xf32) <- (-1x128x7x7xf32)
        relu__125 = paddle._C_ops.relu_(batch_norm__768)

        # pd_op.conv2d: (-1x1568x7x7xf32) <- (-1x128x7x7xf32, 1568x128x1x1xf32)
        conv2d_129 = paddle._C_ops.conv2d(relu__125, parameter_676, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_189 = [1, 1568, 1, 1]

        # pd_op.reshape: (1x1568x1x1xf32, 0x1568xf32) <- (1568xf32, 4xi64)
        reshape_62, reshape_63 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_677, full_int_array_189), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1568x7x7xf32) <- (-1x1568x7x7xf32, 1x1568x1x1xf32)
        add__61 = paddle._C_ops.add_(conv2d_129, reshape_62)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_190 = [-1, 32, 49, 7, 7]

        # pd_op.reshape_: (-1x32x49x7x7xf32, 0x-1x1568x7x7xf32) <- (-1x1568x7x7xf32, 5xi64)
        reshape__186, reshape__187 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__61, full_int_array_190), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_191 = [2]

        # pd_op.unsqueeze_: (-1x32x1x49x7x7xf32, None) <- (-1x32x49x7x7xf32, 1xi64)
        unsqueeze__62, unsqueeze__63 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(reshape__186, full_int_array_191), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unfold: (-1x25088x49xf32) <- (-1x512x14x14xf32)
        unfold_31 = paddle._C_ops.unfold(relu__124, [7, 7], [2, 2], [3, 3, 3, 3], [1, 1])

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_192 = [-1, 32, 16, 49, 7, 7]

        # pd_op.reshape_: (-1x32x16x49x7x7xf32, 0x-1x25088x49xf32) <- (-1x25088x49xf32, 6xi64)
        reshape__188, reshape__189 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_31, full_int_array_192), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x32x16x49x7x7xf32) <- (-1x32x1x49x7x7xf32, -1x32x16x49x7x7xf32)
        multiply_31 = paddle._C_ops.multiply(unsqueeze__62, reshape__188)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_193 = [3]

        # pd_op.sum: (-1x32x16x7x7xf32) <- (-1x32x16x49x7x7xf32, 1xi64)
        sum_31 = paddle._C_ops.sum(multiply_31, full_int_array_193, None, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_194 = [-1, 512, 7, 7]

        # pd_op.reshape_: (-1x512x7x7xf32, 0x-1x32x16x7x7xf32) <- (-1x32x16x7x7xf32, 4xi64)
        reshape__190, reshape__191 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sum_31, full_int_array_194), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__774, batch_norm__775, batch_norm__776, batch_norm__777, batch_norm__778, batch_norm__779 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__190, parameter_678, parameter_679, parameter_680, parameter_681, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        relu__126 = paddle._C_ops.relu_(batch_norm__774)

        # pd_op.conv2d: (-1x2048x7x7xf32) <- (-1x512x7x7xf32, 2048x512x1x1xf32)
        conv2d_130 = paddle._C_ops.conv2d(relu__126, parameter_682, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x2048x7x7xf32, 2048xf32, 2048xf32, xf32, xf32, None) <- (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        batch_norm__780, batch_norm__781, batch_norm__782, batch_norm__783, batch_norm__784, batch_norm__785 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_130, parameter_683, parameter_684, parameter_685, parameter_686, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x2048x7x7xf32) <- (-1x1024x14x14xf32, 2048x1024x1x1xf32)
        conv2d_131 = paddle._C_ops.conv2d(relu__123, parameter_687, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x2048x7x7xf32, 2048xf32, 2048xf32, xf32, xf32, None) <- (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        batch_norm__786, batch_norm__787, batch_norm__788, batch_norm__789, batch_norm__790, batch_norm__791 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_131, parameter_688, parameter_689, parameter_690, parameter_691, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32, -1x2048x7x7xf32)
        add__62 = paddle._C_ops.add_(batch_norm__780, batch_norm__786)

        # pd_op.relu_: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32)
        relu__127 = paddle._C_ops.relu_(add__62)

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x2048x7x7xf32, 512x2048x1x1xf32)
        conv2d_132 = paddle._C_ops.conv2d(relu__127, parameter_692, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__792, batch_norm__793, batch_norm__794, batch_norm__795, batch_norm__796, batch_norm__797 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_132, parameter_693, parameter_694, parameter_695, parameter_696, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        relu__128 = paddle._C_ops.relu_(batch_norm__792)

        # pd_op.conv2d: (-1x128x7x7xf32) <- (-1x512x7x7xf32, 128x512x1x1xf32)
        conv2d_133 = paddle._C_ops.conv2d(relu__128, parameter_697, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x7x7xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__798, batch_norm__799, batch_norm__800, batch_norm__801, batch_norm__802, batch_norm__803 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_133, parameter_698, parameter_699, parameter_700, parameter_701, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x7x7xf32) <- (-1x128x7x7xf32)
        relu__129 = paddle._C_ops.relu_(batch_norm__798)

        # pd_op.conv2d: (-1x1568x7x7xf32) <- (-1x128x7x7xf32, 1568x128x1x1xf32)
        conv2d_134 = paddle._C_ops.conv2d(relu__129, parameter_702, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_195 = [1, 1568, 1, 1]

        # pd_op.reshape: (1x1568x1x1xf32, 0x1568xf32) <- (1568xf32, 4xi64)
        reshape_64, reshape_65 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_703, full_int_array_195), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1568x7x7xf32) <- (-1x1568x7x7xf32, 1x1568x1x1xf32)
        add__63 = paddle._C_ops.add_(conv2d_134, reshape_64)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_196 = [-1, 32, 49, 7, 7]

        # pd_op.reshape_: (-1x32x49x7x7xf32, 0x-1x1568x7x7xf32) <- (-1x1568x7x7xf32, 5xi64)
        reshape__192, reshape__193 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__63, full_int_array_196), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_197 = [2]

        # pd_op.unsqueeze_: (-1x32x1x49x7x7xf32, None) <- (-1x32x49x7x7xf32, 1xi64)
        unsqueeze__64, unsqueeze__65 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(reshape__192, full_int_array_197), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unfold: (-1x25088x49xf32) <- (-1x512x7x7xf32)
        unfold_32 = paddle._C_ops.unfold(relu__128, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_198 = [-1, 32, 16, 49, 7, 7]

        # pd_op.reshape_: (-1x32x16x49x7x7xf32, 0x-1x25088x49xf32) <- (-1x25088x49xf32, 6xi64)
        reshape__194, reshape__195 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_32, full_int_array_198), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x32x16x49x7x7xf32) <- (-1x32x1x49x7x7xf32, -1x32x16x49x7x7xf32)
        multiply_32 = paddle._C_ops.multiply(unsqueeze__64, reshape__194)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_199 = [3]

        # pd_op.sum: (-1x32x16x7x7xf32) <- (-1x32x16x49x7x7xf32, 1xi64)
        sum_32 = paddle._C_ops.sum(multiply_32, full_int_array_199, None, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_200 = [-1, 512, 7, 7]

        # pd_op.reshape_: (-1x512x7x7xf32, 0x-1x32x16x7x7xf32) <- (-1x32x16x7x7xf32, 4xi64)
        reshape__196, reshape__197 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sum_32, full_int_array_200), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__804, batch_norm__805, batch_norm__806, batch_norm__807, batch_norm__808, batch_norm__809 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__196, parameter_704, parameter_705, parameter_706, parameter_707, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        relu__130 = paddle._C_ops.relu_(batch_norm__804)

        # pd_op.conv2d: (-1x2048x7x7xf32) <- (-1x512x7x7xf32, 2048x512x1x1xf32)
        conv2d_135 = paddle._C_ops.conv2d(relu__130, parameter_708, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x2048x7x7xf32, 2048xf32, 2048xf32, xf32, xf32, None) <- (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        batch_norm__810, batch_norm__811, batch_norm__812, batch_norm__813, batch_norm__814, batch_norm__815 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_135, parameter_709, parameter_710, parameter_711, parameter_712, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32, -1x2048x7x7xf32)
        add__64 = paddle._C_ops.add_(batch_norm__810, relu__127)

        # pd_op.relu_: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32)
        relu__131 = paddle._C_ops.relu_(add__64)

        # pd_op.conv2d: (-1x512x7x7xf32) <- (-1x2048x7x7xf32, 512x2048x1x1xf32)
        conv2d_136 = paddle._C_ops.conv2d(relu__131, parameter_713, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__816, batch_norm__817, batch_norm__818, batch_norm__819, batch_norm__820, batch_norm__821 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_136, parameter_714, parameter_715, parameter_716, parameter_717, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        relu__132 = paddle._C_ops.relu_(batch_norm__816)

        # pd_op.conv2d: (-1x128x7x7xf32) <- (-1x512x7x7xf32, 128x512x1x1xf32)
        conv2d_137 = paddle._C_ops.conv2d(relu__132, parameter_718, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x7x7xf32, 128xf32, 128xf32, xf32, xf32, None) <- (-1x128x7x7xf32, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__822, batch_norm__823, batch_norm__824, batch_norm__825, batch_norm__826, batch_norm__827 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_137, parameter_719, parameter_720, parameter_721, parameter_722, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x7x7xf32) <- (-1x128x7x7xf32)
        relu__133 = paddle._C_ops.relu_(batch_norm__822)

        # pd_op.conv2d: (-1x1568x7x7xf32) <- (-1x128x7x7xf32, 1568x128x1x1xf32)
        conv2d_138 = paddle._C_ops.conv2d(relu__133, parameter_723, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_201 = [1, 1568, 1, 1]

        # pd_op.reshape: (1x1568x1x1xf32, 0x1568xf32) <- (1568xf32, 4xi64)
        reshape_66, reshape_67 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_724, full_int_array_201), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x1568x7x7xf32) <- (-1x1568x7x7xf32, 1x1568x1x1xf32)
        add__65 = paddle._C_ops.add_(conv2d_138, reshape_66)

        # pd_op.full_int_array: (5xi64) <- ()
        full_int_array_202 = [-1, 32, 49, 7, 7]

        # pd_op.reshape_: (-1x32x49x7x7xf32, 0x-1x1568x7x7xf32) <- (-1x1568x7x7xf32, 5xi64)
        reshape__198, reshape__199 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__65, full_int_array_202), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_203 = [2]

        # pd_op.unsqueeze_: (-1x32x1x49x7x7xf32, None) <- (-1x32x49x7x7xf32, 1xi64)
        unsqueeze__66, unsqueeze__67 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze_(reshape__198, full_int_array_203), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.unfold: (-1x25088x49xf32) <- (-1x512x7x7xf32)
        unfold_33 = paddle._C_ops.unfold(relu__132, [7, 7], [1, 1], [3, 3, 3, 3], [1, 1])

        # pd_op.full_int_array: (6xi64) <- ()
        full_int_array_204 = [-1, 32, 16, 49, 7, 7]

        # pd_op.reshape_: (-1x32x16x49x7x7xf32, 0x-1x25088x49xf32) <- (-1x25088x49xf32, 6xi64)
        reshape__200, reshape__201 = (lambda x, f: f(x))(paddle._C_ops.reshape_(unfold_33, full_int_array_204), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.multiply: (-1x32x16x49x7x7xf32) <- (-1x32x1x49x7x7xf32, -1x32x16x49x7x7xf32)
        multiply_33 = paddle._C_ops.multiply(unsqueeze__66, reshape__200)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_205 = [3]

        # pd_op.sum: (-1x32x16x7x7xf32) <- (-1x32x16x49x7x7xf32, 1xi64)
        sum_33 = paddle._C_ops.sum(multiply_33, full_int_array_205, None, False)

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_206 = [-1, 512, 7, 7]

        # pd_op.reshape_: (-1x512x7x7xf32, 0x-1x32x16x7x7xf32) <- (-1x32x16x7x7xf32, 4xi64)
        reshape__202, reshape__203 = (lambda x, f: f(x))(paddle._C_ops.reshape_(sum_33, full_int_array_206), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.batch_norm_: (-1x512x7x7xf32, 512xf32, 512xf32, xf32, xf32, None) <- (-1x512x7x7xf32, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__828, batch_norm__829, batch_norm__830, batch_norm__831, batch_norm__832, batch_norm__833 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(reshape__202, parameter_725, parameter_726, parameter_727, parameter_728, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x7x7xf32) <- (-1x512x7x7xf32)
        relu__134 = paddle._C_ops.relu_(batch_norm__828)

        # pd_op.conv2d: (-1x2048x7x7xf32) <- (-1x512x7x7xf32, 2048x512x1x1xf32)
        conv2d_139 = paddle._C_ops.conv2d(relu__134, parameter_729, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x2048x7x7xf32, 2048xf32, 2048xf32, xf32, xf32, None) <- (-1x2048x7x7xf32, 2048xf32, 2048xf32, 2048xf32, 2048xf32)
        batch_norm__834, batch_norm__835, batch_norm__836, batch_norm__837, batch_norm__838, batch_norm__839 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_139, parameter_730, parameter_731, parameter_732, parameter_733, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32, -1x2048x7x7xf32)
        add__66 = paddle._C_ops.add_(batch_norm__834, relu__131)

        # pd_op.relu_: (-1x2048x7x7xf32) <- (-1x2048x7x7xf32)
        relu__135 = paddle._C_ops.relu_(add__66)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_207 = [1, 1]

        # pd_op.pool2d: (-1x2048x1x1xf32) <- (-1x2048x7x7xf32, 2xi64)
        pool2d_4 = paddle._C_ops.pool2d(relu__135, full_int_array_207, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.flatten_: (-1x2048xf32, None) <- (-1x2048x1x1xf32)
        flatten__0, flatten__1 = (lambda x, f: f(x))(paddle._C_ops.flatten_(pool2d_4, 1, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x1000xf32) <- (-1x2048xf32, 2048x1000xf32)
        matmul_0 = paddle._C_ops.matmul(flatten__0, parameter_734, False, False)

        # pd_op.add_: (-1x1000xf32) <- (-1x1000xf32, 1000xf32)
        add__67 = paddle._C_ops.add_(matmul_0, parameter_735)

        # pd_op.softmax_: (-1x1000xf32) <- (-1x1000xf32)
        softmax__0 = paddle._C_ops.softmax_(add__67, -1)
        return softmax__0



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

    def forward(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_11, parameter_15, parameter_12, parameter_14, parameter_13, parameter_16, parameter_20, parameter_17, parameter_19, parameter_18, parameter_21, parameter_25, parameter_22, parameter_24, parameter_23, parameter_26, parameter_30, parameter_27, parameter_29, parameter_28, parameter_31, parameter_32, parameter_36, parameter_33, parameter_35, parameter_34, parameter_37, parameter_41, parameter_38, parameter_40, parameter_39, parameter_42, parameter_46, parameter_43, parameter_45, parameter_44, parameter_47, parameter_51, parameter_48, parameter_50, parameter_49, parameter_52, parameter_56, parameter_53, parameter_55, parameter_54, parameter_57, parameter_58, parameter_62, parameter_59, parameter_61, parameter_60, parameter_63, parameter_67, parameter_64, parameter_66, parameter_65, parameter_68, parameter_72, parameter_69, parameter_71, parameter_70, parameter_73, parameter_77, parameter_74, parameter_76, parameter_75, parameter_78, parameter_79, parameter_83, parameter_80, parameter_82, parameter_81, parameter_84, parameter_88, parameter_85, parameter_87, parameter_86, parameter_89, parameter_93, parameter_90, parameter_92, parameter_91, parameter_94, parameter_98, parameter_95, parameter_97, parameter_96, parameter_99, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_114, parameter_111, parameter_113, parameter_112, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_126, parameter_130, parameter_127, parameter_129, parameter_128, parameter_131, parameter_135, parameter_132, parameter_134, parameter_133, parameter_136, parameter_140, parameter_137, parameter_139, parameter_138, parameter_141, parameter_145, parameter_142, parameter_144, parameter_143, parameter_146, parameter_147, parameter_151, parameter_148, parameter_150, parameter_149, parameter_152, parameter_156, parameter_153, parameter_155, parameter_154, parameter_157, parameter_161, parameter_158, parameter_160, parameter_159, parameter_162, parameter_166, parameter_163, parameter_165, parameter_164, parameter_167, parameter_168, parameter_172, parameter_169, parameter_171, parameter_170, parameter_173, parameter_177, parameter_174, parameter_176, parameter_175, parameter_178, parameter_182, parameter_179, parameter_181, parameter_180, parameter_183, parameter_187, parameter_184, parameter_186, parameter_185, parameter_188, parameter_189, parameter_193, parameter_190, parameter_192, parameter_191, parameter_194, parameter_198, parameter_195, parameter_197, parameter_196, parameter_199, parameter_203, parameter_200, parameter_202, parameter_201, parameter_204, parameter_208, parameter_205, parameter_207, parameter_206, parameter_209, parameter_213, parameter_210, parameter_212, parameter_211, parameter_214, parameter_215, parameter_219, parameter_216, parameter_218, parameter_217, parameter_220, parameter_224, parameter_221, parameter_223, parameter_222, parameter_225, parameter_229, parameter_226, parameter_228, parameter_227, parameter_230, parameter_234, parameter_231, parameter_233, parameter_232, parameter_235, parameter_236, parameter_240, parameter_237, parameter_239, parameter_238, parameter_241, parameter_245, parameter_242, parameter_244, parameter_243, parameter_246, parameter_250, parameter_247, parameter_249, parameter_248, parameter_251, parameter_255, parameter_252, parameter_254, parameter_253, parameter_256, parameter_257, parameter_261, parameter_258, parameter_260, parameter_259, parameter_262, parameter_266, parameter_263, parameter_265, parameter_264, parameter_267, parameter_271, parameter_268, parameter_270, parameter_269, parameter_272, parameter_276, parameter_273, parameter_275, parameter_274, parameter_277, parameter_278, parameter_282, parameter_279, parameter_281, parameter_280, parameter_283, parameter_287, parameter_284, parameter_286, parameter_285, parameter_288, parameter_292, parameter_289, parameter_291, parameter_290, parameter_293, parameter_297, parameter_294, parameter_296, parameter_295, parameter_298, parameter_299, parameter_303, parameter_300, parameter_302, parameter_301, parameter_304, parameter_308, parameter_305, parameter_307, parameter_306, parameter_309, parameter_313, parameter_310, parameter_312, parameter_311, parameter_314, parameter_318, parameter_315, parameter_317, parameter_316, parameter_319, parameter_320, parameter_324, parameter_321, parameter_323, parameter_322, parameter_325, parameter_329, parameter_326, parameter_328, parameter_327, parameter_330, parameter_334, parameter_331, parameter_333, parameter_332, parameter_335, parameter_339, parameter_336, parameter_338, parameter_337, parameter_340, parameter_341, parameter_345, parameter_342, parameter_344, parameter_343, parameter_346, parameter_350, parameter_347, parameter_349, parameter_348, parameter_351, parameter_355, parameter_352, parameter_354, parameter_353, parameter_356, parameter_360, parameter_357, parameter_359, parameter_358, parameter_361, parameter_362, parameter_366, parameter_363, parameter_365, parameter_364, parameter_367, parameter_371, parameter_368, parameter_370, parameter_369, parameter_372, parameter_376, parameter_373, parameter_375, parameter_374, parameter_377, parameter_381, parameter_378, parameter_380, parameter_379, parameter_382, parameter_383, parameter_387, parameter_384, parameter_386, parameter_385, parameter_388, parameter_392, parameter_389, parameter_391, parameter_390, parameter_393, parameter_397, parameter_394, parameter_396, parameter_395, parameter_398, parameter_402, parameter_399, parameter_401, parameter_400, parameter_403, parameter_404, parameter_408, parameter_405, parameter_407, parameter_406, parameter_409, parameter_413, parameter_410, parameter_412, parameter_411, parameter_414, parameter_418, parameter_415, parameter_417, parameter_416, parameter_419, parameter_423, parameter_420, parameter_422, parameter_421, parameter_424, parameter_425, parameter_429, parameter_426, parameter_428, parameter_427, parameter_430, parameter_434, parameter_431, parameter_433, parameter_432, parameter_435, parameter_439, parameter_436, parameter_438, parameter_437, parameter_440, parameter_444, parameter_441, parameter_443, parameter_442, parameter_445, parameter_446, parameter_450, parameter_447, parameter_449, parameter_448, parameter_451, parameter_455, parameter_452, parameter_454, parameter_453, parameter_456, parameter_460, parameter_457, parameter_459, parameter_458, parameter_461, parameter_465, parameter_462, parameter_464, parameter_463, parameter_466, parameter_467, parameter_471, parameter_468, parameter_470, parameter_469, parameter_472, parameter_476, parameter_473, parameter_475, parameter_474, parameter_477, parameter_481, parameter_478, parameter_480, parameter_479, parameter_482, parameter_486, parameter_483, parameter_485, parameter_484, parameter_487, parameter_488, parameter_492, parameter_489, parameter_491, parameter_490, parameter_493, parameter_497, parameter_494, parameter_496, parameter_495, parameter_498, parameter_502, parameter_499, parameter_501, parameter_500, parameter_503, parameter_507, parameter_504, parameter_506, parameter_505, parameter_508, parameter_509, parameter_513, parameter_510, parameter_512, parameter_511, parameter_514, parameter_518, parameter_515, parameter_517, parameter_516, parameter_519, parameter_523, parameter_520, parameter_522, parameter_521, parameter_524, parameter_528, parameter_525, parameter_527, parameter_526, parameter_529, parameter_530, parameter_534, parameter_531, parameter_533, parameter_532, parameter_535, parameter_539, parameter_536, parameter_538, parameter_537, parameter_540, parameter_544, parameter_541, parameter_543, parameter_542, parameter_545, parameter_549, parameter_546, parameter_548, parameter_547, parameter_550, parameter_551, parameter_555, parameter_552, parameter_554, parameter_553, parameter_556, parameter_560, parameter_557, parameter_559, parameter_558, parameter_561, parameter_565, parameter_562, parameter_564, parameter_563, parameter_566, parameter_570, parameter_567, parameter_569, parameter_568, parameter_571, parameter_572, parameter_576, parameter_573, parameter_575, parameter_574, parameter_577, parameter_581, parameter_578, parameter_580, parameter_579, parameter_582, parameter_586, parameter_583, parameter_585, parameter_584, parameter_587, parameter_591, parameter_588, parameter_590, parameter_589, parameter_592, parameter_593, parameter_597, parameter_594, parameter_596, parameter_595, parameter_598, parameter_602, parameter_599, parameter_601, parameter_600, parameter_603, parameter_607, parameter_604, parameter_606, parameter_605, parameter_608, parameter_612, parameter_609, parameter_611, parameter_610, parameter_613, parameter_614, parameter_618, parameter_615, parameter_617, parameter_616, parameter_619, parameter_623, parameter_620, parameter_622, parameter_621, parameter_624, parameter_628, parameter_625, parameter_627, parameter_626, parameter_629, parameter_633, parameter_630, parameter_632, parameter_631, parameter_634, parameter_635, parameter_639, parameter_636, parameter_638, parameter_637, parameter_640, parameter_644, parameter_641, parameter_643, parameter_642, parameter_645, parameter_649, parameter_646, parameter_648, parameter_647, parameter_650, parameter_654, parameter_651, parameter_653, parameter_652, parameter_655, parameter_656, parameter_660, parameter_657, parameter_659, parameter_658, parameter_661, parameter_665, parameter_662, parameter_664, parameter_663, parameter_666, parameter_670, parameter_667, parameter_669, parameter_668, parameter_671, parameter_675, parameter_672, parameter_674, parameter_673, parameter_676, parameter_677, parameter_681, parameter_678, parameter_680, parameter_679, parameter_682, parameter_686, parameter_683, parameter_685, parameter_684, parameter_687, parameter_691, parameter_688, parameter_690, parameter_689, parameter_692, parameter_696, parameter_693, parameter_695, parameter_694, parameter_697, parameter_701, parameter_698, parameter_700, parameter_699, parameter_702, parameter_703, parameter_707, parameter_704, parameter_706, parameter_705, parameter_708, parameter_712, parameter_709, parameter_711, parameter_710, parameter_713, parameter_717, parameter_714, parameter_716, parameter_715, parameter_718, parameter_722, parameter_719, parameter_721, parameter_720, parameter_723, parameter_724, parameter_728, parameter_725, parameter_727, parameter_726, parameter_729, parameter_733, parameter_730, parameter_732, parameter_731, parameter_734, parameter_735, feed_0):
        return self.builtin_module_1732_0_0(parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_11, parameter_15, parameter_12, parameter_14, parameter_13, parameter_16, parameter_20, parameter_17, parameter_19, parameter_18, parameter_21, parameter_25, parameter_22, parameter_24, parameter_23, parameter_26, parameter_30, parameter_27, parameter_29, parameter_28, parameter_31, parameter_32, parameter_36, parameter_33, parameter_35, parameter_34, parameter_37, parameter_41, parameter_38, parameter_40, parameter_39, parameter_42, parameter_46, parameter_43, parameter_45, parameter_44, parameter_47, parameter_51, parameter_48, parameter_50, parameter_49, parameter_52, parameter_56, parameter_53, parameter_55, parameter_54, parameter_57, parameter_58, parameter_62, parameter_59, parameter_61, parameter_60, parameter_63, parameter_67, parameter_64, parameter_66, parameter_65, parameter_68, parameter_72, parameter_69, parameter_71, parameter_70, parameter_73, parameter_77, parameter_74, parameter_76, parameter_75, parameter_78, parameter_79, parameter_83, parameter_80, parameter_82, parameter_81, parameter_84, parameter_88, parameter_85, parameter_87, parameter_86, parameter_89, parameter_93, parameter_90, parameter_92, parameter_91, parameter_94, parameter_98, parameter_95, parameter_97, parameter_96, parameter_99, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_114, parameter_111, parameter_113, parameter_112, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_126, parameter_130, parameter_127, parameter_129, parameter_128, parameter_131, parameter_135, parameter_132, parameter_134, parameter_133, parameter_136, parameter_140, parameter_137, parameter_139, parameter_138, parameter_141, parameter_145, parameter_142, parameter_144, parameter_143, parameter_146, parameter_147, parameter_151, parameter_148, parameter_150, parameter_149, parameter_152, parameter_156, parameter_153, parameter_155, parameter_154, parameter_157, parameter_161, parameter_158, parameter_160, parameter_159, parameter_162, parameter_166, parameter_163, parameter_165, parameter_164, parameter_167, parameter_168, parameter_172, parameter_169, parameter_171, parameter_170, parameter_173, parameter_177, parameter_174, parameter_176, parameter_175, parameter_178, parameter_182, parameter_179, parameter_181, parameter_180, parameter_183, parameter_187, parameter_184, parameter_186, parameter_185, parameter_188, parameter_189, parameter_193, parameter_190, parameter_192, parameter_191, parameter_194, parameter_198, parameter_195, parameter_197, parameter_196, parameter_199, parameter_203, parameter_200, parameter_202, parameter_201, parameter_204, parameter_208, parameter_205, parameter_207, parameter_206, parameter_209, parameter_213, parameter_210, parameter_212, parameter_211, parameter_214, parameter_215, parameter_219, parameter_216, parameter_218, parameter_217, parameter_220, parameter_224, parameter_221, parameter_223, parameter_222, parameter_225, parameter_229, parameter_226, parameter_228, parameter_227, parameter_230, parameter_234, parameter_231, parameter_233, parameter_232, parameter_235, parameter_236, parameter_240, parameter_237, parameter_239, parameter_238, parameter_241, parameter_245, parameter_242, parameter_244, parameter_243, parameter_246, parameter_250, parameter_247, parameter_249, parameter_248, parameter_251, parameter_255, parameter_252, parameter_254, parameter_253, parameter_256, parameter_257, parameter_261, parameter_258, parameter_260, parameter_259, parameter_262, parameter_266, parameter_263, parameter_265, parameter_264, parameter_267, parameter_271, parameter_268, parameter_270, parameter_269, parameter_272, parameter_276, parameter_273, parameter_275, parameter_274, parameter_277, parameter_278, parameter_282, parameter_279, parameter_281, parameter_280, parameter_283, parameter_287, parameter_284, parameter_286, parameter_285, parameter_288, parameter_292, parameter_289, parameter_291, parameter_290, parameter_293, parameter_297, parameter_294, parameter_296, parameter_295, parameter_298, parameter_299, parameter_303, parameter_300, parameter_302, parameter_301, parameter_304, parameter_308, parameter_305, parameter_307, parameter_306, parameter_309, parameter_313, parameter_310, parameter_312, parameter_311, parameter_314, parameter_318, parameter_315, parameter_317, parameter_316, parameter_319, parameter_320, parameter_324, parameter_321, parameter_323, parameter_322, parameter_325, parameter_329, parameter_326, parameter_328, parameter_327, parameter_330, parameter_334, parameter_331, parameter_333, parameter_332, parameter_335, parameter_339, parameter_336, parameter_338, parameter_337, parameter_340, parameter_341, parameter_345, parameter_342, parameter_344, parameter_343, parameter_346, parameter_350, parameter_347, parameter_349, parameter_348, parameter_351, parameter_355, parameter_352, parameter_354, parameter_353, parameter_356, parameter_360, parameter_357, parameter_359, parameter_358, parameter_361, parameter_362, parameter_366, parameter_363, parameter_365, parameter_364, parameter_367, parameter_371, parameter_368, parameter_370, parameter_369, parameter_372, parameter_376, parameter_373, parameter_375, parameter_374, parameter_377, parameter_381, parameter_378, parameter_380, parameter_379, parameter_382, parameter_383, parameter_387, parameter_384, parameter_386, parameter_385, parameter_388, parameter_392, parameter_389, parameter_391, parameter_390, parameter_393, parameter_397, parameter_394, parameter_396, parameter_395, parameter_398, parameter_402, parameter_399, parameter_401, parameter_400, parameter_403, parameter_404, parameter_408, parameter_405, parameter_407, parameter_406, parameter_409, parameter_413, parameter_410, parameter_412, parameter_411, parameter_414, parameter_418, parameter_415, parameter_417, parameter_416, parameter_419, parameter_423, parameter_420, parameter_422, parameter_421, parameter_424, parameter_425, parameter_429, parameter_426, parameter_428, parameter_427, parameter_430, parameter_434, parameter_431, parameter_433, parameter_432, parameter_435, parameter_439, parameter_436, parameter_438, parameter_437, parameter_440, parameter_444, parameter_441, parameter_443, parameter_442, parameter_445, parameter_446, parameter_450, parameter_447, parameter_449, parameter_448, parameter_451, parameter_455, parameter_452, parameter_454, parameter_453, parameter_456, parameter_460, parameter_457, parameter_459, parameter_458, parameter_461, parameter_465, parameter_462, parameter_464, parameter_463, parameter_466, parameter_467, parameter_471, parameter_468, parameter_470, parameter_469, parameter_472, parameter_476, parameter_473, parameter_475, parameter_474, parameter_477, parameter_481, parameter_478, parameter_480, parameter_479, parameter_482, parameter_486, parameter_483, parameter_485, parameter_484, parameter_487, parameter_488, parameter_492, parameter_489, parameter_491, parameter_490, parameter_493, parameter_497, parameter_494, parameter_496, parameter_495, parameter_498, parameter_502, parameter_499, parameter_501, parameter_500, parameter_503, parameter_507, parameter_504, parameter_506, parameter_505, parameter_508, parameter_509, parameter_513, parameter_510, parameter_512, parameter_511, parameter_514, parameter_518, parameter_515, parameter_517, parameter_516, parameter_519, parameter_523, parameter_520, parameter_522, parameter_521, parameter_524, parameter_528, parameter_525, parameter_527, parameter_526, parameter_529, parameter_530, parameter_534, parameter_531, parameter_533, parameter_532, parameter_535, parameter_539, parameter_536, parameter_538, parameter_537, parameter_540, parameter_544, parameter_541, parameter_543, parameter_542, parameter_545, parameter_549, parameter_546, parameter_548, parameter_547, parameter_550, parameter_551, parameter_555, parameter_552, parameter_554, parameter_553, parameter_556, parameter_560, parameter_557, parameter_559, parameter_558, parameter_561, parameter_565, parameter_562, parameter_564, parameter_563, parameter_566, parameter_570, parameter_567, parameter_569, parameter_568, parameter_571, parameter_572, parameter_576, parameter_573, parameter_575, parameter_574, parameter_577, parameter_581, parameter_578, parameter_580, parameter_579, parameter_582, parameter_586, parameter_583, parameter_585, parameter_584, parameter_587, parameter_591, parameter_588, parameter_590, parameter_589, parameter_592, parameter_593, parameter_597, parameter_594, parameter_596, parameter_595, parameter_598, parameter_602, parameter_599, parameter_601, parameter_600, parameter_603, parameter_607, parameter_604, parameter_606, parameter_605, parameter_608, parameter_612, parameter_609, parameter_611, parameter_610, parameter_613, parameter_614, parameter_618, parameter_615, parameter_617, parameter_616, parameter_619, parameter_623, parameter_620, parameter_622, parameter_621, parameter_624, parameter_628, parameter_625, parameter_627, parameter_626, parameter_629, parameter_633, parameter_630, parameter_632, parameter_631, parameter_634, parameter_635, parameter_639, parameter_636, parameter_638, parameter_637, parameter_640, parameter_644, parameter_641, parameter_643, parameter_642, parameter_645, parameter_649, parameter_646, parameter_648, parameter_647, parameter_650, parameter_654, parameter_651, parameter_653, parameter_652, parameter_655, parameter_656, parameter_660, parameter_657, parameter_659, parameter_658, parameter_661, parameter_665, parameter_662, parameter_664, parameter_663, parameter_666, parameter_670, parameter_667, parameter_669, parameter_668, parameter_671, parameter_675, parameter_672, parameter_674, parameter_673, parameter_676, parameter_677, parameter_681, parameter_678, parameter_680, parameter_679, parameter_682, parameter_686, parameter_683, parameter_685, parameter_684, parameter_687, parameter_691, parameter_688, parameter_690, parameter_689, parameter_692, parameter_696, parameter_693, parameter_695, parameter_694, parameter_697, parameter_701, parameter_698, parameter_700, parameter_699, parameter_702, parameter_703, parameter_707, parameter_704, parameter_706, parameter_705, parameter_708, parameter_712, parameter_709, parameter_711, parameter_710, parameter_713, parameter_717, parameter_714, parameter_716, parameter_715, parameter_718, parameter_722, parameter_719, parameter_721, parameter_720, parameter_723, parameter_724, parameter_728, parameter_725, parameter_727, parameter_726, parameter_729, parameter_733, parameter_730, parameter_732, parameter_731, parameter_734, parameter_735, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_1732_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([32, 3, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([8, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([8], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([8], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([8], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([8], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([18, 8, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([64, 32, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([64, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([196, 16, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([196], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([196, 16, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([196], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([16, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([196, 16, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([196], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([128, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([392, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([392], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([512, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([512, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([128, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
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
            paddle.uniform([392, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([392], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([512, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([128, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_139
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([392, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([392], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_149
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
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
            paddle.uniform([32, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([392, 32, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([392], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([512, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([784, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_189
            paddle.uniform([784], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([1024, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_209
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_213
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_214
            paddle.uniform([784, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_215
            paddle.uniform([784], dtype='float32', min=0, max=0.5),
            # parameter_219
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_218
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_217
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_224
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_221
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_223
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_225
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_229
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_227
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_234
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_233
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([784, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([784], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_239
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_238
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_241
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_245
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_244
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_243
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_246
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_250
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_249
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_248
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_251
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_255
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_252
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_254
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_253
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_256
            paddle.uniform([784, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_257
            paddle.uniform([784], dtype='float32', min=0, max=0.5),
            # parameter_261
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_258
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_259
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_262
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_266
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_263
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_265
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_264
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_267
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_271
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_268
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_270
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_269
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_272
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_276
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_273
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_275
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_274
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_277
            paddle.uniform([784, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_278
            paddle.uniform([784], dtype='float32', min=0, max=0.5),
            # parameter_282
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_279
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_281
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_280
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_283
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_287
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_284
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_286
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_285
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_288
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_292
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_289
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_291
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_290
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_293
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_297
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_294
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_296
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_295
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_298
            paddle.uniform([784, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_299
            paddle.uniform([784], dtype='float32', min=0, max=0.5),
            # parameter_303
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_300
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_302
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_301
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_304
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_308
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_305
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_307
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_306
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_309
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_313
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_310
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_312
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_311
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_314
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_318
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_315
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_317
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_316
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_319
            paddle.uniform([784, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_320
            paddle.uniform([784], dtype='float32', min=0, max=0.5),
            # parameter_324
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_321
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_323
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_322
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_325
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_329
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_326
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_328
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_327
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_330
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_334
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_331
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_333
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_332
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_335
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_339
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_336
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_338
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_337
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_340
            paddle.uniform([784, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_341
            paddle.uniform([784], dtype='float32', min=0, max=0.5),
            # parameter_345
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_342
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_344
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_343
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_346
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_350
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_347
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_349
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_348
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_351
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_355
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_352
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_354
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_353
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_356
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_360
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_357
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_359
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_358
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_361
            paddle.uniform([784, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_362
            paddle.uniform([784], dtype='float32', min=0, max=0.5),
            # parameter_366
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_363
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_365
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_364
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_367
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_371
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_368
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_370
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_369
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_372
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_376
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_373
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_375
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_374
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_377
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_381
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_378
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_380
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_379
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_382
            paddle.uniform([784, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_383
            paddle.uniform([784], dtype='float32', min=0, max=0.5),
            # parameter_387
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_384
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_386
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_385
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_388
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_392
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_389
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_391
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_390
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_393
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_397
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_394
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_396
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_395
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_398
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_402
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_399
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_401
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_400
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_403
            paddle.uniform([784, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_404
            paddle.uniform([784], dtype='float32', min=0, max=0.5),
            # parameter_408
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_405
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_407
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_406
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_409
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_413
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_410
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_412
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_411
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_414
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_418
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_415
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_417
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_416
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_419
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_423
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_420
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_422
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_421
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_424
            paddle.uniform([784, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_425
            paddle.uniform([784], dtype='float32', min=0, max=0.5),
            # parameter_429
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_426
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_428
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_427
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_430
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_434
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_431
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_433
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_432
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_435
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_439
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_436
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_438
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_437
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_440
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_444
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_441
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_443
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_442
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_445
            paddle.uniform([784, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_446
            paddle.uniform([784], dtype='float32', min=0, max=0.5),
            # parameter_450
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_447
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_449
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_448
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_451
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_455
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_452
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_454
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_453
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_456
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_460
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_457
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_459
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_458
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_461
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_465
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_462
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_464
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_463
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_466
            paddle.uniform([784, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_467
            paddle.uniform([784], dtype='float32', min=0, max=0.5),
            # parameter_471
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_468
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_470
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_469
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_472
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_476
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_473
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_475
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_474
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_477
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_481
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_478
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_480
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_479
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_482
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_486
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_483
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_485
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_484
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_487
            paddle.uniform([784, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_488
            paddle.uniform([784], dtype='float32', min=0, max=0.5),
            # parameter_492
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_489
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_491
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_490
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_493
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_497
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_494
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_496
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_495
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_498
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_502
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_499
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_501
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_500
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_503
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_507
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_504
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_506
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_505
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_508
            paddle.uniform([784, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_509
            paddle.uniform([784], dtype='float32', min=0, max=0.5),
            # parameter_513
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_510
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_512
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_511
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_514
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_518
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_515
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_517
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_516
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_519
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_523
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_520
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_522
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_521
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_524
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_528
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_525
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_527
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_526
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_529
            paddle.uniform([784, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_530
            paddle.uniform([784], dtype='float32', min=0, max=0.5),
            # parameter_534
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_531
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_533
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_532
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_535
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_539
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_536
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_538
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_537
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_540
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_544
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_541
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_543
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_542
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_545
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_549
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_546
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_548
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_547
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_550
            paddle.uniform([784, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_551
            paddle.uniform([784], dtype='float32', min=0, max=0.5),
            # parameter_555
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_552
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_554
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_553
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_556
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_560
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_557
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_559
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_558
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_561
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_565
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_562
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_564
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_563
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_566
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_570
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_567
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_569
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_568
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_571
            paddle.uniform([784, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_572
            paddle.uniform([784], dtype='float32', min=0, max=0.5),
            # parameter_576
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_573
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_575
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_574
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_577
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_581
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_578
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_580
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_579
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_582
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_586
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_583
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_585
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_584
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_587
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_591
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_588
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_590
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_589
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_592
            paddle.uniform([784, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_593
            paddle.uniform([784], dtype='float32', min=0, max=0.5),
            # parameter_597
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_594
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_596
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_595
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_598
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_602
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_599
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_601
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_600
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_603
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_607
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_604
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_606
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_605
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_608
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_612
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_609
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_611
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_610
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_613
            paddle.uniform([784, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_614
            paddle.uniform([784], dtype='float32', min=0, max=0.5),
            # parameter_618
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_615
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_617
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_616
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
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
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_633
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_630
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_632
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_631
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_634
            paddle.uniform([784, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_635
            paddle.uniform([784], dtype='float32', min=0, max=0.5),
            # parameter_639
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_636
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_638
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_637
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_640
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_644
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_641
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_643
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_642
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_645
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_649
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_646
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_648
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_647
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_650
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_654
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_651
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_653
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_652
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_655
            paddle.uniform([784, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_656
            paddle.uniform([784], dtype='float32', min=0, max=0.5),
            # parameter_660
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_657
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_659
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_658
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_661
            paddle.uniform([1024, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_665
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_662
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_664
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_663
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_666
            paddle.uniform([512, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_670
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_667
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_669
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_668
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_671
            paddle.uniform([128, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_675
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_672
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_674
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_673
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_676
            paddle.uniform([1568, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_677
            paddle.uniform([1568], dtype='float32', min=0, max=0.5),
            # parameter_681
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_678
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_680
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_679
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_682
            paddle.uniform([2048, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_686
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_683
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_685
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_684
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_687
            paddle.uniform([2048, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_691
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_688
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_690
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_689
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_692
            paddle.uniform([512, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_696
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_693
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_695
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_694
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_697
            paddle.uniform([128, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_701
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_698
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_700
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_699
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_702
            paddle.uniform([1568, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_703
            paddle.uniform([1568], dtype='float32', min=0, max=0.5),
            # parameter_707
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_704
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_706
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_705
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_708
            paddle.uniform([2048, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_712
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_709
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_711
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_710
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_713
            paddle.uniform([512, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_717
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_714
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_716
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_715
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_718
            paddle.uniform([128, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_722
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_719
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_721
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_720
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_723
            paddle.uniform([1568, 128, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_724
            paddle.uniform([1568], dtype='float32', min=0, max=0.5),
            # parameter_728
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_725
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_727
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_726
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_729
            paddle.uniform([2048, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_733
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_730
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_732
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_731
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            # parameter_734
            paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
            # parameter_735
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
            paddle.static.InputSpec(shape=[32, 3, 3, 3], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[8, 32, 1, 1], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[8], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[8], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[8], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[8], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[18, 8, 1, 1], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[64, 32, 3, 3], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[196, 16, 1, 1], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[196], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[196, 16, 1, 1], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[196], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[16, 64, 1, 1], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[196, 16, 1, 1], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[196], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[128, 256, 1, 1], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[392, 32, 1, 1], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[392], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[512, 256, 1, 1], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[128], dtype='float32'),
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
            paddle.static.InputSpec(shape=[392, 32, 1, 1], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[392], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_139
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[392, 32, 1, 1], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[392], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_149
            paddle.static.InputSpec(shape=[128], dtype='float32'),
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
            paddle.static.InputSpec(shape=[32, 128, 1, 1], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[392, 32, 1, 1], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[392], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[512, 128, 1, 1], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[256, 512, 1, 1], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[784, 64, 1, 1], dtype='float32'),
            # parameter_189
            paddle.static.InputSpec(shape=[784], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[1024, 512, 1, 1], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_209
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_213
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_214
            paddle.static.InputSpec(shape=[784, 64, 1, 1], dtype='float32'),
            # parameter_215
            paddle.static.InputSpec(shape=[784], dtype='float32'),
            # parameter_219
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_218
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_217
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_224
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_221
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_223
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_225
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_229
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_227
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_234
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_233
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[784, 64, 1, 1], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[784], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_239
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_238
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_241
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_245
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_244
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_243
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_246
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_250
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_249
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_248
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_251
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_255
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_252
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_254
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_253
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_256
            paddle.static.InputSpec(shape=[784, 64, 1, 1], dtype='float32'),
            # parameter_257
            paddle.static.InputSpec(shape=[784], dtype='float32'),
            # parameter_261
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_258
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_259
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_262
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_266
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_263
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_265
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_264
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_267
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_271
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_268
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_270
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_269
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_272
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_276
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_273
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_275
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_274
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_277
            paddle.static.InputSpec(shape=[784, 64, 1, 1], dtype='float32'),
            # parameter_278
            paddle.static.InputSpec(shape=[784], dtype='float32'),
            # parameter_282
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_279
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_281
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_280
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_283
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_287
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_284
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_286
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_285
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_288
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_292
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_289
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_291
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_290
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_293
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_297
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_294
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_296
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_295
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_298
            paddle.static.InputSpec(shape=[784, 64, 1, 1], dtype='float32'),
            # parameter_299
            paddle.static.InputSpec(shape=[784], dtype='float32'),
            # parameter_303
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_300
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_302
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_301
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_304
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_308
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_305
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_307
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_306
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_309
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_313
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_310
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_312
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_311
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_314
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_318
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_315
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_317
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_316
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_319
            paddle.static.InputSpec(shape=[784, 64, 1, 1], dtype='float32'),
            # parameter_320
            paddle.static.InputSpec(shape=[784], dtype='float32'),
            # parameter_324
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_321
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_323
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_322
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_325
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_329
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_326
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_328
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_327
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_330
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_334
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_331
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_333
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_332
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_335
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_339
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_336
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_338
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_337
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_340
            paddle.static.InputSpec(shape=[784, 64, 1, 1], dtype='float32'),
            # parameter_341
            paddle.static.InputSpec(shape=[784], dtype='float32'),
            # parameter_345
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_342
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_344
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_343
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_346
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_350
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_347
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_349
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_348
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_351
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_355
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_352
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_354
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_353
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_356
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_360
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_357
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_359
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_358
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_361
            paddle.static.InputSpec(shape=[784, 64, 1, 1], dtype='float32'),
            # parameter_362
            paddle.static.InputSpec(shape=[784], dtype='float32'),
            # parameter_366
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_363
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_365
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_364
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_367
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_371
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_368
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_370
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_369
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_372
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_376
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_373
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_375
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_374
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_377
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_381
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_378
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_380
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_379
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_382
            paddle.static.InputSpec(shape=[784, 64, 1, 1], dtype='float32'),
            # parameter_383
            paddle.static.InputSpec(shape=[784], dtype='float32'),
            # parameter_387
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_384
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_386
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_385
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_388
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_392
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_389
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_391
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_390
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_393
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_397
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_394
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_396
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_395
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_398
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_402
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_399
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_401
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_400
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_403
            paddle.static.InputSpec(shape=[784, 64, 1, 1], dtype='float32'),
            # parameter_404
            paddle.static.InputSpec(shape=[784], dtype='float32'),
            # parameter_408
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_405
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_407
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_406
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_409
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_413
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_410
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_412
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_411
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_414
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_418
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_415
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_417
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_416
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_419
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_423
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_420
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_422
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_421
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_424
            paddle.static.InputSpec(shape=[784, 64, 1, 1], dtype='float32'),
            # parameter_425
            paddle.static.InputSpec(shape=[784], dtype='float32'),
            # parameter_429
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_426
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_428
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_427
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_430
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_434
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_431
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_433
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_432
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_435
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_439
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_436
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_438
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_437
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_440
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_444
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_441
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_443
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_442
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_445
            paddle.static.InputSpec(shape=[784, 64, 1, 1], dtype='float32'),
            # parameter_446
            paddle.static.InputSpec(shape=[784], dtype='float32'),
            # parameter_450
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_447
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_449
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_448
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_451
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_455
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_452
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_454
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_453
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_456
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_460
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_457
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_459
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_458
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_461
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_465
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_462
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_464
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_463
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_466
            paddle.static.InputSpec(shape=[784, 64, 1, 1], dtype='float32'),
            # parameter_467
            paddle.static.InputSpec(shape=[784], dtype='float32'),
            # parameter_471
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_468
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_470
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_469
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_472
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_476
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_473
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_475
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_474
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_477
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_481
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_478
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_480
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_479
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_482
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_486
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_483
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_485
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_484
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_487
            paddle.static.InputSpec(shape=[784, 64, 1, 1], dtype='float32'),
            # parameter_488
            paddle.static.InputSpec(shape=[784], dtype='float32'),
            # parameter_492
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_489
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_491
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_490
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_493
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_497
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_494
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_496
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_495
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_498
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_502
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_499
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_501
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_500
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_503
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_507
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_504
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_506
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_505
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_508
            paddle.static.InputSpec(shape=[784, 64, 1, 1], dtype='float32'),
            # parameter_509
            paddle.static.InputSpec(shape=[784], dtype='float32'),
            # parameter_513
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_510
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_512
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_511
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_514
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_518
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_515
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_517
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_516
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_519
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_523
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_520
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_522
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_521
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_524
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_528
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_525
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_527
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_526
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_529
            paddle.static.InputSpec(shape=[784, 64, 1, 1], dtype='float32'),
            # parameter_530
            paddle.static.InputSpec(shape=[784], dtype='float32'),
            # parameter_534
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_531
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_533
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_532
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_535
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_539
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_536
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_538
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_537
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_540
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_544
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_541
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_543
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_542
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_545
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_549
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_546
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_548
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_547
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_550
            paddle.static.InputSpec(shape=[784, 64, 1, 1], dtype='float32'),
            # parameter_551
            paddle.static.InputSpec(shape=[784], dtype='float32'),
            # parameter_555
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_552
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_554
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_553
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_556
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_560
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_557
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_559
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_558
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_561
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_565
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_562
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_564
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_563
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_566
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_570
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_567
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_569
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_568
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_571
            paddle.static.InputSpec(shape=[784, 64, 1, 1], dtype='float32'),
            # parameter_572
            paddle.static.InputSpec(shape=[784], dtype='float32'),
            # parameter_576
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_573
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_575
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_574
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_577
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_581
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_578
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_580
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_579
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_582
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_586
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_583
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_585
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_584
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_587
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_591
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_588
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_590
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_589
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_592
            paddle.static.InputSpec(shape=[784, 64, 1, 1], dtype='float32'),
            # parameter_593
            paddle.static.InputSpec(shape=[784], dtype='float32'),
            # parameter_597
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_594
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_596
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_595
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_598
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_602
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_599
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_601
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_600
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_603
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_607
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_604
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_606
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_605
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_608
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_612
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_609
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_611
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_610
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_613
            paddle.static.InputSpec(shape=[784, 64, 1, 1], dtype='float32'),
            # parameter_614
            paddle.static.InputSpec(shape=[784], dtype='float32'),
            # parameter_618
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_615
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_617
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_616
            paddle.static.InputSpec(shape=[256], dtype='float32'),
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
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_633
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_630
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_632
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_631
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_634
            paddle.static.InputSpec(shape=[784, 64, 1, 1], dtype='float32'),
            # parameter_635
            paddle.static.InputSpec(shape=[784], dtype='float32'),
            # parameter_639
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_636
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_638
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_637
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_640
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_644
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_641
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_643
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_642
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_645
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_649
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_646
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_648
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_647
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_650
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_654
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_651
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_653
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_652
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_655
            paddle.static.InputSpec(shape=[784, 64, 1, 1], dtype='float32'),
            # parameter_656
            paddle.static.InputSpec(shape=[784], dtype='float32'),
            # parameter_660
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_657
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_659
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_658
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_661
            paddle.static.InputSpec(shape=[1024, 256, 1, 1], dtype='float32'),
            # parameter_665
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_662
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_664
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_663
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_666
            paddle.static.InputSpec(shape=[512, 1024, 1, 1], dtype='float32'),
            # parameter_670
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_667
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_669
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_668
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_671
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float32'),
            # parameter_675
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_672
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_674
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_673
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_676
            paddle.static.InputSpec(shape=[1568, 128, 1, 1], dtype='float32'),
            # parameter_677
            paddle.static.InputSpec(shape=[1568], dtype='float32'),
            # parameter_681
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_678
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_680
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_679
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_682
            paddle.static.InputSpec(shape=[2048, 512, 1, 1], dtype='float32'),
            # parameter_686
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_683
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_685
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_684
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_687
            paddle.static.InputSpec(shape=[2048, 1024, 1, 1], dtype='float32'),
            # parameter_691
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_688
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_690
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_689
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_692
            paddle.static.InputSpec(shape=[512, 2048, 1, 1], dtype='float32'),
            # parameter_696
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_693
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_695
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_694
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_697
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float32'),
            # parameter_701
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_698
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_700
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_699
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_702
            paddle.static.InputSpec(shape=[1568, 128, 1, 1], dtype='float32'),
            # parameter_703
            paddle.static.InputSpec(shape=[1568], dtype='float32'),
            # parameter_707
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_704
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_706
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_705
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_708
            paddle.static.InputSpec(shape=[2048, 512, 1, 1], dtype='float32'),
            # parameter_712
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_709
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_711
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_710
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_713
            paddle.static.InputSpec(shape=[512, 2048, 1, 1], dtype='float32'),
            # parameter_717
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_714
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_716
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_715
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_718
            paddle.static.InputSpec(shape=[128, 512, 1, 1], dtype='float32'),
            # parameter_722
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_719
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_721
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_720
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_723
            paddle.static.InputSpec(shape=[1568, 128, 1, 1], dtype='float32'),
            # parameter_724
            paddle.static.InputSpec(shape=[1568], dtype='float32'),
            # parameter_728
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_725
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_727
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_726
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_729
            paddle.static.InputSpec(shape=[2048, 512, 1, 1], dtype='float32'),
            # parameter_733
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_730
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_732
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_731
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            # parameter_734
            paddle.static.InputSpec(shape=[2048, 1000], dtype='float32'),
            # parameter_735
            paddle.static.InputSpec(shape=[1000], dtype='float32'),
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