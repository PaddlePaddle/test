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
    return [1931][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_3282_0_0(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_114, parameter_111, parameter_113, parameter_112, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_154, parameter_151, parameter_153, parameter_152, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_169, parameter_166, parameter_168, parameter_167, parameter_170, parameter_174, parameter_171, parameter_173, parameter_172, parameter_175, parameter_179, parameter_176, parameter_178, parameter_177, parameter_180, parameter_184, parameter_181, parameter_183, parameter_182, parameter_185, parameter_189, parameter_186, parameter_188, parameter_187, parameter_190, parameter_194, parameter_191, parameter_193, parameter_192, parameter_195, parameter_199, parameter_196, parameter_198, parameter_197, parameter_200, parameter_204, parameter_201, parameter_203, parameter_202, parameter_205, parameter_209, parameter_206, parameter_208, parameter_207, parameter_210, parameter_214, parameter_211, parameter_213, parameter_212, parameter_215, parameter_219, parameter_216, parameter_218, parameter_217, parameter_220, parameter_224, parameter_221, parameter_223, parameter_222, parameter_225, parameter_229, parameter_226, parameter_228, parameter_227, parameter_230, parameter_234, parameter_231, parameter_233, parameter_232, parameter_235, parameter_239, parameter_236, parameter_238, parameter_237, parameter_240, parameter_244, parameter_241, parameter_243, parameter_242, parameter_245, parameter_249, parameter_246, parameter_248, parameter_247, parameter_250, parameter_254, parameter_251, parameter_253, parameter_252, parameter_255, parameter_259, parameter_256, parameter_258, parameter_257, parameter_260, parameter_264, parameter_261, parameter_263, parameter_262, parameter_265, parameter_269, parameter_266, parameter_268, parameter_267, parameter_270, parameter_274, parameter_271, parameter_273, parameter_272, parameter_275, parameter_279, parameter_276, parameter_278, parameter_277, parameter_280, parameter_284, parameter_281, parameter_283, parameter_282, parameter_285, parameter_289, parameter_286, parameter_288, parameter_287, parameter_290, parameter_294, parameter_291, parameter_293, parameter_292, parameter_295, parameter_299, parameter_296, parameter_298, parameter_297, parameter_300, parameter_304, parameter_301, parameter_303, parameter_302, parameter_305, parameter_309, parameter_306, parameter_308, parameter_307, parameter_310, parameter_314, parameter_311, parameter_313, parameter_312, parameter_315, parameter_319, parameter_316, parameter_318, parameter_317, parameter_320, parameter_324, parameter_321, parameter_323, parameter_322, parameter_325, parameter_329, parameter_326, parameter_328, parameter_327, parameter_330, parameter_334, parameter_331, parameter_333, parameter_332, parameter_335, parameter_339, parameter_336, parameter_338, parameter_337, parameter_340, parameter_344, parameter_341, parameter_343, parameter_342, parameter_345, parameter_349, parameter_346, parameter_348, parameter_347, parameter_350, parameter_354, parameter_351, parameter_353, parameter_352, parameter_355, parameter_359, parameter_356, parameter_358, parameter_357, parameter_360, parameter_364, parameter_361, parameter_363, parameter_362, parameter_365, parameter_369, parameter_366, parameter_368, parameter_367, parameter_370, parameter_374, parameter_371, parameter_373, parameter_372, parameter_375, parameter_379, parameter_376, parameter_378, parameter_377, parameter_380, parameter_384, parameter_381, parameter_383, parameter_382, parameter_385, parameter_389, parameter_386, parameter_388, parameter_387, parameter_390, parameter_394, parameter_391, parameter_393, parameter_392, parameter_395, parameter_399, parameter_396, parameter_398, parameter_397, parameter_400, parameter_404, parameter_401, parameter_403, parameter_402, parameter_405, parameter_409, parameter_406, parameter_408, parameter_407, parameter_410, parameter_414, parameter_411, parameter_413, parameter_412, parameter_415, parameter_419, parameter_416, parameter_418, parameter_417, parameter_420, parameter_424, parameter_421, parameter_423, parameter_422, parameter_425, parameter_429, parameter_426, parameter_428, parameter_427, parameter_430, parameter_434, parameter_431, parameter_433, parameter_432, parameter_435, parameter_439, parameter_436, parameter_438, parameter_437, parameter_440, parameter_444, parameter_441, parameter_443, parameter_442, parameter_445, parameter_449, parameter_446, parameter_448, parameter_447, parameter_450, parameter_454, parameter_451, parameter_453, parameter_452, parameter_455, parameter_459, parameter_456, parameter_458, parameter_457, parameter_460, parameter_464, parameter_461, parameter_463, parameter_462, parameter_465, parameter_469, parameter_466, parameter_468, parameter_467, parameter_470, parameter_474, parameter_471, parameter_473, parameter_472, parameter_475, parameter_479, parameter_476, parameter_478, parameter_477, parameter_480, parameter_484, parameter_481, parameter_483, parameter_482, parameter_485, parameter_489, parameter_486, parameter_488, parameter_487, parameter_490, parameter_494, parameter_491, parameter_493, parameter_492, parameter_495, parameter_499, parameter_496, parameter_498, parameter_497, parameter_500, parameter_504, parameter_501, parameter_503, parameter_502, parameter_505, parameter_509, parameter_506, parameter_508, parameter_507, parameter_510, parameter_514, parameter_511, parameter_513, parameter_512, parameter_515, parameter_519, parameter_516, parameter_518, parameter_517, parameter_520, parameter_524, parameter_521, parameter_523, parameter_522, parameter_525, parameter_529, parameter_526, parameter_528, parameter_527, parameter_530, parameter_534, parameter_531, parameter_533, parameter_532, parameter_535, parameter_539, parameter_536, parameter_538, parameter_537, parameter_540, parameter_544, parameter_541, parameter_543, parameter_542, parameter_545, parameter_549, parameter_546, parameter_548, parameter_547, parameter_550, parameter_554, parameter_551, parameter_553, parameter_552, parameter_555, parameter_559, parameter_556, parameter_558, parameter_557, parameter_560, parameter_564, parameter_561, parameter_563, parameter_562, parameter_565, parameter_569, parameter_566, parameter_568, parameter_567, parameter_570, parameter_574, parameter_571, parameter_573, parameter_572, parameter_575, parameter_579, parameter_576, parameter_578, parameter_577, parameter_580, parameter_584, parameter_581, parameter_583, parameter_582, parameter_585, parameter_589, parameter_586, parameter_588, parameter_587, parameter_590, parameter_594, parameter_591, parameter_593, parameter_592, parameter_595, parameter_599, parameter_596, parameter_598, parameter_597, parameter_600, parameter_604, parameter_601, parameter_603, parameter_602, parameter_605, parameter_609, parameter_606, parameter_608, parameter_607, parameter_610, parameter_614, parameter_611, parameter_613, parameter_612, parameter_615, parameter_619, parameter_616, parameter_618, parameter_617, parameter_620, parameter_624, parameter_621, parameter_623, parameter_622, parameter_625, parameter_629, parameter_626, parameter_628, parameter_627, parameter_630, parameter_634, parameter_631, parameter_633, parameter_632, parameter_635, parameter_639, parameter_636, parameter_638, parameter_637, parameter_640, parameter_644, parameter_641, parameter_643, parameter_642, parameter_645, parameter_649, parameter_646, parameter_648, parameter_647, parameter_650, parameter_654, parameter_651, parameter_653, parameter_652, parameter_655, parameter_659, parameter_656, parameter_658, parameter_657, parameter_660, parameter_664, parameter_661, parameter_663, parameter_662, parameter_665, parameter_669, parameter_666, parameter_668, parameter_667, parameter_670, parameter_674, parameter_671, parameter_673, parameter_672, parameter_675, parameter_679, parameter_676, parameter_678, parameter_677, parameter_680, parameter_684, parameter_681, parameter_683, parameter_682, parameter_685, parameter_689, parameter_686, parameter_688, parameter_687, parameter_690, parameter_694, parameter_691, parameter_693, parameter_692, parameter_695, parameter_699, parameter_696, parameter_698, parameter_697, parameter_700, parameter_704, parameter_701, parameter_703, parameter_702, parameter_705, parameter_709, parameter_706, parameter_708, parameter_707, parameter_710, parameter_714, parameter_711, parameter_713, parameter_712, parameter_715, parameter_719, parameter_716, parameter_718, parameter_717, parameter_720, parameter_724, parameter_721, parameter_723, parameter_722, parameter_725, parameter_729, parameter_726, parameter_728, parameter_727, parameter_730, parameter_734, parameter_731, parameter_733, parameter_732, parameter_735, parameter_739, parameter_736, parameter_738, parameter_737, parameter_740, parameter_744, parameter_741, parameter_743, parameter_742, parameter_745, parameter_749, parameter_746, parameter_748, parameter_747, parameter_750, parameter_754, parameter_751, parameter_753, parameter_752, parameter_755, parameter_759, parameter_756, parameter_758, parameter_757, parameter_760, parameter_764, parameter_761, parameter_763, parameter_762, parameter_765, parameter_769, parameter_766, parameter_768, parameter_767, parameter_770, parameter_774, parameter_771, parameter_773, parameter_772, parameter_775, parameter_779, parameter_776, parameter_778, parameter_777, parameter_780, parameter_784, parameter_781, parameter_783, parameter_782, parameter_785, parameter_789, parameter_786, parameter_788, parameter_787, parameter_790, parameter_794, parameter_791, parameter_793, parameter_792, parameter_795, parameter_799, parameter_796, parameter_798, parameter_797, parameter_800, parameter_804, parameter_801, parameter_803, parameter_802, parameter_805, parameter_809, parameter_806, parameter_808, parameter_807, parameter_810, parameter_814, parameter_811, parameter_813, parameter_812, parameter_815, parameter_819, parameter_816, parameter_818, parameter_817, parameter_820, parameter_824, parameter_821, parameter_823, parameter_822, parameter_825, parameter_829, parameter_826, parameter_828, parameter_827, parameter_830, parameter_834, parameter_831, parameter_833, parameter_832, parameter_835, parameter_839, parameter_836, parameter_838, parameter_837, parameter_840, parameter_844, parameter_841, parameter_843, parameter_842, parameter_845, parameter_849, parameter_846, parameter_848, parameter_847, parameter_850, parameter_854, parameter_851, parameter_853, parameter_852, parameter_855, parameter_859, parameter_856, parameter_858, parameter_857, parameter_860, parameter_864, parameter_861, parameter_863, parameter_862, parameter_865, parameter_869, parameter_866, parameter_868, parameter_867, parameter_870, parameter_874, parameter_871, parameter_873, parameter_872, parameter_875, parameter_879, parameter_876, parameter_878, parameter_877, parameter_880, parameter_884, parameter_881, parameter_883, parameter_882, parameter_885, parameter_889, parameter_886, parameter_888, parameter_887, parameter_890, parameter_894, parameter_891, parameter_893, parameter_892, parameter_895, parameter_899, parameter_896, parameter_898, parameter_897, parameter_900, parameter_904, parameter_901, parameter_903, parameter_902, parameter_905, parameter_909, parameter_906, parameter_908, parameter_907, parameter_910, parameter_914, parameter_911, parameter_913, parameter_912, parameter_915, parameter_919, parameter_916, parameter_918, parameter_917, parameter_920, parameter_924, parameter_921, parameter_923, parameter_922, parameter_925, parameter_929, parameter_926, parameter_928, parameter_927, parameter_930, parameter_934, parameter_931, parameter_933, parameter_932, parameter_935, parameter_939, parameter_936, parameter_938, parameter_937, parameter_940, parameter_944, parameter_941, parameter_943, parameter_942, parameter_945, parameter_949, parameter_946, parameter_948, parameter_947, parameter_950, parameter_954, parameter_951, parameter_953, parameter_952, parameter_955, parameter_959, parameter_956, parameter_958, parameter_957, parameter_960, parameter_964, parameter_961, parameter_963, parameter_962, parameter_965, parameter_969, parameter_966, parameter_968, parameter_967, parameter_970, parameter_974, parameter_971, parameter_973, parameter_972, parameter_975, parameter_979, parameter_976, parameter_978, parameter_977, parameter_980, parameter_984, parameter_981, parameter_983, parameter_982, parameter_985, parameter_989, parameter_986, parameter_988, parameter_987, parameter_990, parameter_994, parameter_991, parameter_993, parameter_992, parameter_995, parameter_999, parameter_996, parameter_998, parameter_997, parameter_1000, parameter_1004, parameter_1001, parameter_1003, parameter_1002, parameter_1005, parameter_1009, parameter_1006, parameter_1008, parameter_1007, parameter_1010, parameter_1014, parameter_1011, parameter_1013, parameter_1012, parameter_1015, parameter_1019, parameter_1016, parameter_1018, parameter_1017, parameter_1020, parameter_1024, parameter_1021, parameter_1023, parameter_1022, parameter_1025, parameter_1029, parameter_1026, parameter_1028, parameter_1027, parameter_1030, parameter_1034, parameter_1031, parameter_1033, parameter_1032, parameter_1035, parameter_1039, parameter_1036, parameter_1038, parameter_1037, parameter_1040, parameter_1044, parameter_1041, parameter_1043, parameter_1042, parameter_1045, parameter_1049, parameter_1046, parameter_1048, parameter_1047, parameter_1050, parameter_1054, parameter_1051, parameter_1053, parameter_1052, parameter_1055, parameter_1059, parameter_1056, parameter_1058, parameter_1057, parameter_1060, parameter_1064, parameter_1061, parameter_1063, parameter_1062, parameter_1065, parameter_1069, parameter_1066, parameter_1068, parameter_1067, parameter_1070, parameter_1074, parameter_1071, parameter_1073, parameter_1072, parameter_1075, parameter_1079, parameter_1076, parameter_1078, parameter_1077, parameter_1080, parameter_1084, parameter_1081, parameter_1083, parameter_1082, parameter_1085, parameter_1089, parameter_1086, parameter_1088, parameter_1087, parameter_1090, parameter_1094, parameter_1091, parameter_1093, parameter_1092, parameter_1095, parameter_1099, parameter_1096, parameter_1098, parameter_1097, parameter_1100, parameter_1104, parameter_1101, parameter_1103, parameter_1102, parameter_1105, parameter_1109, parameter_1106, parameter_1108, parameter_1107, parameter_1110, parameter_1114, parameter_1111, parameter_1113, parameter_1112, parameter_1115, parameter_1119, parameter_1116, parameter_1118, parameter_1117, parameter_1120, parameter_1124, parameter_1121, parameter_1123, parameter_1122, parameter_1125, parameter_1129, parameter_1126, parameter_1128, parameter_1127, parameter_1130, parameter_1134, parameter_1131, parameter_1133, parameter_1132, parameter_1135, parameter_1139, parameter_1136, parameter_1138, parameter_1137, parameter_1140, parameter_1144, parameter_1141, parameter_1143, parameter_1142, parameter_1145, parameter_1149, parameter_1146, parameter_1148, parameter_1147, parameter_1150, parameter_1154, parameter_1151, parameter_1153, parameter_1152, parameter_1155, parameter_1159, parameter_1156, parameter_1158, parameter_1157, parameter_1160, parameter_1164, parameter_1161, parameter_1163, parameter_1162, parameter_1165, parameter_1169, parameter_1166, parameter_1168, parameter_1167, parameter_1170, parameter_1174, parameter_1171, parameter_1173, parameter_1172, parameter_1175, parameter_1179, parameter_1176, parameter_1178, parameter_1177, parameter_1180, parameter_1184, parameter_1181, parameter_1183, parameter_1182, parameter_1185, parameter_1189, parameter_1186, parameter_1188, parameter_1187, parameter_1190, parameter_1194, parameter_1191, parameter_1193, parameter_1192, parameter_1195, parameter_1199, parameter_1196, parameter_1198, parameter_1197, parameter_1200, parameter_1204, parameter_1201, parameter_1203, parameter_1202, parameter_1205, parameter_1209, parameter_1206, parameter_1208, parameter_1207, parameter_1210, parameter_1214, parameter_1211, parameter_1213, parameter_1212, parameter_1215, parameter_1219, parameter_1216, parameter_1218, parameter_1217, parameter_1220, parameter_1224, parameter_1221, parameter_1223, parameter_1222, parameter_1225, parameter_1229, parameter_1226, parameter_1228, parameter_1227, parameter_1230, parameter_1234, parameter_1231, parameter_1233, parameter_1232, parameter_1235, parameter_1239, parameter_1236, parameter_1238, parameter_1237, parameter_1240, parameter_1244, parameter_1241, parameter_1243, parameter_1242, parameter_1245, parameter_1249, parameter_1246, parameter_1248, parameter_1247, parameter_1250, parameter_1254, parameter_1251, parameter_1253, parameter_1252, parameter_1255, parameter_1259, parameter_1256, parameter_1258, parameter_1257, parameter_1260, parameter_1264, parameter_1261, parameter_1263, parameter_1262, parameter_1265, parameter_1269, parameter_1266, parameter_1268, parameter_1267, parameter_1270, parameter_1274, parameter_1271, parameter_1273, parameter_1272, parameter_1275, parameter_1279, parameter_1276, parameter_1278, parameter_1277, parameter_1280, parameter_1284, parameter_1281, parameter_1283, parameter_1282, parameter_1285, parameter_1289, parameter_1286, parameter_1288, parameter_1287, parameter_1290, parameter_1294, parameter_1291, parameter_1293, parameter_1292, parameter_1295, parameter_1299, parameter_1296, parameter_1298, parameter_1297, parameter_1300, parameter_1304, parameter_1301, parameter_1303, parameter_1302, parameter_1305, parameter_1309, parameter_1306, parameter_1308, parameter_1307, parameter_1310, parameter_1314, parameter_1311, parameter_1313, parameter_1312, parameter_1315, parameter_1319, parameter_1316, parameter_1318, parameter_1317, parameter_1320, parameter_1324, parameter_1321, parameter_1323, parameter_1322, parameter_1325, parameter_1329, parameter_1326, parameter_1328, parameter_1327, parameter_1330, parameter_1334, parameter_1331, parameter_1333, parameter_1332, parameter_1335, parameter_1339, parameter_1336, parameter_1338, parameter_1337, parameter_1340, parameter_1341, feed_0):

        # pd_op.cast: (-1x3x128x96xf16) <- (-1x3x128x96xf32)
        cast_0 = paddle._C_ops.cast(feed_0, paddle.float16)

        # pd_op.conv2d: (-1x32x64x48xf16) <- (-1x3x128x96xf16, 32x3x3x3xf16)
        conv2d_0 = paddle._C_ops.conv2d(cast_0, parameter_0, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x64x48xf16, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x64x48xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_0, parameter_1, parameter_2, parameter_3, parameter_4, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x32x64x48xf16) <- (-1x32x64x48xf16)
        relu__0 = paddle._C_ops.relu_(batch_norm__0)

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x16x64x48xf16, -1x16x64x48xf16]) <- (-1x32x64x48xf16, 1xi32)
        split_with_num_0 = paddle._C_ops.split_with_num(relu__0, 2, full_0)

        # builtin.slice: (-1x16x64x48xf16) <- ([-1x16x64x48xf16, -1x16x64x48xf16])
        slice_0 = split_with_num_0[0]

        # pd_op.depthwise_conv2d: (-1x16x32x24xf16) <- (-1x16x64x48xf16, 16x1x3x3xf16)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(slice_0, parameter_5, [2, 2], [1, 1], 'EXPLICIT', 16, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x16x32x24xf16, 16xf32, 16xf32, xf32, xf32, None) <- (-1x16x32x24xf16, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_0, parameter_6, parameter_7, parameter_8, parameter_9, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x16x32x24xf16) <- (-1x16x32x24xf16, 16x16x1x1xf16)
        conv2d_1 = paddle._C_ops.conv2d(batch_norm__6, parameter_10, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x16x32x24xf16, 16xf32, 16xf32, xf32, xf32, None) <- (-1x16x32x24xf16, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_1, parameter_11, parameter_12, parameter_13, parameter_14, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x16x32x24xf16) <- (-1x16x32x24xf16)
        relu__1 = paddle._C_ops.relu_(batch_norm__12)

        # builtin.slice: (-1x16x64x48xf16) <- ([-1x16x64x48xf16, -1x16x64x48xf16])
        slice_1 = split_with_num_0[1]

        # pd_op.conv2d: (-1x32x64x48xf16) <- (-1x16x64x48xf16, 32x16x1x1xf16)
        conv2d_2 = paddle._C_ops.conv2d(slice_1, parameter_15, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x64x48xf16, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x64x48xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_2, parameter_16, parameter_17, parameter_18, parameter_19, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x32x64x48xf16) <- (-1x32x64x48xf16)
        relu__2 = paddle._C_ops.relu_(batch_norm__18)

        # pd_op.depthwise_conv2d: (-1x32x32x24xf16) <- (-1x32x64x48xf16, 32x1x3x3xf16)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(relu__2, parameter_20, [2, 2], [1, 1], 'EXPLICIT', 32, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x32x32x24xf16, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x32x24xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_1, parameter_21, parameter_22, parameter_23, parameter_24, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x16x32x24xf16) <- (-1x32x32x24xf16, 16x32x1x1xf16)
        conv2d_3 = paddle._C_ops.conv2d(batch_norm__24, parameter_25, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x16x32x24xf16, 16xf32, 16xf32, xf32, xf32, None) <- (-1x16x32x24xf16, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_3, parameter_26, parameter_27, parameter_28, parameter_29, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x16x32x24xf16) <- (-1x16x32x24xf16)
        relu__3 = paddle._C_ops.relu_(batch_norm__30)

        # builtin.combine: ([-1x16x32x24xf16, -1x16x32x24xf16]) <- (-1x16x32x24xf16, -1x16x32x24xf16)
        combine_0 = [relu__1, relu__3]

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x32x32x24xf16) <- ([-1x16x32x24xf16, -1x16x32x24xf16], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_1)

        # pd_op.shape: (4xi32) <- (-1x32x32x24xf16)
        shape_0 = paddle._C_ops.shape(paddle.cast(concat_0, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(shape_0, [0], full_int_array_0, full_int_array_1, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_1 = [slice_2, full_2, full_3, full_4, full_5]

        # pd_op.reshape_: (-1x2x16x32x24xf16, 0x-1x32x32x24xf16) <- (-1x32x32x24xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_0, [x.reshape([]) for x in combine_1]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x16x2x32x24xf16) <- (-1x2x16x32x24xf16)
        transpose_0 = paddle._C_ops.transpose(reshape__0, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_8 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_2 = [slice_2, full_6, full_7, full_8]

        # pd_op.reshape_: (-1x32x32x24xf16, 0x-1x16x2x32x24xf16) <- (-1x16x2x32x24xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_0, [x.reshape([]) for x in combine_2]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x32x32x24xf16) <- (-1x32x32x24xf16, 32x1x3x3xf16)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(reshape__2, parameter_30, [1, 1], [1, 1], 'EXPLICIT', 32, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x32x32x24xf16, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x32x24xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_2, parameter_31, parameter_32, parameter_33, parameter_34, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x40x32x24xf16) <- (-1x32x32x24xf16, 40x32x1x1xf16)
        conv2d_4 = paddle._C_ops.conv2d(batch_norm__36, parameter_35, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x32x24xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x32x24xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_4, parameter_36, parameter_37, parameter_38, parameter_39, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x32x24xf16) <- (-1x40x32x24xf16)
        relu__4 = paddle._C_ops.relu_(batch_norm__42)

        # pd_op.depthwise_conv2d: (-1x32x16x12xf16) <- (-1x32x32x24xf16, 32x1x3x3xf16)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(reshape__2, parameter_40, [2, 2], [1, 1], 'EXPLICIT', 32, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x32x16x12xf16, 32xf32, 32xf32, xf32, xf32, None) <- (-1x32x16x12xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_3, parameter_41, parameter_42, parameter_43, parameter_44, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x80x16x12xf16) <- (-1x32x16x12xf16, 80x32x1x1xf16)
        conv2d_5 = paddle._C_ops.conv2d(batch_norm__48, parameter_45, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x16x12xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x16x12xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_5, parameter_46, parameter_47, parameter_48, parameter_49, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x80x16x12xf16) <- (-1x80x16x12xf16)
        relu__5 = paddle._C_ops.relu_(batch_norm__54)

        # pd_op.full: (1xi32) <- ()
        full_9 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x20x32x24xf16, -1x20x32x24xf16]) <- (-1x40x32x24xf16, 1xi32)
        split_with_num_1 = paddle._C_ops.split_with_num(relu__4, 2, full_9)

        # builtin.slice: (-1x20x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16])
        slice_3 = split_with_num_1[1]

        # pd_op.conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x20x1x1xf16)
        conv2d_6 = paddle._C_ops.conv2d(slice_3, parameter_50, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_6, parameter_51, parameter_52, parameter_53, parameter_54, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x20x32x24xf16) <- (-1x20x32x24xf16)
        relu__6 = paddle._C_ops.relu_(batch_norm__60)

        # pd_op.depthwise_conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x1x3x3xf16)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(relu__6, parameter_55, [1, 1], [1, 1], 'EXPLICIT', 20, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_4, parameter_56, parameter_57, parameter_58, parameter_59, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x20x1x1xf16)
        conv2d_7 = paddle._C_ops.conv2d(batch_norm__66, parameter_60, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_7, parameter_61, parameter_62, parameter_63, parameter_64, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x20x32x24xf16) <- (-1x20x32x24xf16)
        relu__7 = paddle._C_ops.relu_(batch_norm__72)

        # builtin.slice: (-1x20x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16])
        slice_4 = split_with_num_1[0]

        # builtin.combine: ([-1x20x32x24xf16, -1x20x32x24xf16]) <- (-1x20x32x24xf16, -1x20x32x24xf16)
        combine_3 = [slice_4, relu__7]

        # pd_op.full: (1xi32) <- ()
        full_10 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x40x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_3, full_10)

        # pd_op.shape: (4xi32) <- (-1x40x32x24xf16)
        shape_1 = paddle._C_ops.shape(paddle.cast(concat_1, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(shape_1, [0], full_int_array_2, full_int_array_3, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_11 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_12 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_13 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_14 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_4 = [slice_5, full_11, full_12, full_13, full_14]

        # pd_op.reshape_: (-1x2x20x32x24xf16, 0x-1x40x32x24xf16) <- (-1x40x32x24xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_1, [x.reshape([]) for x in combine_4]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x20x2x32x24xf16) <- (-1x2x20x32x24xf16)
        transpose_1 = paddle._C_ops.transpose(reshape__4, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_15 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_16 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_17 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_5 = [slice_5, full_15, full_16, full_17]

        # pd_op.reshape_: (-1x40x32x24xf16, 0x-1x20x2x32x24xf16) <- (-1x20x2x32x24xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_1, [x.reshape([]) for x in combine_5]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_18 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x20x32x24xf16, -1x20x32x24xf16]) <- (-1x40x32x24xf16, 1xi32)
        split_with_num_2 = paddle._C_ops.split_with_num(reshape__6, 2, full_18)

        # builtin.slice: (-1x20x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16])
        slice_6 = split_with_num_2[1]

        # pd_op.conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x20x1x1xf16)
        conv2d_8 = paddle._C_ops.conv2d(slice_6, parameter_65, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_8, parameter_66, parameter_67, parameter_68, parameter_69, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x20x32x24xf16) <- (-1x20x32x24xf16)
        relu__8 = paddle._C_ops.relu_(batch_norm__78)

        # pd_op.depthwise_conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x1x3x3xf16)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(relu__8, parameter_70, [1, 1], [1, 1], 'EXPLICIT', 20, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_5, parameter_71, parameter_72, parameter_73, parameter_74, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x20x1x1xf16)
        conv2d_9 = paddle._C_ops.conv2d(batch_norm__84, parameter_75, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_9, parameter_76, parameter_77, parameter_78, parameter_79, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x20x32x24xf16) <- (-1x20x32x24xf16)
        relu__9 = paddle._C_ops.relu_(batch_norm__90)

        # builtin.slice: (-1x20x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16])
        slice_7 = split_with_num_2[0]

        # builtin.combine: ([-1x20x32x24xf16, -1x20x32x24xf16]) <- (-1x20x32x24xf16, -1x20x32x24xf16)
        combine_6 = [slice_7, relu__9]

        # pd_op.full: (1xi32) <- ()
        full_19 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x40x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_6, full_19)

        # pd_op.shape: (4xi32) <- (-1x40x32x24xf16)
        shape_2 = paddle._C_ops.shape(paddle.cast(concat_2, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(shape_2, [0], full_int_array_4, full_int_array_5, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_20 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_21 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_22 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_23 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_7 = [slice_8, full_20, full_21, full_22, full_23]

        # pd_op.reshape_: (-1x2x20x32x24xf16, 0x-1x40x32x24xf16) <- (-1x40x32x24xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_2, [x.reshape([]) for x in combine_7]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x20x2x32x24xf16) <- (-1x2x20x32x24xf16)
        transpose_2 = paddle._C_ops.transpose(reshape__8, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_24 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_25 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_26 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_8 = [slice_8, full_24, full_25, full_26]

        # pd_op.reshape_: (-1x40x32x24xf16, 0x-1x20x2x32x24xf16) <- (-1x20x2x32x24xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_2, [x.reshape([]) for x in combine_8]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_27 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x40x16x12xf16, -1x40x16x12xf16]) <- (-1x80x16x12xf16, 1xi32)
        split_with_num_3 = paddle._C_ops.split_with_num(relu__5, 2, full_27)

        # builtin.slice: (-1x40x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16])
        slice_9 = split_with_num_3[1]

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_10 = paddle._C_ops.conv2d(slice_9, parameter_80, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_10, parameter_81, parameter_82, parameter_83, parameter_84, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__10 = paddle._C_ops.relu_(batch_norm__96)

        # pd_op.depthwise_conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x1x3x3xf16)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(relu__10, parameter_85, [1, 1], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_6, parameter_86, parameter_87, parameter_88, parameter_89, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_11 = paddle._C_ops.conv2d(batch_norm__102, parameter_90, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_11, parameter_91, parameter_92, parameter_93, parameter_94, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__11 = paddle._C_ops.relu_(batch_norm__108)

        # builtin.slice: (-1x40x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16])
        slice_10 = split_with_num_3[0]

        # builtin.combine: ([-1x40x16x12xf16, -1x40x16x12xf16]) <- (-1x40x16x12xf16, -1x40x16x12xf16)
        combine_9 = [slice_10, relu__11]

        # pd_op.full: (1xi32) <- ()
        full_28 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x80x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_9, full_28)

        # pd_op.shape: (4xi32) <- (-1x80x16x12xf16)
        shape_3 = paddle._C_ops.shape(paddle.cast(concat_3, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(shape_3, [0], full_int_array_6, full_int_array_7, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_29 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_30 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_31 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_32 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_10 = [slice_11, full_29, full_30, full_31, full_32]

        # pd_op.reshape_: (-1x2x40x16x12xf16, 0x-1x80x16x12xf16) <- (-1x80x16x12xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_3, [x.reshape([]) for x in combine_10]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x2x16x12xf16) <- (-1x2x40x16x12xf16)
        transpose_3 = paddle._C_ops.transpose(reshape__12, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_33 = paddle._C_ops.full([1], float('80'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_34 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_35 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_11 = [slice_11, full_33, full_34, full_35]

        # pd_op.reshape_: (-1x80x16x12xf16, 0x-1x40x2x16x12xf16) <- (-1x40x2x16x12xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_3, [x.reshape([]) for x in combine_11]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_36 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x40x16x12xf16, -1x40x16x12xf16]) <- (-1x80x16x12xf16, 1xi32)
        split_with_num_4 = paddle._C_ops.split_with_num(reshape__14, 2, full_36)

        # builtin.slice: (-1x40x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16])
        slice_12 = split_with_num_4[1]

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_12 = paddle._C_ops.conv2d(slice_12, parameter_95, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_12, parameter_96, parameter_97, parameter_98, parameter_99, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__12 = paddle._C_ops.relu_(batch_norm__114)

        # pd_op.depthwise_conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x1x3x3xf16)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(relu__12, parameter_100, [1, 1], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_7, parameter_101, parameter_102, parameter_103, parameter_104, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_13 = paddle._C_ops.conv2d(batch_norm__120, parameter_105, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_13, parameter_106, parameter_107, parameter_108, parameter_109, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__13 = paddle._C_ops.relu_(batch_norm__126)

        # builtin.slice: (-1x40x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16])
        slice_13 = split_with_num_4[0]

        # builtin.combine: ([-1x40x16x12xf16, -1x40x16x12xf16]) <- (-1x40x16x12xf16, -1x40x16x12xf16)
        combine_12 = [slice_13, relu__13]

        # pd_op.full: (1xi32) <- ()
        full_37 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x80x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_12, full_37)

        # pd_op.shape: (4xi32) <- (-1x80x16x12xf16)
        shape_4 = paddle._C_ops.shape(paddle.cast(concat_4, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(shape_4, [0], full_int_array_8, full_int_array_9, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_38 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_39 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_40 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_41 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_13 = [slice_14, full_38, full_39, full_40, full_41]

        # pd_op.reshape_: (-1x2x40x16x12xf16, 0x-1x80x16x12xf16) <- (-1x80x16x12xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__16, reshape__17 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_4, [x.reshape([]) for x in combine_13]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x2x16x12xf16) <- (-1x2x40x16x12xf16)
        transpose_4 = paddle._C_ops.transpose(reshape__16, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_42 = paddle._C_ops.full([1], float('80'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_43 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_44 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_14 = [slice_14, full_42, full_43, full_44]

        # pd_op.reshape_: (-1x80x16x12xf16, 0x-1x40x2x16x12xf16) <- (-1x40x2x16x12xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__18, reshape__19 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_4, [x.reshape([]) for x in combine_14]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x40x32x24xf16) <- (-1x40x32x24xf16, -1x40x32x24xf16)
        add__0 = paddle._C_ops.add_(reshape__10, reshape__10)

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x80x16x12xf16, 40x80x1x1xf16)
        conv2d_14 = paddle._C_ops.conv2d(reshape__18, parameter_110, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_14, parameter_111, parameter_112, parameter_113, parameter_114, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.nearest_interp: (-1x40x32x24xf16) <- (-1x40x16x12xf16, None, None, None)
        nearest_interp_0 = paddle._C_ops.nearest_interp(batch_norm__132, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 0)

        # pd_op.add_: (-1x40x32x24xf16) <- (-1x40x32x24xf16, -1x40x32x24xf16)
        add__1 = paddle._C_ops.add_(add__0, nearest_interp_0)

        # pd_op.relu: (-1x40x32x24xf16) <- (-1x40x32x24xf16)
        relu_0 = paddle._C_ops.relu(add__1)

        # pd_op.depthwise_conv2d: (-1x40x16x12xf16) <- (-1x40x32x24xf16, 40x1x3x3xf16)
        depthwise_conv2d_8 = paddle._C_ops.depthwise_conv2d(add__1, parameter_115, [2, 2], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_8, parameter_116, parameter_117, parameter_118, parameter_119, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x80x16x12xf16) <- (-1x40x16x12xf16, 80x40x1x1xf16)
        conv2d_15 = paddle._C_ops.conv2d(batch_norm__138, parameter_120, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x16x12xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x16x12xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_15, parameter_121, parameter_122, parameter_123, parameter_124, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x80x16x12xf16) <- (-1x80x16x12xf16, -1x80x16x12xf16)
        add__2 = paddle._C_ops.add_(batch_norm__144, batch_norm__144)

        # pd_op.add_: (-1x80x16x12xf16) <- (-1x80x16x12xf16, -1x80x16x12xf16)
        add__3 = paddle._C_ops.add_(add__2, reshape__18)

        # pd_op.relu_: (-1x80x16x12xf16) <- (-1x80x16x12xf16)
        relu__14 = paddle._C_ops.relu_(add__3)

        # pd_op.full: (1xi32) <- ()
        full_45 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x20x32x24xf16, -1x20x32x24xf16]) <- (-1x40x32x24xf16, 1xi32)
        split_with_num_5 = paddle._C_ops.split_with_num(relu_0, 2, full_45)

        # builtin.slice: (-1x20x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16])
        slice_15 = split_with_num_5[1]

        # pd_op.conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x20x1x1xf16)
        conv2d_16 = paddle._C_ops.conv2d(slice_15, parameter_125, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_16, parameter_126, parameter_127, parameter_128, parameter_129, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x20x32x24xf16) <- (-1x20x32x24xf16)
        relu__15 = paddle._C_ops.relu_(batch_norm__150)

        # pd_op.depthwise_conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x1x3x3xf16)
        depthwise_conv2d_9 = paddle._C_ops.depthwise_conv2d(relu__15, parameter_130, [1, 1], [1, 1], 'EXPLICIT', 20, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_9, parameter_131, parameter_132, parameter_133, parameter_134, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x20x1x1xf16)
        conv2d_17 = paddle._C_ops.conv2d(batch_norm__156, parameter_135, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_17, parameter_136, parameter_137, parameter_138, parameter_139, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x20x32x24xf16) <- (-1x20x32x24xf16)
        relu__16 = paddle._C_ops.relu_(batch_norm__162)

        # builtin.slice: (-1x20x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16])
        slice_16 = split_with_num_5[0]

        # builtin.combine: ([-1x20x32x24xf16, -1x20x32x24xf16]) <- (-1x20x32x24xf16, -1x20x32x24xf16)
        combine_15 = [slice_16, relu__16]

        # pd_op.full: (1xi32) <- ()
        full_46 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x40x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_15, full_46)

        # pd_op.shape: (4xi32) <- (-1x40x32x24xf16)
        shape_5 = paddle._C_ops.shape(paddle.cast(concat_5, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_10 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_11 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(shape_5, [0], full_int_array_10, full_int_array_11, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_47 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_48 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_49 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_50 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_16 = [slice_17, full_47, full_48, full_49, full_50]

        # pd_op.reshape_: (-1x2x20x32x24xf16, 0x-1x40x32x24xf16) <- (-1x40x32x24xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__20, reshape__21 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_5, [x.reshape([]) for x in combine_16]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x20x2x32x24xf16) <- (-1x2x20x32x24xf16)
        transpose_5 = paddle._C_ops.transpose(reshape__20, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_51 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_52 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_53 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_17 = [slice_17, full_51, full_52, full_53]

        # pd_op.reshape_: (-1x40x32x24xf16, 0x-1x20x2x32x24xf16) <- (-1x20x2x32x24xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__22, reshape__23 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_5, [x.reshape([]) for x in combine_17]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_54 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x20x32x24xf16, -1x20x32x24xf16]) <- (-1x40x32x24xf16, 1xi32)
        split_with_num_6 = paddle._C_ops.split_with_num(reshape__22, 2, full_54)

        # builtin.slice: (-1x20x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16])
        slice_18 = split_with_num_6[1]

        # pd_op.conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x20x1x1xf16)
        conv2d_18 = paddle._C_ops.conv2d(slice_18, parameter_140, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_18, parameter_141, parameter_142, parameter_143, parameter_144, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x20x32x24xf16) <- (-1x20x32x24xf16)
        relu__17 = paddle._C_ops.relu_(batch_norm__168)

        # pd_op.depthwise_conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x1x3x3xf16)
        depthwise_conv2d_10 = paddle._C_ops.depthwise_conv2d(relu__17, parameter_145, [1, 1], [1, 1], 'EXPLICIT', 20, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_10, parameter_146, parameter_147, parameter_148, parameter_149, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x20x1x1xf16)
        conv2d_19 = paddle._C_ops.conv2d(batch_norm__174, parameter_150, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__180, batch_norm__181, batch_norm__182, batch_norm__183, batch_norm__184, batch_norm__185 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_19, parameter_151, parameter_152, parameter_153, parameter_154, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x20x32x24xf16) <- (-1x20x32x24xf16)
        relu__18 = paddle._C_ops.relu_(batch_norm__180)

        # builtin.slice: (-1x20x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16])
        slice_19 = split_with_num_6[0]

        # builtin.combine: ([-1x20x32x24xf16, -1x20x32x24xf16]) <- (-1x20x32x24xf16, -1x20x32x24xf16)
        combine_18 = [slice_19, relu__18]

        # pd_op.full: (1xi32) <- ()
        full_55 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x40x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_18, full_55)

        # pd_op.shape: (4xi32) <- (-1x40x32x24xf16)
        shape_6 = paddle._C_ops.shape(paddle.cast(concat_6, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_12 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_13 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(shape_6, [0], full_int_array_12, full_int_array_13, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_56 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_57 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_58 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_59 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_19 = [slice_20, full_56, full_57, full_58, full_59]

        # pd_op.reshape_: (-1x2x20x32x24xf16, 0x-1x40x32x24xf16) <- (-1x40x32x24xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__24, reshape__25 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_6, [x.reshape([]) for x in combine_19]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x20x2x32x24xf16) <- (-1x2x20x32x24xf16)
        transpose_6 = paddle._C_ops.transpose(reshape__24, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_60 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_61 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_62 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_20 = [slice_20, full_60, full_61, full_62]

        # pd_op.reshape_: (-1x40x32x24xf16, 0x-1x20x2x32x24xf16) <- (-1x20x2x32x24xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__26, reshape__27 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_6, [x.reshape([]) for x in combine_20]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_63 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x40x16x12xf16, -1x40x16x12xf16]) <- (-1x80x16x12xf16, 1xi32)
        split_with_num_7 = paddle._C_ops.split_with_num(relu__14, 2, full_63)

        # builtin.slice: (-1x40x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16])
        slice_21 = split_with_num_7[1]

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_20 = paddle._C_ops.conv2d(slice_21, parameter_155, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__186, batch_norm__187, batch_norm__188, batch_norm__189, batch_norm__190, batch_norm__191 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_20, parameter_156, parameter_157, parameter_158, parameter_159, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__19 = paddle._C_ops.relu_(batch_norm__186)

        # pd_op.depthwise_conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x1x3x3xf16)
        depthwise_conv2d_11 = paddle._C_ops.depthwise_conv2d(relu__19, parameter_160, [1, 1], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__192, batch_norm__193, batch_norm__194, batch_norm__195, batch_norm__196, batch_norm__197 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_11, parameter_161, parameter_162, parameter_163, parameter_164, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_21 = paddle._C_ops.conv2d(batch_norm__192, parameter_165, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__198, batch_norm__199, batch_norm__200, batch_norm__201, batch_norm__202, batch_norm__203 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_21, parameter_166, parameter_167, parameter_168, parameter_169, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__20 = paddle._C_ops.relu_(batch_norm__198)

        # builtin.slice: (-1x40x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16])
        slice_22 = split_with_num_7[0]

        # builtin.combine: ([-1x40x16x12xf16, -1x40x16x12xf16]) <- (-1x40x16x12xf16, -1x40x16x12xf16)
        combine_21 = [slice_22, relu__20]

        # pd_op.full: (1xi32) <- ()
        full_64 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x80x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_21, full_64)

        # pd_op.shape: (4xi32) <- (-1x80x16x12xf16)
        shape_7 = paddle._C_ops.shape(paddle.cast(concat_7, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_14 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_15 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(shape_7, [0], full_int_array_14, full_int_array_15, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_65 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_66 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_67 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_68 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_22 = [slice_23, full_65, full_66, full_67, full_68]

        # pd_op.reshape_: (-1x2x40x16x12xf16, 0x-1x80x16x12xf16) <- (-1x80x16x12xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__28, reshape__29 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_7, [x.reshape([]) for x in combine_22]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x2x16x12xf16) <- (-1x2x40x16x12xf16)
        transpose_7 = paddle._C_ops.transpose(reshape__28, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_69 = paddle._C_ops.full([1], float('80'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_70 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_71 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_23 = [slice_23, full_69, full_70, full_71]

        # pd_op.reshape_: (-1x80x16x12xf16, 0x-1x40x2x16x12xf16) <- (-1x40x2x16x12xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__30, reshape__31 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_7, [x.reshape([]) for x in combine_23]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_72 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x40x16x12xf16, -1x40x16x12xf16]) <- (-1x80x16x12xf16, 1xi32)
        split_with_num_8 = paddle._C_ops.split_with_num(reshape__30, 2, full_72)

        # builtin.slice: (-1x40x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16])
        slice_24 = split_with_num_8[1]

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_22 = paddle._C_ops.conv2d(slice_24, parameter_170, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__204, batch_norm__205, batch_norm__206, batch_norm__207, batch_norm__208, batch_norm__209 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_22, parameter_171, parameter_172, parameter_173, parameter_174, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__21 = paddle._C_ops.relu_(batch_norm__204)

        # pd_op.depthwise_conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x1x3x3xf16)
        depthwise_conv2d_12 = paddle._C_ops.depthwise_conv2d(relu__21, parameter_175, [1, 1], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__210, batch_norm__211, batch_norm__212, batch_norm__213, batch_norm__214, batch_norm__215 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_12, parameter_176, parameter_177, parameter_178, parameter_179, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_23 = paddle._C_ops.conv2d(batch_norm__210, parameter_180, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__216, batch_norm__217, batch_norm__218, batch_norm__219, batch_norm__220, batch_norm__221 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_23, parameter_181, parameter_182, parameter_183, parameter_184, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__22 = paddle._C_ops.relu_(batch_norm__216)

        # builtin.slice: (-1x40x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16])
        slice_25 = split_with_num_8[0]

        # builtin.combine: ([-1x40x16x12xf16, -1x40x16x12xf16]) <- (-1x40x16x12xf16, -1x40x16x12xf16)
        combine_24 = [slice_25, relu__22]

        # pd_op.full: (1xi32) <- ()
        full_73 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x80x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16], 1xi32)
        concat_8 = paddle._C_ops.concat(combine_24, full_73)

        # pd_op.shape: (4xi32) <- (-1x80x16x12xf16)
        shape_8 = paddle._C_ops.shape(paddle.cast(concat_8, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_16 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_17 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(shape_8, [0], full_int_array_16, full_int_array_17, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_74 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_75 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_76 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_77 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_25 = [slice_26, full_74, full_75, full_76, full_77]

        # pd_op.reshape_: (-1x2x40x16x12xf16, 0x-1x80x16x12xf16) <- (-1x80x16x12xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__32, reshape__33 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_8, [x.reshape([]) for x in combine_25]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x2x16x12xf16) <- (-1x2x40x16x12xf16)
        transpose_8 = paddle._C_ops.transpose(reshape__32, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_78 = paddle._C_ops.full([1], float('80'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_79 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_80 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_26 = [slice_26, full_78, full_79, full_80]

        # pd_op.reshape_: (-1x80x16x12xf16, 0x-1x40x2x16x12xf16) <- (-1x40x2x16x12xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__34, reshape__35 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_8, [x.reshape([]) for x in combine_26]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x40x32x24xf16) <- (-1x40x32x24xf16, -1x40x32x24xf16)
        add__4 = paddle._C_ops.add_(reshape__26, reshape__26)

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x80x16x12xf16, 40x80x1x1xf16)
        conv2d_24 = paddle._C_ops.conv2d(reshape__34, parameter_185, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__222, batch_norm__223, batch_norm__224, batch_norm__225, batch_norm__226, batch_norm__227 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_24, parameter_186, parameter_187, parameter_188, parameter_189, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.nearest_interp: (-1x40x32x24xf16) <- (-1x40x16x12xf16, None, None, None)
        nearest_interp_1 = paddle._C_ops.nearest_interp(batch_norm__222, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 0)

        # pd_op.add_: (-1x40x32x24xf16) <- (-1x40x32x24xf16, -1x40x32x24xf16)
        add__5 = paddle._C_ops.add_(add__4, nearest_interp_1)

        # pd_op.relu: (-1x40x32x24xf16) <- (-1x40x32x24xf16)
        relu_1 = paddle._C_ops.relu(add__5)

        # pd_op.depthwise_conv2d: (-1x40x16x12xf16) <- (-1x40x32x24xf16, 40x1x3x3xf16)
        depthwise_conv2d_13 = paddle._C_ops.depthwise_conv2d(add__5, parameter_190, [2, 2], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__228, batch_norm__229, batch_norm__230, batch_norm__231, batch_norm__232, batch_norm__233 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_13, parameter_191, parameter_192, parameter_193, parameter_194, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x80x16x12xf16) <- (-1x40x16x12xf16, 80x40x1x1xf16)
        conv2d_25 = paddle._C_ops.conv2d(batch_norm__228, parameter_195, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x16x12xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x16x12xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__234, batch_norm__235, batch_norm__236, batch_norm__237, batch_norm__238, batch_norm__239 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_25, parameter_196, parameter_197, parameter_198, parameter_199, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x80x16x12xf16) <- (-1x80x16x12xf16, -1x80x16x12xf16)
        add__6 = paddle._C_ops.add_(batch_norm__234, batch_norm__234)

        # pd_op.add_: (-1x80x16x12xf16) <- (-1x80x16x12xf16, -1x80x16x12xf16)
        add__7 = paddle._C_ops.add_(add__6, reshape__34)

        # pd_op.relu_: (-1x80x16x12xf16) <- (-1x80x16x12xf16)
        relu__23 = paddle._C_ops.relu_(add__7)

        # pd_op.depthwise_conv2d: (-1x80x8x6xf16) <- (-1x80x16x12xf16, 80x1x3x3xf16)
        depthwise_conv2d_14 = paddle._C_ops.depthwise_conv2d(relu__23, parameter_200, [2, 2], [1, 1], 'EXPLICIT', 80, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__240, batch_norm__241, batch_norm__242, batch_norm__243, batch_norm__244, batch_norm__245 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_14, parameter_201, parameter_202, parameter_203, parameter_204, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x160x8x6xf16) <- (-1x80x8x6xf16, 160x80x1x1xf16)
        conv2d_26 = paddle._C_ops.conv2d(batch_norm__240, parameter_205, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x160x8x6xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x8x6xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__246, batch_norm__247, batch_norm__248, batch_norm__249, batch_norm__250, batch_norm__251 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_26, parameter_206, parameter_207, parameter_208, parameter_209, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x160x8x6xf16) <- (-1x160x8x6xf16)
        relu__24 = paddle._C_ops.relu_(batch_norm__246)

        # pd_op.full: (1xi32) <- ()
        full_81 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x20x32x24xf16, -1x20x32x24xf16]) <- (-1x40x32x24xf16, 1xi32)
        split_with_num_9 = paddle._C_ops.split_with_num(relu_1, 2, full_81)

        # builtin.slice: (-1x20x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16])
        slice_27 = split_with_num_9[1]

        # pd_op.conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x20x1x1xf16)
        conv2d_27 = paddle._C_ops.conv2d(slice_27, parameter_210, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__252, batch_norm__253, batch_norm__254, batch_norm__255, batch_norm__256, batch_norm__257 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_27, parameter_211, parameter_212, parameter_213, parameter_214, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x20x32x24xf16) <- (-1x20x32x24xf16)
        relu__25 = paddle._C_ops.relu_(batch_norm__252)

        # pd_op.depthwise_conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x1x3x3xf16)
        depthwise_conv2d_15 = paddle._C_ops.depthwise_conv2d(relu__25, parameter_215, [1, 1], [1, 1], 'EXPLICIT', 20, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__258, batch_norm__259, batch_norm__260, batch_norm__261, batch_norm__262, batch_norm__263 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_15, parameter_216, parameter_217, parameter_218, parameter_219, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x20x1x1xf16)
        conv2d_28 = paddle._C_ops.conv2d(batch_norm__258, parameter_220, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__264, batch_norm__265, batch_norm__266, batch_norm__267, batch_norm__268, batch_norm__269 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_28, parameter_221, parameter_222, parameter_223, parameter_224, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x20x32x24xf16) <- (-1x20x32x24xf16)
        relu__26 = paddle._C_ops.relu_(batch_norm__264)

        # builtin.slice: (-1x20x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16])
        slice_28 = split_with_num_9[0]

        # builtin.combine: ([-1x20x32x24xf16, -1x20x32x24xf16]) <- (-1x20x32x24xf16, -1x20x32x24xf16)
        combine_27 = [slice_28, relu__26]

        # pd_op.full: (1xi32) <- ()
        full_82 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x40x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16], 1xi32)
        concat_9 = paddle._C_ops.concat(combine_27, full_82)

        # pd_op.shape: (4xi32) <- (-1x40x32x24xf16)
        shape_9 = paddle._C_ops.shape(paddle.cast(concat_9, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_18 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_19 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(shape_9, [0], full_int_array_18, full_int_array_19, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_83 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_84 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_85 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_86 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_28 = [slice_29, full_83, full_84, full_85, full_86]

        # pd_op.reshape_: (-1x2x20x32x24xf16, 0x-1x40x32x24xf16) <- (-1x40x32x24xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__36, reshape__37 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_9, [x.reshape([]) for x in combine_28]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x20x2x32x24xf16) <- (-1x2x20x32x24xf16)
        transpose_9 = paddle._C_ops.transpose(reshape__36, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_87 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_88 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_89 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_29 = [slice_29, full_87, full_88, full_89]

        # pd_op.reshape_: (-1x40x32x24xf16, 0x-1x20x2x32x24xf16) <- (-1x20x2x32x24xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__38, reshape__39 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_9, [x.reshape([]) for x in combine_29]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_90 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x20x32x24xf16, -1x20x32x24xf16]) <- (-1x40x32x24xf16, 1xi32)
        split_with_num_10 = paddle._C_ops.split_with_num(reshape__38, 2, full_90)

        # builtin.slice: (-1x20x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16])
        slice_30 = split_with_num_10[1]

        # pd_op.conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x20x1x1xf16)
        conv2d_29 = paddle._C_ops.conv2d(slice_30, parameter_225, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__270, batch_norm__271, batch_norm__272, batch_norm__273, batch_norm__274, batch_norm__275 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_29, parameter_226, parameter_227, parameter_228, parameter_229, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x20x32x24xf16) <- (-1x20x32x24xf16)
        relu__27 = paddle._C_ops.relu_(batch_norm__270)

        # pd_op.depthwise_conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x1x3x3xf16)
        depthwise_conv2d_16 = paddle._C_ops.depthwise_conv2d(relu__27, parameter_230, [1, 1], [1, 1], 'EXPLICIT', 20, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__276, batch_norm__277, batch_norm__278, batch_norm__279, batch_norm__280, batch_norm__281 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_16, parameter_231, parameter_232, parameter_233, parameter_234, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x20x1x1xf16)
        conv2d_30 = paddle._C_ops.conv2d(batch_norm__276, parameter_235, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__282, batch_norm__283, batch_norm__284, batch_norm__285, batch_norm__286, batch_norm__287 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_30, parameter_236, parameter_237, parameter_238, parameter_239, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x20x32x24xf16) <- (-1x20x32x24xf16)
        relu__28 = paddle._C_ops.relu_(batch_norm__282)

        # builtin.slice: (-1x20x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16])
        slice_31 = split_with_num_10[0]

        # builtin.combine: ([-1x20x32x24xf16, -1x20x32x24xf16]) <- (-1x20x32x24xf16, -1x20x32x24xf16)
        combine_30 = [slice_31, relu__28]

        # pd_op.full: (1xi32) <- ()
        full_91 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x40x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16], 1xi32)
        concat_10 = paddle._C_ops.concat(combine_30, full_91)

        # pd_op.shape: (4xi32) <- (-1x40x32x24xf16)
        shape_10 = paddle._C_ops.shape(paddle.cast(concat_10, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_20 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_21 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(shape_10, [0], full_int_array_20, full_int_array_21, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_92 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_93 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_94 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_95 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_31 = [slice_32, full_92, full_93, full_94, full_95]

        # pd_op.reshape_: (-1x2x20x32x24xf16, 0x-1x40x32x24xf16) <- (-1x40x32x24xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__40, reshape__41 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_10, [x.reshape([]) for x in combine_31]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x20x2x32x24xf16) <- (-1x2x20x32x24xf16)
        transpose_10 = paddle._C_ops.transpose(reshape__40, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_96 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_97 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_98 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_32 = [slice_32, full_96, full_97, full_98]

        # pd_op.reshape_: (-1x40x32x24xf16, 0x-1x20x2x32x24xf16) <- (-1x20x2x32x24xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__42, reshape__43 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_10, [x.reshape([]) for x in combine_32]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_99 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x40x16x12xf16, -1x40x16x12xf16]) <- (-1x80x16x12xf16, 1xi32)
        split_with_num_11 = paddle._C_ops.split_with_num(relu__23, 2, full_99)

        # builtin.slice: (-1x40x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16])
        slice_33 = split_with_num_11[1]

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_31 = paddle._C_ops.conv2d(slice_33, parameter_240, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__288, batch_norm__289, batch_norm__290, batch_norm__291, batch_norm__292, batch_norm__293 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_31, parameter_241, parameter_242, parameter_243, parameter_244, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__29 = paddle._C_ops.relu_(batch_norm__288)

        # pd_op.depthwise_conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x1x3x3xf16)
        depthwise_conv2d_17 = paddle._C_ops.depthwise_conv2d(relu__29, parameter_245, [1, 1], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__294, batch_norm__295, batch_norm__296, batch_norm__297, batch_norm__298, batch_norm__299 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_17, parameter_246, parameter_247, parameter_248, parameter_249, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_32 = paddle._C_ops.conv2d(batch_norm__294, parameter_250, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__300, batch_norm__301, batch_norm__302, batch_norm__303, batch_norm__304, batch_norm__305 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_32, parameter_251, parameter_252, parameter_253, parameter_254, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__30 = paddle._C_ops.relu_(batch_norm__300)

        # builtin.slice: (-1x40x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16])
        slice_34 = split_with_num_11[0]

        # builtin.combine: ([-1x40x16x12xf16, -1x40x16x12xf16]) <- (-1x40x16x12xf16, -1x40x16x12xf16)
        combine_33 = [slice_34, relu__30]

        # pd_op.full: (1xi32) <- ()
        full_100 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x80x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16], 1xi32)
        concat_11 = paddle._C_ops.concat(combine_33, full_100)

        # pd_op.shape: (4xi32) <- (-1x80x16x12xf16)
        shape_11 = paddle._C_ops.shape(paddle.cast(concat_11, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_22 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_23 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_35 = paddle._C_ops.slice(shape_11, [0], full_int_array_22, full_int_array_23, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_101 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_102 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_103 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_104 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_34 = [slice_35, full_101, full_102, full_103, full_104]

        # pd_op.reshape_: (-1x2x40x16x12xf16, 0x-1x80x16x12xf16) <- (-1x80x16x12xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__44, reshape__45 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_11, [x.reshape([]) for x in combine_34]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x2x16x12xf16) <- (-1x2x40x16x12xf16)
        transpose_11 = paddle._C_ops.transpose(reshape__44, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_105 = paddle._C_ops.full([1], float('80'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_106 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_107 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_35 = [slice_35, full_105, full_106, full_107]

        # pd_op.reshape_: (-1x80x16x12xf16, 0x-1x40x2x16x12xf16) <- (-1x40x2x16x12xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__46, reshape__47 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_11, [x.reshape([]) for x in combine_35]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_108 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x40x16x12xf16, -1x40x16x12xf16]) <- (-1x80x16x12xf16, 1xi32)
        split_with_num_12 = paddle._C_ops.split_with_num(reshape__46, 2, full_108)

        # builtin.slice: (-1x40x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16])
        slice_36 = split_with_num_12[1]

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_33 = paddle._C_ops.conv2d(slice_36, parameter_255, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__306, batch_norm__307, batch_norm__308, batch_norm__309, batch_norm__310, batch_norm__311 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_33, parameter_256, parameter_257, parameter_258, parameter_259, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__31 = paddle._C_ops.relu_(batch_norm__306)

        # pd_op.depthwise_conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x1x3x3xf16)
        depthwise_conv2d_18 = paddle._C_ops.depthwise_conv2d(relu__31, parameter_260, [1, 1], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__312, batch_norm__313, batch_norm__314, batch_norm__315, batch_norm__316, batch_norm__317 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_18, parameter_261, parameter_262, parameter_263, parameter_264, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_34 = paddle._C_ops.conv2d(batch_norm__312, parameter_265, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__318, batch_norm__319, batch_norm__320, batch_norm__321, batch_norm__322, batch_norm__323 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_34, parameter_266, parameter_267, parameter_268, parameter_269, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__32 = paddle._C_ops.relu_(batch_norm__318)

        # builtin.slice: (-1x40x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16])
        slice_37 = split_with_num_12[0]

        # builtin.combine: ([-1x40x16x12xf16, -1x40x16x12xf16]) <- (-1x40x16x12xf16, -1x40x16x12xf16)
        combine_36 = [slice_37, relu__32]

        # pd_op.full: (1xi32) <- ()
        full_109 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x80x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16], 1xi32)
        concat_12 = paddle._C_ops.concat(combine_36, full_109)

        # pd_op.shape: (4xi32) <- (-1x80x16x12xf16)
        shape_12 = paddle._C_ops.shape(paddle.cast(concat_12, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_24 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_25 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_38 = paddle._C_ops.slice(shape_12, [0], full_int_array_24, full_int_array_25, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_110 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_111 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_112 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_113 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_37 = [slice_38, full_110, full_111, full_112, full_113]

        # pd_op.reshape_: (-1x2x40x16x12xf16, 0x-1x80x16x12xf16) <- (-1x80x16x12xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__48, reshape__49 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_12, [x.reshape([]) for x in combine_37]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x2x16x12xf16) <- (-1x2x40x16x12xf16)
        transpose_12 = paddle._C_ops.transpose(reshape__48, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_114 = paddle._C_ops.full([1], float('80'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_115 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_116 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_38 = [slice_38, full_114, full_115, full_116]

        # pd_op.reshape_: (-1x80x16x12xf16, 0x-1x40x2x16x12xf16) <- (-1x40x2x16x12xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__50, reshape__51 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_12, [x.reshape([]) for x in combine_38]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_117 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x80x8x6xf16, -1x80x8x6xf16]) <- (-1x160x8x6xf16, 1xi32)
        split_with_num_13 = paddle._C_ops.split_with_num(relu__24, 2, full_117)

        # builtin.slice: (-1x80x8x6xf16) <- ([-1x80x8x6xf16, -1x80x8x6xf16])
        slice_39 = split_with_num_13[1]

        # pd_op.conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x80x1x1xf16)
        conv2d_35 = paddle._C_ops.conv2d(slice_39, parameter_270, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__324, batch_norm__325, batch_norm__326, batch_norm__327, batch_norm__328, batch_norm__329 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_35, parameter_271, parameter_272, parameter_273, parameter_274, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x80x8x6xf16) <- (-1x80x8x6xf16)
        relu__33 = paddle._C_ops.relu_(batch_norm__324)

        # pd_op.depthwise_conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x1x3x3xf16)
        depthwise_conv2d_19 = paddle._C_ops.depthwise_conv2d(relu__33, parameter_275, [1, 1], [1, 1], 'EXPLICIT', 80, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__330, batch_norm__331, batch_norm__332, batch_norm__333, batch_norm__334, batch_norm__335 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_19, parameter_276, parameter_277, parameter_278, parameter_279, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x80x1x1xf16)
        conv2d_36 = paddle._C_ops.conv2d(batch_norm__330, parameter_280, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__336, batch_norm__337, batch_norm__338, batch_norm__339, batch_norm__340, batch_norm__341 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_36, parameter_281, parameter_282, parameter_283, parameter_284, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x80x8x6xf16) <- (-1x80x8x6xf16)
        relu__34 = paddle._C_ops.relu_(batch_norm__336)

        # builtin.slice: (-1x80x8x6xf16) <- ([-1x80x8x6xf16, -1x80x8x6xf16])
        slice_40 = split_with_num_13[0]

        # builtin.combine: ([-1x80x8x6xf16, -1x80x8x6xf16]) <- (-1x80x8x6xf16, -1x80x8x6xf16)
        combine_39 = [slice_40, relu__34]

        # pd_op.full: (1xi32) <- ()
        full_118 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x160x8x6xf16) <- ([-1x80x8x6xf16, -1x80x8x6xf16], 1xi32)
        concat_13 = paddle._C_ops.concat(combine_39, full_118)

        # pd_op.shape: (4xi32) <- (-1x160x8x6xf16)
        shape_13 = paddle._C_ops.shape(paddle.cast(concat_13, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_26 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_27 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_41 = paddle._C_ops.slice(shape_13, [0], full_int_array_26, full_int_array_27, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_119 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_120 = paddle._C_ops.full([1], float('80'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_121 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_122 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_40 = [slice_41, full_119, full_120, full_121, full_122]

        # pd_op.reshape_: (-1x2x80x8x6xf16, 0x-1x160x8x6xf16) <- (-1x160x8x6xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__52, reshape__53 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_13, [x.reshape([]) for x in combine_40]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x80x2x8x6xf16) <- (-1x2x80x8x6xf16)
        transpose_13 = paddle._C_ops.transpose(reshape__52, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_123 = paddle._C_ops.full([1], float('160'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_124 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_125 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_41 = [slice_41, full_123, full_124, full_125]

        # pd_op.reshape_: (-1x160x8x6xf16, 0x-1x80x2x8x6xf16) <- (-1x80x2x8x6xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__54, reshape__55 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_13, [x.reshape([]) for x in combine_41]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_126 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x80x8x6xf16, -1x80x8x6xf16]) <- (-1x160x8x6xf16, 1xi32)
        split_with_num_14 = paddle._C_ops.split_with_num(reshape__54, 2, full_126)

        # builtin.slice: (-1x80x8x6xf16) <- ([-1x80x8x6xf16, -1x80x8x6xf16])
        slice_42 = split_with_num_14[1]

        # pd_op.conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x80x1x1xf16)
        conv2d_37 = paddle._C_ops.conv2d(slice_42, parameter_285, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__342, batch_norm__343, batch_norm__344, batch_norm__345, batch_norm__346, batch_norm__347 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_37, parameter_286, parameter_287, parameter_288, parameter_289, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x80x8x6xf16) <- (-1x80x8x6xf16)
        relu__35 = paddle._C_ops.relu_(batch_norm__342)

        # pd_op.depthwise_conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x1x3x3xf16)
        depthwise_conv2d_20 = paddle._C_ops.depthwise_conv2d(relu__35, parameter_290, [1, 1], [1, 1], 'EXPLICIT', 80, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__348, batch_norm__349, batch_norm__350, batch_norm__351, batch_norm__352, batch_norm__353 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_20, parameter_291, parameter_292, parameter_293, parameter_294, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x80x1x1xf16)
        conv2d_38 = paddle._C_ops.conv2d(batch_norm__348, parameter_295, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__354, batch_norm__355, batch_norm__356, batch_norm__357, batch_norm__358, batch_norm__359 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_38, parameter_296, parameter_297, parameter_298, parameter_299, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x80x8x6xf16) <- (-1x80x8x6xf16)
        relu__36 = paddle._C_ops.relu_(batch_norm__354)

        # builtin.slice: (-1x80x8x6xf16) <- ([-1x80x8x6xf16, -1x80x8x6xf16])
        slice_43 = split_with_num_14[0]

        # builtin.combine: ([-1x80x8x6xf16, -1x80x8x6xf16]) <- (-1x80x8x6xf16, -1x80x8x6xf16)
        combine_42 = [slice_43, relu__36]

        # pd_op.full: (1xi32) <- ()
        full_127 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x160x8x6xf16) <- ([-1x80x8x6xf16, -1x80x8x6xf16], 1xi32)
        concat_14 = paddle._C_ops.concat(combine_42, full_127)

        # pd_op.shape: (4xi32) <- (-1x160x8x6xf16)
        shape_14 = paddle._C_ops.shape(paddle.cast(concat_14, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_28 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_29 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_44 = paddle._C_ops.slice(shape_14, [0], full_int_array_28, full_int_array_29, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_128 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_129 = paddle._C_ops.full([1], float('80'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_130 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_131 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_43 = [slice_44, full_128, full_129, full_130, full_131]

        # pd_op.reshape_: (-1x2x80x8x6xf16, 0x-1x160x8x6xf16) <- (-1x160x8x6xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__56, reshape__57 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_14, [x.reshape([]) for x in combine_43]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x80x2x8x6xf16) <- (-1x2x80x8x6xf16)
        transpose_14 = paddle._C_ops.transpose(reshape__56, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_132 = paddle._C_ops.full([1], float('160'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_133 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_134 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_44 = [slice_44, full_132, full_133, full_134]

        # pd_op.reshape_: (-1x160x8x6xf16, 0x-1x80x2x8x6xf16) <- (-1x80x2x8x6xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__58, reshape__59 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_14, [x.reshape([]) for x in combine_44]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x40x32x24xf16) <- (-1x40x32x24xf16, -1x40x32x24xf16)
        add__8 = paddle._C_ops.add_(reshape__42, reshape__42)

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x80x16x12xf16, 40x80x1x1xf16)
        conv2d_39 = paddle._C_ops.conv2d(reshape__50, parameter_300, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__360, batch_norm__361, batch_norm__362, batch_norm__363, batch_norm__364, batch_norm__365 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_39, parameter_301, parameter_302, parameter_303, parameter_304, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.nearest_interp: (-1x40x32x24xf16) <- (-1x40x16x12xf16, None, None, None)
        nearest_interp_2 = paddle._C_ops.nearest_interp(batch_norm__360, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 0)

        # pd_op.add_: (-1x40x32x24xf16) <- (-1x40x32x24xf16, -1x40x32x24xf16)
        add__9 = paddle._C_ops.add_(add__8, nearest_interp_2)

        # pd_op.conv2d: (-1x40x8x6xf16) <- (-1x160x8x6xf16, 40x160x1x1xf16)
        conv2d_40 = paddle._C_ops.conv2d(reshape__58, parameter_305, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x8x6xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x8x6xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__366, batch_norm__367, batch_norm__368, batch_norm__369, batch_norm__370, batch_norm__371 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_40, parameter_306, parameter_307, parameter_308, parameter_309, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.nearest_interp: (-1x40x32x24xf16) <- (-1x40x8x6xf16, None, None, None)
        nearest_interp_3 = paddle._C_ops.nearest_interp(batch_norm__366, None, None, None, 'NCHW', -1, -1, -1, [float('4'), float('4')], 'nearest', False, 0)

        # pd_op.add_: (-1x40x32x24xf16) <- (-1x40x32x24xf16, -1x40x32x24xf16)
        add__10 = paddle._C_ops.add_(add__9, nearest_interp_3)

        # pd_op.relu: (-1x40x32x24xf16) <- (-1x40x32x24xf16)
        relu_2 = paddle._C_ops.relu(add__10)

        # pd_op.depthwise_conv2d: (-1x40x16x12xf16) <- (-1x40x32x24xf16, 40x1x3x3xf16)
        depthwise_conv2d_21 = paddle._C_ops.depthwise_conv2d(add__10, parameter_310, [2, 2], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__372, batch_norm__373, batch_norm__374, batch_norm__375, batch_norm__376, batch_norm__377 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_21, parameter_311, parameter_312, parameter_313, parameter_314, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x80x16x12xf16) <- (-1x40x16x12xf16, 80x40x1x1xf16)
        conv2d_41 = paddle._C_ops.conv2d(batch_norm__372, parameter_315, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x16x12xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x16x12xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__378, batch_norm__379, batch_norm__380, batch_norm__381, batch_norm__382, batch_norm__383 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_41, parameter_316, parameter_317, parameter_318, parameter_319, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x80x16x12xf16) <- (-1x80x16x12xf16, -1x80x16x12xf16)
        add__11 = paddle._C_ops.add_(batch_norm__378, batch_norm__378)

        # pd_op.add_: (-1x80x16x12xf16) <- (-1x80x16x12xf16, -1x80x16x12xf16)
        add__12 = paddle._C_ops.add_(add__11, reshape__50)

        # pd_op.conv2d: (-1x80x8x6xf16) <- (-1x160x8x6xf16, 80x160x1x1xf16)
        conv2d_42 = paddle._C_ops.conv2d(reshape__58, parameter_320, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__384, batch_norm__385, batch_norm__386, batch_norm__387, batch_norm__388, batch_norm__389 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_42, parameter_321, parameter_322, parameter_323, parameter_324, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.nearest_interp: (-1x80x16x12xf16) <- (-1x80x8x6xf16, None, None, None)
        nearest_interp_4 = paddle._C_ops.nearest_interp(batch_norm__384, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 0)

        # pd_op.add_: (-1x80x16x12xf16) <- (-1x80x16x12xf16, -1x80x16x12xf16)
        add__13 = paddle._C_ops.add_(add__12, nearest_interp_4)

        # pd_op.relu_: (-1x80x16x12xf16) <- (-1x80x16x12xf16)
        relu__37 = paddle._C_ops.relu_(add__13)

        # pd_op.depthwise_conv2d: (-1x40x16x12xf16) <- (-1x40x32x24xf16, 40x1x3x3xf16)
        depthwise_conv2d_22 = paddle._C_ops.depthwise_conv2d(add__10, parameter_325, [2, 2], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__390, batch_norm__391, batch_norm__392, batch_norm__393, batch_norm__394, batch_norm__395 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_22, parameter_326, parameter_327, parameter_328, parameter_329, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_43 = paddle._C_ops.conv2d(batch_norm__390, parameter_330, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__396, batch_norm__397, batch_norm__398, batch_norm__399, batch_norm__400, batch_norm__401 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_43, parameter_331, parameter_332, parameter_333, parameter_334, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__38 = paddle._C_ops.relu_(batch_norm__396)

        # pd_op.depthwise_conv2d: (-1x40x8x6xf16) <- (-1x40x16x12xf16, 40x1x3x3xf16)
        depthwise_conv2d_23 = paddle._C_ops.depthwise_conv2d(relu__38, parameter_335, [2, 2], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x8x6xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x8x6xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__402, batch_norm__403, batch_norm__404, batch_norm__405, batch_norm__406, batch_norm__407 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_23, parameter_336, parameter_337, parameter_338, parameter_339, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x160x8x6xf16) <- (-1x40x8x6xf16, 160x40x1x1xf16)
        conv2d_44 = paddle._C_ops.conv2d(batch_norm__402, parameter_340, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x160x8x6xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x8x6xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__408, batch_norm__409, batch_norm__410, batch_norm__411, batch_norm__412, batch_norm__413 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_44, parameter_341, parameter_342, parameter_343, parameter_344, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x160x8x6xf16) <- (-1x160x8x6xf16, -1x160x8x6xf16)
        add__14 = paddle._C_ops.add_(batch_norm__408, batch_norm__408)

        # pd_op.depthwise_conv2d: (-1x80x8x6xf16) <- (-1x80x16x12xf16, 80x1x3x3xf16)
        depthwise_conv2d_24 = paddle._C_ops.depthwise_conv2d(reshape__50, parameter_345, [2, 2], [1, 1], 'EXPLICIT', 80, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__414, batch_norm__415, batch_norm__416, batch_norm__417, batch_norm__418, batch_norm__419 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_24, parameter_346, parameter_347, parameter_348, parameter_349, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x160x8x6xf16) <- (-1x80x8x6xf16, 160x80x1x1xf16)
        conv2d_45 = paddle._C_ops.conv2d(batch_norm__414, parameter_350, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x160x8x6xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x8x6xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__420, batch_norm__421, batch_norm__422, batch_norm__423, batch_norm__424, batch_norm__425 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_45, parameter_351, parameter_352, parameter_353, parameter_354, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x160x8x6xf16) <- (-1x160x8x6xf16, -1x160x8x6xf16)
        add__15 = paddle._C_ops.add_(add__14, batch_norm__420)

        # pd_op.add_: (-1x160x8x6xf16) <- (-1x160x8x6xf16, -1x160x8x6xf16)
        add__16 = paddle._C_ops.add_(add__15, reshape__58)

        # pd_op.relu_: (-1x160x8x6xf16) <- (-1x160x8x6xf16)
        relu__39 = paddle._C_ops.relu_(add__16)

        # pd_op.full: (1xi32) <- ()
        full_135 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x20x32x24xf16, -1x20x32x24xf16]) <- (-1x40x32x24xf16, 1xi32)
        split_with_num_15 = paddle._C_ops.split_with_num(relu_2, 2, full_135)

        # builtin.slice: (-1x20x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16])
        slice_45 = split_with_num_15[1]

        # pd_op.conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x20x1x1xf16)
        conv2d_46 = paddle._C_ops.conv2d(slice_45, parameter_355, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__426, batch_norm__427, batch_norm__428, batch_norm__429, batch_norm__430, batch_norm__431 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_46, parameter_356, parameter_357, parameter_358, parameter_359, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x20x32x24xf16) <- (-1x20x32x24xf16)
        relu__40 = paddle._C_ops.relu_(batch_norm__426)

        # pd_op.depthwise_conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x1x3x3xf16)
        depthwise_conv2d_25 = paddle._C_ops.depthwise_conv2d(relu__40, parameter_360, [1, 1], [1, 1], 'EXPLICIT', 20, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__432, batch_norm__433, batch_norm__434, batch_norm__435, batch_norm__436, batch_norm__437 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_25, parameter_361, parameter_362, parameter_363, parameter_364, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x20x1x1xf16)
        conv2d_47 = paddle._C_ops.conv2d(batch_norm__432, parameter_365, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__438, batch_norm__439, batch_norm__440, batch_norm__441, batch_norm__442, batch_norm__443 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_47, parameter_366, parameter_367, parameter_368, parameter_369, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x20x32x24xf16) <- (-1x20x32x24xf16)
        relu__41 = paddle._C_ops.relu_(batch_norm__438)

        # builtin.slice: (-1x20x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16])
        slice_46 = split_with_num_15[0]

        # builtin.combine: ([-1x20x32x24xf16, -1x20x32x24xf16]) <- (-1x20x32x24xf16, -1x20x32x24xf16)
        combine_45 = [slice_46, relu__41]

        # pd_op.full: (1xi32) <- ()
        full_136 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x40x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16], 1xi32)
        concat_15 = paddle._C_ops.concat(combine_45, full_136)

        # pd_op.shape: (4xi32) <- (-1x40x32x24xf16)
        shape_15 = paddle._C_ops.shape(paddle.cast(concat_15, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_30 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_31 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_47 = paddle._C_ops.slice(shape_15, [0], full_int_array_30, full_int_array_31, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_137 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_138 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_139 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_140 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_46 = [slice_47, full_137, full_138, full_139, full_140]

        # pd_op.reshape_: (-1x2x20x32x24xf16, 0x-1x40x32x24xf16) <- (-1x40x32x24xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__60, reshape__61 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_15, [x.reshape([]) for x in combine_46]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x20x2x32x24xf16) <- (-1x2x20x32x24xf16)
        transpose_15 = paddle._C_ops.transpose(reshape__60, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_141 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_142 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_143 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_47 = [slice_47, full_141, full_142, full_143]

        # pd_op.reshape_: (-1x40x32x24xf16, 0x-1x20x2x32x24xf16) <- (-1x20x2x32x24xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__62, reshape__63 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_15, [x.reshape([]) for x in combine_47]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_144 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x20x32x24xf16, -1x20x32x24xf16]) <- (-1x40x32x24xf16, 1xi32)
        split_with_num_16 = paddle._C_ops.split_with_num(reshape__62, 2, full_144)

        # builtin.slice: (-1x20x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16])
        slice_48 = split_with_num_16[1]

        # pd_op.conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x20x1x1xf16)
        conv2d_48 = paddle._C_ops.conv2d(slice_48, parameter_370, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__444, batch_norm__445, batch_norm__446, batch_norm__447, batch_norm__448, batch_norm__449 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_48, parameter_371, parameter_372, parameter_373, parameter_374, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x20x32x24xf16) <- (-1x20x32x24xf16)
        relu__42 = paddle._C_ops.relu_(batch_norm__444)

        # pd_op.depthwise_conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x1x3x3xf16)
        depthwise_conv2d_26 = paddle._C_ops.depthwise_conv2d(relu__42, parameter_375, [1, 1], [1, 1], 'EXPLICIT', 20, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__450, batch_norm__451, batch_norm__452, batch_norm__453, batch_norm__454, batch_norm__455 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_26, parameter_376, parameter_377, parameter_378, parameter_379, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x20x1x1xf16)
        conv2d_49 = paddle._C_ops.conv2d(batch_norm__450, parameter_380, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__456, batch_norm__457, batch_norm__458, batch_norm__459, batch_norm__460, batch_norm__461 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_49, parameter_381, parameter_382, parameter_383, parameter_384, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x20x32x24xf16) <- (-1x20x32x24xf16)
        relu__43 = paddle._C_ops.relu_(batch_norm__456)

        # builtin.slice: (-1x20x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16])
        slice_49 = split_with_num_16[0]

        # builtin.combine: ([-1x20x32x24xf16, -1x20x32x24xf16]) <- (-1x20x32x24xf16, -1x20x32x24xf16)
        combine_48 = [slice_49, relu__43]

        # pd_op.full: (1xi32) <- ()
        full_145 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x40x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16], 1xi32)
        concat_16 = paddle._C_ops.concat(combine_48, full_145)

        # pd_op.shape: (4xi32) <- (-1x40x32x24xf16)
        shape_16 = paddle._C_ops.shape(paddle.cast(concat_16, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_32 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_33 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_50 = paddle._C_ops.slice(shape_16, [0], full_int_array_32, full_int_array_33, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_146 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_147 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_148 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_149 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_49 = [slice_50, full_146, full_147, full_148, full_149]

        # pd_op.reshape_: (-1x2x20x32x24xf16, 0x-1x40x32x24xf16) <- (-1x40x32x24xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__64, reshape__65 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_16, [x.reshape([]) for x in combine_49]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x20x2x32x24xf16) <- (-1x2x20x32x24xf16)
        transpose_16 = paddle._C_ops.transpose(reshape__64, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_150 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_151 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_152 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_50 = [slice_50, full_150, full_151, full_152]

        # pd_op.reshape_: (-1x40x32x24xf16, 0x-1x20x2x32x24xf16) <- (-1x20x2x32x24xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__66, reshape__67 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_16, [x.reshape([]) for x in combine_50]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_153 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x40x16x12xf16, -1x40x16x12xf16]) <- (-1x80x16x12xf16, 1xi32)
        split_with_num_17 = paddle._C_ops.split_with_num(relu__37, 2, full_153)

        # builtin.slice: (-1x40x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16])
        slice_51 = split_with_num_17[1]

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_50 = paddle._C_ops.conv2d(slice_51, parameter_385, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__462, batch_norm__463, batch_norm__464, batch_norm__465, batch_norm__466, batch_norm__467 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_50, parameter_386, parameter_387, parameter_388, parameter_389, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__44 = paddle._C_ops.relu_(batch_norm__462)

        # pd_op.depthwise_conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x1x3x3xf16)
        depthwise_conv2d_27 = paddle._C_ops.depthwise_conv2d(relu__44, parameter_390, [1, 1], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__468, batch_norm__469, batch_norm__470, batch_norm__471, batch_norm__472, batch_norm__473 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_27, parameter_391, parameter_392, parameter_393, parameter_394, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_51 = paddle._C_ops.conv2d(batch_norm__468, parameter_395, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__474, batch_norm__475, batch_norm__476, batch_norm__477, batch_norm__478, batch_norm__479 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_51, parameter_396, parameter_397, parameter_398, parameter_399, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__45 = paddle._C_ops.relu_(batch_norm__474)

        # builtin.slice: (-1x40x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16])
        slice_52 = split_with_num_17[0]

        # builtin.combine: ([-1x40x16x12xf16, -1x40x16x12xf16]) <- (-1x40x16x12xf16, -1x40x16x12xf16)
        combine_51 = [slice_52, relu__45]

        # pd_op.full: (1xi32) <- ()
        full_154 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x80x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16], 1xi32)
        concat_17 = paddle._C_ops.concat(combine_51, full_154)

        # pd_op.shape: (4xi32) <- (-1x80x16x12xf16)
        shape_17 = paddle._C_ops.shape(paddle.cast(concat_17, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_34 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_35 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_53 = paddle._C_ops.slice(shape_17, [0], full_int_array_34, full_int_array_35, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_155 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_156 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_157 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_158 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_52 = [slice_53, full_155, full_156, full_157, full_158]

        # pd_op.reshape_: (-1x2x40x16x12xf16, 0x-1x80x16x12xf16) <- (-1x80x16x12xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__68, reshape__69 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_17, [x.reshape([]) for x in combine_52]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x2x16x12xf16) <- (-1x2x40x16x12xf16)
        transpose_17 = paddle._C_ops.transpose(reshape__68, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_159 = paddle._C_ops.full([1], float('80'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_160 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_161 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_53 = [slice_53, full_159, full_160, full_161]

        # pd_op.reshape_: (-1x80x16x12xf16, 0x-1x40x2x16x12xf16) <- (-1x40x2x16x12xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__70, reshape__71 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_17, [x.reshape([]) for x in combine_53]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_162 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x40x16x12xf16, -1x40x16x12xf16]) <- (-1x80x16x12xf16, 1xi32)
        split_with_num_18 = paddle._C_ops.split_with_num(reshape__70, 2, full_162)

        # builtin.slice: (-1x40x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16])
        slice_54 = split_with_num_18[1]

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_52 = paddle._C_ops.conv2d(slice_54, parameter_400, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__480, batch_norm__481, batch_norm__482, batch_norm__483, batch_norm__484, batch_norm__485 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_52, parameter_401, parameter_402, parameter_403, parameter_404, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__46 = paddle._C_ops.relu_(batch_norm__480)

        # pd_op.depthwise_conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x1x3x3xf16)
        depthwise_conv2d_28 = paddle._C_ops.depthwise_conv2d(relu__46, parameter_405, [1, 1], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__486, batch_norm__487, batch_norm__488, batch_norm__489, batch_norm__490, batch_norm__491 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_28, parameter_406, parameter_407, parameter_408, parameter_409, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_53 = paddle._C_ops.conv2d(batch_norm__486, parameter_410, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__492, batch_norm__493, batch_norm__494, batch_norm__495, batch_norm__496, batch_norm__497 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_53, parameter_411, parameter_412, parameter_413, parameter_414, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__47 = paddle._C_ops.relu_(batch_norm__492)

        # builtin.slice: (-1x40x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16])
        slice_55 = split_with_num_18[0]

        # builtin.combine: ([-1x40x16x12xf16, -1x40x16x12xf16]) <- (-1x40x16x12xf16, -1x40x16x12xf16)
        combine_54 = [slice_55, relu__47]

        # pd_op.full: (1xi32) <- ()
        full_163 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x80x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16], 1xi32)
        concat_18 = paddle._C_ops.concat(combine_54, full_163)

        # pd_op.shape: (4xi32) <- (-1x80x16x12xf16)
        shape_18 = paddle._C_ops.shape(paddle.cast(concat_18, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_36 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_37 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_56 = paddle._C_ops.slice(shape_18, [0], full_int_array_36, full_int_array_37, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_164 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_165 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_166 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_167 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_55 = [slice_56, full_164, full_165, full_166, full_167]

        # pd_op.reshape_: (-1x2x40x16x12xf16, 0x-1x80x16x12xf16) <- (-1x80x16x12xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__72, reshape__73 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_18, [x.reshape([]) for x in combine_55]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x2x16x12xf16) <- (-1x2x40x16x12xf16)
        transpose_18 = paddle._C_ops.transpose(reshape__72, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_168 = paddle._C_ops.full([1], float('80'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_169 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_170 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_56 = [slice_56, full_168, full_169, full_170]

        # pd_op.reshape_: (-1x80x16x12xf16, 0x-1x40x2x16x12xf16) <- (-1x40x2x16x12xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__74, reshape__75 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_18, [x.reshape([]) for x in combine_56]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_171 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x80x8x6xf16, -1x80x8x6xf16]) <- (-1x160x8x6xf16, 1xi32)
        split_with_num_19 = paddle._C_ops.split_with_num(relu__39, 2, full_171)

        # builtin.slice: (-1x80x8x6xf16) <- ([-1x80x8x6xf16, -1x80x8x6xf16])
        slice_57 = split_with_num_19[1]

        # pd_op.conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x80x1x1xf16)
        conv2d_54 = paddle._C_ops.conv2d(slice_57, parameter_415, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__498, batch_norm__499, batch_norm__500, batch_norm__501, batch_norm__502, batch_norm__503 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_54, parameter_416, parameter_417, parameter_418, parameter_419, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x80x8x6xf16) <- (-1x80x8x6xf16)
        relu__48 = paddle._C_ops.relu_(batch_norm__498)

        # pd_op.depthwise_conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x1x3x3xf16)
        depthwise_conv2d_29 = paddle._C_ops.depthwise_conv2d(relu__48, parameter_420, [1, 1], [1, 1], 'EXPLICIT', 80, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__504, batch_norm__505, batch_norm__506, batch_norm__507, batch_norm__508, batch_norm__509 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_29, parameter_421, parameter_422, parameter_423, parameter_424, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x80x1x1xf16)
        conv2d_55 = paddle._C_ops.conv2d(batch_norm__504, parameter_425, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__510, batch_norm__511, batch_norm__512, batch_norm__513, batch_norm__514, batch_norm__515 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_55, parameter_426, parameter_427, parameter_428, parameter_429, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x80x8x6xf16) <- (-1x80x8x6xf16)
        relu__49 = paddle._C_ops.relu_(batch_norm__510)

        # builtin.slice: (-1x80x8x6xf16) <- ([-1x80x8x6xf16, -1x80x8x6xf16])
        slice_58 = split_with_num_19[0]

        # builtin.combine: ([-1x80x8x6xf16, -1x80x8x6xf16]) <- (-1x80x8x6xf16, -1x80x8x6xf16)
        combine_57 = [slice_58, relu__49]

        # pd_op.full: (1xi32) <- ()
        full_172 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x160x8x6xf16) <- ([-1x80x8x6xf16, -1x80x8x6xf16], 1xi32)
        concat_19 = paddle._C_ops.concat(combine_57, full_172)

        # pd_op.shape: (4xi32) <- (-1x160x8x6xf16)
        shape_19 = paddle._C_ops.shape(paddle.cast(concat_19, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_38 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_39 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_59 = paddle._C_ops.slice(shape_19, [0], full_int_array_38, full_int_array_39, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_173 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_174 = paddle._C_ops.full([1], float('80'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_175 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_176 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_58 = [slice_59, full_173, full_174, full_175, full_176]

        # pd_op.reshape_: (-1x2x80x8x6xf16, 0x-1x160x8x6xf16) <- (-1x160x8x6xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__76, reshape__77 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_19, [x.reshape([]) for x in combine_58]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x80x2x8x6xf16) <- (-1x2x80x8x6xf16)
        transpose_19 = paddle._C_ops.transpose(reshape__76, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_177 = paddle._C_ops.full([1], float('160'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_178 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_179 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_59 = [slice_59, full_177, full_178, full_179]

        # pd_op.reshape_: (-1x160x8x6xf16, 0x-1x80x2x8x6xf16) <- (-1x80x2x8x6xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__78, reshape__79 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_19, [x.reshape([]) for x in combine_59]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_180 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x80x8x6xf16, -1x80x8x6xf16]) <- (-1x160x8x6xf16, 1xi32)
        split_with_num_20 = paddle._C_ops.split_with_num(reshape__78, 2, full_180)

        # builtin.slice: (-1x80x8x6xf16) <- ([-1x80x8x6xf16, -1x80x8x6xf16])
        slice_60 = split_with_num_20[1]

        # pd_op.conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x80x1x1xf16)
        conv2d_56 = paddle._C_ops.conv2d(slice_60, parameter_430, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__516, batch_norm__517, batch_norm__518, batch_norm__519, batch_norm__520, batch_norm__521 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_56, parameter_431, parameter_432, parameter_433, parameter_434, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x80x8x6xf16) <- (-1x80x8x6xf16)
        relu__50 = paddle._C_ops.relu_(batch_norm__516)

        # pd_op.depthwise_conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x1x3x3xf16)
        depthwise_conv2d_30 = paddle._C_ops.depthwise_conv2d(relu__50, parameter_435, [1, 1], [1, 1], 'EXPLICIT', 80, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__522, batch_norm__523, batch_norm__524, batch_norm__525, batch_norm__526, batch_norm__527 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_30, parameter_436, parameter_437, parameter_438, parameter_439, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x80x1x1xf16)
        conv2d_57 = paddle._C_ops.conv2d(batch_norm__522, parameter_440, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__528, batch_norm__529, batch_norm__530, batch_norm__531, batch_norm__532, batch_norm__533 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_57, parameter_441, parameter_442, parameter_443, parameter_444, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x80x8x6xf16) <- (-1x80x8x6xf16)
        relu__51 = paddle._C_ops.relu_(batch_norm__528)

        # builtin.slice: (-1x80x8x6xf16) <- ([-1x80x8x6xf16, -1x80x8x6xf16])
        slice_61 = split_with_num_20[0]

        # builtin.combine: ([-1x80x8x6xf16, -1x80x8x6xf16]) <- (-1x80x8x6xf16, -1x80x8x6xf16)
        combine_60 = [slice_61, relu__51]

        # pd_op.full: (1xi32) <- ()
        full_181 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x160x8x6xf16) <- ([-1x80x8x6xf16, -1x80x8x6xf16], 1xi32)
        concat_20 = paddle._C_ops.concat(combine_60, full_181)

        # pd_op.shape: (4xi32) <- (-1x160x8x6xf16)
        shape_20 = paddle._C_ops.shape(paddle.cast(concat_20, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_40 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_41 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_62 = paddle._C_ops.slice(shape_20, [0], full_int_array_40, full_int_array_41, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_182 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_183 = paddle._C_ops.full([1], float('80'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_184 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_185 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_61 = [slice_62, full_182, full_183, full_184, full_185]

        # pd_op.reshape_: (-1x2x80x8x6xf16, 0x-1x160x8x6xf16) <- (-1x160x8x6xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__80, reshape__81 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_20, [x.reshape([]) for x in combine_61]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x80x2x8x6xf16) <- (-1x2x80x8x6xf16)
        transpose_20 = paddle._C_ops.transpose(reshape__80, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_186 = paddle._C_ops.full([1], float('160'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_187 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_188 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_62 = [slice_62, full_186, full_187, full_188]

        # pd_op.reshape_: (-1x160x8x6xf16, 0x-1x80x2x8x6xf16) <- (-1x80x2x8x6xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__82, reshape__83 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_20, [x.reshape([]) for x in combine_62]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x40x32x24xf16) <- (-1x40x32x24xf16, -1x40x32x24xf16)
        add__17 = paddle._C_ops.add_(reshape__66, reshape__66)

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x80x16x12xf16, 40x80x1x1xf16)
        conv2d_58 = paddle._C_ops.conv2d(reshape__74, parameter_445, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__534, batch_norm__535, batch_norm__536, batch_norm__537, batch_norm__538, batch_norm__539 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_58, parameter_446, parameter_447, parameter_448, parameter_449, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.nearest_interp: (-1x40x32x24xf16) <- (-1x40x16x12xf16, None, None, None)
        nearest_interp_5 = paddle._C_ops.nearest_interp(batch_norm__534, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 0)

        # pd_op.add_: (-1x40x32x24xf16) <- (-1x40x32x24xf16, -1x40x32x24xf16)
        add__18 = paddle._C_ops.add_(add__17, nearest_interp_5)

        # pd_op.conv2d: (-1x40x8x6xf16) <- (-1x160x8x6xf16, 40x160x1x1xf16)
        conv2d_59 = paddle._C_ops.conv2d(reshape__82, parameter_450, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x8x6xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x8x6xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__540, batch_norm__541, batch_norm__542, batch_norm__543, batch_norm__544, batch_norm__545 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_59, parameter_451, parameter_452, parameter_453, parameter_454, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.nearest_interp: (-1x40x32x24xf16) <- (-1x40x8x6xf16, None, None, None)
        nearest_interp_6 = paddle._C_ops.nearest_interp(batch_norm__540, None, None, None, 'NCHW', -1, -1, -1, [float('4'), float('4')], 'nearest', False, 0)

        # pd_op.add_: (-1x40x32x24xf16) <- (-1x40x32x24xf16, -1x40x32x24xf16)
        add__19 = paddle._C_ops.add_(add__18, nearest_interp_6)

        # pd_op.relu: (-1x40x32x24xf16) <- (-1x40x32x24xf16)
        relu_3 = paddle._C_ops.relu(add__19)

        # pd_op.depthwise_conv2d: (-1x40x16x12xf16) <- (-1x40x32x24xf16, 40x1x3x3xf16)
        depthwise_conv2d_31 = paddle._C_ops.depthwise_conv2d(add__19, parameter_455, [2, 2], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__546, batch_norm__547, batch_norm__548, batch_norm__549, batch_norm__550, batch_norm__551 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_31, parameter_456, parameter_457, parameter_458, parameter_459, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x80x16x12xf16) <- (-1x40x16x12xf16, 80x40x1x1xf16)
        conv2d_60 = paddle._C_ops.conv2d(batch_norm__546, parameter_460, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x16x12xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x16x12xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__552, batch_norm__553, batch_norm__554, batch_norm__555, batch_norm__556, batch_norm__557 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_60, parameter_461, parameter_462, parameter_463, parameter_464, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x80x16x12xf16) <- (-1x80x16x12xf16, -1x80x16x12xf16)
        add__20 = paddle._C_ops.add_(batch_norm__552, batch_norm__552)

        # pd_op.add_: (-1x80x16x12xf16) <- (-1x80x16x12xf16, -1x80x16x12xf16)
        add__21 = paddle._C_ops.add_(add__20, reshape__74)

        # pd_op.conv2d: (-1x80x8x6xf16) <- (-1x160x8x6xf16, 80x160x1x1xf16)
        conv2d_61 = paddle._C_ops.conv2d(reshape__82, parameter_465, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__558, batch_norm__559, batch_norm__560, batch_norm__561, batch_norm__562, batch_norm__563 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_61, parameter_466, parameter_467, parameter_468, parameter_469, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.nearest_interp: (-1x80x16x12xf16) <- (-1x80x8x6xf16, None, None, None)
        nearest_interp_7 = paddle._C_ops.nearest_interp(batch_norm__558, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 0)

        # pd_op.add_: (-1x80x16x12xf16) <- (-1x80x16x12xf16, -1x80x16x12xf16)
        add__22 = paddle._C_ops.add_(add__21, nearest_interp_7)

        # pd_op.relu_: (-1x80x16x12xf16) <- (-1x80x16x12xf16)
        relu__52 = paddle._C_ops.relu_(add__22)

        # pd_op.depthwise_conv2d: (-1x40x16x12xf16) <- (-1x40x32x24xf16, 40x1x3x3xf16)
        depthwise_conv2d_32 = paddle._C_ops.depthwise_conv2d(add__19, parameter_470, [2, 2], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__564, batch_norm__565, batch_norm__566, batch_norm__567, batch_norm__568, batch_norm__569 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_32, parameter_471, parameter_472, parameter_473, parameter_474, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_62 = paddle._C_ops.conv2d(batch_norm__564, parameter_475, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__570, batch_norm__571, batch_norm__572, batch_norm__573, batch_norm__574, batch_norm__575 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_62, parameter_476, parameter_477, parameter_478, parameter_479, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__53 = paddle._C_ops.relu_(batch_norm__570)

        # pd_op.depthwise_conv2d: (-1x40x8x6xf16) <- (-1x40x16x12xf16, 40x1x3x3xf16)
        depthwise_conv2d_33 = paddle._C_ops.depthwise_conv2d(relu__53, parameter_480, [2, 2], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x8x6xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x8x6xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__576, batch_norm__577, batch_norm__578, batch_norm__579, batch_norm__580, batch_norm__581 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_33, parameter_481, parameter_482, parameter_483, parameter_484, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x160x8x6xf16) <- (-1x40x8x6xf16, 160x40x1x1xf16)
        conv2d_63 = paddle._C_ops.conv2d(batch_norm__576, parameter_485, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x160x8x6xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x8x6xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__582, batch_norm__583, batch_norm__584, batch_norm__585, batch_norm__586, batch_norm__587 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_63, parameter_486, parameter_487, parameter_488, parameter_489, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x160x8x6xf16) <- (-1x160x8x6xf16, -1x160x8x6xf16)
        add__23 = paddle._C_ops.add_(batch_norm__582, batch_norm__582)

        # pd_op.depthwise_conv2d: (-1x80x8x6xf16) <- (-1x80x16x12xf16, 80x1x3x3xf16)
        depthwise_conv2d_34 = paddle._C_ops.depthwise_conv2d(reshape__74, parameter_490, [2, 2], [1, 1], 'EXPLICIT', 80, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__588, batch_norm__589, batch_norm__590, batch_norm__591, batch_norm__592, batch_norm__593 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_34, parameter_491, parameter_492, parameter_493, parameter_494, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x160x8x6xf16) <- (-1x80x8x6xf16, 160x80x1x1xf16)
        conv2d_64 = paddle._C_ops.conv2d(batch_norm__588, parameter_495, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x160x8x6xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x8x6xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__594, batch_norm__595, batch_norm__596, batch_norm__597, batch_norm__598, batch_norm__599 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_64, parameter_496, parameter_497, parameter_498, parameter_499, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x160x8x6xf16) <- (-1x160x8x6xf16, -1x160x8x6xf16)
        add__24 = paddle._C_ops.add_(add__23, batch_norm__594)

        # pd_op.add_: (-1x160x8x6xf16) <- (-1x160x8x6xf16, -1x160x8x6xf16)
        add__25 = paddle._C_ops.add_(add__24, reshape__82)

        # pd_op.relu_: (-1x160x8x6xf16) <- (-1x160x8x6xf16)
        relu__54 = paddle._C_ops.relu_(add__25)

        # pd_op.full: (1xi32) <- ()
        full_189 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x20x32x24xf16, -1x20x32x24xf16]) <- (-1x40x32x24xf16, 1xi32)
        split_with_num_21 = paddle._C_ops.split_with_num(relu_3, 2, full_189)

        # builtin.slice: (-1x20x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16])
        slice_63 = split_with_num_21[1]

        # pd_op.conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x20x1x1xf16)
        conv2d_65 = paddle._C_ops.conv2d(slice_63, parameter_500, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__600, batch_norm__601, batch_norm__602, batch_norm__603, batch_norm__604, batch_norm__605 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_65, parameter_501, parameter_502, parameter_503, parameter_504, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x20x32x24xf16) <- (-1x20x32x24xf16)
        relu__55 = paddle._C_ops.relu_(batch_norm__600)

        # pd_op.depthwise_conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x1x3x3xf16)
        depthwise_conv2d_35 = paddle._C_ops.depthwise_conv2d(relu__55, parameter_505, [1, 1], [1, 1], 'EXPLICIT', 20, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__606, batch_norm__607, batch_norm__608, batch_norm__609, batch_norm__610, batch_norm__611 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_35, parameter_506, parameter_507, parameter_508, parameter_509, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x20x1x1xf16)
        conv2d_66 = paddle._C_ops.conv2d(batch_norm__606, parameter_510, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__612, batch_norm__613, batch_norm__614, batch_norm__615, batch_norm__616, batch_norm__617 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_66, parameter_511, parameter_512, parameter_513, parameter_514, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x20x32x24xf16) <- (-1x20x32x24xf16)
        relu__56 = paddle._C_ops.relu_(batch_norm__612)

        # builtin.slice: (-1x20x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16])
        slice_64 = split_with_num_21[0]

        # builtin.combine: ([-1x20x32x24xf16, -1x20x32x24xf16]) <- (-1x20x32x24xf16, -1x20x32x24xf16)
        combine_63 = [slice_64, relu__56]

        # pd_op.full: (1xi32) <- ()
        full_190 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x40x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16], 1xi32)
        concat_21 = paddle._C_ops.concat(combine_63, full_190)

        # pd_op.shape: (4xi32) <- (-1x40x32x24xf16)
        shape_21 = paddle._C_ops.shape(paddle.cast(concat_21, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_42 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_43 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_65 = paddle._C_ops.slice(shape_21, [0], full_int_array_42, full_int_array_43, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_191 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_192 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_193 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_194 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_64 = [slice_65, full_191, full_192, full_193, full_194]

        # pd_op.reshape_: (-1x2x20x32x24xf16, 0x-1x40x32x24xf16) <- (-1x40x32x24xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__84, reshape__85 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_21, [x.reshape([]) for x in combine_64]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x20x2x32x24xf16) <- (-1x2x20x32x24xf16)
        transpose_21 = paddle._C_ops.transpose(reshape__84, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_195 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_196 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_197 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_65 = [slice_65, full_195, full_196, full_197]

        # pd_op.reshape_: (-1x40x32x24xf16, 0x-1x20x2x32x24xf16) <- (-1x20x2x32x24xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__86, reshape__87 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_21, [x.reshape([]) for x in combine_65]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_198 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x20x32x24xf16, -1x20x32x24xf16]) <- (-1x40x32x24xf16, 1xi32)
        split_with_num_22 = paddle._C_ops.split_with_num(reshape__86, 2, full_198)

        # builtin.slice: (-1x20x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16])
        slice_66 = split_with_num_22[1]

        # pd_op.conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x20x1x1xf16)
        conv2d_67 = paddle._C_ops.conv2d(slice_66, parameter_515, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__618, batch_norm__619, batch_norm__620, batch_norm__621, batch_norm__622, batch_norm__623 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_67, parameter_516, parameter_517, parameter_518, parameter_519, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x20x32x24xf16) <- (-1x20x32x24xf16)
        relu__57 = paddle._C_ops.relu_(batch_norm__618)

        # pd_op.depthwise_conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x1x3x3xf16)
        depthwise_conv2d_36 = paddle._C_ops.depthwise_conv2d(relu__57, parameter_520, [1, 1], [1, 1], 'EXPLICIT', 20, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__624, batch_norm__625, batch_norm__626, batch_norm__627, batch_norm__628, batch_norm__629 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_36, parameter_521, parameter_522, parameter_523, parameter_524, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x20x1x1xf16)
        conv2d_68 = paddle._C_ops.conv2d(batch_norm__624, parameter_525, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__630, batch_norm__631, batch_norm__632, batch_norm__633, batch_norm__634, batch_norm__635 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_68, parameter_526, parameter_527, parameter_528, parameter_529, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x20x32x24xf16) <- (-1x20x32x24xf16)
        relu__58 = paddle._C_ops.relu_(batch_norm__630)

        # builtin.slice: (-1x20x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16])
        slice_67 = split_with_num_22[0]

        # builtin.combine: ([-1x20x32x24xf16, -1x20x32x24xf16]) <- (-1x20x32x24xf16, -1x20x32x24xf16)
        combine_66 = [slice_67, relu__58]

        # pd_op.full: (1xi32) <- ()
        full_199 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x40x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16], 1xi32)
        concat_22 = paddle._C_ops.concat(combine_66, full_199)

        # pd_op.shape: (4xi32) <- (-1x40x32x24xf16)
        shape_22 = paddle._C_ops.shape(paddle.cast(concat_22, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_44 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_45 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_68 = paddle._C_ops.slice(shape_22, [0], full_int_array_44, full_int_array_45, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_200 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_201 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_202 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_203 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_67 = [slice_68, full_200, full_201, full_202, full_203]

        # pd_op.reshape_: (-1x2x20x32x24xf16, 0x-1x40x32x24xf16) <- (-1x40x32x24xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__88, reshape__89 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_22, [x.reshape([]) for x in combine_67]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x20x2x32x24xf16) <- (-1x2x20x32x24xf16)
        transpose_22 = paddle._C_ops.transpose(reshape__88, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_204 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_205 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_206 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_68 = [slice_68, full_204, full_205, full_206]

        # pd_op.reshape_: (-1x40x32x24xf16, 0x-1x20x2x32x24xf16) <- (-1x20x2x32x24xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__90, reshape__91 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_22, [x.reshape([]) for x in combine_68]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_207 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x40x16x12xf16, -1x40x16x12xf16]) <- (-1x80x16x12xf16, 1xi32)
        split_with_num_23 = paddle._C_ops.split_with_num(relu__52, 2, full_207)

        # builtin.slice: (-1x40x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16])
        slice_69 = split_with_num_23[1]

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_69 = paddle._C_ops.conv2d(slice_69, parameter_530, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__636, batch_norm__637, batch_norm__638, batch_norm__639, batch_norm__640, batch_norm__641 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_69, parameter_531, parameter_532, parameter_533, parameter_534, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__59 = paddle._C_ops.relu_(batch_norm__636)

        # pd_op.depthwise_conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x1x3x3xf16)
        depthwise_conv2d_37 = paddle._C_ops.depthwise_conv2d(relu__59, parameter_535, [1, 1], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__642, batch_norm__643, batch_norm__644, batch_norm__645, batch_norm__646, batch_norm__647 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_37, parameter_536, parameter_537, parameter_538, parameter_539, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_70 = paddle._C_ops.conv2d(batch_norm__642, parameter_540, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__648, batch_norm__649, batch_norm__650, batch_norm__651, batch_norm__652, batch_norm__653 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_70, parameter_541, parameter_542, parameter_543, parameter_544, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__60 = paddle._C_ops.relu_(batch_norm__648)

        # builtin.slice: (-1x40x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16])
        slice_70 = split_with_num_23[0]

        # builtin.combine: ([-1x40x16x12xf16, -1x40x16x12xf16]) <- (-1x40x16x12xf16, -1x40x16x12xf16)
        combine_69 = [slice_70, relu__60]

        # pd_op.full: (1xi32) <- ()
        full_208 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x80x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16], 1xi32)
        concat_23 = paddle._C_ops.concat(combine_69, full_208)

        # pd_op.shape: (4xi32) <- (-1x80x16x12xf16)
        shape_23 = paddle._C_ops.shape(paddle.cast(concat_23, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_46 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_47 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_71 = paddle._C_ops.slice(shape_23, [0], full_int_array_46, full_int_array_47, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_209 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_210 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_211 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_212 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_70 = [slice_71, full_209, full_210, full_211, full_212]

        # pd_op.reshape_: (-1x2x40x16x12xf16, 0x-1x80x16x12xf16) <- (-1x80x16x12xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__92, reshape__93 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_23, [x.reshape([]) for x in combine_70]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x2x16x12xf16) <- (-1x2x40x16x12xf16)
        transpose_23 = paddle._C_ops.transpose(reshape__92, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_213 = paddle._C_ops.full([1], float('80'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_214 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_215 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_71 = [slice_71, full_213, full_214, full_215]

        # pd_op.reshape_: (-1x80x16x12xf16, 0x-1x40x2x16x12xf16) <- (-1x40x2x16x12xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__94, reshape__95 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_23, [x.reshape([]) for x in combine_71]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_216 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x40x16x12xf16, -1x40x16x12xf16]) <- (-1x80x16x12xf16, 1xi32)
        split_with_num_24 = paddle._C_ops.split_with_num(reshape__94, 2, full_216)

        # builtin.slice: (-1x40x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16])
        slice_72 = split_with_num_24[1]

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_71 = paddle._C_ops.conv2d(slice_72, parameter_545, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__654, batch_norm__655, batch_norm__656, batch_norm__657, batch_norm__658, batch_norm__659 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_71, parameter_546, parameter_547, parameter_548, parameter_549, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__61 = paddle._C_ops.relu_(batch_norm__654)

        # pd_op.depthwise_conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x1x3x3xf16)
        depthwise_conv2d_38 = paddle._C_ops.depthwise_conv2d(relu__61, parameter_550, [1, 1], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__660, batch_norm__661, batch_norm__662, batch_norm__663, batch_norm__664, batch_norm__665 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_38, parameter_551, parameter_552, parameter_553, parameter_554, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_72 = paddle._C_ops.conv2d(batch_norm__660, parameter_555, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__666, batch_norm__667, batch_norm__668, batch_norm__669, batch_norm__670, batch_norm__671 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_72, parameter_556, parameter_557, parameter_558, parameter_559, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__62 = paddle._C_ops.relu_(batch_norm__666)

        # builtin.slice: (-1x40x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16])
        slice_73 = split_with_num_24[0]

        # builtin.combine: ([-1x40x16x12xf16, -1x40x16x12xf16]) <- (-1x40x16x12xf16, -1x40x16x12xf16)
        combine_72 = [slice_73, relu__62]

        # pd_op.full: (1xi32) <- ()
        full_217 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x80x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16], 1xi32)
        concat_24 = paddle._C_ops.concat(combine_72, full_217)

        # pd_op.shape: (4xi32) <- (-1x80x16x12xf16)
        shape_24 = paddle._C_ops.shape(paddle.cast(concat_24, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_48 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_49 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_74 = paddle._C_ops.slice(shape_24, [0], full_int_array_48, full_int_array_49, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_218 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_219 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_220 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_221 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_73 = [slice_74, full_218, full_219, full_220, full_221]

        # pd_op.reshape_: (-1x2x40x16x12xf16, 0x-1x80x16x12xf16) <- (-1x80x16x12xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__96, reshape__97 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_24, [x.reshape([]) for x in combine_73]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x2x16x12xf16) <- (-1x2x40x16x12xf16)
        transpose_24 = paddle._C_ops.transpose(reshape__96, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_222 = paddle._C_ops.full([1], float('80'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_223 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_224 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_74 = [slice_74, full_222, full_223, full_224]

        # pd_op.reshape_: (-1x80x16x12xf16, 0x-1x40x2x16x12xf16) <- (-1x40x2x16x12xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__98, reshape__99 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_24, [x.reshape([]) for x in combine_74]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_225 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x80x8x6xf16, -1x80x8x6xf16]) <- (-1x160x8x6xf16, 1xi32)
        split_with_num_25 = paddle._C_ops.split_with_num(relu__54, 2, full_225)

        # builtin.slice: (-1x80x8x6xf16) <- ([-1x80x8x6xf16, -1x80x8x6xf16])
        slice_75 = split_with_num_25[1]

        # pd_op.conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x80x1x1xf16)
        conv2d_73 = paddle._C_ops.conv2d(slice_75, parameter_560, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__672, batch_norm__673, batch_norm__674, batch_norm__675, batch_norm__676, batch_norm__677 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_73, parameter_561, parameter_562, parameter_563, parameter_564, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x80x8x6xf16) <- (-1x80x8x6xf16)
        relu__63 = paddle._C_ops.relu_(batch_norm__672)

        # pd_op.depthwise_conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x1x3x3xf16)
        depthwise_conv2d_39 = paddle._C_ops.depthwise_conv2d(relu__63, parameter_565, [1, 1], [1, 1], 'EXPLICIT', 80, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__678, batch_norm__679, batch_norm__680, batch_norm__681, batch_norm__682, batch_norm__683 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_39, parameter_566, parameter_567, parameter_568, parameter_569, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x80x1x1xf16)
        conv2d_74 = paddle._C_ops.conv2d(batch_norm__678, parameter_570, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__684, batch_norm__685, batch_norm__686, batch_norm__687, batch_norm__688, batch_norm__689 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_74, parameter_571, parameter_572, parameter_573, parameter_574, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x80x8x6xf16) <- (-1x80x8x6xf16)
        relu__64 = paddle._C_ops.relu_(batch_norm__684)

        # builtin.slice: (-1x80x8x6xf16) <- ([-1x80x8x6xf16, -1x80x8x6xf16])
        slice_76 = split_with_num_25[0]

        # builtin.combine: ([-1x80x8x6xf16, -1x80x8x6xf16]) <- (-1x80x8x6xf16, -1x80x8x6xf16)
        combine_75 = [slice_76, relu__64]

        # pd_op.full: (1xi32) <- ()
        full_226 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x160x8x6xf16) <- ([-1x80x8x6xf16, -1x80x8x6xf16], 1xi32)
        concat_25 = paddle._C_ops.concat(combine_75, full_226)

        # pd_op.shape: (4xi32) <- (-1x160x8x6xf16)
        shape_25 = paddle._C_ops.shape(paddle.cast(concat_25, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_50 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_51 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_77 = paddle._C_ops.slice(shape_25, [0], full_int_array_50, full_int_array_51, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_227 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_228 = paddle._C_ops.full([1], float('80'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_229 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_230 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_76 = [slice_77, full_227, full_228, full_229, full_230]

        # pd_op.reshape_: (-1x2x80x8x6xf16, 0x-1x160x8x6xf16) <- (-1x160x8x6xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__100, reshape__101 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_25, [x.reshape([]) for x in combine_76]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x80x2x8x6xf16) <- (-1x2x80x8x6xf16)
        transpose_25 = paddle._C_ops.transpose(reshape__100, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_231 = paddle._C_ops.full([1], float('160'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_232 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_233 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_77 = [slice_77, full_231, full_232, full_233]

        # pd_op.reshape_: (-1x160x8x6xf16, 0x-1x80x2x8x6xf16) <- (-1x80x2x8x6xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__102, reshape__103 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_25, [x.reshape([]) for x in combine_77]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_234 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x80x8x6xf16, -1x80x8x6xf16]) <- (-1x160x8x6xf16, 1xi32)
        split_with_num_26 = paddle._C_ops.split_with_num(reshape__102, 2, full_234)

        # builtin.slice: (-1x80x8x6xf16) <- ([-1x80x8x6xf16, -1x80x8x6xf16])
        slice_78 = split_with_num_26[1]

        # pd_op.conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x80x1x1xf16)
        conv2d_75 = paddle._C_ops.conv2d(slice_78, parameter_575, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__690, batch_norm__691, batch_norm__692, batch_norm__693, batch_norm__694, batch_norm__695 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_75, parameter_576, parameter_577, parameter_578, parameter_579, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x80x8x6xf16) <- (-1x80x8x6xf16)
        relu__65 = paddle._C_ops.relu_(batch_norm__690)

        # pd_op.depthwise_conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x1x3x3xf16)
        depthwise_conv2d_40 = paddle._C_ops.depthwise_conv2d(relu__65, parameter_580, [1, 1], [1, 1], 'EXPLICIT', 80, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__696, batch_norm__697, batch_norm__698, batch_norm__699, batch_norm__700, batch_norm__701 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_40, parameter_581, parameter_582, parameter_583, parameter_584, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x80x1x1xf16)
        conv2d_76 = paddle._C_ops.conv2d(batch_norm__696, parameter_585, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__702, batch_norm__703, batch_norm__704, batch_norm__705, batch_norm__706, batch_norm__707 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_76, parameter_586, parameter_587, parameter_588, parameter_589, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x80x8x6xf16) <- (-1x80x8x6xf16)
        relu__66 = paddle._C_ops.relu_(batch_norm__702)

        # builtin.slice: (-1x80x8x6xf16) <- ([-1x80x8x6xf16, -1x80x8x6xf16])
        slice_79 = split_with_num_26[0]

        # builtin.combine: ([-1x80x8x6xf16, -1x80x8x6xf16]) <- (-1x80x8x6xf16, -1x80x8x6xf16)
        combine_78 = [slice_79, relu__66]

        # pd_op.full: (1xi32) <- ()
        full_235 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x160x8x6xf16) <- ([-1x80x8x6xf16, -1x80x8x6xf16], 1xi32)
        concat_26 = paddle._C_ops.concat(combine_78, full_235)

        # pd_op.shape: (4xi32) <- (-1x160x8x6xf16)
        shape_26 = paddle._C_ops.shape(paddle.cast(concat_26, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_52 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_53 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_80 = paddle._C_ops.slice(shape_26, [0], full_int_array_52, full_int_array_53, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_236 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_237 = paddle._C_ops.full([1], float('80'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_238 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_239 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_79 = [slice_80, full_236, full_237, full_238, full_239]

        # pd_op.reshape_: (-1x2x80x8x6xf16, 0x-1x160x8x6xf16) <- (-1x160x8x6xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__104, reshape__105 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_26, [x.reshape([]) for x in combine_79]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x80x2x8x6xf16) <- (-1x2x80x8x6xf16)
        transpose_26 = paddle._C_ops.transpose(reshape__104, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_240 = paddle._C_ops.full([1], float('160'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_241 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_242 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_80 = [slice_80, full_240, full_241, full_242]

        # pd_op.reshape_: (-1x160x8x6xf16, 0x-1x80x2x8x6xf16) <- (-1x80x2x8x6xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__106, reshape__107 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_26, [x.reshape([]) for x in combine_80]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x40x32x24xf16) <- (-1x40x32x24xf16, -1x40x32x24xf16)
        add__26 = paddle._C_ops.add_(reshape__90, reshape__90)

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x80x16x12xf16, 40x80x1x1xf16)
        conv2d_77 = paddle._C_ops.conv2d(reshape__98, parameter_590, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__708, batch_norm__709, batch_norm__710, batch_norm__711, batch_norm__712, batch_norm__713 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_77, parameter_591, parameter_592, parameter_593, parameter_594, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.nearest_interp: (-1x40x32x24xf16) <- (-1x40x16x12xf16, None, None, None)
        nearest_interp_8 = paddle._C_ops.nearest_interp(batch_norm__708, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 0)

        # pd_op.add_: (-1x40x32x24xf16) <- (-1x40x32x24xf16, -1x40x32x24xf16)
        add__27 = paddle._C_ops.add_(add__26, nearest_interp_8)

        # pd_op.conv2d: (-1x40x8x6xf16) <- (-1x160x8x6xf16, 40x160x1x1xf16)
        conv2d_78 = paddle._C_ops.conv2d(reshape__106, parameter_595, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x8x6xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x8x6xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__714, batch_norm__715, batch_norm__716, batch_norm__717, batch_norm__718, batch_norm__719 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_78, parameter_596, parameter_597, parameter_598, parameter_599, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.nearest_interp: (-1x40x32x24xf16) <- (-1x40x8x6xf16, None, None, None)
        nearest_interp_9 = paddle._C_ops.nearest_interp(batch_norm__714, None, None, None, 'NCHW', -1, -1, -1, [float('4'), float('4')], 'nearest', False, 0)

        # pd_op.add_: (-1x40x32x24xf16) <- (-1x40x32x24xf16, -1x40x32x24xf16)
        add__28 = paddle._C_ops.add_(add__27, nearest_interp_9)

        # pd_op.relu: (-1x40x32x24xf16) <- (-1x40x32x24xf16)
        relu_4 = paddle._C_ops.relu(add__28)

        # pd_op.depthwise_conv2d: (-1x40x16x12xf16) <- (-1x40x32x24xf16, 40x1x3x3xf16)
        depthwise_conv2d_41 = paddle._C_ops.depthwise_conv2d(add__28, parameter_600, [2, 2], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__720, batch_norm__721, batch_norm__722, batch_norm__723, batch_norm__724, batch_norm__725 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_41, parameter_601, parameter_602, parameter_603, parameter_604, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x80x16x12xf16) <- (-1x40x16x12xf16, 80x40x1x1xf16)
        conv2d_79 = paddle._C_ops.conv2d(batch_norm__720, parameter_605, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x16x12xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x16x12xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__726, batch_norm__727, batch_norm__728, batch_norm__729, batch_norm__730, batch_norm__731 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_79, parameter_606, parameter_607, parameter_608, parameter_609, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x80x16x12xf16) <- (-1x80x16x12xf16, -1x80x16x12xf16)
        add__29 = paddle._C_ops.add_(batch_norm__726, batch_norm__726)

        # pd_op.add_: (-1x80x16x12xf16) <- (-1x80x16x12xf16, -1x80x16x12xf16)
        add__30 = paddle._C_ops.add_(add__29, reshape__98)

        # pd_op.conv2d: (-1x80x8x6xf16) <- (-1x160x8x6xf16, 80x160x1x1xf16)
        conv2d_80 = paddle._C_ops.conv2d(reshape__106, parameter_610, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__732, batch_norm__733, batch_norm__734, batch_norm__735, batch_norm__736, batch_norm__737 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_80, parameter_611, parameter_612, parameter_613, parameter_614, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.nearest_interp: (-1x80x16x12xf16) <- (-1x80x8x6xf16, None, None, None)
        nearest_interp_10 = paddle._C_ops.nearest_interp(batch_norm__732, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 0)

        # pd_op.add_: (-1x80x16x12xf16) <- (-1x80x16x12xf16, -1x80x16x12xf16)
        add__31 = paddle._C_ops.add_(add__30, nearest_interp_10)

        # pd_op.relu_: (-1x80x16x12xf16) <- (-1x80x16x12xf16)
        relu__67 = paddle._C_ops.relu_(add__31)

        # pd_op.depthwise_conv2d: (-1x40x16x12xf16) <- (-1x40x32x24xf16, 40x1x3x3xf16)
        depthwise_conv2d_42 = paddle._C_ops.depthwise_conv2d(add__28, parameter_615, [2, 2], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__738, batch_norm__739, batch_norm__740, batch_norm__741, batch_norm__742, batch_norm__743 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_42, parameter_616, parameter_617, parameter_618, parameter_619, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_81 = paddle._C_ops.conv2d(batch_norm__738, parameter_620, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__744, batch_norm__745, batch_norm__746, batch_norm__747, batch_norm__748, batch_norm__749 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_81, parameter_621, parameter_622, parameter_623, parameter_624, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__68 = paddle._C_ops.relu_(batch_norm__744)

        # pd_op.depthwise_conv2d: (-1x40x8x6xf16) <- (-1x40x16x12xf16, 40x1x3x3xf16)
        depthwise_conv2d_43 = paddle._C_ops.depthwise_conv2d(relu__68, parameter_625, [2, 2], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x8x6xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x8x6xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__750, batch_norm__751, batch_norm__752, batch_norm__753, batch_norm__754, batch_norm__755 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_43, parameter_626, parameter_627, parameter_628, parameter_629, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x160x8x6xf16) <- (-1x40x8x6xf16, 160x40x1x1xf16)
        conv2d_82 = paddle._C_ops.conv2d(batch_norm__750, parameter_630, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x160x8x6xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x8x6xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__756, batch_norm__757, batch_norm__758, batch_norm__759, batch_norm__760, batch_norm__761 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_82, parameter_631, parameter_632, parameter_633, parameter_634, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x160x8x6xf16) <- (-1x160x8x6xf16, -1x160x8x6xf16)
        add__32 = paddle._C_ops.add_(batch_norm__756, batch_norm__756)

        # pd_op.depthwise_conv2d: (-1x80x8x6xf16) <- (-1x80x16x12xf16, 80x1x3x3xf16)
        depthwise_conv2d_44 = paddle._C_ops.depthwise_conv2d(reshape__98, parameter_635, [2, 2], [1, 1], 'EXPLICIT', 80, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__762, batch_norm__763, batch_norm__764, batch_norm__765, batch_norm__766, batch_norm__767 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_44, parameter_636, parameter_637, parameter_638, parameter_639, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x160x8x6xf16) <- (-1x80x8x6xf16, 160x80x1x1xf16)
        conv2d_83 = paddle._C_ops.conv2d(batch_norm__762, parameter_640, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x160x8x6xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x8x6xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__768, batch_norm__769, batch_norm__770, batch_norm__771, batch_norm__772, batch_norm__773 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_83, parameter_641, parameter_642, parameter_643, parameter_644, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x160x8x6xf16) <- (-1x160x8x6xf16, -1x160x8x6xf16)
        add__33 = paddle._C_ops.add_(add__32, batch_norm__768)

        # pd_op.add_: (-1x160x8x6xf16) <- (-1x160x8x6xf16, -1x160x8x6xf16)
        add__34 = paddle._C_ops.add_(add__33, reshape__106)

        # pd_op.relu_: (-1x160x8x6xf16) <- (-1x160x8x6xf16)
        relu__69 = paddle._C_ops.relu_(add__34)

        # pd_op.full: (1xi32) <- ()
        full_243 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x20x32x24xf16, -1x20x32x24xf16]) <- (-1x40x32x24xf16, 1xi32)
        split_with_num_27 = paddle._C_ops.split_with_num(relu_4, 2, full_243)

        # builtin.slice: (-1x20x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16])
        slice_81 = split_with_num_27[1]

        # pd_op.conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x20x1x1xf16)
        conv2d_84 = paddle._C_ops.conv2d(slice_81, parameter_645, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__774, batch_norm__775, batch_norm__776, batch_norm__777, batch_norm__778, batch_norm__779 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_84, parameter_646, parameter_647, parameter_648, parameter_649, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x20x32x24xf16) <- (-1x20x32x24xf16)
        relu__70 = paddle._C_ops.relu_(batch_norm__774)

        # pd_op.depthwise_conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x1x3x3xf16)
        depthwise_conv2d_45 = paddle._C_ops.depthwise_conv2d(relu__70, parameter_650, [1, 1], [1, 1], 'EXPLICIT', 20, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__780, batch_norm__781, batch_norm__782, batch_norm__783, batch_norm__784, batch_norm__785 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_45, parameter_651, parameter_652, parameter_653, parameter_654, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x20x1x1xf16)
        conv2d_85 = paddle._C_ops.conv2d(batch_norm__780, parameter_655, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__786, batch_norm__787, batch_norm__788, batch_norm__789, batch_norm__790, batch_norm__791 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_85, parameter_656, parameter_657, parameter_658, parameter_659, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x20x32x24xf16) <- (-1x20x32x24xf16)
        relu__71 = paddle._C_ops.relu_(batch_norm__786)

        # builtin.slice: (-1x20x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16])
        slice_82 = split_with_num_27[0]

        # builtin.combine: ([-1x20x32x24xf16, -1x20x32x24xf16]) <- (-1x20x32x24xf16, -1x20x32x24xf16)
        combine_81 = [slice_82, relu__71]

        # pd_op.full: (1xi32) <- ()
        full_244 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x40x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16], 1xi32)
        concat_27 = paddle._C_ops.concat(combine_81, full_244)

        # pd_op.shape: (4xi32) <- (-1x40x32x24xf16)
        shape_27 = paddle._C_ops.shape(paddle.cast(concat_27, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_54 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_55 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_83 = paddle._C_ops.slice(shape_27, [0], full_int_array_54, full_int_array_55, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_245 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_246 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_247 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_248 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_82 = [slice_83, full_245, full_246, full_247, full_248]

        # pd_op.reshape_: (-1x2x20x32x24xf16, 0x-1x40x32x24xf16) <- (-1x40x32x24xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__108, reshape__109 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_27, [x.reshape([]) for x in combine_82]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x20x2x32x24xf16) <- (-1x2x20x32x24xf16)
        transpose_27 = paddle._C_ops.transpose(reshape__108, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_249 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_250 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_251 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_83 = [slice_83, full_249, full_250, full_251]

        # pd_op.reshape_: (-1x40x32x24xf16, 0x-1x20x2x32x24xf16) <- (-1x20x2x32x24xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__110, reshape__111 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_27, [x.reshape([]) for x in combine_83]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_252 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x20x32x24xf16, -1x20x32x24xf16]) <- (-1x40x32x24xf16, 1xi32)
        split_with_num_28 = paddle._C_ops.split_with_num(reshape__110, 2, full_252)

        # builtin.slice: (-1x20x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16])
        slice_84 = split_with_num_28[1]

        # pd_op.conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x20x1x1xf16)
        conv2d_86 = paddle._C_ops.conv2d(slice_84, parameter_660, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__792, batch_norm__793, batch_norm__794, batch_norm__795, batch_norm__796, batch_norm__797 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_86, parameter_661, parameter_662, parameter_663, parameter_664, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x20x32x24xf16) <- (-1x20x32x24xf16)
        relu__72 = paddle._C_ops.relu_(batch_norm__792)

        # pd_op.depthwise_conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x1x3x3xf16)
        depthwise_conv2d_46 = paddle._C_ops.depthwise_conv2d(relu__72, parameter_665, [1, 1], [1, 1], 'EXPLICIT', 20, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__798, batch_norm__799, batch_norm__800, batch_norm__801, batch_norm__802, batch_norm__803 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_46, parameter_666, parameter_667, parameter_668, parameter_669, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x20x1x1xf16)
        conv2d_87 = paddle._C_ops.conv2d(batch_norm__798, parameter_670, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__804, batch_norm__805, batch_norm__806, batch_norm__807, batch_norm__808, batch_norm__809 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_87, parameter_671, parameter_672, parameter_673, parameter_674, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x20x32x24xf16) <- (-1x20x32x24xf16)
        relu__73 = paddle._C_ops.relu_(batch_norm__804)

        # builtin.slice: (-1x20x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16])
        slice_85 = split_with_num_28[0]

        # builtin.combine: ([-1x20x32x24xf16, -1x20x32x24xf16]) <- (-1x20x32x24xf16, -1x20x32x24xf16)
        combine_84 = [slice_85, relu__73]

        # pd_op.full: (1xi32) <- ()
        full_253 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x40x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16], 1xi32)
        concat_28 = paddle._C_ops.concat(combine_84, full_253)

        # pd_op.shape: (4xi32) <- (-1x40x32x24xf16)
        shape_28 = paddle._C_ops.shape(paddle.cast(concat_28, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_56 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_57 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_86 = paddle._C_ops.slice(shape_28, [0], full_int_array_56, full_int_array_57, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_254 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_255 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_256 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_257 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_85 = [slice_86, full_254, full_255, full_256, full_257]

        # pd_op.reshape_: (-1x2x20x32x24xf16, 0x-1x40x32x24xf16) <- (-1x40x32x24xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__112, reshape__113 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_28, [x.reshape([]) for x in combine_85]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x20x2x32x24xf16) <- (-1x2x20x32x24xf16)
        transpose_28 = paddle._C_ops.transpose(reshape__112, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_258 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_259 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_260 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_86 = [slice_86, full_258, full_259, full_260]

        # pd_op.reshape_: (-1x40x32x24xf16, 0x-1x20x2x32x24xf16) <- (-1x20x2x32x24xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__114, reshape__115 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_28, [x.reshape([]) for x in combine_86]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_261 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x40x16x12xf16, -1x40x16x12xf16]) <- (-1x80x16x12xf16, 1xi32)
        split_with_num_29 = paddle._C_ops.split_with_num(relu__67, 2, full_261)

        # builtin.slice: (-1x40x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16])
        slice_87 = split_with_num_29[1]

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_88 = paddle._C_ops.conv2d(slice_87, parameter_675, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__810, batch_norm__811, batch_norm__812, batch_norm__813, batch_norm__814, batch_norm__815 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_88, parameter_676, parameter_677, parameter_678, parameter_679, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__74 = paddle._C_ops.relu_(batch_norm__810)

        # pd_op.depthwise_conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x1x3x3xf16)
        depthwise_conv2d_47 = paddle._C_ops.depthwise_conv2d(relu__74, parameter_680, [1, 1], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__816, batch_norm__817, batch_norm__818, batch_norm__819, batch_norm__820, batch_norm__821 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_47, parameter_681, parameter_682, parameter_683, parameter_684, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_89 = paddle._C_ops.conv2d(batch_norm__816, parameter_685, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__822, batch_norm__823, batch_norm__824, batch_norm__825, batch_norm__826, batch_norm__827 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_89, parameter_686, parameter_687, parameter_688, parameter_689, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__75 = paddle._C_ops.relu_(batch_norm__822)

        # builtin.slice: (-1x40x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16])
        slice_88 = split_with_num_29[0]

        # builtin.combine: ([-1x40x16x12xf16, -1x40x16x12xf16]) <- (-1x40x16x12xf16, -1x40x16x12xf16)
        combine_87 = [slice_88, relu__75]

        # pd_op.full: (1xi32) <- ()
        full_262 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x80x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16], 1xi32)
        concat_29 = paddle._C_ops.concat(combine_87, full_262)

        # pd_op.shape: (4xi32) <- (-1x80x16x12xf16)
        shape_29 = paddle._C_ops.shape(paddle.cast(concat_29, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_58 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_59 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_89 = paddle._C_ops.slice(shape_29, [0], full_int_array_58, full_int_array_59, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_263 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_264 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_265 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_266 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_88 = [slice_89, full_263, full_264, full_265, full_266]

        # pd_op.reshape_: (-1x2x40x16x12xf16, 0x-1x80x16x12xf16) <- (-1x80x16x12xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__116, reshape__117 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_29, [x.reshape([]) for x in combine_88]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x2x16x12xf16) <- (-1x2x40x16x12xf16)
        transpose_29 = paddle._C_ops.transpose(reshape__116, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_267 = paddle._C_ops.full([1], float('80'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_268 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_269 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_89 = [slice_89, full_267, full_268, full_269]

        # pd_op.reshape_: (-1x80x16x12xf16, 0x-1x40x2x16x12xf16) <- (-1x40x2x16x12xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__118, reshape__119 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_29, [x.reshape([]) for x in combine_89]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_270 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x40x16x12xf16, -1x40x16x12xf16]) <- (-1x80x16x12xf16, 1xi32)
        split_with_num_30 = paddle._C_ops.split_with_num(reshape__118, 2, full_270)

        # builtin.slice: (-1x40x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16])
        slice_90 = split_with_num_30[1]

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_90 = paddle._C_ops.conv2d(slice_90, parameter_690, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__828, batch_norm__829, batch_norm__830, batch_norm__831, batch_norm__832, batch_norm__833 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_90, parameter_691, parameter_692, parameter_693, parameter_694, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__76 = paddle._C_ops.relu_(batch_norm__828)

        # pd_op.depthwise_conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x1x3x3xf16)
        depthwise_conv2d_48 = paddle._C_ops.depthwise_conv2d(relu__76, parameter_695, [1, 1], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__834, batch_norm__835, batch_norm__836, batch_norm__837, batch_norm__838, batch_norm__839 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_48, parameter_696, parameter_697, parameter_698, parameter_699, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_91 = paddle._C_ops.conv2d(batch_norm__834, parameter_700, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__840, batch_norm__841, batch_norm__842, batch_norm__843, batch_norm__844, batch_norm__845 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_91, parameter_701, parameter_702, parameter_703, parameter_704, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__77 = paddle._C_ops.relu_(batch_norm__840)

        # builtin.slice: (-1x40x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16])
        slice_91 = split_with_num_30[0]

        # builtin.combine: ([-1x40x16x12xf16, -1x40x16x12xf16]) <- (-1x40x16x12xf16, -1x40x16x12xf16)
        combine_90 = [slice_91, relu__77]

        # pd_op.full: (1xi32) <- ()
        full_271 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x80x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16], 1xi32)
        concat_30 = paddle._C_ops.concat(combine_90, full_271)

        # pd_op.shape: (4xi32) <- (-1x80x16x12xf16)
        shape_30 = paddle._C_ops.shape(paddle.cast(concat_30, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_60 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_61 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_92 = paddle._C_ops.slice(shape_30, [0], full_int_array_60, full_int_array_61, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_272 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_273 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_274 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_275 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_91 = [slice_92, full_272, full_273, full_274, full_275]

        # pd_op.reshape_: (-1x2x40x16x12xf16, 0x-1x80x16x12xf16) <- (-1x80x16x12xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__120, reshape__121 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_30, [x.reshape([]) for x in combine_91]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x2x16x12xf16) <- (-1x2x40x16x12xf16)
        transpose_30 = paddle._C_ops.transpose(reshape__120, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_276 = paddle._C_ops.full([1], float('80'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_277 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_278 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_92 = [slice_92, full_276, full_277, full_278]

        # pd_op.reshape_: (-1x80x16x12xf16, 0x-1x40x2x16x12xf16) <- (-1x40x2x16x12xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__122, reshape__123 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_30, [x.reshape([]) for x in combine_92]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_279 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x80x8x6xf16, -1x80x8x6xf16]) <- (-1x160x8x6xf16, 1xi32)
        split_with_num_31 = paddle._C_ops.split_with_num(relu__69, 2, full_279)

        # builtin.slice: (-1x80x8x6xf16) <- ([-1x80x8x6xf16, -1x80x8x6xf16])
        slice_93 = split_with_num_31[1]

        # pd_op.conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x80x1x1xf16)
        conv2d_92 = paddle._C_ops.conv2d(slice_93, parameter_705, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__846, batch_norm__847, batch_norm__848, batch_norm__849, batch_norm__850, batch_norm__851 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_92, parameter_706, parameter_707, parameter_708, parameter_709, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x80x8x6xf16) <- (-1x80x8x6xf16)
        relu__78 = paddle._C_ops.relu_(batch_norm__846)

        # pd_op.depthwise_conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x1x3x3xf16)
        depthwise_conv2d_49 = paddle._C_ops.depthwise_conv2d(relu__78, parameter_710, [1, 1], [1, 1], 'EXPLICIT', 80, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__852, batch_norm__853, batch_norm__854, batch_norm__855, batch_norm__856, batch_norm__857 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_49, parameter_711, parameter_712, parameter_713, parameter_714, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x80x1x1xf16)
        conv2d_93 = paddle._C_ops.conv2d(batch_norm__852, parameter_715, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__858, batch_norm__859, batch_norm__860, batch_norm__861, batch_norm__862, batch_norm__863 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_93, parameter_716, parameter_717, parameter_718, parameter_719, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x80x8x6xf16) <- (-1x80x8x6xf16)
        relu__79 = paddle._C_ops.relu_(batch_norm__858)

        # builtin.slice: (-1x80x8x6xf16) <- ([-1x80x8x6xf16, -1x80x8x6xf16])
        slice_94 = split_with_num_31[0]

        # builtin.combine: ([-1x80x8x6xf16, -1x80x8x6xf16]) <- (-1x80x8x6xf16, -1x80x8x6xf16)
        combine_93 = [slice_94, relu__79]

        # pd_op.full: (1xi32) <- ()
        full_280 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x160x8x6xf16) <- ([-1x80x8x6xf16, -1x80x8x6xf16], 1xi32)
        concat_31 = paddle._C_ops.concat(combine_93, full_280)

        # pd_op.shape: (4xi32) <- (-1x160x8x6xf16)
        shape_31 = paddle._C_ops.shape(paddle.cast(concat_31, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_62 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_63 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_95 = paddle._C_ops.slice(shape_31, [0], full_int_array_62, full_int_array_63, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_281 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_282 = paddle._C_ops.full([1], float('80'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_283 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_284 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_94 = [slice_95, full_281, full_282, full_283, full_284]

        # pd_op.reshape_: (-1x2x80x8x6xf16, 0x-1x160x8x6xf16) <- (-1x160x8x6xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__124, reshape__125 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_31, [x.reshape([]) for x in combine_94]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x80x2x8x6xf16) <- (-1x2x80x8x6xf16)
        transpose_31 = paddle._C_ops.transpose(reshape__124, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_285 = paddle._C_ops.full([1], float('160'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_286 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_287 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_95 = [slice_95, full_285, full_286, full_287]

        # pd_op.reshape_: (-1x160x8x6xf16, 0x-1x80x2x8x6xf16) <- (-1x80x2x8x6xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__126, reshape__127 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_31, [x.reshape([]) for x in combine_95]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_288 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x80x8x6xf16, -1x80x8x6xf16]) <- (-1x160x8x6xf16, 1xi32)
        split_with_num_32 = paddle._C_ops.split_with_num(reshape__126, 2, full_288)

        # builtin.slice: (-1x80x8x6xf16) <- ([-1x80x8x6xf16, -1x80x8x6xf16])
        slice_96 = split_with_num_32[1]

        # pd_op.conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x80x1x1xf16)
        conv2d_94 = paddle._C_ops.conv2d(slice_96, parameter_720, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__864, batch_norm__865, batch_norm__866, batch_norm__867, batch_norm__868, batch_norm__869 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_94, parameter_721, parameter_722, parameter_723, parameter_724, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x80x8x6xf16) <- (-1x80x8x6xf16)
        relu__80 = paddle._C_ops.relu_(batch_norm__864)

        # pd_op.depthwise_conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x1x3x3xf16)
        depthwise_conv2d_50 = paddle._C_ops.depthwise_conv2d(relu__80, parameter_725, [1, 1], [1, 1], 'EXPLICIT', 80, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__870, batch_norm__871, batch_norm__872, batch_norm__873, batch_norm__874, batch_norm__875 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_50, parameter_726, parameter_727, parameter_728, parameter_729, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x80x1x1xf16)
        conv2d_95 = paddle._C_ops.conv2d(batch_norm__870, parameter_730, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__876, batch_norm__877, batch_norm__878, batch_norm__879, batch_norm__880, batch_norm__881 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_95, parameter_731, parameter_732, parameter_733, parameter_734, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x80x8x6xf16) <- (-1x80x8x6xf16)
        relu__81 = paddle._C_ops.relu_(batch_norm__876)

        # builtin.slice: (-1x80x8x6xf16) <- ([-1x80x8x6xf16, -1x80x8x6xf16])
        slice_97 = split_with_num_32[0]

        # builtin.combine: ([-1x80x8x6xf16, -1x80x8x6xf16]) <- (-1x80x8x6xf16, -1x80x8x6xf16)
        combine_96 = [slice_97, relu__81]

        # pd_op.full: (1xi32) <- ()
        full_289 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x160x8x6xf16) <- ([-1x80x8x6xf16, -1x80x8x6xf16], 1xi32)
        concat_32 = paddle._C_ops.concat(combine_96, full_289)

        # pd_op.shape: (4xi32) <- (-1x160x8x6xf16)
        shape_32 = paddle._C_ops.shape(paddle.cast(concat_32, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_64 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_65 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_98 = paddle._C_ops.slice(shape_32, [0], full_int_array_64, full_int_array_65, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_290 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_291 = paddle._C_ops.full([1], float('80'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_292 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_293 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_97 = [slice_98, full_290, full_291, full_292, full_293]

        # pd_op.reshape_: (-1x2x80x8x6xf16, 0x-1x160x8x6xf16) <- (-1x160x8x6xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__128, reshape__129 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_32, [x.reshape([]) for x in combine_97]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x80x2x8x6xf16) <- (-1x2x80x8x6xf16)
        transpose_32 = paddle._C_ops.transpose(reshape__128, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_294 = paddle._C_ops.full([1], float('160'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_295 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_296 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_98 = [slice_98, full_294, full_295, full_296]

        # pd_op.reshape_: (-1x160x8x6xf16, 0x-1x80x2x8x6xf16) <- (-1x80x2x8x6xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__130, reshape__131 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_32, [x.reshape([]) for x in combine_98]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x40x32x24xf16) <- (-1x40x32x24xf16, -1x40x32x24xf16)
        add__35 = paddle._C_ops.add_(reshape__114, reshape__114)

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x80x16x12xf16, 40x80x1x1xf16)
        conv2d_96 = paddle._C_ops.conv2d(reshape__122, parameter_735, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__882, batch_norm__883, batch_norm__884, batch_norm__885, batch_norm__886, batch_norm__887 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_96, parameter_736, parameter_737, parameter_738, parameter_739, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.nearest_interp: (-1x40x32x24xf16) <- (-1x40x16x12xf16, None, None, None)
        nearest_interp_11 = paddle._C_ops.nearest_interp(batch_norm__882, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 0)

        # pd_op.add_: (-1x40x32x24xf16) <- (-1x40x32x24xf16, -1x40x32x24xf16)
        add__36 = paddle._C_ops.add_(add__35, nearest_interp_11)

        # pd_op.conv2d: (-1x40x8x6xf16) <- (-1x160x8x6xf16, 40x160x1x1xf16)
        conv2d_97 = paddle._C_ops.conv2d(reshape__130, parameter_740, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x8x6xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x8x6xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__888, batch_norm__889, batch_norm__890, batch_norm__891, batch_norm__892, batch_norm__893 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_97, parameter_741, parameter_742, parameter_743, parameter_744, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.nearest_interp: (-1x40x32x24xf16) <- (-1x40x8x6xf16, None, None, None)
        nearest_interp_12 = paddle._C_ops.nearest_interp(batch_norm__888, None, None, None, 'NCHW', -1, -1, -1, [float('4'), float('4')], 'nearest', False, 0)

        # pd_op.add_: (-1x40x32x24xf16) <- (-1x40x32x24xf16, -1x40x32x24xf16)
        add__37 = paddle._C_ops.add_(add__36, nearest_interp_12)

        # pd_op.relu: (-1x40x32x24xf16) <- (-1x40x32x24xf16)
        relu_5 = paddle._C_ops.relu(add__37)

        # pd_op.depthwise_conv2d: (-1x40x16x12xf16) <- (-1x40x32x24xf16, 40x1x3x3xf16)
        depthwise_conv2d_51 = paddle._C_ops.depthwise_conv2d(add__37, parameter_745, [2, 2], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__894, batch_norm__895, batch_norm__896, batch_norm__897, batch_norm__898, batch_norm__899 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_51, parameter_746, parameter_747, parameter_748, parameter_749, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x80x16x12xf16) <- (-1x40x16x12xf16, 80x40x1x1xf16)
        conv2d_98 = paddle._C_ops.conv2d(batch_norm__894, parameter_750, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x16x12xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x16x12xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__900, batch_norm__901, batch_norm__902, batch_norm__903, batch_norm__904, batch_norm__905 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_98, parameter_751, parameter_752, parameter_753, parameter_754, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x80x16x12xf16) <- (-1x80x16x12xf16, -1x80x16x12xf16)
        add__38 = paddle._C_ops.add_(batch_norm__900, batch_norm__900)

        # pd_op.add_: (-1x80x16x12xf16) <- (-1x80x16x12xf16, -1x80x16x12xf16)
        add__39 = paddle._C_ops.add_(add__38, reshape__122)

        # pd_op.conv2d: (-1x80x8x6xf16) <- (-1x160x8x6xf16, 80x160x1x1xf16)
        conv2d_99 = paddle._C_ops.conv2d(reshape__130, parameter_755, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__906, batch_norm__907, batch_norm__908, batch_norm__909, batch_norm__910, batch_norm__911 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_99, parameter_756, parameter_757, parameter_758, parameter_759, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.nearest_interp: (-1x80x16x12xf16) <- (-1x80x8x6xf16, None, None, None)
        nearest_interp_13 = paddle._C_ops.nearest_interp(batch_norm__906, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 0)

        # pd_op.add_: (-1x80x16x12xf16) <- (-1x80x16x12xf16, -1x80x16x12xf16)
        add__40 = paddle._C_ops.add_(add__39, nearest_interp_13)

        # pd_op.relu_: (-1x80x16x12xf16) <- (-1x80x16x12xf16)
        relu__82 = paddle._C_ops.relu_(add__40)

        # pd_op.depthwise_conv2d: (-1x40x16x12xf16) <- (-1x40x32x24xf16, 40x1x3x3xf16)
        depthwise_conv2d_52 = paddle._C_ops.depthwise_conv2d(add__37, parameter_760, [2, 2], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__912, batch_norm__913, batch_norm__914, batch_norm__915, batch_norm__916, batch_norm__917 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_52, parameter_761, parameter_762, parameter_763, parameter_764, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_100 = paddle._C_ops.conv2d(batch_norm__912, parameter_765, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__918, batch_norm__919, batch_norm__920, batch_norm__921, batch_norm__922, batch_norm__923 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_100, parameter_766, parameter_767, parameter_768, parameter_769, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__83 = paddle._C_ops.relu_(batch_norm__918)

        # pd_op.depthwise_conv2d: (-1x40x8x6xf16) <- (-1x40x16x12xf16, 40x1x3x3xf16)
        depthwise_conv2d_53 = paddle._C_ops.depthwise_conv2d(relu__83, parameter_770, [2, 2], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x8x6xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x8x6xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__924, batch_norm__925, batch_norm__926, batch_norm__927, batch_norm__928, batch_norm__929 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_53, parameter_771, parameter_772, parameter_773, parameter_774, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x160x8x6xf16) <- (-1x40x8x6xf16, 160x40x1x1xf16)
        conv2d_101 = paddle._C_ops.conv2d(batch_norm__924, parameter_775, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x160x8x6xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x8x6xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__930, batch_norm__931, batch_norm__932, batch_norm__933, batch_norm__934, batch_norm__935 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_101, parameter_776, parameter_777, parameter_778, parameter_779, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x160x8x6xf16) <- (-1x160x8x6xf16, -1x160x8x6xf16)
        add__41 = paddle._C_ops.add_(batch_norm__930, batch_norm__930)

        # pd_op.depthwise_conv2d: (-1x80x8x6xf16) <- (-1x80x16x12xf16, 80x1x3x3xf16)
        depthwise_conv2d_54 = paddle._C_ops.depthwise_conv2d(reshape__122, parameter_780, [2, 2], [1, 1], 'EXPLICIT', 80, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__936, batch_norm__937, batch_norm__938, batch_norm__939, batch_norm__940, batch_norm__941 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_54, parameter_781, parameter_782, parameter_783, parameter_784, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x160x8x6xf16) <- (-1x80x8x6xf16, 160x80x1x1xf16)
        conv2d_102 = paddle._C_ops.conv2d(batch_norm__936, parameter_785, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x160x8x6xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x8x6xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__942, batch_norm__943, batch_norm__944, batch_norm__945, batch_norm__946, batch_norm__947 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_102, parameter_786, parameter_787, parameter_788, parameter_789, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x160x8x6xf16) <- (-1x160x8x6xf16, -1x160x8x6xf16)
        add__42 = paddle._C_ops.add_(add__41, batch_norm__942)

        # pd_op.add_: (-1x160x8x6xf16) <- (-1x160x8x6xf16, -1x160x8x6xf16)
        add__43 = paddle._C_ops.add_(add__42, reshape__130)

        # pd_op.relu_: (-1x160x8x6xf16) <- (-1x160x8x6xf16)
        relu__84 = paddle._C_ops.relu_(add__43)

        # pd_op.depthwise_conv2d: (-1x160x4x3xf16) <- (-1x160x8x6xf16, 160x1x3x3xf16)
        depthwise_conv2d_55 = paddle._C_ops.depthwise_conv2d(relu__84, parameter_790, [2, 2], [1, 1], 'EXPLICIT', 160, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x160x4x3xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x4x3xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__948, batch_norm__949, batch_norm__950, batch_norm__951, batch_norm__952, batch_norm__953 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_55, parameter_791, parameter_792, parameter_793, parameter_794, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x320x4x3xf16) <- (-1x160x4x3xf16, 320x160x1x1xf16)
        conv2d_103 = paddle._C_ops.conv2d(batch_norm__948, parameter_795, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x320x4x3xf16, 320xf32, 320xf32, xf32, xf32, None) <- (-1x320x4x3xf16, 320xf32, 320xf32, 320xf32, 320xf32)
        batch_norm__954, batch_norm__955, batch_norm__956, batch_norm__957, batch_norm__958, batch_norm__959 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_103, parameter_796, parameter_797, parameter_798, parameter_799, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x320x4x3xf16) <- (-1x320x4x3xf16)
        relu__85 = paddle._C_ops.relu_(batch_norm__954)

        # pd_op.full: (1xi32) <- ()
        full_297 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x20x32x24xf16, -1x20x32x24xf16]) <- (-1x40x32x24xf16, 1xi32)
        split_with_num_33 = paddle._C_ops.split_with_num(relu_5, 2, full_297)

        # builtin.slice: (-1x20x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16])
        slice_99 = split_with_num_33[1]

        # pd_op.conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x20x1x1xf16)
        conv2d_104 = paddle._C_ops.conv2d(slice_99, parameter_800, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__960, batch_norm__961, batch_norm__962, batch_norm__963, batch_norm__964, batch_norm__965 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_104, parameter_801, parameter_802, parameter_803, parameter_804, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x20x32x24xf16) <- (-1x20x32x24xf16)
        relu__86 = paddle._C_ops.relu_(batch_norm__960)

        # pd_op.depthwise_conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x1x3x3xf16)
        depthwise_conv2d_56 = paddle._C_ops.depthwise_conv2d(relu__86, parameter_805, [1, 1], [1, 1], 'EXPLICIT', 20, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__966, batch_norm__967, batch_norm__968, batch_norm__969, batch_norm__970, batch_norm__971 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_56, parameter_806, parameter_807, parameter_808, parameter_809, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x20x1x1xf16)
        conv2d_105 = paddle._C_ops.conv2d(batch_norm__966, parameter_810, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__972, batch_norm__973, batch_norm__974, batch_norm__975, batch_norm__976, batch_norm__977 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_105, parameter_811, parameter_812, parameter_813, parameter_814, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x20x32x24xf16) <- (-1x20x32x24xf16)
        relu__87 = paddle._C_ops.relu_(batch_norm__972)

        # builtin.slice: (-1x20x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16])
        slice_100 = split_with_num_33[0]

        # builtin.combine: ([-1x20x32x24xf16, -1x20x32x24xf16]) <- (-1x20x32x24xf16, -1x20x32x24xf16)
        combine_99 = [slice_100, relu__87]

        # pd_op.full: (1xi32) <- ()
        full_298 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x40x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16], 1xi32)
        concat_33 = paddle._C_ops.concat(combine_99, full_298)

        # pd_op.shape: (4xi32) <- (-1x40x32x24xf16)
        shape_33 = paddle._C_ops.shape(paddle.cast(concat_33, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_66 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_67 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_101 = paddle._C_ops.slice(shape_33, [0], full_int_array_66, full_int_array_67, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_299 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_300 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_301 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_302 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_100 = [slice_101, full_299, full_300, full_301, full_302]

        # pd_op.reshape_: (-1x2x20x32x24xf16, 0x-1x40x32x24xf16) <- (-1x40x32x24xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__132, reshape__133 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_33, [x.reshape([]) for x in combine_100]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x20x2x32x24xf16) <- (-1x2x20x32x24xf16)
        transpose_33 = paddle._C_ops.transpose(reshape__132, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_303 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_304 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_305 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_101 = [slice_101, full_303, full_304, full_305]

        # pd_op.reshape_: (-1x40x32x24xf16, 0x-1x20x2x32x24xf16) <- (-1x20x2x32x24xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__134, reshape__135 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_33, [x.reshape([]) for x in combine_101]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_306 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x20x32x24xf16, -1x20x32x24xf16]) <- (-1x40x32x24xf16, 1xi32)
        split_with_num_34 = paddle._C_ops.split_with_num(reshape__134, 2, full_306)

        # builtin.slice: (-1x20x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16])
        slice_102 = split_with_num_34[1]

        # pd_op.conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x20x1x1xf16)
        conv2d_106 = paddle._C_ops.conv2d(slice_102, parameter_815, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__978, batch_norm__979, batch_norm__980, batch_norm__981, batch_norm__982, batch_norm__983 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_106, parameter_816, parameter_817, parameter_818, parameter_819, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x20x32x24xf16) <- (-1x20x32x24xf16)
        relu__88 = paddle._C_ops.relu_(batch_norm__978)

        # pd_op.depthwise_conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x1x3x3xf16)
        depthwise_conv2d_57 = paddle._C_ops.depthwise_conv2d(relu__88, parameter_820, [1, 1], [1, 1], 'EXPLICIT', 20, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__984, batch_norm__985, batch_norm__986, batch_norm__987, batch_norm__988, batch_norm__989 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_57, parameter_821, parameter_822, parameter_823, parameter_824, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x20x1x1xf16)
        conv2d_107 = paddle._C_ops.conv2d(batch_norm__984, parameter_825, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__990, batch_norm__991, batch_norm__992, batch_norm__993, batch_norm__994, batch_norm__995 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_107, parameter_826, parameter_827, parameter_828, parameter_829, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x20x32x24xf16) <- (-1x20x32x24xf16)
        relu__89 = paddle._C_ops.relu_(batch_norm__990)

        # builtin.slice: (-1x20x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16])
        slice_103 = split_with_num_34[0]

        # builtin.combine: ([-1x20x32x24xf16, -1x20x32x24xf16]) <- (-1x20x32x24xf16, -1x20x32x24xf16)
        combine_102 = [slice_103, relu__89]

        # pd_op.full: (1xi32) <- ()
        full_307 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x40x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16], 1xi32)
        concat_34 = paddle._C_ops.concat(combine_102, full_307)

        # pd_op.shape: (4xi32) <- (-1x40x32x24xf16)
        shape_34 = paddle._C_ops.shape(paddle.cast(concat_34, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_68 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_69 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_104 = paddle._C_ops.slice(shape_34, [0], full_int_array_68, full_int_array_69, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_308 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_309 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_310 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_311 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_103 = [slice_104, full_308, full_309, full_310, full_311]

        # pd_op.reshape_: (-1x2x20x32x24xf16, 0x-1x40x32x24xf16) <- (-1x40x32x24xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__136, reshape__137 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_34, [x.reshape([]) for x in combine_103]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x20x2x32x24xf16) <- (-1x2x20x32x24xf16)
        transpose_34 = paddle._C_ops.transpose(reshape__136, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_312 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_313 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_314 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_104 = [slice_104, full_312, full_313, full_314]

        # pd_op.reshape_: (-1x40x32x24xf16, 0x-1x20x2x32x24xf16) <- (-1x20x2x32x24xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__138, reshape__139 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_34, [x.reshape([]) for x in combine_104]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_315 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x40x16x12xf16, -1x40x16x12xf16]) <- (-1x80x16x12xf16, 1xi32)
        split_with_num_35 = paddle._C_ops.split_with_num(relu__82, 2, full_315)

        # builtin.slice: (-1x40x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16])
        slice_105 = split_with_num_35[1]

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_108 = paddle._C_ops.conv2d(slice_105, parameter_830, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__996, batch_norm__997, batch_norm__998, batch_norm__999, batch_norm__1000, batch_norm__1001 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_108, parameter_831, parameter_832, parameter_833, parameter_834, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__90 = paddle._C_ops.relu_(batch_norm__996)

        # pd_op.depthwise_conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x1x3x3xf16)
        depthwise_conv2d_58 = paddle._C_ops.depthwise_conv2d(relu__90, parameter_835, [1, 1], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1002, batch_norm__1003, batch_norm__1004, batch_norm__1005, batch_norm__1006, batch_norm__1007 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_58, parameter_836, parameter_837, parameter_838, parameter_839, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_109 = paddle._C_ops.conv2d(batch_norm__1002, parameter_840, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1008, batch_norm__1009, batch_norm__1010, batch_norm__1011, batch_norm__1012, batch_norm__1013 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_109, parameter_841, parameter_842, parameter_843, parameter_844, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__91 = paddle._C_ops.relu_(batch_norm__1008)

        # builtin.slice: (-1x40x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16])
        slice_106 = split_with_num_35[0]

        # builtin.combine: ([-1x40x16x12xf16, -1x40x16x12xf16]) <- (-1x40x16x12xf16, -1x40x16x12xf16)
        combine_105 = [slice_106, relu__91]

        # pd_op.full: (1xi32) <- ()
        full_316 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x80x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16], 1xi32)
        concat_35 = paddle._C_ops.concat(combine_105, full_316)

        # pd_op.shape: (4xi32) <- (-1x80x16x12xf16)
        shape_35 = paddle._C_ops.shape(paddle.cast(concat_35, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_70 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_71 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_107 = paddle._C_ops.slice(shape_35, [0], full_int_array_70, full_int_array_71, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_317 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_318 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_319 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_320 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_106 = [slice_107, full_317, full_318, full_319, full_320]

        # pd_op.reshape_: (-1x2x40x16x12xf16, 0x-1x80x16x12xf16) <- (-1x80x16x12xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__140, reshape__141 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_35, [x.reshape([]) for x in combine_106]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x2x16x12xf16) <- (-1x2x40x16x12xf16)
        transpose_35 = paddle._C_ops.transpose(reshape__140, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_321 = paddle._C_ops.full([1], float('80'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_322 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_323 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_107 = [slice_107, full_321, full_322, full_323]

        # pd_op.reshape_: (-1x80x16x12xf16, 0x-1x40x2x16x12xf16) <- (-1x40x2x16x12xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__142, reshape__143 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_35, [x.reshape([]) for x in combine_107]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_324 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x40x16x12xf16, -1x40x16x12xf16]) <- (-1x80x16x12xf16, 1xi32)
        split_with_num_36 = paddle._C_ops.split_with_num(reshape__142, 2, full_324)

        # builtin.slice: (-1x40x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16])
        slice_108 = split_with_num_36[1]

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_110 = paddle._C_ops.conv2d(slice_108, parameter_845, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1014, batch_norm__1015, batch_norm__1016, batch_norm__1017, batch_norm__1018, batch_norm__1019 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_110, parameter_846, parameter_847, parameter_848, parameter_849, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__92 = paddle._C_ops.relu_(batch_norm__1014)

        # pd_op.depthwise_conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x1x3x3xf16)
        depthwise_conv2d_59 = paddle._C_ops.depthwise_conv2d(relu__92, parameter_850, [1, 1], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1020, batch_norm__1021, batch_norm__1022, batch_norm__1023, batch_norm__1024, batch_norm__1025 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_59, parameter_851, parameter_852, parameter_853, parameter_854, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_111 = paddle._C_ops.conv2d(batch_norm__1020, parameter_855, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1026, batch_norm__1027, batch_norm__1028, batch_norm__1029, batch_norm__1030, batch_norm__1031 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_111, parameter_856, parameter_857, parameter_858, parameter_859, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__93 = paddle._C_ops.relu_(batch_norm__1026)

        # builtin.slice: (-1x40x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16])
        slice_109 = split_with_num_36[0]

        # builtin.combine: ([-1x40x16x12xf16, -1x40x16x12xf16]) <- (-1x40x16x12xf16, -1x40x16x12xf16)
        combine_108 = [slice_109, relu__93]

        # pd_op.full: (1xi32) <- ()
        full_325 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x80x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16], 1xi32)
        concat_36 = paddle._C_ops.concat(combine_108, full_325)

        # pd_op.shape: (4xi32) <- (-1x80x16x12xf16)
        shape_36 = paddle._C_ops.shape(paddle.cast(concat_36, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_72 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_73 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_110 = paddle._C_ops.slice(shape_36, [0], full_int_array_72, full_int_array_73, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_326 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_327 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_328 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_329 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_109 = [slice_110, full_326, full_327, full_328, full_329]

        # pd_op.reshape_: (-1x2x40x16x12xf16, 0x-1x80x16x12xf16) <- (-1x80x16x12xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__144, reshape__145 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_36, [x.reshape([]) for x in combine_109]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x2x16x12xf16) <- (-1x2x40x16x12xf16)
        transpose_36 = paddle._C_ops.transpose(reshape__144, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_330 = paddle._C_ops.full([1], float('80'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_331 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_332 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_110 = [slice_110, full_330, full_331, full_332]

        # pd_op.reshape_: (-1x80x16x12xf16, 0x-1x40x2x16x12xf16) <- (-1x40x2x16x12xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__146, reshape__147 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_36, [x.reshape([]) for x in combine_110]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_333 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x80x8x6xf16, -1x80x8x6xf16]) <- (-1x160x8x6xf16, 1xi32)
        split_with_num_37 = paddle._C_ops.split_with_num(relu__84, 2, full_333)

        # builtin.slice: (-1x80x8x6xf16) <- ([-1x80x8x6xf16, -1x80x8x6xf16])
        slice_111 = split_with_num_37[1]

        # pd_op.conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x80x1x1xf16)
        conv2d_112 = paddle._C_ops.conv2d(slice_111, parameter_860, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__1032, batch_norm__1033, batch_norm__1034, batch_norm__1035, batch_norm__1036, batch_norm__1037 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_112, parameter_861, parameter_862, parameter_863, parameter_864, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x80x8x6xf16) <- (-1x80x8x6xf16)
        relu__94 = paddle._C_ops.relu_(batch_norm__1032)

        # pd_op.depthwise_conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x1x3x3xf16)
        depthwise_conv2d_60 = paddle._C_ops.depthwise_conv2d(relu__94, parameter_865, [1, 1], [1, 1], 'EXPLICIT', 80, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__1038, batch_norm__1039, batch_norm__1040, batch_norm__1041, batch_norm__1042, batch_norm__1043 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_60, parameter_866, parameter_867, parameter_868, parameter_869, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x80x1x1xf16)
        conv2d_113 = paddle._C_ops.conv2d(batch_norm__1038, parameter_870, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__1044, batch_norm__1045, batch_norm__1046, batch_norm__1047, batch_norm__1048, batch_norm__1049 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_113, parameter_871, parameter_872, parameter_873, parameter_874, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x80x8x6xf16) <- (-1x80x8x6xf16)
        relu__95 = paddle._C_ops.relu_(batch_norm__1044)

        # builtin.slice: (-1x80x8x6xf16) <- ([-1x80x8x6xf16, -1x80x8x6xf16])
        slice_112 = split_with_num_37[0]

        # builtin.combine: ([-1x80x8x6xf16, -1x80x8x6xf16]) <- (-1x80x8x6xf16, -1x80x8x6xf16)
        combine_111 = [slice_112, relu__95]

        # pd_op.full: (1xi32) <- ()
        full_334 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x160x8x6xf16) <- ([-1x80x8x6xf16, -1x80x8x6xf16], 1xi32)
        concat_37 = paddle._C_ops.concat(combine_111, full_334)

        # pd_op.shape: (4xi32) <- (-1x160x8x6xf16)
        shape_37 = paddle._C_ops.shape(paddle.cast(concat_37, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_74 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_75 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_113 = paddle._C_ops.slice(shape_37, [0], full_int_array_74, full_int_array_75, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_335 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_336 = paddle._C_ops.full([1], float('80'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_337 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_338 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_112 = [slice_113, full_335, full_336, full_337, full_338]

        # pd_op.reshape_: (-1x2x80x8x6xf16, 0x-1x160x8x6xf16) <- (-1x160x8x6xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__148, reshape__149 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_37, [x.reshape([]) for x in combine_112]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x80x2x8x6xf16) <- (-1x2x80x8x6xf16)
        transpose_37 = paddle._C_ops.transpose(reshape__148, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_339 = paddle._C_ops.full([1], float('160'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_340 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_341 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_113 = [slice_113, full_339, full_340, full_341]

        # pd_op.reshape_: (-1x160x8x6xf16, 0x-1x80x2x8x6xf16) <- (-1x80x2x8x6xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__150, reshape__151 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_37, [x.reshape([]) for x in combine_113]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_342 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x80x8x6xf16, -1x80x8x6xf16]) <- (-1x160x8x6xf16, 1xi32)
        split_with_num_38 = paddle._C_ops.split_with_num(reshape__150, 2, full_342)

        # builtin.slice: (-1x80x8x6xf16) <- ([-1x80x8x6xf16, -1x80x8x6xf16])
        slice_114 = split_with_num_38[1]

        # pd_op.conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x80x1x1xf16)
        conv2d_114 = paddle._C_ops.conv2d(slice_114, parameter_875, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__1050, batch_norm__1051, batch_norm__1052, batch_norm__1053, batch_norm__1054, batch_norm__1055 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_114, parameter_876, parameter_877, parameter_878, parameter_879, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x80x8x6xf16) <- (-1x80x8x6xf16)
        relu__96 = paddle._C_ops.relu_(batch_norm__1050)

        # pd_op.depthwise_conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x1x3x3xf16)
        depthwise_conv2d_61 = paddle._C_ops.depthwise_conv2d(relu__96, parameter_880, [1, 1], [1, 1], 'EXPLICIT', 80, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__1056, batch_norm__1057, batch_norm__1058, batch_norm__1059, batch_norm__1060, batch_norm__1061 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_61, parameter_881, parameter_882, parameter_883, parameter_884, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x80x1x1xf16)
        conv2d_115 = paddle._C_ops.conv2d(batch_norm__1056, parameter_885, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__1062, batch_norm__1063, batch_norm__1064, batch_norm__1065, batch_norm__1066, batch_norm__1067 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_115, parameter_886, parameter_887, parameter_888, parameter_889, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x80x8x6xf16) <- (-1x80x8x6xf16)
        relu__97 = paddle._C_ops.relu_(batch_norm__1062)

        # builtin.slice: (-1x80x8x6xf16) <- ([-1x80x8x6xf16, -1x80x8x6xf16])
        slice_115 = split_with_num_38[0]

        # builtin.combine: ([-1x80x8x6xf16, -1x80x8x6xf16]) <- (-1x80x8x6xf16, -1x80x8x6xf16)
        combine_114 = [slice_115, relu__97]

        # pd_op.full: (1xi32) <- ()
        full_343 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x160x8x6xf16) <- ([-1x80x8x6xf16, -1x80x8x6xf16], 1xi32)
        concat_38 = paddle._C_ops.concat(combine_114, full_343)

        # pd_op.shape: (4xi32) <- (-1x160x8x6xf16)
        shape_38 = paddle._C_ops.shape(paddle.cast(concat_38, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_76 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_77 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_116 = paddle._C_ops.slice(shape_38, [0], full_int_array_76, full_int_array_77, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_344 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_345 = paddle._C_ops.full([1], float('80'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_346 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_347 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_115 = [slice_116, full_344, full_345, full_346, full_347]

        # pd_op.reshape_: (-1x2x80x8x6xf16, 0x-1x160x8x6xf16) <- (-1x160x8x6xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__152, reshape__153 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_38, [x.reshape([]) for x in combine_115]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x80x2x8x6xf16) <- (-1x2x80x8x6xf16)
        transpose_38 = paddle._C_ops.transpose(reshape__152, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_348 = paddle._C_ops.full([1], float('160'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_349 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_350 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_116 = [slice_116, full_348, full_349, full_350]

        # pd_op.reshape_: (-1x160x8x6xf16, 0x-1x80x2x8x6xf16) <- (-1x80x2x8x6xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__154, reshape__155 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_38, [x.reshape([]) for x in combine_116]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_351 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x160x4x3xf16, -1x160x4x3xf16]) <- (-1x320x4x3xf16, 1xi32)
        split_with_num_39 = paddle._C_ops.split_with_num(relu__85, 2, full_351)

        # builtin.slice: (-1x160x4x3xf16) <- ([-1x160x4x3xf16, -1x160x4x3xf16])
        slice_117 = split_with_num_39[1]

        # pd_op.conv2d: (-1x160x4x3xf16) <- (-1x160x4x3xf16, 160x160x1x1xf16)
        conv2d_116 = paddle._C_ops.conv2d(slice_117, parameter_890, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x160x4x3xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x4x3xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__1068, batch_norm__1069, batch_norm__1070, batch_norm__1071, batch_norm__1072, batch_norm__1073 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_116, parameter_891, parameter_892, parameter_893, parameter_894, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x160x4x3xf16) <- (-1x160x4x3xf16)
        relu__98 = paddle._C_ops.relu_(batch_norm__1068)

        # pd_op.depthwise_conv2d: (-1x160x4x3xf16) <- (-1x160x4x3xf16, 160x1x3x3xf16)
        depthwise_conv2d_62 = paddle._C_ops.depthwise_conv2d(relu__98, parameter_895, [1, 1], [1, 1], 'EXPLICIT', 160, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x160x4x3xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x4x3xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__1074, batch_norm__1075, batch_norm__1076, batch_norm__1077, batch_norm__1078, batch_norm__1079 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_62, parameter_896, parameter_897, parameter_898, parameter_899, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x160x4x3xf16) <- (-1x160x4x3xf16, 160x160x1x1xf16)
        conv2d_117 = paddle._C_ops.conv2d(batch_norm__1074, parameter_900, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x160x4x3xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x4x3xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__1080, batch_norm__1081, batch_norm__1082, batch_norm__1083, batch_norm__1084, batch_norm__1085 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_117, parameter_901, parameter_902, parameter_903, parameter_904, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x160x4x3xf16) <- (-1x160x4x3xf16)
        relu__99 = paddle._C_ops.relu_(batch_norm__1080)

        # builtin.slice: (-1x160x4x3xf16) <- ([-1x160x4x3xf16, -1x160x4x3xf16])
        slice_118 = split_with_num_39[0]

        # builtin.combine: ([-1x160x4x3xf16, -1x160x4x3xf16]) <- (-1x160x4x3xf16, -1x160x4x3xf16)
        combine_117 = [slice_118, relu__99]

        # pd_op.full: (1xi32) <- ()
        full_352 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x320x4x3xf16) <- ([-1x160x4x3xf16, -1x160x4x3xf16], 1xi32)
        concat_39 = paddle._C_ops.concat(combine_117, full_352)

        # pd_op.shape: (4xi32) <- (-1x320x4x3xf16)
        shape_39 = paddle._C_ops.shape(paddle.cast(concat_39, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_78 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_79 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_119 = paddle._C_ops.slice(shape_39, [0], full_int_array_78, full_int_array_79, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_353 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_354 = paddle._C_ops.full([1], float('160'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_355 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_356 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_118 = [slice_119, full_353, full_354, full_355, full_356]

        # pd_op.reshape_: (-1x2x160x4x3xf16, 0x-1x320x4x3xf16) <- (-1x320x4x3xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__156, reshape__157 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_39, [x.reshape([]) for x in combine_118]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x160x2x4x3xf16) <- (-1x2x160x4x3xf16)
        transpose_39 = paddle._C_ops.transpose(reshape__156, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_357 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_358 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_359 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_119 = [slice_119, full_357, full_358, full_359]

        # pd_op.reshape_: (-1x320x4x3xf16, 0x-1x160x2x4x3xf16) <- (-1x160x2x4x3xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__158, reshape__159 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_39, [x.reshape([]) for x in combine_119]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_360 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x160x4x3xf16, -1x160x4x3xf16]) <- (-1x320x4x3xf16, 1xi32)
        split_with_num_40 = paddle._C_ops.split_with_num(reshape__158, 2, full_360)

        # builtin.slice: (-1x160x4x3xf16) <- ([-1x160x4x3xf16, -1x160x4x3xf16])
        slice_120 = split_with_num_40[1]

        # pd_op.conv2d: (-1x160x4x3xf16) <- (-1x160x4x3xf16, 160x160x1x1xf16)
        conv2d_118 = paddle._C_ops.conv2d(slice_120, parameter_905, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x160x4x3xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x4x3xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__1086, batch_norm__1087, batch_norm__1088, batch_norm__1089, batch_norm__1090, batch_norm__1091 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_118, parameter_906, parameter_907, parameter_908, parameter_909, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x160x4x3xf16) <- (-1x160x4x3xf16)
        relu__100 = paddle._C_ops.relu_(batch_norm__1086)

        # pd_op.depthwise_conv2d: (-1x160x4x3xf16) <- (-1x160x4x3xf16, 160x1x3x3xf16)
        depthwise_conv2d_63 = paddle._C_ops.depthwise_conv2d(relu__100, parameter_910, [1, 1], [1, 1], 'EXPLICIT', 160, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x160x4x3xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x4x3xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__1092, batch_norm__1093, batch_norm__1094, batch_norm__1095, batch_norm__1096, batch_norm__1097 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_63, parameter_911, parameter_912, parameter_913, parameter_914, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x160x4x3xf16) <- (-1x160x4x3xf16, 160x160x1x1xf16)
        conv2d_119 = paddle._C_ops.conv2d(batch_norm__1092, parameter_915, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x160x4x3xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x4x3xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__1098, batch_norm__1099, batch_norm__1100, batch_norm__1101, batch_norm__1102, batch_norm__1103 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_119, parameter_916, parameter_917, parameter_918, parameter_919, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x160x4x3xf16) <- (-1x160x4x3xf16)
        relu__101 = paddle._C_ops.relu_(batch_norm__1098)

        # builtin.slice: (-1x160x4x3xf16) <- ([-1x160x4x3xf16, -1x160x4x3xf16])
        slice_121 = split_with_num_40[0]

        # builtin.combine: ([-1x160x4x3xf16, -1x160x4x3xf16]) <- (-1x160x4x3xf16, -1x160x4x3xf16)
        combine_120 = [slice_121, relu__101]

        # pd_op.full: (1xi32) <- ()
        full_361 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x320x4x3xf16) <- ([-1x160x4x3xf16, -1x160x4x3xf16], 1xi32)
        concat_40 = paddle._C_ops.concat(combine_120, full_361)

        # pd_op.shape: (4xi32) <- (-1x320x4x3xf16)
        shape_40 = paddle._C_ops.shape(paddle.cast(concat_40, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_80 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_81 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_122 = paddle._C_ops.slice(shape_40, [0], full_int_array_80, full_int_array_81, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_362 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_363 = paddle._C_ops.full([1], float('160'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_364 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_365 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_121 = [slice_122, full_362, full_363, full_364, full_365]

        # pd_op.reshape_: (-1x2x160x4x3xf16, 0x-1x320x4x3xf16) <- (-1x320x4x3xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__160, reshape__161 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_40, [x.reshape([]) for x in combine_121]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x160x2x4x3xf16) <- (-1x2x160x4x3xf16)
        transpose_40 = paddle._C_ops.transpose(reshape__160, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_366 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_367 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_368 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_122 = [slice_122, full_366, full_367, full_368]

        # pd_op.reshape_: (-1x320x4x3xf16, 0x-1x160x2x4x3xf16) <- (-1x160x2x4x3xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__162, reshape__163 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_40, [x.reshape([]) for x in combine_122]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x40x32x24xf16) <- (-1x40x32x24xf16, -1x40x32x24xf16)
        add__44 = paddle._C_ops.add_(reshape__138, reshape__138)

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x80x16x12xf16, 40x80x1x1xf16)
        conv2d_120 = paddle._C_ops.conv2d(reshape__146, parameter_920, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1104, batch_norm__1105, batch_norm__1106, batch_norm__1107, batch_norm__1108, batch_norm__1109 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_120, parameter_921, parameter_922, parameter_923, parameter_924, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.nearest_interp: (-1x40x32x24xf16) <- (-1x40x16x12xf16, None, None, None)
        nearest_interp_14 = paddle._C_ops.nearest_interp(batch_norm__1104, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 0)

        # pd_op.add_: (-1x40x32x24xf16) <- (-1x40x32x24xf16, -1x40x32x24xf16)
        add__45 = paddle._C_ops.add_(add__44, nearest_interp_14)

        # pd_op.conv2d: (-1x40x8x6xf16) <- (-1x160x8x6xf16, 40x160x1x1xf16)
        conv2d_121 = paddle._C_ops.conv2d(reshape__154, parameter_925, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x8x6xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x8x6xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1110, batch_norm__1111, batch_norm__1112, batch_norm__1113, batch_norm__1114, batch_norm__1115 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_121, parameter_926, parameter_927, parameter_928, parameter_929, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.nearest_interp: (-1x40x32x24xf16) <- (-1x40x8x6xf16, None, None, None)
        nearest_interp_15 = paddle._C_ops.nearest_interp(batch_norm__1110, None, None, None, 'NCHW', -1, -1, -1, [float('4'), float('4')], 'nearest', False, 0)

        # pd_op.add_: (-1x40x32x24xf16) <- (-1x40x32x24xf16, -1x40x32x24xf16)
        add__46 = paddle._C_ops.add_(add__45, nearest_interp_15)

        # pd_op.conv2d: (-1x40x4x3xf16) <- (-1x320x4x3xf16, 40x320x1x1xf16)
        conv2d_122 = paddle._C_ops.conv2d(reshape__162, parameter_930, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x4x3xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x4x3xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1116, batch_norm__1117, batch_norm__1118, batch_norm__1119, batch_norm__1120, batch_norm__1121 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_122, parameter_931, parameter_932, parameter_933, parameter_934, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.nearest_interp: (-1x40x32x24xf16) <- (-1x40x4x3xf16, None, None, None)
        nearest_interp_16 = paddle._C_ops.nearest_interp(batch_norm__1116, None, None, None, 'NCHW', -1, -1, -1, [float('8'), float('8')], 'nearest', False, 0)

        # pd_op.add_: (-1x40x32x24xf16) <- (-1x40x32x24xf16, -1x40x32x24xf16)
        add__47 = paddle._C_ops.add_(add__46, nearest_interp_16)

        # pd_op.relu: (-1x40x32x24xf16) <- (-1x40x32x24xf16)
        relu_6 = paddle._C_ops.relu(add__47)

        # pd_op.depthwise_conv2d: (-1x40x16x12xf16) <- (-1x40x32x24xf16, 40x1x3x3xf16)
        depthwise_conv2d_64 = paddle._C_ops.depthwise_conv2d(add__47, parameter_935, [2, 2], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1122, batch_norm__1123, batch_norm__1124, batch_norm__1125, batch_norm__1126, batch_norm__1127 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_64, parameter_936, parameter_937, parameter_938, parameter_939, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x80x16x12xf16) <- (-1x40x16x12xf16, 80x40x1x1xf16)
        conv2d_123 = paddle._C_ops.conv2d(batch_norm__1122, parameter_940, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x16x12xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x16x12xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__1128, batch_norm__1129, batch_norm__1130, batch_norm__1131, batch_norm__1132, batch_norm__1133 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_123, parameter_941, parameter_942, parameter_943, parameter_944, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x80x16x12xf16) <- (-1x80x16x12xf16, -1x80x16x12xf16)
        add__48 = paddle._C_ops.add_(batch_norm__1128, batch_norm__1128)

        # pd_op.add_: (-1x80x16x12xf16) <- (-1x80x16x12xf16, -1x80x16x12xf16)
        add__49 = paddle._C_ops.add_(add__48, reshape__146)

        # pd_op.conv2d: (-1x80x8x6xf16) <- (-1x160x8x6xf16, 80x160x1x1xf16)
        conv2d_124 = paddle._C_ops.conv2d(reshape__154, parameter_945, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__1134, batch_norm__1135, batch_norm__1136, batch_norm__1137, batch_norm__1138, batch_norm__1139 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_124, parameter_946, parameter_947, parameter_948, parameter_949, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.nearest_interp: (-1x80x16x12xf16) <- (-1x80x8x6xf16, None, None, None)
        nearest_interp_17 = paddle._C_ops.nearest_interp(batch_norm__1134, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 0)

        # pd_op.add_: (-1x80x16x12xf16) <- (-1x80x16x12xf16, -1x80x16x12xf16)
        add__50 = paddle._C_ops.add_(add__49, nearest_interp_17)

        # pd_op.conv2d: (-1x80x4x3xf16) <- (-1x320x4x3xf16, 80x320x1x1xf16)
        conv2d_125 = paddle._C_ops.conv2d(reshape__162, parameter_950, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x4x3xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x4x3xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__1140, batch_norm__1141, batch_norm__1142, batch_norm__1143, batch_norm__1144, batch_norm__1145 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_125, parameter_951, parameter_952, parameter_953, parameter_954, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.nearest_interp: (-1x80x16x12xf16) <- (-1x80x4x3xf16, None, None, None)
        nearest_interp_18 = paddle._C_ops.nearest_interp(batch_norm__1140, None, None, None, 'NCHW', -1, -1, -1, [float('4'), float('4')], 'nearest', False, 0)

        # pd_op.add_: (-1x80x16x12xf16) <- (-1x80x16x12xf16, -1x80x16x12xf16)
        add__51 = paddle._C_ops.add_(add__50, nearest_interp_18)

        # pd_op.relu_: (-1x80x16x12xf16) <- (-1x80x16x12xf16)
        relu__102 = paddle._C_ops.relu_(add__51)

        # pd_op.depthwise_conv2d: (-1x40x16x12xf16) <- (-1x40x32x24xf16, 40x1x3x3xf16)
        depthwise_conv2d_65 = paddle._C_ops.depthwise_conv2d(add__47, parameter_955, [2, 2], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1146, batch_norm__1147, batch_norm__1148, batch_norm__1149, batch_norm__1150, batch_norm__1151 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_65, parameter_956, parameter_957, parameter_958, parameter_959, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_126 = paddle._C_ops.conv2d(batch_norm__1146, parameter_960, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1152, batch_norm__1153, batch_norm__1154, batch_norm__1155, batch_norm__1156, batch_norm__1157 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_126, parameter_961, parameter_962, parameter_963, parameter_964, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__103 = paddle._C_ops.relu_(batch_norm__1152)

        # pd_op.depthwise_conv2d: (-1x40x8x6xf16) <- (-1x40x16x12xf16, 40x1x3x3xf16)
        depthwise_conv2d_66 = paddle._C_ops.depthwise_conv2d(relu__103, parameter_965, [2, 2], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x8x6xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x8x6xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1158, batch_norm__1159, batch_norm__1160, batch_norm__1161, batch_norm__1162, batch_norm__1163 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_66, parameter_966, parameter_967, parameter_968, parameter_969, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x160x8x6xf16) <- (-1x40x8x6xf16, 160x40x1x1xf16)
        conv2d_127 = paddle._C_ops.conv2d(batch_norm__1158, parameter_970, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x160x8x6xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x8x6xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__1164, batch_norm__1165, batch_norm__1166, batch_norm__1167, batch_norm__1168, batch_norm__1169 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_127, parameter_971, parameter_972, parameter_973, parameter_974, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x160x8x6xf16) <- (-1x160x8x6xf16, -1x160x8x6xf16)
        add__52 = paddle._C_ops.add_(batch_norm__1164, batch_norm__1164)

        # pd_op.depthwise_conv2d: (-1x80x8x6xf16) <- (-1x80x16x12xf16, 80x1x3x3xf16)
        depthwise_conv2d_67 = paddle._C_ops.depthwise_conv2d(reshape__146, parameter_975, [2, 2], [1, 1], 'EXPLICIT', 80, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__1170, batch_norm__1171, batch_norm__1172, batch_norm__1173, batch_norm__1174, batch_norm__1175 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_67, parameter_976, parameter_977, parameter_978, parameter_979, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x160x8x6xf16) <- (-1x80x8x6xf16, 160x80x1x1xf16)
        conv2d_128 = paddle._C_ops.conv2d(batch_norm__1170, parameter_980, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x160x8x6xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x8x6xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__1176, batch_norm__1177, batch_norm__1178, batch_norm__1179, batch_norm__1180, batch_norm__1181 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_128, parameter_981, parameter_982, parameter_983, parameter_984, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x160x8x6xf16) <- (-1x160x8x6xf16, -1x160x8x6xf16)
        add__53 = paddle._C_ops.add_(add__52, batch_norm__1176)

        # pd_op.add_: (-1x160x8x6xf16) <- (-1x160x8x6xf16, -1x160x8x6xf16)
        add__54 = paddle._C_ops.add_(add__53, reshape__154)

        # pd_op.conv2d: (-1x160x4x3xf16) <- (-1x320x4x3xf16, 160x320x1x1xf16)
        conv2d_129 = paddle._C_ops.conv2d(reshape__162, parameter_985, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x160x4x3xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x4x3xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__1182, batch_norm__1183, batch_norm__1184, batch_norm__1185, batch_norm__1186, batch_norm__1187 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_129, parameter_986, parameter_987, parameter_988, parameter_989, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.nearest_interp: (-1x160x8x6xf16) <- (-1x160x4x3xf16, None, None, None)
        nearest_interp_19 = paddle._C_ops.nearest_interp(batch_norm__1182, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 0)

        # pd_op.add_: (-1x160x8x6xf16) <- (-1x160x8x6xf16, -1x160x8x6xf16)
        add__55 = paddle._C_ops.add_(add__54, nearest_interp_19)

        # pd_op.relu_: (-1x160x8x6xf16) <- (-1x160x8x6xf16)
        relu__104 = paddle._C_ops.relu_(add__55)

        # pd_op.depthwise_conv2d: (-1x40x16x12xf16) <- (-1x40x32x24xf16, 40x1x3x3xf16)
        depthwise_conv2d_68 = paddle._C_ops.depthwise_conv2d(add__47, parameter_990, [2, 2], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1188, batch_norm__1189, batch_norm__1190, batch_norm__1191, batch_norm__1192, batch_norm__1193 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_68, parameter_991, parameter_992, parameter_993, parameter_994, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_130 = paddle._C_ops.conv2d(batch_norm__1188, parameter_995, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1194, batch_norm__1195, batch_norm__1196, batch_norm__1197, batch_norm__1198, batch_norm__1199 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_130, parameter_996, parameter_997, parameter_998, parameter_999, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__105 = paddle._C_ops.relu_(batch_norm__1194)

        # pd_op.depthwise_conv2d: (-1x40x8x6xf16) <- (-1x40x16x12xf16, 40x1x3x3xf16)
        depthwise_conv2d_69 = paddle._C_ops.depthwise_conv2d(relu__105, parameter_1000, [2, 2], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x8x6xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x8x6xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1200, batch_norm__1201, batch_norm__1202, batch_norm__1203, batch_norm__1204, batch_norm__1205 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_69, parameter_1001, parameter_1002, parameter_1003, parameter_1004, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x40x8x6xf16) <- (-1x40x8x6xf16, 40x40x1x1xf16)
        conv2d_131 = paddle._C_ops.conv2d(batch_norm__1200, parameter_1005, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x8x6xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x8x6xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1206, batch_norm__1207, batch_norm__1208, batch_norm__1209, batch_norm__1210, batch_norm__1211 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_131, parameter_1006, parameter_1007, parameter_1008, parameter_1009, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x8x6xf16) <- (-1x40x8x6xf16)
        relu__106 = paddle._C_ops.relu_(batch_norm__1206)

        # pd_op.depthwise_conv2d: (-1x40x4x3xf16) <- (-1x40x8x6xf16, 40x1x3x3xf16)
        depthwise_conv2d_70 = paddle._C_ops.depthwise_conv2d(relu__106, parameter_1010, [2, 2], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x4x3xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x4x3xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1212, batch_norm__1213, batch_norm__1214, batch_norm__1215, batch_norm__1216, batch_norm__1217 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_70, parameter_1011, parameter_1012, parameter_1013, parameter_1014, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x320x4x3xf16) <- (-1x40x4x3xf16, 320x40x1x1xf16)
        conv2d_132 = paddle._C_ops.conv2d(batch_norm__1212, parameter_1015, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x320x4x3xf16, 320xf32, 320xf32, xf32, xf32, None) <- (-1x320x4x3xf16, 320xf32, 320xf32, 320xf32, 320xf32)
        batch_norm__1218, batch_norm__1219, batch_norm__1220, batch_norm__1221, batch_norm__1222, batch_norm__1223 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_132, parameter_1016, parameter_1017, parameter_1018, parameter_1019, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x320x4x3xf16) <- (-1x320x4x3xf16, -1x320x4x3xf16)
        add__56 = paddle._C_ops.add_(batch_norm__1218, batch_norm__1218)

        # pd_op.depthwise_conv2d: (-1x80x8x6xf16) <- (-1x80x16x12xf16, 80x1x3x3xf16)
        depthwise_conv2d_71 = paddle._C_ops.depthwise_conv2d(reshape__146, parameter_1020, [2, 2], [1, 1], 'EXPLICIT', 80, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__1224, batch_norm__1225, batch_norm__1226, batch_norm__1227, batch_norm__1228, batch_norm__1229 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_71, parameter_1021, parameter_1022, parameter_1023, parameter_1024, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x80x1x1xf16)
        conv2d_133 = paddle._C_ops.conv2d(batch_norm__1224, parameter_1025, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__1230, batch_norm__1231, batch_norm__1232, batch_norm__1233, batch_norm__1234, batch_norm__1235 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_133, parameter_1026, parameter_1027, parameter_1028, parameter_1029, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x80x8x6xf16) <- (-1x80x8x6xf16)
        relu__107 = paddle._C_ops.relu_(batch_norm__1230)

        # pd_op.depthwise_conv2d: (-1x80x4x3xf16) <- (-1x80x8x6xf16, 80x1x3x3xf16)
        depthwise_conv2d_72 = paddle._C_ops.depthwise_conv2d(relu__107, parameter_1030, [2, 2], [1, 1], 'EXPLICIT', 80, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x80x4x3xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x4x3xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__1236, batch_norm__1237, batch_norm__1238, batch_norm__1239, batch_norm__1240, batch_norm__1241 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_72, parameter_1031, parameter_1032, parameter_1033, parameter_1034, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x320x4x3xf16) <- (-1x80x4x3xf16, 320x80x1x1xf16)
        conv2d_134 = paddle._C_ops.conv2d(batch_norm__1236, parameter_1035, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x320x4x3xf16, 320xf32, 320xf32, xf32, xf32, None) <- (-1x320x4x3xf16, 320xf32, 320xf32, 320xf32, 320xf32)
        batch_norm__1242, batch_norm__1243, batch_norm__1244, batch_norm__1245, batch_norm__1246, batch_norm__1247 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_134, parameter_1036, parameter_1037, parameter_1038, parameter_1039, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x320x4x3xf16) <- (-1x320x4x3xf16, -1x320x4x3xf16)
        add__57 = paddle._C_ops.add_(add__56, batch_norm__1242)

        # pd_op.depthwise_conv2d: (-1x160x4x3xf16) <- (-1x160x8x6xf16, 160x1x3x3xf16)
        depthwise_conv2d_73 = paddle._C_ops.depthwise_conv2d(reshape__154, parameter_1040, [2, 2], [1, 1], 'EXPLICIT', 160, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x160x4x3xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x4x3xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__1248, batch_norm__1249, batch_norm__1250, batch_norm__1251, batch_norm__1252, batch_norm__1253 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_73, parameter_1041, parameter_1042, parameter_1043, parameter_1044, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x320x4x3xf16) <- (-1x160x4x3xf16, 320x160x1x1xf16)
        conv2d_135 = paddle._C_ops.conv2d(batch_norm__1248, parameter_1045, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x320x4x3xf16, 320xf32, 320xf32, xf32, xf32, None) <- (-1x320x4x3xf16, 320xf32, 320xf32, 320xf32, 320xf32)
        batch_norm__1254, batch_norm__1255, batch_norm__1256, batch_norm__1257, batch_norm__1258, batch_norm__1259 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_135, parameter_1046, parameter_1047, parameter_1048, parameter_1049, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x320x4x3xf16) <- (-1x320x4x3xf16, -1x320x4x3xf16)
        add__58 = paddle._C_ops.add_(add__57, batch_norm__1254)

        # pd_op.add_: (-1x320x4x3xf16) <- (-1x320x4x3xf16, -1x320x4x3xf16)
        add__59 = paddle._C_ops.add_(add__58, reshape__162)

        # pd_op.relu_: (-1x320x4x3xf16) <- (-1x320x4x3xf16)
        relu__108 = paddle._C_ops.relu_(add__59)

        # pd_op.full: (1xi32) <- ()
        full_369 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x20x32x24xf16, -1x20x32x24xf16]) <- (-1x40x32x24xf16, 1xi32)
        split_with_num_41 = paddle._C_ops.split_with_num(relu_6, 2, full_369)

        # builtin.slice: (-1x20x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16])
        slice_123 = split_with_num_41[1]

        # pd_op.conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x20x1x1xf16)
        conv2d_136 = paddle._C_ops.conv2d(slice_123, parameter_1050, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__1260, batch_norm__1261, batch_norm__1262, batch_norm__1263, batch_norm__1264, batch_norm__1265 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_136, parameter_1051, parameter_1052, parameter_1053, parameter_1054, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x20x32x24xf16) <- (-1x20x32x24xf16)
        relu__109 = paddle._C_ops.relu_(batch_norm__1260)

        # pd_op.depthwise_conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x1x3x3xf16)
        depthwise_conv2d_74 = paddle._C_ops.depthwise_conv2d(relu__109, parameter_1055, [1, 1], [1, 1], 'EXPLICIT', 20, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__1266, batch_norm__1267, batch_norm__1268, batch_norm__1269, batch_norm__1270, batch_norm__1271 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_74, parameter_1056, parameter_1057, parameter_1058, parameter_1059, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x20x1x1xf16)
        conv2d_137 = paddle._C_ops.conv2d(batch_norm__1266, parameter_1060, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__1272, batch_norm__1273, batch_norm__1274, batch_norm__1275, batch_norm__1276, batch_norm__1277 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_137, parameter_1061, parameter_1062, parameter_1063, parameter_1064, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x20x32x24xf16) <- (-1x20x32x24xf16)
        relu__110 = paddle._C_ops.relu_(batch_norm__1272)

        # builtin.slice: (-1x20x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16])
        slice_124 = split_with_num_41[0]

        # builtin.combine: ([-1x20x32x24xf16, -1x20x32x24xf16]) <- (-1x20x32x24xf16, -1x20x32x24xf16)
        combine_123 = [slice_124, relu__110]

        # pd_op.full: (1xi32) <- ()
        full_370 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x40x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16], 1xi32)
        concat_41 = paddle._C_ops.concat(combine_123, full_370)

        # pd_op.shape: (4xi32) <- (-1x40x32x24xf16)
        shape_41 = paddle._C_ops.shape(paddle.cast(concat_41, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_82 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_83 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_125 = paddle._C_ops.slice(shape_41, [0], full_int_array_82, full_int_array_83, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_371 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_372 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_373 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_374 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_124 = [slice_125, full_371, full_372, full_373, full_374]

        # pd_op.reshape_: (-1x2x20x32x24xf16, 0x-1x40x32x24xf16) <- (-1x40x32x24xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__164, reshape__165 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_41, [x.reshape([]) for x in combine_124]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x20x2x32x24xf16) <- (-1x2x20x32x24xf16)
        transpose_41 = paddle._C_ops.transpose(reshape__164, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_375 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_376 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_377 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_125 = [slice_125, full_375, full_376, full_377]

        # pd_op.reshape_: (-1x40x32x24xf16, 0x-1x20x2x32x24xf16) <- (-1x20x2x32x24xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__166, reshape__167 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_41, [x.reshape([]) for x in combine_125]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_378 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x20x32x24xf16, -1x20x32x24xf16]) <- (-1x40x32x24xf16, 1xi32)
        split_with_num_42 = paddle._C_ops.split_with_num(reshape__166, 2, full_378)

        # builtin.slice: (-1x20x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16])
        slice_126 = split_with_num_42[1]

        # pd_op.conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x20x1x1xf16)
        conv2d_138 = paddle._C_ops.conv2d(slice_126, parameter_1065, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__1278, batch_norm__1279, batch_norm__1280, batch_norm__1281, batch_norm__1282, batch_norm__1283 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_138, parameter_1066, parameter_1067, parameter_1068, parameter_1069, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x20x32x24xf16) <- (-1x20x32x24xf16)
        relu__111 = paddle._C_ops.relu_(batch_norm__1278)

        # pd_op.depthwise_conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x1x3x3xf16)
        depthwise_conv2d_75 = paddle._C_ops.depthwise_conv2d(relu__111, parameter_1070, [1, 1], [1, 1], 'EXPLICIT', 20, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__1284, batch_norm__1285, batch_norm__1286, batch_norm__1287, batch_norm__1288, batch_norm__1289 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_75, parameter_1071, parameter_1072, parameter_1073, parameter_1074, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x20x32x24xf16) <- (-1x20x32x24xf16, 20x20x1x1xf16)
        conv2d_139 = paddle._C_ops.conv2d(batch_norm__1284, parameter_1075, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x20x32x24xf16, 20xf32, 20xf32, xf32, xf32, None) <- (-1x20x32x24xf16, 20xf32, 20xf32, 20xf32, 20xf32)
        batch_norm__1290, batch_norm__1291, batch_norm__1292, batch_norm__1293, batch_norm__1294, batch_norm__1295 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_139, parameter_1076, parameter_1077, parameter_1078, parameter_1079, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x20x32x24xf16) <- (-1x20x32x24xf16)
        relu__112 = paddle._C_ops.relu_(batch_norm__1290)

        # builtin.slice: (-1x20x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16])
        slice_127 = split_with_num_42[0]

        # builtin.combine: ([-1x20x32x24xf16, -1x20x32x24xf16]) <- (-1x20x32x24xf16, -1x20x32x24xf16)
        combine_126 = [slice_127, relu__112]

        # pd_op.full: (1xi32) <- ()
        full_379 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x40x32x24xf16) <- ([-1x20x32x24xf16, -1x20x32x24xf16], 1xi32)
        concat_42 = paddle._C_ops.concat(combine_126, full_379)

        # pd_op.shape: (4xi32) <- (-1x40x32x24xf16)
        shape_42 = paddle._C_ops.shape(paddle.cast(concat_42, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_84 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_85 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_128 = paddle._C_ops.slice(shape_42, [0], full_int_array_84, full_int_array_85, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_380 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_381 = paddle._C_ops.full([1], float('20'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_382 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_383 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_127 = [slice_128, full_380, full_381, full_382, full_383]

        # pd_op.reshape_: (-1x2x20x32x24xf16, 0x-1x40x32x24xf16) <- (-1x40x32x24xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__168, reshape__169 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_42, [x.reshape([]) for x in combine_127]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x20x2x32x24xf16) <- (-1x2x20x32x24xf16)
        transpose_42 = paddle._C_ops.transpose(reshape__168, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_384 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_385 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_386 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_128 = [slice_128, full_384, full_385, full_386]

        # pd_op.reshape_: (-1x40x32x24xf16, 0x-1x20x2x32x24xf16) <- (-1x20x2x32x24xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__170, reshape__171 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_42, [x.reshape([]) for x in combine_128]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_387 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x40x16x12xf16, -1x40x16x12xf16]) <- (-1x80x16x12xf16, 1xi32)
        split_with_num_43 = paddle._C_ops.split_with_num(relu__102, 2, full_387)

        # builtin.slice: (-1x40x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16])
        slice_129 = split_with_num_43[1]

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_140 = paddle._C_ops.conv2d(slice_129, parameter_1080, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1296, batch_norm__1297, batch_norm__1298, batch_norm__1299, batch_norm__1300, batch_norm__1301 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_140, parameter_1081, parameter_1082, parameter_1083, parameter_1084, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__113 = paddle._C_ops.relu_(batch_norm__1296)

        # pd_op.depthwise_conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x1x3x3xf16)
        depthwise_conv2d_76 = paddle._C_ops.depthwise_conv2d(relu__113, parameter_1085, [1, 1], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1302, batch_norm__1303, batch_norm__1304, batch_norm__1305, batch_norm__1306, batch_norm__1307 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_76, parameter_1086, parameter_1087, parameter_1088, parameter_1089, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_141 = paddle._C_ops.conv2d(batch_norm__1302, parameter_1090, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1308, batch_norm__1309, batch_norm__1310, batch_norm__1311, batch_norm__1312, batch_norm__1313 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_141, parameter_1091, parameter_1092, parameter_1093, parameter_1094, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__114 = paddle._C_ops.relu_(batch_norm__1308)

        # builtin.slice: (-1x40x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16])
        slice_130 = split_with_num_43[0]

        # builtin.combine: ([-1x40x16x12xf16, -1x40x16x12xf16]) <- (-1x40x16x12xf16, -1x40x16x12xf16)
        combine_129 = [slice_130, relu__114]

        # pd_op.full: (1xi32) <- ()
        full_388 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x80x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16], 1xi32)
        concat_43 = paddle._C_ops.concat(combine_129, full_388)

        # pd_op.shape: (4xi32) <- (-1x80x16x12xf16)
        shape_43 = paddle._C_ops.shape(paddle.cast(concat_43, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_86 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_87 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_131 = paddle._C_ops.slice(shape_43, [0], full_int_array_86, full_int_array_87, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_389 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_390 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_391 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_392 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_130 = [slice_131, full_389, full_390, full_391, full_392]

        # pd_op.reshape_: (-1x2x40x16x12xf16, 0x-1x80x16x12xf16) <- (-1x80x16x12xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__172, reshape__173 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_43, [x.reshape([]) for x in combine_130]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x2x16x12xf16) <- (-1x2x40x16x12xf16)
        transpose_43 = paddle._C_ops.transpose(reshape__172, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_393 = paddle._C_ops.full([1], float('80'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_394 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_395 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_131 = [slice_131, full_393, full_394, full_395]

        # pd_op.reshape_: (-1x80x16x12xf16, 0x-1x40x2x16x12xf16) <- (-1x40x2x16x12xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__174, reshape__175 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_43, [x.reshape([]) for x in combine_131]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_396 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x40x16x12xf16, -1x40x16x12xf16]) <- (-1x80x16x12xf16, 1xi32)
        split_with_num_44 = paddle._C_ops.split_with_num(reshape__174, 2, full_396)

        # builtin.slice: (-1x40x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16])
        slice_132 = split_with_num_44[1]

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_142 = paddle._C_ops.conv2d(slice_132, parameter_1095, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1314, batch_norm__1315, batch_norm__1316, batch_norm__1317, batch_norm__1318, batch_norm__1319 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_142, parameter_1096, parameter_1097, parameter_1098, parameter_1099, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__115 = paddle._C_ops.relu_(batch_norm__1314)

        # pd_op.depthwise_conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x1x3x3xf16)
        depthwise_conv2d_77 = paddle._C_ops.depthwise_conv2d(relu__115, parameter_1100, [1, 1], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1320, batch_norm__1321, batch_norm__1322, batch_norm__1323, batch_norm__1324, batch_norm__1325 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_77, parameter_1101, parameter_1102, parameter_1103, parameter_1104, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_143 = paddle._C_ops.conv2d(batch_norm__1320, parameter_1105, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1326, batch_norm__1327, batch_norm__1328, batch_norm__1329, batch_norm__1330, batch_norm__1331 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_143, parameter_1106, parameter_1107, parameter_1108, parameter_1109, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__116 = paddle._C_ops.relu_(batch_norm__1326)

        # builtin.slice: (-1x40x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16])
        slice_133 = split_with_num_44[0]

        # builtin.combine: ([-1x40x16x12xf16, -1x40x16x12xf16]) <- (-1x40x16x12xf16, -1x40x16x12xf16)
        combine_132 = [slice_133, relu__116]

        # pd_op.full: (1xi32) <- ()
        full_397 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x80x16x12xf16) <- ([-1x40x16x12xf16, -1x40x16x12xf16], 1xi32)
        concat_44 = paddle._C_ops.concat(combine_132, full_397)

        # pd_op.shape: (4xi32) <- (-1x80x16x12xf16)
        shape_44 = paddle._C_ops.shape(paddle.cast(concat_44, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_88 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_89 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_134 = paddle._C_ops.slice(shape_44, [0], full_int_array_88, full_int_array_89, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_398 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_399 = paddle._C_ops.full([1], float('40'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_400 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_401 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_133 = [slice_134, full_398, full_399, full_400, full_401]

        # pd_op.reshape_: (-1x2x40x16x12xf16, 0x-1x80x16x12xf16) <- (-1x80x16x12xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__176, reshape__177 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_44, [x.reshape([]) for x in combine_133]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x40x2x16x12xf16) <- (-1x2x40x16x12xf16)
        transpose_44 = paddle._C_ops.transpose(reshape__176, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_402 = paddle._C_ops.full([1], float('80'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_403 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_404 = paddle._C_ops.full([1], float('12'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_134 = [slice_134, full_402, full_403, full_404]

        # pd_op.reshape_: (-1x80x16x12xf16, 0x-1x40x2x16x12xf16) <- (-1x40x2x16x12xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__178, reshape__179 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_44, [x.reshape([]) for x in combine_134]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_405 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x80x8x6xf16, -1x80x8x6xf16]) <- (-1x160x8x6xf16, 1xi32)
        split_with_num_45 = paddle._C_ops.split_with_num(relu__104, 2, full_405)

        # builtin.slice: (-1x80x8x6xf16) <- ([-1x80x8x6xf16, -1x80x8x6xf16])
        slice_135 = split_with_num_45[1]

        # pd_op.conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x80x1x1xf16)
        conv2d_144 = paddle._C_ops.conv2d(slice_135, parameter_1110, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__1332, batch_norm__1333, batch_norm__1334, batch_norm__1335, batch_norm__1336, batch_norm__1337 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_144, parameter_1111, parameter_1112, parameter_1113, parameter_1114, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x80x8x6xf16) <- (-1x80x8x6xf16)
        relu__117 = paddle._C_ops.relu_(batch_norm__1332)

        # pd_op.depthwise_conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x1x3x3xf16)
        depthwise_conv2d_78 = paddle._C_ops.depthwise_conv2d(relu__117, parameter_1115, [1, 1], [1, 1], 'EXPLICIT', 80, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__1338, batch_norm__1339, batch_norm__1340, batch_norm__1341, batch_norm__1342, batch_norm__1343 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_78, parameter_1116, parameter_1117, parameter_1118, parameter_1119, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x80x1x1xf16)
        conv2d_145 = paddle._C_ops.conv2d(batch_norm__1338, parameter_1120, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__1344, batch_norm__1345, batch_norm__1346, batch_norm__1347, batch_norm__1348, batch_norm__1349 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_145, parameter_1121, parameter_1122, parameter_1123, parameter_1124, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x80x8x6xf16) <- (-1x80x8x6xf16)
        relu__118 = paddle._C_ops.relu_(batch_norm__1344)

        # builtin.slice: (-1x80x8x6xf16) <- ([-1x80x8x6xf16, -1x80x8x6xf16])
        slice_136 = split_with_num_45[0]

        # builtin.combine: ([-1x80x8x6xf16, -1x80x8x6xf16]) <- (-1x80x8x6xf16, -1x80x8x6xf16)
        combine_135 = [slice_136, relu__118]

        # pd_op.full: (1xi32) <- ()
        full_406 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x160x8x6xf16) <- ([-1x80x8x6xf16, -1x80x8x6xf16], 1xi32)
        concat_45 = paddle._C_ops.concat(combine_135, full_406)

        # pd_op.shape: (4xi32) <- (-1x160x8x6xf16)
        shape_45 = paddle._C_ops.shape(paddle.cast(concat_45, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_90 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_91 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_137 = paddle._C_ops.slice(shape_45, [0], full_int_array_90, full_int_array_91, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_407 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_408 = paddle._C_ops.full([1], float('80'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_409 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_410 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_136 = [slice_137, full_407, full_408, full_409, full_410]

        # pd_op.reshape_: (-1x2x80x8x6xf16, 0x-1x160x8x6xf16) <- (-1x160x8x6xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__180, reshape__181 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_45, [x.reshape([]) for x in combine_136]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x80x2x8x6xf16) <- (-1x2x80x8x6xf16)
        transpose_45 = paddle._C_ops.transpose(reshape__180, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_411 = paddle._C_ops.full([1], float('160'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_412 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_413 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_137 = [slice_137, full_411, full_412, full_413]

        # pd_op.reshape_: (-1x160x8x6xf16, 0x-1x80x2x8x6xf16) <- (-1x80x2x8x6xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__182, reshape__183 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_45, [x.reshape([]) for x in combine_137]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_414 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x80x8x6xf16, -1x80x8x6xf16]) <- (-1x160x8x6xf16, 1xi32)
        split_with_num_46 = paddle._C_ops.split_with_num(reshape__182, 2, full_414)

        # builtin.slice: (-1x80x8x6xf16) <- ([-1x80x8x6xf16, -1x80x8x6xf16])
        slice_138 = split_with_num_46[1]

        # pd_op.conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x80x1x1xf16)
        conv2d_146 = paddle._C_ops.conv2d(slice_138, parameter_1125, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__1350, batch_norm__1351, batch_norm__1352, batch_norm__1353, batch_norm__1354, batch_norm__1355 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_146, parameter_1126, parameter_1127, parameter_1128, parameter_1129, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x80x8x6xf16) <- (-1x80x8x6xf16)
        relu__119 = paddle._C_ops.relu_(batch_norm__1350)

        # pd_op.depthwise_conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x1x3x3xf16)
        depthwise_conv2d_79 = paddle._C_ops.depthwise_conv2d(relu__119, parameter_1130, [1, 1], [1, 1], 'EXPLICIT', 80, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__1356, batch_norm__1357, batch_norm__1358, batch_norm__1359, batch_norm__1360, batch_norm__1361 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_79, parameter_1131, parameter_1132, parameter_1133, parameter_1134, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x80x1x1xf16)
        conv2d_147 = paddle._C_ops.conv2d(batch_norm__1356, parameter_1135, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__1362, batch_norm__1363, batch_norm__1364, batch_norm__1365, batch_norm__1366, batch_norm__1367 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_147, parameter_1136, parameter_1137, parameter_1138, parameter_1139, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x80x8x6xf16) <- (-1x80x8x6xf16)
        relu__120 = paddle._C_ops.relu_(batch_norm__1362)

        # builtin.slice: (-1x80x8x6xf16) <- ([-1x80x8x6xf16, -1x80x8x6xf16])
        slice_139 = split_with_num_46[0]

        # builtin.combine: ([-1x80x8x6xf16, -1x80x8x6xf16]) <- (-1x80x8x6xf16, -1x80x8x6xf16)
        combine_138 = [slice_139, relu__120]

        # pd_op.full: (1xi32) <- ()
        full_415 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x160x8x6xf16) <- ([-1x80x8x6xf16, -1x80x8x6xf16], 1xi32)
        concat_46 = paddle._C_ops.concat(combine_138, full_415)

        # pd_op.shape: (4xi32) <- (-1x160x8x6xf16)
        shape_46 = paddle._C_ops.shape(paddle.cast(concat_46, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_92 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_93 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_140 = paddle._C_ops.slice(shape_46, [0], full_int_array_92, full_int_array_93, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_416 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_417 = paddle._C_ops.full([1], float('80'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_418 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_419 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_139 = [slice_140, full_416, full_417, full_418, full_419]

        # pd_op.reshape_: (-1x2x80x8x6xf16, 0x-1x160x8x6xf16) <- (-1x160x8x6xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__184, reshape__185 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_46, [x.reshape([]) for x in combine_139]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x80x2x8x6xf16) <- (-1x2x80x8x6xf16)
        transpose_46 = paddle._C_ops.transpose(reshape__184, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_420 = paddle._C_ops.full([1], float('160'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_421 = paddle._C_ops.full([1], float('8'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_422 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_140 = [slice_140, full_420, full_421, full_422]

        # pd_op.reshape_: (-1x160x8x6xf16, 0x-1x80x2x8x6xf16) <- (-1x80x2x8x6xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__186, reshape__187 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_46, [x.reshape([]) for x in combine_140]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_423 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x160x4x3xf16, -1x160x4x3xf16]) <- (-1x320x4x3xf16, 1xi32)
        split_with_num_47 = paddle._C_ops.split_with_num(relu__108, 2, full_423)

        # builtin.slice: (-1x160x4x3xf16) <- ([-1x160x4x3xf16, -1x160x4x3xf16])
        slice_141 = split_with_num_47[1]

        # pd_op.conv2d: (-1x160x4x3xf16) <- (-1x160x4x3xf16, 160x160x1x1xf16)
        conv2d_148 = paddle._C_ops.conv2d(slice_141, parameter_1140, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x160x4x3xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x4x3xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__1368, batch_norm__1369, batch_norm__1370, batch_norm__1371, batch_norm__1372, batch_norm__1373 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_148, parameter_1141, parameter_1142, parameter_1143, parameter_1144, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x160x4x3xf16) <- (-1x160x4x3xf16)
        relu__121 = paddle._C_ops.relu_(batch_norm__1368)

        # pd_op.depthwise_conv2d: (-1x160x4x3xf16) <- (-1x160x4x3xf16, 160x1x3x3xf16)
        depthwise_conv2d_80 = paddle._C_ops.depthwise_conv2d(relu__121, parameter_1145, [1, 1], [1, 1], 'EXPLICIT', 160, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x160x4x3xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x4x3xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__1374, batch_norm__1375, batch_norm__1376, batch_norm__1377, batch_norm__1378, batch_norm__1379 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_80, parameter_1146, parameter_1147, parameter_1148, parameter_1149, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x160x4x3xf16) <- (-1x160x4x3xf16, 160x160x1x1xf16)
        conv2d_149 = paddle._C_ops.conv2d(batch_norm__1374, parameter_1150, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x160x4x3xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x4x3xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__1380, batch_norm__1381, batch_norm__1382, batch_norm__1383, batch_norm__1384, batch_norm__1385 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_149, parameter_1151, parameter_1152, parameter_1153, parameter_1154, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x160x4x3xf16) <- (-1x160x4x3xf16)
        relu__122 = paddle._C_ops.relu_(batch_norm__1380)

        # builtin.slice: (-1x160x4x3xf16) <- ([-1x160x4x3xf16, -1x160x4x3xf16])
        slice_142 = split_with_num_47[0]

        # builtin.combine: ([-1x160x4x3xf16, -1x160x4x3xf16]) <- (-1x160x4x3xf16, -1x160x4x3xf16)
        combine_141 = [slice_142, relu__122]

        # pd_op.full: (1xi32) <- ()
        full_424 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x320x4x3xf16) <- ([-1x160x4x3xf16, -1x160x4x3xf16], 1xi32)
        concat_47 = paddle._C_ops.concat(combine_141, full_424)

        # pd_op.shape: (4xi32) <- (-1x320x4x3xf16)
        shape_47 = paddle._C_ops.shape(paddle.cast(concat_47, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_94 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_95 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_143 = paddle._C_ops.slice(shape_47, [0], full_int_array_94, full_int_array_95, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_425 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_426 = paddle._C_ops.full([1], float('160'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_427 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_428 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_142 = [slice_143, full_425, full_426, full_427, full_428]

        # pd_op.reshape_: (-1x2x160x4x3xf16, 0x-1x320x4x3xf16) <- (-1x320x4x3xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__188, reshape__189 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_47, [x.reshape([]) for x in combine_142]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x160x2x4x3xf16) <- (-1x2x160x4x3xf16)
        transpose_47 = paddle._C_ops.transpose(reshape__188, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_429 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_430 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_431 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_143 = [slice_143, full_429, full_430, full_431]

        # pd_op.reshape_: (-1x320x4x3xf16, 0x-1x160x2x4x3xf16) <- (-1x160x2x4x3xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__190, reshape__191 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_47, [x.reshape([]) for x in combine_143]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi32) <- ()
        full_432 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split_with_num: ([-1x160x4x3xf16, -1x160x4x3xf16]) <- (-1x320x4x3xf16, 1xi32)
        split_with_num_48 = paddle._C_ops.split_with_num(reshape__190, 2, full_432)

        # builtin.slice: (-1x160x4x3xf16) <- ([-1x160x4x3xf16, -1x160x4x3xf16])
        slice_144 = split_with_num_48[1]

        # pd_op.conv2d: (-1x160x4x3xf16) <- (-1x160x4x3xf16, 160x160x1x1xf16)
        conv2d_150 = paddle._C_ops.conv2d(slice_144, parameter_1155, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x160x4x3xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x4x3xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__1386, batch_norm__1387, batch_norm__1388, batch_norm__1389, batch_norm__1390, batch_norm__1391 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_150, parameter_1156, parameter_1157, parameter_1158, parameter_1159, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x160x4x3xf16) <- (-1x160x4x3xf16)
        relu__123 = paddle._C_ops.relu_(batch_norm__1386)

        # pd_op.depthwise_conv2d: (-1x160x4x3xf16) <- (-1x160x4x3xf16, 160x1x3x3xf16)
        depthwise_conv2d_81 = paddle._C_ops.depthwise_conv2d(relu__123, parameter_1160, [1, 1], [1, 1], 'EXPLICIT', 160, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x160x4x3xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x4x3xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__1392, batch_norm__1393, batch_norm__1394, batch_norm__1395, batch_norm__1396, batch_norm__1397 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_81, parameter_1161, parameter_1162, parameter_1163, parameter_1164, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x160x4x3xf16) <- (-1x160x4x3xf16, 160x160x1x1xf16)
        conv2d_151 = paddle._C_ops.conv2d(batch_norm__1392, parameter_1165, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x160x4x3xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x4x3xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__1398, batch_norm__1399, batch_norm__1400, batch_norm__1401, batch_norm__1402, batch_norm__1403 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_151, parameter_1166, parameter_1167, parameter_1168, parameter_1169, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x160x4x3xf16) <- (-1x160x4x3xf16)
        relu__124 = paddle._C_ops.relu_(batch_norm__1398)

        # builtin.slice: (-1x160x4x3xf16) <- ([-1x160x4x3xf16, -1x160x4x3xf16])
        slice_145 = split_with_num_48[0]

        # builtin.combine: ([-1x160x4x3xf16, -1x160x4x3xf16]) <- (-1x160x4x3xf16, -1x160x4x3xf16)
        combine_144 = [slice_145, relu__124]

        # pd_op.full: (1xi32) <- ()
        full_433 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x320x4x3xf16) <- ([-1x160x4x3xf16, -1x160x4x3xf16], 1xi32)
        concat_48 = paddle._C_ops.concat(combine_144, full_433)

        # pd_op.shape: (4xi32) <- (-1x320x4x3xf16)
        shape_48 = paddle._C_ops.shape(paddle.cast(concat_48, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_96 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_97 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_146 = paddle._C_ops.slice(shape_48, [0], full_int_array_96, full_int_array_97, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_434 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_435 = paddle._C_ops.full([1], float('160'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_436 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_437 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_145 = [slice_146, full_434, full_435, full_436, full_437]

        # pd_op.reshape_: (-1x2x160x4x3xf16, 0x-1x320x4x3xf16) <- (-1x320x4x3xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__192, reshape__193 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_48, [x.reshape([]) for x in combine_145]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x160x2x4x3xf16) <- (-1x2x160x4x3xf16)
        transpose_48 = paddle._C_ops.transpose(reshape__192, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_438 = paddle._C_ops.full([1], float('320'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_439 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_440 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_146 = [slice_146, full_438, full_439, full_440]

        # pd_op.reshape_: (-1x320x4x3xf16, 0x-1x160x2x4x3xf16) <- (-1x160x2x4x3xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__194, reshape__195 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_48, [x.reshape([]) for x in combine_146]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x40x32x24xf16) <- (-1x40x32x24xf16, -1x40x32x24xf16)
        add__60 = paddle._C_ops.add_(reshape__170, reshape__170)

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x80x16x12xf16, 40x80x1x1xf16)
        conv2d_152 = paddle._C_ops.conv2d(reshape__178, parameter_1170, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1404, batch_norm__1405, batch_norm__1406, batch_norm__1407, batch_norm__1408, batch_norm__1409 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_152, parameter_1171, parameter_1172, parameter_1173, parameter_1174, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.nearest_interp: (-1x40x32x24xf16) <- (-1x40x16x12xf16, None, None, None)
        nearest_interp_20 = paddle._C_ops.nearest_interp(batch_norm__1404, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 0)

        # pd_op.add_: (-1x40x32x24xf16) <- (-1x40x32x24xf16, -1x40x32x24xf16)
        add__61 = paddle._C_ops.add_(add__60, nearest_interp_20)

        # pd_op.conv2d: (-1x40x8x6xf16) <- (-1x160x8x6xf16, 40x160x1x1xf16)
        conv2d_153 = paddle._C_ops.conv2d(reshape__186, parameter_1175, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x8x6xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x8x6xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1410, batch_norm__1411, batch_norm__1412, batch_norm__1413, batch_norm__1414, batch_norm__1415 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_153, parameter_1176, parameter_1177, parameter_1178, parameter_1179, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.nearest_interp: (-1x40x32x24xf16) <- (-1x40x8x6xf16, None, None, None)
        nearest_interp_21 = paddle._C_ops.nearest_interp(batch_norm__1410, None, None, None, 'NCHW', -1, -1, -1, [float('4'), float('4')], 'nearest', False, 0)

        # pd_op.add_: (-1x40x32x24xf16) <- (-1x40x32x24xf16, -1x40x32x24xf16)
        add__62 = paddle._C_ops.add_(add__61, nearest_interp_21)

        # pd_op.conv2d: (-1x40x4x3xf16) <- (-1x320x4x3xf16, 40x320x1x1xf16)
        conv2d_154 = paddle._C_ops.conv2d(reshape__194, parameter_1180, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x4x3xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x4x3xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1416, batch_norm__1417, batch_norm__1418, batch_norm__1419, batch_norm__1420, batch_norm__1421 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_154, parameter_1181, parameter_1182, parameter_1183, parameter_1184, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.nearest_interp: (-1x40x32x24xf16) <- (-1x40x4x3xf16, None, None, None)
        nearest_interp_22 = paddle._C_ops.nearest_interp(batch_norm__1416, None, None, None, 'NCHW', -1, -1, -1, [float('8'), float('8')], 'nearest', False, 0)

        # pd_op.add_: (-1x40x32x24xf16) <- (-1x40x32x24xf16, -1x40x32x24xf16)
        add__63 = paddle._C_ops.add_(add__62, nearest_interp_22)

        # pd_op.relu: (-1x40x32x24xf16) <- (-1x40x32x24xf16)
        relu_7 = paddle._C_ops.relu(add__63)

        # pd_op.depthwise_conv2d: (-1x40x16x12xf16) <- (-1x40x32x24xf16, 40x1x3x3xf16)
        depthwise_conv2d_82 = paddle._C_ops.depthwise_conv2d(add__63, parameter_1185, [2, 2], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1422, batch_norm__1423, batch_norm__1424, batch_norm__1425, batch_norm__1426, batch_norm__1427 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_82, parameter_1186, parameter_1187, parameter_1188, parameter_1189, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x80x16x12xf16) <- (-1x40x16x12xf16, 80x40x1x1xf16)
        conv2d_155 = paddle._C_ops.conv2d(batch_norm__1422, parameter_1190, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x16x12xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x16x12xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__1428, batch_norm__1429, batch_norm__1430, batch_norm__1431, batch_norm__1432, batch_norm__1433 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_155, parameter_1191, parameter_1192, parameter_1193, parameter_1194, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x80x16x12xf16) <- (-1x80x16x12xf16, -1x80x16x12xf16)
        add__64 = paddle._C_ops.add_(batch_norm__1428, batch_norm__1428)

        # pd_op.add_: (-1x80x16x12xf16) <- (-1x80x16x12xf16, -1x80x16x12xf16)
        add__65 = paddle._C_ops.add_(add__64, reshape__178)

        # pd_op.conv2d: (-1x80x8x6xf16) <- (-1x160x8x6xf16, 80x160x1x1xf16)
        conv2d_156 = paddle._C_ops.conv2d(reshape__186, parameter_1195, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__1434, batch_norm__1435, batch_norm__1436, batch_norm__1437, batch_norm__1438, batch_norm__1439 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_156, parameter_1196, parameter_1197, parameter_1198, parameter_1199, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.nearest_interp: (-1x80x16x12xf16) <- (-1x80x8x6xf16, None, None, None)
        nearest_interp_23 = paddle._C_ops.nearest_interp(batch_norm__1434, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 0)

        # pd_op.add_: (-1x80x16x12xf16) <- (-1x80x16x12xf16, -1x80x16x12xf16)
        add__66 = paddle._C_ops.add_(add__65, nearest_interp_23)

        # pd_op.conv2d: (-1x80x4x3xf16) <- (-1x320x4x3xf16, 80x320x1x1xf16)
        conv2d_157 = paddle._C_ops.conv2d(reshape__194, parameter_1200, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x4x3xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x4x3xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__1440, batch_norm__1441, batch_norm__1442, batch_norm__1443, batch_norm__1444, batch_norm__1445 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_157, parameter_1201, parameter_1202, parameter_1203, parameter_1204, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.nearest_interp: (-1x80x16x12xf16) <- (-1x80x4x3xf16, None, None, None)
        nearest_interp_24 = paddle._C_ops.nearest_interp(batch_norm__1440, None, None, None, 'NCHW', -1, -1, -1, [float('4'), float('4')], 'nearest', False, 0)

        # pd_op.add_: (-1x80x16x12xf16) <- (-1x80x16x12xf16, -1x80x16x12xf16)
        add__67 = paddle._C_ops.add_(add__66, nearest_interp_24)

        # pd_op.relu_: (-1x80x16x12xf16) <- (-1x80x16x12xf16)
        relu__125 = paddle._C_ops.relu_(add__67)

        # pd_op.depthwise_conv2d: (-1x40x16x12xf16) <- (-1x40x32x24xf16, 40x1x3x3xf16)
        depthwise_conv2d_83 = paddle._C_ops.depthwise_conv2d(add__63, parameter_1205, [2, 2], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1446, batch_norm__1447, batch_norm__1448, batch_norm__1449, batch_norm__1450, batch_norm__1451 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_83, parameter_1206, parameter_1207, parameter_1208, parameter_1209, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_158 = paddle._C_ops.conv2d(batch_norm__1446, parameter_1210, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1452, batch_norm__1453, batch_norm__1454, batch_norm__1455, batch_norm__1456, batch_norm__1457 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_158, parameter_1211, parameter_1212, parameter_1213, parameter_1214, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__126 = paddle._C_ops.relu_(batch_norm__1452)

        # pd_op.depthwise_conv2d: (-1x40x8x6xf16) <- (-1x40x16x12xf16, 40x1x3x3xf16)
        depthwise_conv2d_84 = paddle._C_ops.depthwise_conv2d(relu__126, parameter_1215, [2, 2], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x8x6xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x8x6xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1458, batch_norm__1459, batch_norm__1460, batch_norm__1461, batch_norm__1462, batch_norm__1463 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_84, parameter_1216, parameter_1217, parameter_1218, parameter_1219, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x160x8x6xf16) <- (-1x40x8x6xf16, 160x40x1x1xf16)
        conv2d_159 = paddle._C_ops.conv2d(batch_norm__1458, parameter_1220, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x160x8x6xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x8x6xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__1464, batch_norm__1465, batch_norm__1466, batch_norm__1467, batch_norm__1468, batch_norm__1469 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_159, parameter_1221, parameter_1222, parameter_1223, parameter_1224, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x160x8x6xf16) <- (-1x160x8x6xf16, -1x160x8x6xf16)
        add__68 = paddle._C_ops.add_(batch_norm__1464, batch_norm__1464)

        # pd_op.depthwise_conv2d: (-1x80x8x6xf16) <- (-1x80x16x12xf16, 80x1x3x3xf16)
        depthwise_conv2d_85 = paddle._C_ops.depthwise_conv2d(reshape__178, parameter_1225, [2, 2], [1, 1], 'EXPLICIT', 80, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__1470, batch_norm__1471, batch_norm__1472, batch_norm__1473, batch_norm__1474, batch_norm__1475 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_85, parameter_1226, parameter_1227, parameter_1228, parameter_1229, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x160x8x6xf16) <- (-1x80x8x6xf16, 160x80x1x1xf16)
        conv2d_160 = paddle._C_ops.conv2d(batch_norm__1470, parameter_1230, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x160x8x6xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x8x6xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__1476, batch_norm__1477, batch_norm__1478, batch_norm__1479, batch_norm__1480, batch_norm__1481 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_160, parameter_1231, parameter_1232, parameter_1233, parameter_1234, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x160x8x6xf16) <- (-1x160x8x6xf16, -1x160x8x6xf16)
        add__69 = paddle._C_ops.add_(add__68, batch_norm__1476)

        # pd_op.add_: (-1x160x8x6xf16) <- (-1x160x8x6xf16, -1x160x8x6xf16)
        add__70 = paddle._C_ops.add_(add__69, reshape__186)

        # pd_op.conv2d: (-1x160x4x3xf16) <- (-1x320x4x3xf16, 160x320x1x1xf16)
        conv2d_161 = paddle._C_ops.conv2d(reshape__194, parameter_1235, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x160x4x3xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x4x3xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__1482, batch_norm__1483, batch_norm__1484, batch_norm__1485, batch_norm__1486, batch_norm__1487 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_161, parameter_1236, parameter_1237, parameter_1238, parameter_1239, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.nearest_interp: (-1x160x8x6xf16) <- (-1x160x4x3xf16, None, None, None)
        nearest_interp_25 = paddle._C_ops.nearest_interp(batch_norm__1482, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 0)

        # pd_op.add_: (-1x160x8x6xf16) <- (-1x160x8x6xf16, -1x160x8x6xf16)
        add__71 = paddle._C_ops.add_(add__70, nearest_interp_25)

        # pd_op.relu_: (-1x160x8x6xf16) <- (-1x160x8x6xf16)
        relu__127 = paddle._C_ops.relu_(add__71)

        # pd_op.depthwise_conv2d: (-1x40x16x12xf16) <- (-1x40x32x24xf16, 40x1x3x3xf16)
        depthwise_conv2d_86 = paddle._C_ops.depthwise_conv2d(add__63, parameter_1240, [2, 2], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1488, batch_norm__1489, batch_norm__1490, batch_norm__1491, batch_norm__1492, batch_norm__1493 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_86, parameter_1241, parameter_1242, parameter_1243, parameter_1244, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x40x16x12xf16, 40x40x1x1xf16)
        conv2d_162 = paddle._C_ops.conv2d(batch_norm__1488, parameter_1245, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1494, batch_norm__1495, batch_norm__1496, batch_norm__1497, batch_norm__1498, batch_norm__1499 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_162, parameter_1246, parameter_1247, parameter_1248, parameter_1249, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__128 = paddle._C_ops.relu_(batch_norm__1494)

        # pd_op.depthwise_conv2d: (-1x40x8x6xf16) <- (-1x40x16x12xf16, 40x1x3x3xf16)
        depthwise_conv2d_87 = paddle._C_ops.depthwise_conv2d(relu__128, parameter_1250, [2, 2], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x8x6xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x8x6xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1500, batch_norm__1501, batch_norm__1502, batch_norm__1503, batch_norm__1504, batch_norm__1505 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_87, parameter_1251, parameter_1252, parameter_1253, parameter_1254, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x40x8x6xf16) <- (-1x40x8x6xf16, 40x40x1x1xf16)
        conv2d_163 = paddle._C_ops.conv2d(batch_norm__1500, parameter_1255, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x8x6xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x8x6xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1506, batch_norm__1507, batch_norm__1508, batch_norm__1509, batch_norm__1510, batch_norm__1511 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_163, parameter_1256, parameter_1257, parameter_1258, parameter_1259, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x8x6xf16) <- (-1x40x8x6xf16)
        relu__129 = paddle._C_ops.relu_(batch_norm__1506)

        # pd_op.depthwise_conv2d: (-1x40x4x3xf16) <- (-1x40x8x6xf16, 40x1x3x3xf16)
        depthwise_conv2d_88 = paddle._C_ops.depthwise_conv2d(relu__129, parameter_1260, [2, 2], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x4x3xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x4x3xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1512, batch_norm__1513, batch_norm__1514, batch_norm__1515, batch_norm__1516, batch_norm__1517 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_88, parameter_1261, parameter_1262, parameter_1263, parameter_1264, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x320x4x3xf16) <- (-1x40x4x3xf16, 320x40x1x1xf16)
        conv2d_164 = paddle._C_ops.conv2d(batch_norm__1512, parameter_1265, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x320x4x3xf16, 320xf32, 320xf32, xf32, xf32, None) <- (-1x320x4x3xf16, 320xf32, 320xf32, 320xf32, 320xf32)
        batch_norm__1518, batch_norm__1519, batch_norm__1520, batch_norm__1521, batch_norm__1522, batch_norm__1523 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_164, parameter_1266, parameter_1267, parameter_1268, parameter_1269, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x320x4x3xf16) <- (-1x320x4x3xf16, -1x320x4x3xf16)
        add__72 = paddle._C_ops.add_(batch_norm__1518, batch_norm__1518)

        # pd_op.depthwise_conv2d: (-1x80x8x6xf16) <- (-1x80x16x12xf16, 80x1x3x3xf16)
        depthwise_conv2d_89 = paddle._C_ops.depthwise_conv2d(reshape__178, parameter_1270, [2, 2], [1, 1], 'EXPLICIT', 80, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__1524, batch_norm__1525, batch_norm__1526, batch_norm__1527, batch_norm__1528, batch_norm__1529 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_89, parameter_1271, parameter_1272, parameter_1273, parameter_1274, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x80x8x6xf16) <- (-1x80x8x6xf16, 80x80x1x1xf16)
        conv2d_165 = paddle._C_ops.conv2d(batch_norm__1524, parameter_1275, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__1530, batch_norm__1531, batch_norm__1532, batch_norm__1533, batch_norm__1534, batch_norm__1535 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_165, parameter_1276, parameter_1277, parameter_1278, parameter_1279, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x80x8x6xf16) <- (-1x80x8x6xf16)
        relu__130 = paddle._C_ops.relu_(batch_norm__1530)

        # pd_op.depthwise_conv2d: (-1x80x4x3xf16) <- (-1x80x8x6xf16, 80x1x3x3xf16)
        depthwise_conv2d_90 = paddle._C_ops.depthwise_conv2d(relu__130, parameter_1280, [2, 2], [1, 1], 'EXPLICIT', 80, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x80x4x3xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x4x3xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__1536, batch_norm__1537, batch_norm__1538, batch_norm__1539, batch_norm__1540, batch_norm__1541 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_90, parameter_1281, parameter_1282, parameter_1283, parameter_1284, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x320x4x3xf16) <- (-1x80x4x3xf16, 320x80x1x1xf16)
        conv2d_166 = paddle._C_ops.conv2d(batch_norm__1536, parameter_1285, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x320x4x3xf16, 320xf32, 320xf32, xf32, xf32, None) <- (-1x320x4x3xf16, 320xf32, 320xf32, 320xf32, 320xf32)
        batch_norm__1542, batch_norm__1543, batch_norm__1544, batch_norm__1545, batch_norm__1546, batch_norm__1547 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_166, parameter_1286, parameter_1287, parameter_1288, parameter_1289, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x320x4x3xf16) <- (-1x320x4x3xf16, -1x320x4x3xf16)
        add__73 = paddle._C_ops.add_(add__72, batch_norm__1542)

        # pd_op.depthwise_conv2d: (-1x160x4x3xf16) <- (-1x160x8x6xf16, 160x1x3x3xf16)
        depthwise_conv2d_91 = paddle._C_ops.depthwise_conv2d(reshape__186, parameter_1290, [2, 2], [1, 1], 'EXPLICIT', 160, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x160x4x3xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x4x3xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__1548, batch_norm__1549, batch_norm__1550, batch_norm__1551, batch_norm__1552, batch_norm__1553 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_91, parameter_1291, parameter_1292, parameter_1293, parameter_1294, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x320x4x3xf16) <- (-1x160x4x3xf16, 320x160x1x1xf16)
        conv2d_167 = paddle._C_ops.conv2d(batch_norm__1548, parameter_1295, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x320x4x3xf16, 320xf32, 320xf32, xf32, xf32, None) <- (-1x320x4x3xf16, 320xf32, 320xf32, 320xf32, 320xf32)
        batch_norm__1554, batch_norm__1555, batch_norm__1556, batch_norm__1557, batch_norm__1558, batch_norm__1559 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_167, parameter_1296, parameter_1297, parameter_1298, parameter_1299, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add_: (-1x320x4x3xf16) <- (-1x320x4x3xf16, -1x320x4x3xf16)
        add__74 = paddle._C_ops.add_(add__73, batch_norm__1554)

        # pd_op.add_: (-1x320x4x3xf16) <- (-1x320x4x3xf16, -1x320x4x3xf16)
        add__75 = paddle._C_ops.add_(add__74, reshape__194)

        # pd_op.relu_: (-1x320x4x3xf16) <- (-1x320x4x3xf16)
        relu__131 = paddle._C_ops.relu_(add__75)

        # pd_op.depthwise_conv2d: (-1x320x4x3xf16) <- (-1x320x4x3xf16, 320x1x3x3xf16)
        depthwise_conv2d_92 = paddle._C_ops.depthwise_conv2d(relu__131, parameter_1300, [1, 1], [1, 1], 'EXPLICIT', 320, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x320x4x3xf16, 320xf32, 320xf32, xf32, xf32, None) <- (-1x320x4x3xf16, 320xf32, 320xf32, 320xf32, 320xf32)
        batch_norm__1560, batch_norm__1561, batch_norm__1562, batch_norm__1563, batch_norm__1564, batch_norm__1565 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_92, parameter_1301, parameter_1302, parameter_1303, parameter_1304, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x160x4x3xf16) <- (-1x320x4x3xf16, 160x320x1x1xf16)
        conv2d_168 = paddle._C_ops.conv2d(batch_norm__1560, parameter_1305, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x160x4x3xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x4x3xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__1566, batch_norm__1567, batch_norm__1568, batch_norm__1569, batch_norm__1570, batch_norm__1571 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_168, parameter_1306, parameter_1307, parameter_1308, parameter_1309, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x160x4x3xf16) <- (-1x160x4x3xf16)
        relu__132 = paddle._C_ops.relu_(batch_norm__1566)

        # pd_op.bilinear_interp: (-1x160x8x6xf16) <- (-1x160x4x3xf16, None, None, None)
        bilinear_interp_0 = paddle._C_ops.bilinear_interp(relu__132, None, None, None, 'NCHW', -1, 8, 6, [], 'bilinear', True, 0)

        # pd_op.add_: (-1x160x8x6xf16) <- (-1x160x8x6xf16, -1x160x8x6xf16)
        add__76 = paddle._C_ops.add_(relu__127, bilinear_interp_0)

        # pd_op.depthwise_conv2d: (-1x160x8x6xf16) <- (-1x160x8x6xf16, 160x1x3x3xf16)
        depthwise_conv2d_93 = paddle._C_ops.depthwise_conv2d(add__76, parameter_1310, [1, 1], [1, 1], 'EXPLICIT', 160, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x160x8x6xf16, 160xf32, 160xf32, xf32, xf32, None) <- (-1x160x8x6xf16, 160xf32, 160xf32, 160xf32, 160xf32)
        batch_norm__1572, batch_norm__1573, batch_norm__1574, batch_norm__1575, batch_norm__1576, batch_norm__1577 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_93, parameter_1311, parameter_1312, parameter_1313, parameter_1314, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x80x8x6xf16) <- (-1x160x8x6xf16, 80x160x1x1xf16)
        conv2d_169 = paddle._C_ops.conv2d(batch_norm__1572, parameter_1315, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x80x8x6xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x8x6xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__1578, batch_norm__1579, batch_norm__1580, batch_norm__1581, batch_norm__1582, batch_norm__1583 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_169, parameter_1316, parameter_1317, parameter_1318, parameter_1319, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x80x8x6xf16) <- (-1x80x8x6xf16)
        relu__133 = paddle._C_ops.relu_(batch_norm__1578)

        # pd_op.bilinear_interp: (-1x80x16x12xf16) <- (-1x80x8x6xf16, None, None, None)
        bilinear_interp_1 = paddle._C_ops.bilinear_interp(relu__133, None, None, None, 'NCHW', -1, 16, 12, [], 'bilinear', True, 0)

        # pd_op.add_: (-1x80x16x12xf16) <- (-1x80x16x12xf16, -1x80x16x12xf16)
        add__77 = paddle._C_ops.add_(relu__125, bilinear_interp_1)

        # pd_op.depthwise_conv2d: (-1x80x16x12xf16) <- (-1x80x16x12xf16, 80x1x3x3xf16)
        depthwise_conv2d_94 = paddle._C_ops.depthwise_conv2d(add__77, parameter_1320, [1, 1], [1, 1], 'EXPLICIT', 80, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x80x16x12xf16, 80xf32, 80xf32, xf32, xf32, None) <- (-1x80x16x12xf16, 80xf32, 80xf32, 80xf32, 80xf32)
        batch_norm__1584, batch_norm__1585, batch_norm__1586, batch_norm__1587, batch_norm__1588, batch_norm__1589 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_94, parameter_1321, parameter_1322, parameter_1323, parameter_1324, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x40x16x12xf16) <- (-1x80x16x12xf16, 40x80x1x1xf16)
        conv2d_170 = paddle._C_ops.conv2d(batch_norm__1584, parameter_1325, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x16x12xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x16x12xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1590, batch_norm__1591, batch_norm__1592, batch_norm__1593, batch_norm__1594, batch_norm__1595 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_170, parameter_1326, parameter_1327, parameter_1328, parameter_1329, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x16x12xf16) <- (-1x40x16x12xf16)
        relu__134 = paddle._C_ops.relu_(batch_norm__1590)

        # pd_op.bilinear_interp: (-1x40x32x24xf16) <- (-1x40x16x12xf16, None, None, None)
        bilinear_interp_2 = paddle._C_ops.bilinear_interp(relu__134, None, None, None, 'NCHW', -1, 32, 24, [], 'bilinear', True, 0)

        # pd_op.add_: (-1x40x32x24xf16) <- (-1x40x32x24xf16, -1x40x32x24xf16)
        add__78 = paddle._C_ops.add_(relu_7, bilinear_interp_2)

        # pd_op.depthwise_conv2d: (-1x40x32x24xf16) <- (-1x40x32x24xf16, 40x1x3x3xf16)
        depthwise_conv2d_95 = paddle._C_ops.depthwise_conv2d(add__78, parameter_1330, [1, 1], [1, 1], 'EXPLICIT', 40, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x40x32x24xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x32x24xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1596, batch_norm__1597, batch_norm__1598, batch_norm__1599, batch_norm__1600, batch_norm__1601 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_95, parameter_1331, parameter_1332, parameter_1333, parameter_1334, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x40x32x24xf16) <- (-1x40x32x24xf16, 40x40x1x1xf16)
        conv2d_171 = paddle._C_ops.conv2d(batch_norm__1596, parameter_1335, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x40x32x24xf16, 40xf32, 40xf32, xf32, xf32, None) <- (-1x40x32x24xf16, 40xf32, 40xf32, 40xf32, 40xf32)
        batch_norm__1602, batch_norm__1603, batch_norm__1604, batch_norm__1605, batch_norm__1606, batch_norm__1607 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_171, parameter_1336, parameter_1337, parameter_1338, parameter_1339, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x40x32x24xf16) <- (-1x40x32x24xf16)
        relu__135 = paddle._C_ops.relu_(batch_norm__1602)

        # pd_op.conv2d: (-1x17x32x24xf16) <- (-1x40x32x24xf16, 17x40x1x1xf16)
        conv2d_172 = paddle._C_ops.conv2d(relu__135, parameter_1340, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_98 = [1, 17, 1, 1]

        # pd_op.reshape: (1x17x1x1xf16, 0x17xf16) <- (17xf16, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_1341, full_int_array_98), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x17x32x24xf16) <- (-1x17x32x24xf16, 1x17x1x1xf16)
        add__79 = paddle._C_ops.add_(conv2d_172, reshape_0)

        # pd_op.shape: (4xi32) <- (-1x17x32x24xf16)
        shape_49 = paddle._C_ops.shape(paddle.cast(add__79, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_99 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_100 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_147 = paddle._C_ops.slice(shape_49, [0], full_int_array_99, full_int_array_100, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_441 = paddle._C_ops.full([1], float('17'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_442 = paddle._C_ops.full([1], float('768'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_147 = [slice_147, full_441, full_442]

        # pd_op.reshape: (-1x17x768xf16, 0x-1x17x32x24xf16) <- (-1x17x32x24xf16, [1xi32, 1xi32, 1xi32])
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__79, [x.reshape([]) for x in combine_147]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full: (1xi64) <- ()
        full_443 = paddle._C_ops.full([1], float('-1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1x17xi64) <- (-1x17x768xf16, 1xi64)
        argmax_0 = paddle._C_ops.argmax(reshape_2, full_443, False, False, paddle.int64)

        # pd_op.cast: (-1x17x32x24xf32) <- (-1x17x32x24xf16)
        cast_1 = paddle._C_ops.cast(add__79, paddle.float32)
        return cast_1, argmax_0



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

    def forward(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_114, parameter_111, parameter_113, parameter_112, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_154, parameter_151, parameter_153, parameter_152, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_169, parameter_166, parameter_168, parameter_167, parameter_170, parameter_174, parameter_171, parameter_173, parameter_172, parameter_175, parameter_179, parameter_176, parameter_178, parameter_177, parameter_180, parameter_184, parameter_181, parameter_183, parameter_182, parameter_185, parameter_189, parameter_186, parameter_188, parameter_187, parameter_190, parameter_194, parameter_191, parameter_193, parameter_192, parameter_195, parameter_199, parameter_196, parameter_198, parameter_197, parameter_200, parameter_204, parameter_201, parameter_203, parameter_202, parameter_205, parameter_209, parameter_206, parameter_208, parameter_207, parameter_210, parameter_214, parameter_211, parameter_213, parameter_212, parameter_215, parameter_219, parameter_216, parameter_218, parameter_217, parameter_220, parameter_224, parameter_221, parameter_223, parameter_222, parameter_225, parameter_229, parameter_226, parameter_228, parameter_227, parameter_230, parameter_234, parameter_231, parameter_233, parameter_232, parameter_235, parameter_239, parameter_236, parameter_238, parameter_237, parameter_240, parameter_244, parameter_241, parameter_243, parameter_242, parameter_245, parameter_249, parameter_246, parameter_248, parameter_247, parameter_250, parameter_254, parameter_251, parameter_253, parameter_252, parameter_255, parameter_259, parameter_256, parameter_258, parameter_257, parameter_260, parameter_264, parameter_261, parameter_263, parameter_262, parameter_265, parameter_269, parameter_266, parameter_268, parameter_267, parameter_270, parameter_274, parameter_271, parameter_273, parameter_272, parameter_275, parameter_279, parameter_276, parameter_278, parameter_277, parameter_280, parameter_284, parameter_281, parameter_283, parameter_282, parameter_285, parameter_289, parameter_286, parameter_288, parameter_287, parameter_290, parameter_294, parameter_291, parameter_293, parameter_292, parameter_295, parameter_299, parameter_296, parameter_298, parameter_297, parameter_300, parameter_304, parameter_301, parameter_303, parameter_302, parameter_305, parameter_309, parameter_306, parameter_308, parameter_307, parameter_310, parameter_314, parameter_311, parameter_313, parameter_312, parameter_315, parameter_319, parameter_316, parameter_318, parameter_317, parameter_320, parameter_324, parameter_321, parameter_323, parameter_322, parameter_325, parameter_329, parameter_326, parameter_328, parameter_327, parameter_330, parameter_334, parameter_331, parameter_333, parameter_332, parameter_335, parameter_339, parameter_336, parameter_338, parameter_337, parameter_340, parameter_344, parameter_341, parameter_343, parameter_342, parameter_345, parameter_349, parameter_346, parameter_348, parameter_347, parameter_350, parameter_354, parameter_351, parameter_353, parameter_352, parameter_355, parameter_359, parameter_356, parameter_358, parameter_357, parameter_360, parameter_364, parameter_361, parameter_363, parameter_362, parameter_365, parameter_369, parameter_366, parameter_368, parameter_367, parameter_370, parameter_374, parameter_371, parameter_373, parameter_372, parameter_375, parameter_379, parameter_376, parameter_378, parameter_377, parameter_380, parameter_384, parameter_381, parameter_383, parameter_382, parameter_385, parameter_389, parameter_386, parameter_388, parameter_387, parameter_390, parameter_394, parameter_391, parameter_393, parameter_392, parameter_395, parameter_399, parameter_396, parameter_398, parameter_397, parameter_400, parameter_404, parameter_401, parameter_403, parameter_402, parameter_405, parameter_409, parameter_406, parameter_408, parameter_407, parameter_410, parameter_414, parameter_411, parameter_413, parameter_412, parameter_415, parameter_419, parameter_416, parameter_418, parameter_417, parameter_420, parameter_424, parameter_421, parameter_423, parameter_422, parameter_425, parameter_429, parameter_426, parameter_428, parameter_427, parameter_430, parameter_434, parameter_431, parameter_433, parameter_432, parameter_435, parameter_439, parameter_436, parameter_438, parameter_437, parameter_440, parameter_444, parameter_441, parameter_443, parameter_442, parameter_445, parameter_449, parameter_446, parameter_448, parameter_447, parameter_450, parameter_454, parameter_451, parameter_453, parameter_452, parameter_455, parameter_459, parameter_456, parameter_458, parameter_457, parameter_460, parameter_464, parameter_461, parameter_463, parameter_462, parameter_465, parameter_469, parameter_466, parameter_468, parameter_467, parameter_470, parameter_474, parameter_471, parameter_473, parameter_472, parameter_475, parameter_479, parameter_476, parameter_478, parameter_477, parameter_480, parameter_484, parameter_481, parameter_483, parameter_482, parameter_485, parameter_489, parameter_486, parameter_488, parameter_487, parameter_490, parameter_494, parameter_491, parameter_493, parameter_492, parameter_495, parameter_499, parameter_496, parameter_498, parameter_497, parameter_500, parameter_504, parameter_501, parameter_503, parameter_502, parameter_505, parameter_509, parameter_506, parameter_508, parameter_507, parameter_510, parameter_514, parameter_511, parameter_513, parameter_512, parameter_515, parameter_519, parameter_516, parameter_518, parameter_517, parameter_520, parameter_524, parameter_521, parameter_523, parameter_522, parameter_525, parameter_529, parameter_526, parameter_528, parameter_527, parameter_530, parameter_534, parameter_531, parameter_533, parameter_532, parameter_535, parameter_539, parameter_536, parameter_538, parameter_537, parameter_540, parameter_544, parameter_541, parameter_543, parameter_542, parameter_545, parameter_549, parameter_546, parameter_548, parameter_547, parameter_550, parameter_554, parameter_551, parameter_553, parameter_552, parameter_555, parameter_559, parameter_556, parameter_558, parameter_557, parameter_560, parameter_564, parameter_561, parameter_563, parameter_562, parameter_565, parameter_569, parameter_566, parameter_568, parameter_567, parameter_570, parameter_574, parameter_571, parameter_573, parameter_572, parameter_575, parameter_579, parameter_576, parameter_578, parameter_577, parameter_580, parameter_584, parameter_581, parameter_583, parameter_582, parameter_585, parameter_589, parameter_586, parameter_588, parameter_587, parameter_590, parameter_594, parameter_591, parameter_593, parameter_592, parameter_595, parameter_599, parameter_596, parameter_598, parameter_597, parameter_600, parameter_604, parameter_601, parameter_603, parameter_602, parameter_605, parameter_609, parameter_606, parameter_608, parameter_607, parameter_610, parameter_614, parameter_611, parameter_613, parameter_612, parameter_615, parameter_619, parameter_616, parameter_618, parameter_617, parameter_620, parameter_624, parameter_621, parameter_623, parameter_622, parameter_625, parameter_629, parameter_626, parameter_628, parameter_627, parameter_630, parameter_634, parameter_631, parameter_633, parameter_632, parameter_635, parameter_639, parameter_636, parameter_638, parameter_637, parameter_640, parameter_644, parameter_641, parameter_643, parameter_642, parameter_645, parameter_649, parameter_646, parameter_648, parameter_647, parameter_650, parameter_654, parameter_651, parameter_653, parameter_652, parameter_655, parameter_659, parameter_656, parameter_658, parameter_657, parameter_660, parameter_664, parameter_661, parameter_663, parameter_662, parameter_665, parameter_669, parameter_666, parameter_668, parameter_667, parameter_670, parameter_674, parameter_671, parameter_673, parameter_672, parameter_675, parameter_679, parameter_676, parameter_678, parameter_677, parameter_680, parameter_684, parameter_681, parameter_683, parameter_682, parameter_685, parameter_689, parameter_686, parameter_688, parameter_687, parameter_690, parameter_694, parameter_691, parameter_693, parameter_692, parameter_695, parameter_699, parameter_696, parameter_698, parameter_697, parameter_700, parameter_704, parameter_701, parameter_703, parameter_702, parameter_705, parameter_709, parameter_706, parameter_708, parameter_707, parameter_710, parameter_714, parameter_711, parameter_713, parameter_712, parameter_715, parameter_719, parameter_716, parameter_718, parameter_717, parameter_720, parameter_724, parameter_721, parameter_723, parameter_722, parameter_725, parameter_729, parameter_726, parameter_728, parameter_727, parameter_730, parameter_734, parameter_731, parameter_733, parameter_732, parameter_735, parameter_739, parameter_736, parameter_738, parameter_737, parameter_740, parameter_744, parameter_741, parameter_743, parameter_742, parameter_745, parameter_749, parameter_746, parameter_748, parameter_747, parameter_750, parameter_754, parameter_751, parameter_753, parameter_752, parameter_755, parameter_759, parameter_756, parameter_758, parameter_757, parameter_760, parameter_764, parameter_761, parameter_763, parameter_762, parameter_765, parameter_769, parameter_766, parameter_768, parameter_767, parameter_770, parameter_774, parameter_771, parameter_773, parameter_772, parameter_775, parameter_779, parameter_776, parameter_778, parameter_777, parameter_780, parameter_784, parameter_781, parameter_783, parameter_782, parameter_785, parameter_789, parameter_786, parameter_788, parameter_787, parameter_790, parameter_794, parameter_791, parameter_793, parameter_792, parameter_795, parameter_799, parameter_796, parameter_798, parameter_797, parameter_800, parameter_804, parameter_801, parameter_803, parameter_802, parameter_805, parameter_809, parameter_806, parameter_808, parameter_807, parameter_810, parameter_814, parameter_811, parameter_813, parameter_812, parameter_815, parameter_819, parameter_816, parameter_818, parameter_817, parameter_820, parameter_824, parameter_821, parameter_823, parameter_822, parameter_825, parameter_829, parameter_826, parameter_828, parameter_827, parameter_830, parameter_834, parameter_831, parameter_833, parameter_832, parameter_835, parameter_839, parameter_836, parameter_838, parameter_837, parameter_840, parameter_844, parameter_841, parameter_843, parameter_842, parameter_845, parameter_849, parameter_846, parameter_848, parameter_847, parameter_850, parameter_854, parameter_851, parameter_853, parameter_852, parameter_855, parameter_859, parameter_856, parameter_858, parameter_857, parameter_860, parameter_864, parameter_861, parameter_863, parameter_862, parameter_865, parameter_869, parameter_866, parameter_868, parameter_867, parameter_870, parameter_874, parameter_871, parameter_873, parameter_872, parameter_875, parameter_879, parameter_876, parameter_878, parameter_877, parameter_880, parameter_884, parameter_881, parameter_883, parameter_882, parameter_885, parameter_889, parameter_886, parameter_888, parameter_887, parameter_890, parameter_894, parameter_891, parameter_893, parameter_892, parameter_895, parameter_899, parameter_896, parameter_898, parameter_897, parameter_900, parameter_904, parameter_901, parameter_903, parameter_902, parameter_905, parameter_909, parameter_906, parameter_908, parameter_907, parameter_910, parameter_914, parameter_911, parameter_913, parameter_912, parameter_915, parameter_919, parameter_916, parameter_918, parameter_917, parameter_920, parameter_924, parameter_921, parameter_923, parameter_922, parameter_925, parameter_929, parameter_926, parameter_928, parameter_927, parameter_930, parameter_934, parameter_931, parameter_933, parameter_932, parameter_935, parameter_939, parameter_936, parameter_938, parameter_937, parameter_940, parameter_944, parameter_941, parameter_943, parameter_942, parameter_945, parameter_949, parameter_946, parameter_948, parameter_947, parameter_950, parameter_954, parameter_951, parameter_953, parameter_952, parameter_955, parameter_959, parameter_956, parameter_958, parameter_957, parameter_960, parameter_964, parameter_961, parameter_963, parameter_962, parameter_965, parameter_969, parameter_966, parameter_968, parameter_967, parameter_970, parameter_974, parameter_971, parameter_973, parameter_972, parameter_975, parameter_979, parameter_976, parameter_978, parameter_977, parameter_980, parameter_984, parameter_981, parameter_983, parameter_982, parameter_985, parameter_989, parameter_986, parameter_988, parameter_987, parameter_990, parameter_994, parameter_991, parameter_993, parameter_992, parameter_995, parameter_999, parameter_996, parameter_998, parameter_997, parameter_1000, parameter_1004, parameter_1001, parameter_1003, parameter_1002, parameter_1005, parameter_1009, parameter_1006, parameter_1008, parameter_1007, parameter_1010, parameter_1014, parameter_1011, parameter_1013, parameter_1012, parameter_1015, parameter_1019, parameter_1016, parameter_1018, parameter_1017, parameter_1020, parameter_1024, parameter_1021, parameter_1023, parameter_1022, parameter_1025, parameter_1029, parameter_1026, parameter_1028, parameter_1027, parameter_1030, parameter_1034, parameter_1031, parameter_1033, parameter_1032, parameter_1035, parameter_1039, parameter_1036, parameter_1038, parameter_1037, parameter_1040, parameter_1044, parameter_1041, parameter_1043, parameter_1042, parameter_1045, parameter_1049, parameter_1046, parameter_1048, parameter_1047, parameter_1050, parameter_1054, parameter_1051, parameter_1053, parameter_1052, parameter_1055, parameter_1059, parameter_1056, parameter_1058, parameter_1057, parameter_1060, parameter_1064, parameter_1061, parameter_1063, parameter_1062, parameter_1065, parameter_1069, parameter_1066, parameter_1068, parameter_1067, parameter_1070, parameter_1074, parameter_1071, parameter_1073, parameter_1072, parameter_1075, parameter_1079, parameter_1076, parameter_1078, parameter_1077, parameter_1080, parameter_1084, parameter_1081, parameter_1083, parameter_1082, parameter_1085, parameter_1089, parameter_1086, parameter_1088, parameter_1087, parameter_1090, parameter_1094, parameter_1091, parameter_1093, parameter_1092, parameter_1095, parameter_1099, parameter_1096, parameter_1098, parameter_1097, parameter_1100, parameter_1104, parameter_1101, parameter_1103, parameter_1102, parameter_1105, parameter_1109, parameter_1106, parameter_1108, parameter_1107, parameter_1110, parameter_1114, parameter_1111, parameter_1113, parameter_1112, parameter_1115, parameter_1119, parameter_1116, parameter_1118, parameter_1117, parameter_1120, parameter_1124, parameter_1121, parameter_1123, parameter_1122, parameter_1125, parameter_1129, parameter_1126, parameter_1128, parameter_1127, parameter_1130, parameter_1134, parameter_1131, parameter_1133, parameter_1132, parameter_1135, parameter_1139, parameter_1136, parameter_1138, parameter_1137, parameter_1140, parameter_1144, parameter_1141, parameter_1143, parameter_1142, parameter_1145, parameter_1149, parameter_1146, parameter_1148, parameter_1147, parameter_1150, parameter_1154, parameter_1151, parameter_1153, parameter_1152, parameter_1155, parameter_1159, parameter_1156, parameter_1158, parameter_1157, parameter_1160, parameter_1164, parameter_1161, parameter_1163, parameter_1162, parameter_1165, parameter_1169, parameter_1166, parameter_1168, parameter_1167, parameter_1170, parameter_1174, parameter_1171, parameter_1173, parameter_1172, parameter_1175, parameter_1179, parameter_1176, parameter_1178, parameter_1177, parameter_1180, parameter_1184, parameter_1181, parameter_1183, parameter_1182, parameter_1185, parameter_1189, parameter_1186, parameter_1188, parameter_1187, parameter_1190, parameter_1194, parameter_1191, parameter_1193, parameter_1192, parameter_1195, parameter_1199, parameter_1196, parameter_1198, parameter_1197, parameter_1200, parameter_1204, parameter_1201, parameter_1203, parameter_1202, parameter_1205, parameter_1209, parameter_1206, parameter_1208, parameter_1207, parameter_1210, parameter_1214, parameter_1211, parameter_1213, parameter_1212, parameter_1215, parameter_1219, parameter_1216, parameter_1218, parameter_1217, parameter_1220, parameter_1224, parameter_1221, parameter_1223, parameter_1222, parameter_1225, parameter_1229, parameter_1226, parameter_1228, parameter_1227, parameter_1230, parameter_1234, parameter_1231, parameter_1233, parameter_1232, parameter_1235, parameter_1239, parameter_1236, parameter_1238, parameter_1237, parameter_1240, parameter_1244, parameter_1241, parameter_1243, parameter_1242, parameter_1245, parameter_1249, parameter_1246, parameter_1248, parameter_1247, parameter_1250, parameter_1254, parameter_1251, parameter_1253, parameter_1252, parameter_1255, parameter_1259, parameter_1256, parameter_1258, parameter_1257, parameter_1260, parameter_1264, parameter_1261, parameter_1263, parameter_1262, parameter_1265, parameter_1269, parameter_1266, parameter_1268, parameter_1267, parameter_1270, parameter_1274, parameter_1271, parameter_1273, parameter_1272, parameter_1275, parameter_1279, parameter_1276, parameter_1278, parameter_1277, parameter_1280, parameter_1284, parameter_1281, parameter_1283, parameter_1282, parameter_1285, parameter_1289, parameter_1286, parameter_1288, parameter_1287, parameter_1290, parameter_1294, parameter_1291, parameter_1293, parameter_1292, parameter_1295, parameter_1299, parameter_1296, parameter_1298, parameter_1297, parameter_1300, parameter_1304, parameter_1301, parameter_1303, parameter_1302, parameter_1305, parameter_1309, parameter_1306, parameter_1308, parameter_1307, parameter_1310, parameter_1314, parameter_1311, parameter_1313, parameter_1312, parameter_1315, parameter_1319, parameter_1316, parameter_1318, parameter_1317, parameter_1320, parameter_1324, parameter_1321, parameter_1323, parameter_1322, parameter_1325, parameter_1329, parameter_1326, parameter_1328, parameter_1327, parameter_1330, parameter_1334, parameter_1331, parameter_1333, parameter_1332, parameter_1335, parameter_1339, parameter_1336, parameter_1338, parameter_1337, parameter_1340, parameter_1341, feed_0):
        return self.builtin_module_3282_0_0(parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_114, parameter_111, parameter_113, parameter_112, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_154, parameter_151, parameter_153, parameter_152, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_169, parameter_166, parameter_168, parameter_167, parameter_170, parameter_174, parameter_171, parameter_173, parameter_172, parameter_175, parameter_179, parameter_176, parameter_178, parameter_177, parameter_180, parameter_184, parameter_181, parameter_183, parameter_182, parameter_185, parameter_189, parameter_186, parameter_188, parameter_187, parameter_190, parameter_194, parameter_191, parameter_193, parameter_192, parameter_195, parameter_199, parameter_196, parameter_198, parameter_197, parameter_200, parameter_204, parameter_201, parameter_203, parameter_202, parameter_205, parameter_209, parameter_206, parameter_208, parameter_207, parameter_210, parameter_214, parameter_211, parameter_213, parameter_212, parameter_215, parameter_219, parameter_216, parameter_218, parameter_217, parameter_220, parameter_224, parameter_221, parameter_223, parameter_222, parameter_225, parameter_229, parameter_226, parameter_228, parameter_227, parameter_230, parameter_234, parameter_231, parameter_233, parameter_232, parameter_235, parameter_239, parameter_236, parameter_238, parameter_237, parameter_240, parameter_244, parameter_241, parameter_243, parameter_242, parameter_245, parameter_249, parameter_246, parameter_248, parameter_247, parameter_250, parameter_254, parameter_251, parameter_253, parameter_252, parameter_255, parameter_259, parameter_256, parameter_258, parameter_257, parameter_260, parameter_264, parameter_261, parameter_263, parameter_262, parameter_265, parameter_269, parameter_266, parameter_268, parameter_267, parameter_270, parameter_274, parameter_271, parameter_273, parameter_272, parameter_275, parameter_279, parameter_276, parameter_278, parameter_277, parameter_280, parameter_284, parameter_281, parameter_283, parameter_282, parameter_285, parameter_289, parameter_286, parameter_288, parameter_287, parameter_290, parameter_294, parameter_291, parameter_293, parameter_292, parameter_295, parameter_299, parameter_296, parameter_298, parameter_297, parameter_300, parameter_304, parameter_301, parameter_303, parameter_302, parameter_305, parameter_309, parameter_306, parameter_308, parameter_307, parameter_310, parameter_314, parameter_311, parameter_313, parameter_312, parameter_315, parameter_319, parameter_316, parameter_318, parameter_317, parameter_320, parameter_324, parameter_321, parameter_323, parameter_322, parameter_325, parameter_329, parameter_326, parameter_328, parameter_327, parameter_330, parameter_334, parameter_331, parameter_333, parameter_332, parameter_335, parameter_339, parameter_336, parameter_338, parameter_337, parameter_340, parameter_344, parameter_341, parameter_343, parameter_342, parameter_345, parameter_349, parameter_346, parameter_348, parameter_347, parameter_350, parameter_354, parameter_351, parameter_353, parameter_352, parameter_355, parameter_359, parameter_356, parameter_358, parameter_357, parameter_360, parameter_364, parameter_361, parameter_363, parameter_362, parameter_365, parameter_369, parameter_366, parameter_368, parameter_367, parameter_370, parameter_374, parameter_371, parameter_373, parameter_372, parameter_375, parameter_379, parameter_376, parameter_378, parameter_377, parameter_380, parameter_384, parameter_381, parameter_383, parameter_382, parameter_385, parameter_389, parameter_386, parameter_388, parameter_387, parameter_390, parameter_394, parameter_391, parameter_393, parameter_392, parameter_395, parameter_399, parameter_396, parameter_398, parameter_397, parameter_400, parameter_404, parameter_401, parameter_403, parameter_402, parameter_405, parameter_409, parameter_406, parameter_408, parameter_407, parameter_410, parameter_414, parameter_411, parameter_413, parameter_412, parameter_415, parameter_419, parameter_416, parameter_418, parameter_417, parameter_420, parameter_424, parameter_421, parameter_423, parameter_422, parameter_425, parameter_429, parameter_426, parameter_428, parameter_427, parameter_430, parameter_434, parameter_431, parameter_433, parameter_432, parameter_435, parameter_439, parameter_436, parameter_438, parameter_437, parameter_440, parameter_444, parameter_441, parameter_443, parameter_442, parameter_445, parameter_449, parameter_446, parameter_448, parameter_447, parameter_450, parameter_454, parameter_451, parameter_453, parameter_452, parameter_455, parameter_459, parameter_456, parameter_458, parameter_457, parameter_460, parameter_464, parameter_461, parameter_463, parameter_462, parameter_465, parameter_469, parameter_466, parameter_468, parameter_467, parameter_470, parameter_474, parameter_471, parameter_473, parameter_472, parameter_475, parameter_479, parameter_476, parameter_478, parameter_477, parameter_480, parameter_484, parameter_481, parameter_483, parameter_482, parameter_485, parameter_489, parameter_486, parameter_488, parameter_487, parameter_490, parameter_494, parameter_491, parameter_493, parameter_492, parameter_495, parameter_499, parameter_496, parameter_498, parameter_497, parameter_500, parameter_504, parameter_501, parameter_503, parameter_502, parameter_505, parameter_509, parameter_506, parameter_508, parameter_507, parameter_510, parameter_514, parameter_511, parameter_513, parameter_512, parameter_515, parameter_519, parameter_516, parameter_518, parameter_517, parameter_520, parameter_524, parameter_521, parameter_523, parameter_522, parameter_525, parameter_529, parameter_526, parameter_528, parameter_527, parameter_530, parameter_534, parameter_531, parameter_533, parameter_532, parameter_535, parameter_539, parameter_536, parameter_538, parameter_537, parameter_540, parameter_544, parameter_541, parameter_543, parameter_542, parameter_545, parameter_549, parameter_546, parameter_548, parameter_547, parameter_550, parameter_554, parameter_551, parameter_553, parameter_552, parameter_555, parameter_559, parameter_556, parameter_558, parameter_557, parameter_560, parameter_564, parameter_561, parameter_563, parameter_562, parameter_565, parameter_569, parameter_566, parameter_568, parameter_567, parameter_570, parameter_574, parameter_571, parameter_573, parameter_572, parameter_575, parameter_579, parameter_576, parameter_578, parameter_577, parameter_580, parameter_584, parameter_581, parameter_583, parameter_582, parameter_585, parameter_589, parameter_586, parameter_588, parameter_587, parameter_590, parameter_594, parameter_591, parameter_593, parameter_592, parameter_595, parameter_599, parameter_596, parameter_598, parameter_597, parameter_600, parameter_604, parameter_601, parameter_603, parameter_602, parameter_605, parameter_609, parameter_606, parameter_608, parameter_607, parameter_610, parameter_614, parameter_611, parameter_613, parameter_612, parameter_615, parameter_619, parameter_616, parameter_618, parameter_617, parameter_620, parameter_624, parameter_621, parameter_623, parameter_622, parameter_625, parameter_629, parameter_626, parameter_628, parameter_627, parameter_630, parameter_634, parameter_631, parameter_633, parameter_632, parameter_635, parameter_639, parameter_636, parameter_638, parameter_637, parameter_640, parameter_644, parameter_641, parameter_643, parameter_642, parameter_645, parameter_649, parameter_646, parameter_648, parameter_647, parameter_650, parameter_654, parameter_651, parameter_653, parameter_652, parameter_655, parameter_659, parameter_656, parameter_658, parameter_657, parameter_660, parameter_664, parameter_661, parameter_663, parameter_662, parameter_665, parameter_669, parameter_666, parameter_668, parameter_667, parameter_670, parameter_674, parameter_671, parameter_673, parameter_672, parameter_675, parameter_679, parameter_676, parameter_678, parameter_677, parameter_680, parameter_684, parameter_681, parameter_683, parameter_682, parameter_685, parameter_689, parameter_686, parameter_688, parameter_687, parameter_690, parameter_694, parameter_691, parameter_693, parameter_692, parameter_695, parameter_699, parameter_696, parameter_698, parameter_697, parameter_700, parameter_704, parameter_701, parameter_703, parameter_702, parameter_705, parameter_709, parameter_706, parameter_708, parameter_707, parameter_710, parameter_714, parameter_711, parameter_713, parameter_712, parameter_715, parameter_719, parameter_716, parameter_718, parameter_717, parameter_720, parameter_724, parameter_721, parameter_723, parameter_722, parameter_725, parameter_729, parameter_726, parameter_728, parameter_727, parameter_730, parameter_734, parameter_731, parameter_733, parameter_732, parameter_735, parameter_739, parameter_736, parameter_738, parameter_737, parameter_740, parameter_744, parameter_741, parameter_743, parameter_742, parameter_745, parameter_749, parameter_746, parameter_748, parameter_747, parameter_750, parameter_754, parameter_751, parameter_753, parameter_752, parameter_755, parameter_759, parameter_756, parameter_758, parameter_757, parameter_760, parameter_764, parameter_761, parameter_763, parameter_762, parameter_765, parameter_769, parameter_766, parameter_768, parameter_767, parameter_770, parameter_774, parameter_771, parameter_773, parameter_772, parameter_775, parameter_779, parameter_776, parameter_778, parameter_777, parameter_780, parameter_784, parameter_781, parameter_783, parameter_782, parameter_785, parameter_789, parameter_786, parameter_788, parameter_787, parameter_790, parameter_794, parameter_791, parameter_793, parameter_792, parameter_795, parameter_799, parameter_796, parameter_798, parameter_797, parameter_800, parameter_804, parameter_801, parameter_803, parameter_802, parameter_805, parameter_809, parameter_806, parameter_808, parameter_807, parameter_810, parameter_814, parameter_811, parameter_813, parameter_812, parameter_815, parameter_819, parameter_816, parameter_818, parameter_817, parameter_820, parameter_824, parameter_821, parameter_823, parameter_822, parameter_825, parameter_829, parameter_826, parameter_828, parameter_827, parameter_830, parameter_834, parameter_831, parameter_833, parameter_832, parameter_835, parameter_839, parameter_836, parameter_838, parameter_837, parameter_840, parameter_844, parameter_841, parameter_843, parameter_842, parameter_845, parameter_849, parameter_846, parameter_848, parameter_847, parameter_850, parameter_854, parameter_851, parameter_853, parameter_852, parameter_855, parameter_859, parameter_856, parameter_858, parameter_857, parameter_860, parameter_864, parameter_861, parameter_863, parameter_862, parameter_865, parameter_869, parameter_866, parameter_868, parameter_867, parameter_870, parameter_874, parameter_871, parameter_873, parameter_872, parameter_875, parameter_879, parameter_876, parameter_878, parameter_877, parameter_880, parameter_884, parameter_881, parameter_883, parameter_882, parameter_885, parameter_889, parameter_886, parameter_888, parameter_887, parameter_890, parameter_894, parameter_891, parameter_893, parameter_892, parameter_895, parameter_899, parameter_896, parameter_898, parameter_897, parameter_900, parameter_904, parameter_901, parameter_903, parameter_902, parameter_905, parameter_909, parameter_906, parameter_908, parameter_907, parameter_910, parameter_914, parameter_911, parameter_913, parameter_912, parameter_915, parameter_919, parameter_916, parameter_918, parameter_917, parameter_920, parameter_924, parameter_921, parameter_923, parameter_922, parameter_925, parameter_929, parameter_926, parameter_928, parameter_927, parameter_930, parameter_934, parameter_931, parameter_933, parameter_932, parameter_935, parameter_939, parameter_936, parameter_938, parameter_937, parameter_940, parameter_944, parameter_941, parameter_943, parameter_942, parameter_945, parameter_949, parameter_946, parameter_948, parameter_947, parameter_950, parameter_954, parameter_951, parameter_953, parameter_952, parameter_955, parameter_959, parameter_956, parameter_958, parameter_957, parameter_960, parameter_964, parameter_961, parameter_963, parameter_962, parameter_965, parameter_969, parameter_966, parameter_968, parameter_967, parameter_970, parameter_974, parameter_971, parameter_973, parameter_972, parameter_975, parameter_979, parameter_976, parameter_978, parameter_977, parameter_980, parameter_984, parameter_981, parameter_983, parameter_982, parameter_985, parameter_989, parameter_986, parameter_988, parameter_987, parameter_990, parameter_994, parameter_991, parameter_993, parameter_992, parameter_995, parameter_999, parameter_996, parameter_998, parameter_997, parameter_1000, parameter_1004, parameter_1001, parameter_1003, parameter_1002, parameter_1005, parameter_1009, parameter_1006, parameter_1008, parameter_1007, parameter_1010, parameter_1014, parameter_1011, parameter_1013, parameter_1012, parameter_1015, parameter_1019, parameter_1016, parameter_1018, parameter_1017, parameter_1020, parameter_1024, parameter_1021, parameter_1023, parameter_1022, parameter_1025, parameter_1029, parameter_1026, parameter_1028, parameter_1027, parameter_1030, parameter_1034, parameter_1031, parameter_1033, parameter_1032, parameter_1035, parameter_1039, parameter_1036, parameter_1038, parameter_1037, parameter_1040, parameter_1044, parameter_1041, parameter_1043, parameter_1042, parameter_1045, parameter_1049, parameter_1046, parameter_1048, parameter_1047, parameter_1050, parameter_1054, parameter_1051, parameter_1053, parameter_1052, parameter_1055, parameter_1059, parameter_1056, parameter_1058, parameter_1057, parameter_1060, parameter_1064, parameter_1061, parameter_1063, parameter_1062, parameter_1065, parameter_1069, parameter_1066, parameter_1068, parameter_1067, parameter_1070, parameter_1074, parameter_1071, parameter_1073, parameter_1072, parameter_1075, parameter_1079, parameter_1076, parameter_1078, parameter_1077, parameter_1080, parameter_1084, parameter_1081, parameter_1083, parameter_1082, parameter_1085, parameter_1089, parameter_1086, parameter_1088, parameter_1087, parameter_1090, parameter_1094, parameter_1091, parameter_1093, parameter_1092, parameter_1095, parameter_1099, parameter_1096, parameter_1098, parameter_1097, parameter_1100, parameter_1104, parameter_1101, parameter_1103, parameter_1102, parameter_1105, parameter_1109, parameter_1106, parameter_1108, parameter_1107, parameter_1110, parameter_1114, parameter_1111, parameter_1113, parameter_1112, parameter_1115, parameter_1119, parameter_1116, parameter_1118, parameter_1117, parameter_1120, parameter_1124, parameter_1121, parameter_1123, parameter_1122, parameter_1125, parameter_1129, parameter_1126, parameter_1128, parameter_1127, parameter_1130, parameter_1134, parameter_1131, parameter_1133, parameter_1132, parameter_1135, parameter_1139, parameter_1136, parameter_1138, parameter_1137, parameter_1140, parameter_1144, parameter_1141, parameter_1143, parameter_1142, parameter_1145, parameter_1149, parameter_1146, parameter_1148, parameter_1147, parameter_1150, parameter_1154, parameter_1151, parameter_1153, parameter_1152, parameter_1155, parameter_1159, parameter_1156, parameter_1158, parameter_1157, parameter_1160, parameter_1164, parameter_1161, parameter_1163, parameter_1162, parameter_1165, parameter_1169, parameter_1166, parameter_1168, parameter_1167, parameter_1170, parameter_1174, parameter_1171, parameter_1173, parameter_1172, parameter_1175, parameter_1179, parameter_1176, parameter_1178, parameter_1177, parameter_1180, parameter_1184, parameter_1181, parameter_1183, parameter_1182, parameter_1185, parameter_1189, parameter_1186, parameter_1188, parameter_1187, parameter_1190, parameter_1194, parameter_1191, parameter_1193, parameter_1192, parameter_1195, parameter_1199, parameter_1196, parameter_1198, parameter_1197, parameter_1200, parameter_1204, parameter_1201, parameter_1203, parameter_1202, parameter_1205, parameter_1209, parameter_1206, parameter_1208, parameter_1207, parameter_1210, parameter_1214, parameter_1211, parameter_1213, parameter_1212, parameter_1215, parameter_1219, parameter_1216, parameter_1218, parameter_1217, parameter_1220, parameter_1224, parameter_1221, parameter_1223, parameter_1222, parameter_1225, parameter_1229, parameter_1226, parameter_1228, parameter_1227, parameter_1230, parameter_1234, parameter_1231, parameter_1233, parameter_1232, parameter_1235, parameter_1239, parameter_1236, parameter_1238, parameter_1237, parameter_1240, parameter_1244, parameter_1241, parameter_1243, parameter_1242, parameter_1245, parameter_1249, parameter_1246, parameter_1248, parameter_1247, parameter_1250, parameter_1254, parameter_1251, parameter_1253, parameter_1252, parameter_1255, parameter_1259, parameter_1256, parameter_1258, parameter_1257, parameter_1260, parameter_1264, parameter_1261, parameter_1263, parameter_1262, parameter_1265, parameter_1269, parameter_1266, parameter_1268, parameter_1267, parameter_1270, parameter_1274, parameter_1271, parameter_1273, parameter_1272, parameter_1275, parameter_1279, parameter_1276, parameter_1278, parameter_1277, parameter_1280, parameter_1284, parameter_1281, parameter_1283, parameter_1282, parameter_1285, parameter_1289, parameter_1286, parameter_1288, parameter_1287, parameter_1290, parameter_1294, parameter_1291, parameter_1293, parameter_1292, parameter_1295, parameter_1299, parameter_1296, parameter_1298, parameter_1297, parameter_1300, parameter_1304, parameter_1301, parameter_1303, parameter_1302, parameter_1305, parameter_1309, parameter_1306, parameter_1308, parameter_1307, parameter_1310, parameter_1314, parameter_1311, parameter_1313, parameter_1312, parameter_1315, parameter_1319, parameter_1316, parameter_1318, parameter_1317, parameter_1320, parameter_1324, parameter_1321, parameter_1323, parameter_1322, parameter_1325, parameter_1329, parameter_1326, parameter_1328, parameter_1327, parameter_1330, parameter_1334, parameter_1331, parameter_1333, parameter_1332, parameter_1335, parameter_1339, parameter_1336, parameter_1338, parameter_1337, parameter_1340, parameter_1341, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_3282_0_0(CinnTestBase, unittest.TestCase):
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
            paddle.uniform([16, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_9
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([16, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_14
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([32, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_19
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([32, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_24
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([16, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_29
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([32, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_34
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([40, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_39
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([32, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_44
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([80, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_49
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([20, 20, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_54
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([20, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_59
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([20, 20, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_64
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([20, 20, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_69
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([20, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_74
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([20, 20, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_79
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_84
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_89
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_94
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_99
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_104
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_109
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([40, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_114
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_119
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([80, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_124
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([20, 20, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_129
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([20, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_134
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([20, 20, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_139
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([20, 20, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_144
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([20, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_149
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([20, 20, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_154
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_159
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_164
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_169
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_174
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_179
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_184
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([40, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_189
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_194
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([80, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_199
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([80, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_204
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([160, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_209
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([20, 20, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_214
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_213
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_215
            paddle.uniform([20, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_219
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_218
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_217
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([20, 20, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_224
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_221
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_223
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_225
            paddle.uniform([20, 20, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_229
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_227
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([20, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_234
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_233
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([20, 20, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_239
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_238
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_244
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_241
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_243
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_245
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_249
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_246
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_248
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_250
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_254
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_251
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_253
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_252
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_255
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_259
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_256
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_258
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_257
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_264
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_261
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_263
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_262
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_265
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_269
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_266
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_268
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_267
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_270
            paddle.uniform([80, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_274
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_271
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_273
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_272
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_275
            paddle.uniform([80, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_279
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_276
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_278
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_277
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_280
            paddle.uniform([80, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_284
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_281
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_283
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_282
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_285
            paddle.uniform([80, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_289
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_286
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_288
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_287
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_290
            paddle.uniform([80, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_294
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_291
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_293
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_292
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_295
            paddle.uniform([80, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_299
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_296
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_298
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_297
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_300
            paddle.uniform([40, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_304
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_301
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_303
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_302
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_305
            paddle.uniform([40, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_309
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_306
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_308
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_307
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_310
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_314
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_311
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_313
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_312
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_315
            paddle.uniform([80, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_319
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_316
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_318
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_317
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_320
            paddle.uniform([80, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_324
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_321
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_323
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_322
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_325
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_329
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_326
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_328
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_327
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_330
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_334
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_331
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_333
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_332
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_335
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_339
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_336
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_338
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_337
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_340
            paddle.uniform([160, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_344
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_341
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_343
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_342
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_345
            paddle.uniform([80, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_349
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_346
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_348
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_347
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_350
            paddle.uniform([160, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_354
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_351
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_353
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_352
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_355
            paddle.uniform([20, 20, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_359
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_356
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_358
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_357
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_360
            paddle.uniform([20, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_364
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_361
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_363
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_362
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_365
            paddle.uniform([20, 20, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_369
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_366
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_368
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_367
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_370
            paddle.uniform([20, 20, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_374
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_371
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_373
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_372
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_375
            paddle.uniform([20, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_379
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_376
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_378
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_377
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_380
            paddle.uniform([20, 20, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_384
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_381
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_383
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_382
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_385
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_389
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_386
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_388
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_387
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_390
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_394
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_391
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_393
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_392
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_395
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_399
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_396
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_398
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_397
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_400
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_404
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_401
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_403
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_402
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_405
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_409
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_406
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_408
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_407
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_410
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_414
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_411
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_413
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_412
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_415
            paddle.uniform([80, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_419
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_416
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_418
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_417
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_420
            paddle.uniform([80, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_424
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_421
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_423
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_422
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_425
            paddle.uniform([80, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_429
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_426
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_428
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_427
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_430
            paddle.uniform([80, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_434
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_431
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_433
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_432
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_435
            paddle.uniform([80, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_439
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_436
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_438
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_437
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_440
            paddle.uniform([80, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_444
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_441
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_443
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_442
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_445
            paddle.uniform([40, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_449
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_446
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_448
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_447
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_450
            paddle.uniform([40, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_454
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_451
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_453
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_452
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_455
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_459
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_456
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_458
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_457
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_460
            paddle.uniform([80, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_464
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_461
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_463
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_462
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_465
            paddle.uniform([80, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_469
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_466
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_468
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_467
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_470
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_474
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_471
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_473
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_472
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_475
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_479
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_476
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_478
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_477
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_480
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_484
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_481
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_483
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_482
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_485
            paddle.uniform([160, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_489
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_486
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_488
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_487
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_490
            paddle.uniform([80, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_494
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_491
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_493
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_492
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_495
            paddle.uniform([160, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_499
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_496
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_498
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_497
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_500
            paddle.uniform([20, 20, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_504
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_501
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_503
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_502
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_505
            paddle.uniform([20, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_509
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_506
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_508
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_507
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_510
            paddle.uniform([20, 20, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_514
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_511
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_513
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_512
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_515
            paddle.uniform([20, 20, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_519
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_516
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_518
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_517
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_520
            paddle.uniform([20, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_524
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_521
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_523
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_522
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_525
            paddle.uniform([20, 20, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_529
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_526
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_528
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_527
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_530
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_534
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_531
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_533
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_532
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_535
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_539
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_536
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_538
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_537
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_540
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_544
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_541
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_543
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_542
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_545
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_549
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_546
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_548
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_547
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_550
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_554
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_551
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_553
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_552
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_555
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_559
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_556
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_558
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_557
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_560
            paddle.uniform([80, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_564
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_561
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_563
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_562
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_565
            paddle.uniform([80, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_569
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_566
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_568
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_567
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_570
            paddle.uniform([80, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_574
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_571
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_573
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_572
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_575
            paddle.uniform([80, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_579
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_576
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_578
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_577
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_580
            paddle.uniform([80, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_584
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_581
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_583
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_582
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_585
            paddle.uniform([80, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_589
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_586
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_588
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_587
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_590
            paddle.uniform([40, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_594
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_591
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_593
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_592
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_595
            paddle.uniform([40, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_599
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_596
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_598
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_597
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_600
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_604
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_601
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_603
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_602
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_605
            paddle.uniform([80, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_609
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_606
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_608
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_607
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_610
            paddle.uniform([80, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_614
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_611
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_613
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_612
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_615
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_619
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_616
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_618
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_617
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_620
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_624
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_621
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_623
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_622
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_625
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_629
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_626
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_628
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_627
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_630
            paddle.uniform([160, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_634
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_631
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_633
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_632
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_635
            paddle.uniform([80, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_639
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_636
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_638
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_637
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_640
            paddle.uniform([160, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_644
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_641
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_643
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_642
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_645
            paddle.uniform([20, 20, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_649
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_646
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_648
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_647
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_650
            paddle.uniform([20, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_654
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_651
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_653
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_652
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_655
            paddle.uniform([20, 20, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_659
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_656
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_658
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_657
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_660
            paddle.uniform([20, 20, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_664
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_661
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_663
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_662
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_665
            paddle.uniform([20, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_669
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_666
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_668
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_667
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_670
            paddle.uniform([20, 20, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_674
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_671
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_673
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_672
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_675
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_679
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_676
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_678
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_677
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_680
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_684
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_681
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_683
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_682
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_685
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_689
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_686
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_688
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_687
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_690
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_694
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_691
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_693
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_692
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_695
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_699
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_696
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_698
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_697
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_700
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_704
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_701
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_703
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_702
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_705
            paddle.uniform([80, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_709
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_706
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_708
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_707
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_710
            paddle.uniform([80, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_714
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_711
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_713
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_712
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_715
            paddle.uniform([80, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_719
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_716
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_718
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_717
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_720
            paddle.uniform([80, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_724
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_721
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_723
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_722
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_725
            paddle.uniform([80, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_729
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_726
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_728
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_727
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_730
            paddle.uniform([80, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_734
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_731
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_733
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_732
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_735
            paddle.uniform([40, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_739
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_736
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_738
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_737
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_740
            paddle.uniform([40, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_744
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_741
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_743
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_742
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_745
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_749
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_746
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_748
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_747
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_750
            paddle.uniform([80, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_754
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_751
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_753
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_752
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_755
            paddle.uniform([80, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_759
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_756
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_758
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_757
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_760
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_764
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_761
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_763
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_762
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_765
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_769
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_766
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_768
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_767
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_770
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_774
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_771
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_773
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_772
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_775
            paddle.uniform([160, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_779
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_776
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_778
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_777
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_780
            paddle.uniform([80, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_784
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_781
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_783
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_782
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_785
            paddle.uniform([160, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_789
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_786
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_788
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_787
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_790
            paddle.uniform([160, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_794
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_791
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_793
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_792
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_795
            paddle.uniform([320, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_799
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_796
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_798
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_797
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_800
            paddle.uniform([20, 20, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_804
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_801
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_803
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_802
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_805
            paddle.uniform([20, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_809
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_806
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_808
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_807
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_810
            paddle.uniform([20, 20, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_814
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_811
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_813
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_812
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_815
            paddle.uniform([20, 20, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_819
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_816
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_818
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_817
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_820
            paddle.uniform([20, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_824
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_821
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_823
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_822
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_825
            paddle.uniform([20, 20, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_829
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_826
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_828
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_827
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_830
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_834
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_831
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_833
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_832
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_835
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_839
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_836
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_838
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_837
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_840
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_844
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_841
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_843
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_842
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_845
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_849
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_846
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_848
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_847
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_850
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_854
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_851
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_853
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_852
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_855
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_859
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_856
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_858
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_857
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_860
            paddle.uniform([80, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_864
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_861
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_863
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_862
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_865
            paddle.uniform([80, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_869
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_866
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_868
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_867
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_870
            paddle.uniform([80, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_874
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_871
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_873
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_872
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_875
            paddle.uniform([80, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_879
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_876
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_878
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_877
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_880
            paddle.uniform([80, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_884
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_881
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_883
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_882
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_885
            paddle.uniform([80, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_889
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_886
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_888
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_887
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_890
            paddle.uniform([160, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_894
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_891
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_893
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_892
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_895
            paddle.uniform([160, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_899
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_896
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_898
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_897
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_900
            paddle.uniform([160, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_904
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_901
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_903
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_902
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_905
            paddle.uniform([160, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_909
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_906
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_908
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_907
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_910
            paddle.uniform([160, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_914
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_911
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_913
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_912
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_915
            paddle.uniform([160, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_919
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_916
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_918
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_917
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_920
            paddle.uniform([40, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_924
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_921
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_923
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_922
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_925
            paddle.uniform([40, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_929
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_926
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_928
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_927
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_930
            paddle.uniform([40, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_934
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_931
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_933
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_932
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_935
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_939
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_936
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_938
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_937
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_940
            paddle.uniform([80, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_944
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_941
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_943
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_942
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_945
            paddle.uniform([80, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_949
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_946
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_948
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_947
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_950
            paddle.uniform([80, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_954
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_951
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_953
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_952
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_955
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_959
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_956
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_958
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_957
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_960
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_964
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_961
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_963
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_962
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_965
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_969
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_966
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_968
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_967
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_970
            paddle.uniform([160, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_974
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_971
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_973
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_972
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_975
            paddle.uniform([80, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_979
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_976
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_978
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_977
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_980
            paddle.uniform([160, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_984
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_981
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_983
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_982
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_985
            paddle.uniform([160, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_989
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_986
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_988
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_987
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_990
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_994
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_991
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_993
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_992
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_995
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_999
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_996
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_998
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_997
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1000
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_1004
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1001
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1003
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1002
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1005
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1009
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1006
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1008
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1007
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1010
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_1014
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1011
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1013
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1012
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1015
            paddle.uniform([320, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1019
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_1016
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_1018
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_1017
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_1020
            paddle.uniform([80, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_1024
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1021
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1023
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1022
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1025
            paddle.uniform([80, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1029
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1026
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1028
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1027
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1030
            paddle.uniform([80, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_1034
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1031
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1033
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1032
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1035
            paddle.uniform([320, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1039
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_1036
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_1038
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_1037
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_1040
            paddle.uniform([160, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_1044
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1041
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1043
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1042
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1045
            paddle.uniform([320, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1049
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_1046
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_1048
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_1047
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_1050
            paddle.uniform([20, 20, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1054
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_1051
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_1053
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_1052
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_1055
            paddle.uniform([20, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_1059
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_1056
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_1058
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_1057
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_1060
            paddle.uniform([20, 20, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1064
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_1061
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_1063
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_1062
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_1065
            paddle.uniform([20, 20, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1069
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_1066
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_1068
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_1067
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_1070
            paddle.uniform([20, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_1074
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_1071
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_1073
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_1072
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_1075
            paddle.uniform([20, 20, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1079
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_1076
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_1078
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_1077
            paddle.uniform([20], dtype='float32', min=0, max=0.5),
            # parameter_1080
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1084
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1081
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1083
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1082
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1085
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_1089
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1086
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1088
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1087
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1090
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1094
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1091
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1093
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1092
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1095
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1099
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1096
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1098
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1097
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1100
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_1104
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1101
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1103
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1102
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1105
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1109
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1106
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1108
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1107
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1110
            paddle.uniform([80, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1114
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1111
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1113
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1112
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1115
            paddle.uniform([80, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_1119
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1116
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1118
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1117
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1120
            paddle.uniform([80, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1124
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1121
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1123
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1122
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1125
            paddle.uniform([80, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1129
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1126
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1128
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1127
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1130
            paddle.uniform([80, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_1134
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1131
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1133
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1132
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1135
            paddle.uniform([80, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1139
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1136
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1138
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1137
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1140
            paddle.uniform([160, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1144
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1141
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1143
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1142
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1145
            paddle.uniform([160, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_1149
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1146
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1148
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1147
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1150
            paddle.uniform([160, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1154
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1151
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1153
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1152
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1155
            paddle.uniform([160, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1159
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1156
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1158
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1157
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1160
            paddle.uniform([160, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_1164
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1161
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1163
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1162
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1165
            paddle.uniform([160, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1169
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1166
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1168
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1167
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1170
            paddle.uniform([40, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1174
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1171
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1173
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1172
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1175
            paddle.uniform([40, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1179
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1176
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1178
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1177
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1180
            paddle.uniform([40, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1184
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1181
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1183
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1182
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1185
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_1189
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1186
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1188
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1187
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1190
            paddle.uniform([80, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1194
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1191
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1193
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1192
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1195
            paddle.uniform([80, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1199
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1196
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1198
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1197
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1200
            paddle.uniform([80, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1204
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1201
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1203
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1202
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1205
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_1209
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1206
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1208
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1207
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1210
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1214
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1211
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1213
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1212
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1215
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_1219
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1216
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1218
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1217
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1220
            paddle.uniform([160, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1224
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1221
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1223
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1222
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1225
            paddle.uniform([80, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_1229
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1226
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1228
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1227
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1230
            paddle.uniform([160, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1234
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1231
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1233
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1232
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1235
            paddle.uniform([160, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1239
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1236
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1238
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1237
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1240
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_1244
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1241
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1243
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1242
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1245
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1249
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1246
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1248
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1247
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1250
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_1254
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1251
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1253
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1252
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1255
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1259
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1256
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1258
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1257
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1260
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_1264
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1261
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1263
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1262
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1265
            paddle.uniform([320, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1269
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_1266
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_1268
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_1267
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_1270
            paddle.uniform([80, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_1274
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1271
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1273
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1272
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1275
            paddle.uniform([80, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1279
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1276
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1278
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1277
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1280
            paddle.uniform([80, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_1284
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1281
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1283
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1282
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1285
            paddle.uniform([320, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1289
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_1286
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_1288
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_1287
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_1290
            paddle.uniform([160, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_1294
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1291
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1293
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1292
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1295
            paddle.uniform([320, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1299
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_1296
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_1298
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_1297
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_1300
            paddle.uniform([320, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_1304
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_1301
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_1303
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_1302
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_1305
            paddle.uniform([160, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1309
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1306
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1308
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1307
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1310
            paddle.uniform([160, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_1314
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1311
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1313
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1312
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            # parameter_1315
            paddle.uniform([80, 160, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1319
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1316
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1318
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1317
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1320
            paddle.uniform([80, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_1324
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1321
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1323
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1322
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            # parameter_1325
            paddle.uniform([40, 80, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1329
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1326
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1328
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1327
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1330
            paddle.uniform([40, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_1334
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1331
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1333
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1332
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1335
            paddle.uniform([40, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1339
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1336
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1338
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1337
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            # parameter_1340
            paddle.uniform([17, 40, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_1341
            paddle.uniform([17], dtype='float16', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 3, 128, 96], dtype='float32', min=0, max=0.5),
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
            paddle.static.InputSpec(shape=[16, 1, 3, 3], dtype='float16'),
            # parameter_9
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[16, 16, 1, 1], dtype='float16'),
            # parameter_14
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[32, 16, 1, 1], dtype='float16'),
            # parameter_19
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[32, 1, 3, 3], dtype='float16'),
            # parameter_24
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[16, 32, 1, 1], dtype='float16'),
            # parameter_29
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[32, 1, 3, 3], dtype='float16'),
            # parameter_34
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[40, 32, 1, 1], dtype='float16'),
            # parameter_39
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[32, 1, 3, 3], dtype='float16'),
            # parameter_44
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[80, 32, 1, 1], dtype='float16'),
            # parameter_49
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[20, 20, 1, 1], dtype='float16'),
            # parameter_54
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[20, 1, 3, 3], dtype='float16'),
            # parameter_59
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[20, 20, 1, 1], dtype='float16'),
            # parameter_64
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[20, 20, 1, 1], dtype='float16'),
            # parameter_69
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[20, 1, 3, 3], dtype='float16'),
            # parameter_74
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[20, 20, 1, 1], dtype='float16'),
            # parameter_79
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_84
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_89
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_94
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_99
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_104
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_109
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[40, 80, 1, 1], dtype='float16'),
            # parameter_114
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_119
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[80, 40, 1, 1], dtype='float16'),
            # parameter_124
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[20, 20, 1, 1], dtype='float16'),
            # parameter_129
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[20, 1, 3, 3], dtype='float16'),
            # parameter_134
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[20, 20, 1, 1], dtype='float16'),
            # parameter_139
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[20, 20, 1, 1], dtype='float16'),
            # parameter_144
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[20, 1, 3, 3], dtype='float16'),
            # parameter_149
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[20, 20, 1, 1], dtype='float16'),
            # parameter_154
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_159
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_164
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_169
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_174
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_179
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_184
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[40, 80, 1, 1], dtype='float16'),
            # parameter_189
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_194
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[80, 40, 1, 1], dtype='float16'),
            # parameter_199
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[80, 1, 3, 3], dtype='float16'),
            # parameter_204
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[160, 80, 1, 1], dtype='float16'),
            # parameter_209
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[20, 20, 1, 1], dtype='float16'),
            # parameter_214
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_213
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_215
            paddle.static.InputSpec(shape=[20, 1, 3, 3], dtype='float16'),
            # parameter_219
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_218
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_217
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[20, 20, 1, 1], dtype='float16'),
            # parameter_224
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_221
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_223
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_225
            paddle.static.InputSpec(shape=[20, 20, 1, 1], dtype='float16'),
            # parameter_229
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_227
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[20, 1, 3, 3], dtype='float16'),
            # parameter_234
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_233
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[20, 20, 1, 1], dtype='float16'),
            # parameter_239
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_238
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_244
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_241
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_243
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_245
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_249
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_246
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_248
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_250
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_254
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_251
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_253
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_252
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_255
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_259
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_256
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_258
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_257
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_264
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_261
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_263
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_262
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_265
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_269
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_266
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_268
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_267
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_270
            paddle.static.InputSpec(shape=[80, 80, 1, 1], dtype='float16'),
            # parameter_274
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_271
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_273
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_272
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_275
            paddle.static.InputSpec(shape=[80, 1, 3, 3], dtype='float16'),
            # parameter_279
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_276
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_278
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_277
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_280
            paddle.static.InputSpec(shape=[80, 80, 1, 1], dtype='float16'),
            # parameter_284
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_281
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_283
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_282
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_285
            paddle.static.InputSpec(shape=[80, 80, 1, 1], dtype='float16'),
            # parameter_289
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_286
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_288
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_287
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_290
            paddle.static.InputSpec(shape=[80, 1, 3, 3], dtype='float16'),
            # parameter_294
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_291
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_293
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_292
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_295
            paddle.static.InputSpec(shape=[80, 80, 1, 1], dtype='float16'),
            # parameter_299
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_296
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_298
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_297
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_300
            paddle.static.InputSpec(shape=[40, 80, 1, 1], dtype='float16'),
            # parameter_304
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_301
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_303
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_302
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_305
            paddle.static.InputSpec(shape=[40, 160, 1, 1], dtype='float16'),
            # parameter_309
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_306
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_308
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_307
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_310
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_314
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_311
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_313
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_312
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_315
            paddle.static.InputSpec(shape=[80, 40, 1, 1], dtype='float16'),
            # parameter_319
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_316
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_318
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_317
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_320
            paddle.static.InputSpec(shape=[80, 160, 1, 1], dtype='float16'),
            # parameter_324
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_321
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_323
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_322
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_325
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_329
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_326
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_328
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_327
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_330
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_334
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_331
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_333
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_332
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_335
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_339
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_336
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_338
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_337
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_340
            paddle.static.InputSpec(shape=[160, 40, 1, 1], dtype='float16'),
            # parameter_344
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_341
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_343
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_342
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_345
            paddle.static.InputSpec(shape=[80, 1, 3, 3], dtype='float16'),
            # parameter_349
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_346
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_348
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_347
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_350
            paddle.static.InputSpec(shape=[160, 80, 1, 1], dtype='float16'),
            # parameter_354
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_351
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_353
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_352
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_355
            paddle.static.InputSpec(shape=[20, 20, 1, 1], dtype='float16'),
            # parameter_359
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_356
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_358
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_357
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_360
            paddle.static.InputSpec(shape=[20, 1, 3, 3], dtype='float16'),
            # parameter_364
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_361
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_363
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_362
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_365
            paddle.static.InputSpec(shape=[20, 20, 1, 1], dtype='float16'),
            # parameter_369
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_366
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_368
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_367
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_370
            paddle.static.InputSpec(shape=[20, 20, 1, 1], dtype='float16'),
            # parameter_374
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_371
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_373
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_372
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_375
            paddle.static.InputSpec(shape=[20, 1, 3, 3], dtype='float16'),
            # parameter_379
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_376
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_378
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_377
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_380
            paddle.static.InputSpec(shape=[20, 20, 1, 1], dtype='float16'),
            # parameter_384
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_381
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_383
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_382
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_385
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_389
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_386
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_388
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_387
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_390
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_394
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_391
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_393
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_392
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_395
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_399
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_396
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_398
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_397
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_400
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_404
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_401
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_403
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_402
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_405
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_409
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_406
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_408
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_407
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_410
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_414
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_411
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_413
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_412
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_415
            paddle.static.InputSpec(shape=[80, 80, 1, 1], dtype='float16'),
            # parameter_419
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_416
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_418
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_417
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_420
            paddle.static.InputSpec(shape=[80, 1, 3, 3], dtype='float16'),
            # parameter_424
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_421
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_423
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_422
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_425
            paddle.static.InputSpec(shape=[80, 80, 1, 1], dtype='float16'),
            # parameter_429
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_426
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_428
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_427
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_430
            paddle.static.InputSpec(shape=[80, 80, 1, 1], dtype='float16'),
            # parameter_434
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_431
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_433
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_432
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_435
            paddle.static.InputSpec(shape=[80, 1, 3, 3], dtype='float16'),
            # parameter_439
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_436
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_438
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_437
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_440
            paddle.static.InputSpec(shape=[80, 80, 1, 1], dtype='float16'),
            # parameter_444
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_441
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_443
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_442
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_445
            paddle.static.InputSpec(shape=[40, 80, 1, 1], dtype='float16'),
            # parameter_449
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_446
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_448
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_447
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_450
            paddle.static.InputSpec(shape=[40, 160, 1, 1], dtype='float16'),
            # parameter_454
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_451
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_453
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_452
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_455
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_459
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_456
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_458
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_457
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_460
            paddle.static.InputSpec(shape=[80, 40, 1, 1], dtype='float16'),
            # parameter_464
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_461
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_463
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_462
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_465
            paddle.static.InputSpec(shape=[80, 160, 1, 1], dtype='float16'),
            # parameter_469
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_466
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_468
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_467
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_470
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_474
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_471
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_473
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_472
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_475
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_479
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_476
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_478
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_477
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_480
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_484
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_481
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_483
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_482
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_485
            paddle.static.InputSpec(shape=[160, 40, 1, 1], dtype='float16'),
            # parameter_489
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_486
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_488
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_487
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_490
            paddle.static.InputSpec(shape=[80, 1, 3, 3], dtype='float16'),
            # parameter_494
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_491
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_493
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_492
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_495
            paddle.static.InputSpec(shape=[160, 80, 1, 1], dtype='float16'),
            # parameter_499
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_496
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_498
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_497
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_500
            paddle.static.InputSpec(shape=[20, 20, 1, 1], dtype='float16'),
            # parameter_504
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_501
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_503
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_502
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_505
            paddle.static.InputSpec(shape=[20, 1, 3, 3], dtype='float16'),
            # parameter_509
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_506
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_508
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_507
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_510
            paddle.static.InputSpec(shape=[20, 20, 1, 1], dtype='float16'),
            # parameter_514
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_511
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_513
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_512
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_515
            paddle.static.InputSpec(shape=[20, 20, 1, 1], dtype='float16'),
            # parameter_519
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_516
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_518
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_517
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_520
            paddle.static.InputSpec(shape=[20, 1, 3, 3], dtype='float16'),
            # parameter_524
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_521
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_523
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_522
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_525
            paddle.static.InputSpec(shape=[20, 20, 1, 1], dtype='float16'),
            # parameter_529
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_526
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_528
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_527
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_530
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_534
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_531
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_533
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_532
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_535
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_539
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_536
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_538
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_537
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_540
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_544
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_541
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_543
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_542
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_545
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_549
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_546
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_548
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_547
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_550
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_554
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_551
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_553
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_552
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_555
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_559
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_556
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_558
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_557
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_560
            paddle.static.InputSpec(shape=[80, 80, 1, 1], dtype='float16'),
            # parameter_564
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_561
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_563
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_562
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_565
            paddle.static.InputSpec(shape=[80, 1, 3, 3], dtype='float16'),
            # parameter_569
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_566
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_568
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_567
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_570
            paddle.static.InputSpec(shape=[80, 80, 1, 1], dtype='float16'),
            # parameter_574
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_571
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_573
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_572
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_575
            paddle.static.InputSpec(shape=[80, 80, 1, 1], dtype='float16'),
            # parameter_579
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_576
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_578
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_577
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_580
            paddle.static.InputSpec(shape=[80, 1, 3, 3], dtype='float16'),
            # parameter_584
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_581
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_583
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_582
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_585
            paddle.static.InputSpec(shape=[80, 80, 1, 1], dtype='float16'),
            # parameter_589
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_586
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_588
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_587
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_590
            paddle.static.InputSpec(shape=[40, 80, 1, 1], dtype='float16'),
            # parameter_594
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_591
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_593
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_592
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_595
            paddle.static.InputSpec(shape=[40, 160, 1, 1], dtype='float16'),
            # parameter_599
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_596
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_598
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_597
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_600
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_604
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_601
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_603
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_602
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_605
            paddle.static.InputSpec(shape=[80, 40, 1, 1], dtype='float16'),
            # parameter_609
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_606
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_608
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_607
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_610
            paddle.static.InputSpec(shape=[80, 160, 1, 1], dtype='float16'),
            # parameter_614
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_611
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_613
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_612
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_615
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_619
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_616
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_618
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_617
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_620
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_624
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_621
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_623
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_622
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_625
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_629
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_626
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_628
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_627
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_630
            paddle.static.InputSpec(shape=[160, 40, 1, 1], dtype='float16'),
            # parameter_634
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_631
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_633
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_632
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_635
            paddle.static.InputSpec(shape=[80, 1, 3, 3], dtype='float16'),
            # parameter_639
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_636
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_638
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_637
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_640
            paddle.static.InputSpec(shape=[160, 80, 1, 1], dtype='float16'),
            # parameter_644
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_641
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_643
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_642
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_645
            paddle.static.InputSpec(shape=[20, 20, 1, 1], dtype='float16'),
            # parameter_649
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_646
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_648
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_647
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_650
            paddle.static.InputSpec(shape=[20, 1, 3, 3], dtype='float16'),
            # parameter_654
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_651
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_653
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_652
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_655
            paddle.static.InputSpec(shape=[20, 20, 1, 1], dtype='float16'),
            # parameter_659
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_656
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_658
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_657
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_660
            paddle.static.InputSpec(shape=[20, 20, 1, 1], dtype='float16'),
            # parameter_664
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_661
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_663
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_662
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_665
            paddle.static.InputSpec(shape=[20, 1, 3, 3], dtype='float16'),
            # parameter_669
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_666
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_668
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_667
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_670
            paddle.static.InputSpec(shape=[20, 20, 1, 1], dtype='float16'),
            # parameter_674
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_671
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_673
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_672
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_675
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_679
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_676
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_678
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_677
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_680
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_684
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_681
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_683
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_682
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_685
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_689
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_686
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_688
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_687
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_690
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_694
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_691
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_693
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_692
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_695
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_699
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_696
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_698
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_697
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_700
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_704
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_701
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_703
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_702
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_705
            paddle.static.InputSpec(shape=[80, 80, 1, 1], dtype='float16'),
            # parameter_709
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_706
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_708
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_707
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_710
            paddle.static.InputSpec(shape=[80, 1, 3, 3], dtype='float16'),
            # parameter_714
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_711
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_713
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_712
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_715
            paddle.static.InputSpec(shape=[80, 80, 1, 1], dtype='float16'),
            # parameter_719
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_716
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_718
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_717
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_720
            paddle.static.InputSpec(shape=[80, 80, 1, 1], dtype='float16'),
            # parameter_724
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_721
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_723
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_722
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_725
            paddle.static.InputSpec(shape=[80, 1, 3, 3], dtype='float16'),
            # parameter_729
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_726
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_728
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_727
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_730
            paddle.static.InputSpec(shape=[80, 80, 1, 1], dtype='float16'),
            # parameter_734
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_731
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_733
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_732
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_735
            paddle.static.InputSpec(shape=[40, 80, 1, 1], dtype='float16'),
            # parameter_739
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_736
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_738
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_737
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_740
            paddle.static.InputSpec(shape=[40, 160, 1, 1], dtype='float16'),
            # parameter_744
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_741
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_743
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_742
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_745
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_749
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_746
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_748
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_747
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_750
            paddle.static.InputSpec(shape=[80, 40, 1, 1], dtype='float16'),
            # parameter_754
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_751
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_753
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_752
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_755
            paddle.static.InputSpec(shape=[80, 160, 1, 1], dtype='float16'),
            # parameter_759
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_756
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_758
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_757
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_760
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_764
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_761
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_763
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_762
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_765
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_769
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_766
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_768
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_767
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_770
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_774
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_771
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_773
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_772
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_775
            paddle.static.InputSpec(shape=[160, 40, 1, 1], dtype='float16'),
            # parameter_779
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_776
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_778
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_777
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_780
            paddle.static.InputSpec(shape=[80, 1, 3, 3], dtype='float16'),
            # parameter_784
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_781
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_783
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_782
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_785
            paddle.static.InputSpec(shape=[160, 80, 1, 1], dtype='float16'),
            # parameter_789
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_786
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_788
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_787
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_790
            paddle.static.InputSpec(shape=[160, 1, 3, 3], dtype='float16'),
            # parameter_794
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_791
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_793
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_792
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_795
            paddle.static.InputSpec(shape=[320, 160, 1, 1], dtype='float16'),
            # parameter_799
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_796
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_798
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_797
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_800
            paddle.static.InputSpec(shape=[20, 20, 1, 1], dtype='float16'),
            # parameter_804
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_801
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_803
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_802
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_805
            paddle.static.InputSpec(shape=[20, 1, 3, 3], dtype='float16'),
            # parameter_809
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_806
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_808
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_807
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_810
            paddle.static.InputSpec(shape=[20, 20, 1, 1], dtype='float16'),
            # parameter_814
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_811
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_813
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_812
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_815
            paddle.static.InputSpec(shape=[20, 20, 1, 1], dtype='float16'),
            # parameter_819
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_816
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_818
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_817
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_820
            paddle.static.InputSpec(shape=[20, 1, 3, 3], dtype='float16'),
            # parameter_824
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_821
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_823
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_822
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_825
            paddle.static.InputSpec(shape=[20, 20, 1, 1], dtype='float16'),
            # parameter_829
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_826
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_828
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_827
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_830
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_834
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_831
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_833
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_832
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_835
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_839
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_836
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_838
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_837
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_840
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_844
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_841
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_843
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_842
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_845
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_849
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_846
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_848
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_847
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_850
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_854
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_851
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_853
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_852
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_855
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_859
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_856
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_858
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_857
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_860
            paddle.static.InputSpec(shape=[80, 80, 1, 1], dtype='float16'),
            # parameter_864
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_861
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_863
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_862
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_865
            paddle.static.InputSpec(shape=[80, 1, 3, 3], dtype='float16'),
            # parameter_869
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_866
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_868
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_867
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_870
            paddle.static.InputSpec(shape=[80, 80, 1, 1], dtype='float16'),
            # parameter_874
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_871
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_873
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_872
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_875
            paddle.static.InputSpec(shape=[80, 80, 1, 1], dtype='float16'),
            # parameter_879
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_876
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_878
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_877
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_880
            paddle.static.InputSpec(shape=[80, 1, 3, 3], dtype='float16'),
            # parameter_884
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_881
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_883
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_882
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_885
            paddle.static.InputSpec(shape=[80, 80, 1, 1], dtype='float16'),
            # parameter_889
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_886
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_888
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_887
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_890
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float16'),
            # parameter_894
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_891
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_893
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_892
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_895
            paddle.static.InputSpec(shape=[160, 1, 3, 3], dtype='float16'),
            # parameter_899
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_896
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_898
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_897
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_900
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float16'),
            # parameter_904
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_901
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_903
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_902
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_905
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float16'),
            # parameter_909
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_906
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_908
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_907
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_910
            paddle.static.InputSpec(shape=[160, 1, 3, 3], dtype='float16'),
            # parameter_914
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_911
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_913
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_912
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_915
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float16'),
            # parameter_919
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_916
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_918
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_917
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_920
            paddle.static.InputSpec(shape=[40, 80, 1, 1], dtype='float16'),
            # parameter_924
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_921
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_923
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_922
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_925
            paddle.static.InputSpec(shape=[40, 160, 1, 1], dtype='float16'),
            # parameter_929
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_926
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_928
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_927
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_930
            paddle.static.InputSpec(shape=[40, 320, 1, 1], dtype='float16'),
            # parameter_934
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_931
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_933
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_932
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_935
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_939
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_936
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_938
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_937
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_940
            paddle.static.InputSpec(shape=[80, 40, 1, 1], dtype='float16'),
            # parameter_944
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_941
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_943
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_942
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_945
            paddle.static.InputSpec(shape=[80, 160, 1, 1], dtype='float16'),
            # parameter_949
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_946
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_948
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_947
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_950
            paddle.static.InputSpec(shape=[80, 320, 1, 1], dtype='float16'),
            # parameter_954
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_951
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_953
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_952
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_955
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_959
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_956
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_958
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_957
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_960
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_964
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_961
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_963
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_962
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_965
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_969
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_966
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_968
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_967
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_970
            paddle.static.InputSpec(shape=[160, 40, 1, 1], dtype='float16'),
            # parameter_974
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_971
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_973
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_972
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_975
            paddle.static.InputSpec(shape=[80, 1, 3, 3], dtype='float16'),
            # parameter_979
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_976
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_978
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_977
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_980
            paddle.static.InputSpec(shape=[160, 80, 1, 1], dtype='float16'),
            # parameter_984
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_981
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_983
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_982
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_985
            paddle.static.InputSpec(shape=[160, 320, 1, 1], dtype='float16'),
            # parameter_989
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_986
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_988
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_987
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_990
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_994
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_991
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_993
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_992
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_995
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_999
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_996
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_998
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_997
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1000
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_1004
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1001
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1003
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1002
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1005
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_1009
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1006
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1008
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1007
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1010
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_1014
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1011
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1013
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1012
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1015
            paddle.static.InputSpec(shape=[320, 40, 1, 1], dtype='float16'),
            # parameter_1019
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_1016
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_1018
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_1017
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_1020
            paddle.static.InputSpec(shape=[80, 1, 3, 3], dtype='float16'),
            # parameter_1024
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1021
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1023
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1022
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1025
            paddle.static.InputSpec(shape=[80, 80, 1, 1], dtype='float16'),
            # parameter_1029
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1026
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1028
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1027
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1030
            paddle.static.InputSpec(shape=[80, 1, 3, 3], dtype='float16'),
            # parameter_1034
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1031
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1033
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1032
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1035
            paddle.static.InputSpec(shape=[320, 80, 1, 1], dtype='float16'),
            # parameter_1039
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_1036
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_1038
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_1037
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_1040
            paddle.static.InputSpec(shape=[160, 1, 3, 3], dtype='float16'),
            # parameter_1044
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1041
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1043
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1042
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1045
            paddle.static.InputSpec(shape=[320, 160, 1, 1], dtype='float16'),
            # parameter_1049
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_1046
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_1048
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_1047
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_1050
            paddle.static.InputSpec(shape=[20, 20, 1, 1], dtype='float16'),
            # parameter_1054
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_1051
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_1053
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_1052
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_1055
            paddle.static.InputSpec(shape=[20, 1, 3, 3], dtype='float16'),
            # parameter_1059
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_1056
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_1058
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_1057
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_1060
            paddle.static.InputSpec(shape=[20, 20, 1, 1], dtype='float16'),
            # parameter_1064
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_1061
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_1063
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_1062
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_1065
            paddle.static.InputSpec(shape=[20, 20, 1, 1], dtype='float16'),
            # parameter_1069
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_1066
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_1068
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_1067
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_1070
            paddle.static.InputSpec(shape=[20, 1, 3, 3], dtype='float16'),
            # parameter_1074
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_1071
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_1073
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_1072
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_1075
            paddle.static.InputSpec(shape=[20, 20, 1, 1], dtype='float16'),
            # parameter_1079
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_1076
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_1078
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_1077
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            # parameter_1080
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_1084
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1081
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1083
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1082
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1085
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_1089
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1086
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1088
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1087
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1090
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_1094
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1091
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1093
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1092
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1095
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_1099
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1096
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1098
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1097
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1100
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_1104
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1101
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1103
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1102
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1105
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_1109
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1106
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1108
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1107
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1110
            paddle.static.InputSpec(shape=[80, 80, 1, 1], dtype='float16'),
            # parameter_1114
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1111
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1113
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1112
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1115
            paddle.static.InputSpec(shape=[80, 1, 3, 3], dtype='float16'),
            # parameter_1119
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1116
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1118
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1117
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1120
            paddle.static.InputSpec(shape=[80, 80, 1, 1], dtype='float16'),
            # parameter_1124
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1121
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1123
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1122
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1125
            paddle.static.InputSpec(shape=[80, 80, 1, 1], dtype='float16'),
            # parameter_1129
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1126
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1128
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1127
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1130
            paddle.static.InputSpec(shape=[80, 1, 3, 3], dtype='float16'),
            # parameter_1134
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1131
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1133
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1132
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1135
            paddle.static.InputSpec(shape=[80, 80, 1, 1], dtype='float16'),
            # parameter_1139
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1136
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1138
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1137
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1140
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float16'),
            # parameter_1144
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1141
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1143
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1142
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1145
            paddle.static.InputSpec(shape=[160, 1, 3, 3], dtype='float16'),
            # parameter_1149
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1146
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1148
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1147
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1150
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float16'),
            # parameter_1154
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1151
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1153
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1152
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1155
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float16'),
            # parameter_1159
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1156
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1158
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1157
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1160
            paddle.static.InputSpec(shape=[160, 1, 3, 3], dtype='float16'),
            # parameter_1164
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1161
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1163
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1162
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1165
            paddle.static.InputSpec(shape=[160, 160, 1, 1], dtype='float16'),
            # parameter_1169
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1166
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1168
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1167
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1170
            paddle.static.InputSpec(shape=[40, 80, 1, 1], dtype='float16'),
            # parameter_1174
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1171
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1173
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1172
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1175
            paddle.static.InputSpec(shape=[40, 160, 1, 1], dtype='float16'),
            # parameter_1179
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1176
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1178
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1177
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1180
            paddle.static.InputSpec(shape=[40, 320, 1, 1], dtype='float16'),
            # parameter_1184
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1181
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1183
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1182
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1185
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_1189
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1186
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1188
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1187
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1190
            paddle.static.InputSpec(shape=[80, 40, 1, 1], dtype='float16'),
            # parameter_1194
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1191
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1193
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1192
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1195
            paddle.static.InputSpec(shape=[80, 160, 1, 1], dtype='float16'),
            # parameter_1199
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1196
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1198
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1197
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1200
            paddle.static.InputSpec(shape=[80, 320, 1, 1], dtype='float16'),
            # parameter_1204
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1201
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1203
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1202
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1205
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_1209
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1206
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1208
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1207
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1210
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_1214
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1211
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1213
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1212
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1215
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_1219
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1216
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1218
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1217
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1220
            paddle.static.InputSpec(shape=[160, 40, 1, 1], dtype='float16'),
            # parameter_1224
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1221
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1223
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1222
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1225
            paddle.static.InputSpec(shape=[80, 1, 3, 3], dtype='float16'),
            # parameter_1229
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1226
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1228
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1227
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1230
            paddle.static.InputSpec(shape=[160, 80, 1, 1], dtype='float16'),
            # parameter_1234
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1231
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1233
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1232
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1235
            paddle.static.InputSpec(shape=[160, 320, 1, 1], dtype='float16'),
            # parameter_1239
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1236
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1238
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1237
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1240
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_1244
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1241
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1243
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1242
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1245
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_1249
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1246
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1248
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1247
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1250
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_1254
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1251
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1253
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1252
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1255
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_1259
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1256
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1258
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1257
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1260
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_1264
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1261
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1263
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1262
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1265
            paddle.static.InputSpec(shape=[320, 40, 1, 1], dtype='float16'),
            # parameter_1269
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_1266
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_1268
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_1267
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_1270
            paddle.static.InputSpec(shape=[80, 1, 3, 3], dtype='float16'),
            # parameter_1274
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1271
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1273
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1272
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1275
            paddle.static.InputSpec(shape=[80, 80, 1, 1], dtype='float16'),
            # parameter_1279
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1276
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1278
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1277
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1280
            paddle.static.InputSpec(shape=[80, 1, 3, 3], dtype='float16'),
            # parameter_1284
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1281
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1283
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1282
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1285
            paddle.static.InputSpec(shape=[320, 80, 1, 1], dtype='float16'),
            # parameter_1289
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_1286
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_1288
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_1287
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_1290
            paddle.static.InputSpec(shape=[160, 1, 3, 3], dtype='float16'),
            # parameter_1294
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1291
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1293
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1292
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1295
            paddle.static.InputSpec(shape=[320, 160, 1, 1], dtype='float16'),
            # parameter_1299
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_1296
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_1298
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_1297
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_1300
            paddle.static.InputSpec(shape=[320, 1, 3, 3], dtype='float16'),
            # parameter_1304
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_1301
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_1303
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_1302
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_1305
            paddle.static.InputSpec(shape=[160, 320, 1, 1], dtype='float16'),
            # parameter_1309
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1306
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1308
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1307
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1310
            paddle.static.InputSpec(shape=[160, 1, 3, 3], dtype='float16'),
            # parameter_1314
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1311
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1313
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1312
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            # parameter_1315
            paddle.static.InputSpec(shape=[80, 160, 1, 1], dtype='float16'),
            # parameter_1319
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1316
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1318
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1317
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1320
            paddle.static.InputSpec(shape=[80, 1, 3, 3], dtype='float16'),
            # parameter_1324
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1321
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1323
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1322
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            # parameter_1325
            paddle.static.InputSpec(shape=[40, 80, 1, 1], dtype='float16'),
            # parameter_1329
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1326
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1328
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1327
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1330
            paddle.static.InputSpec(shape=[40, 1, 3, 3], dtype='float16'),
            # parameter_1334
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1331
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1333
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1332
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1335
            paddle.static.InputSpec(shape=[40, 40, 1, 1], dtype='float16'),
            # parameter_1339
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1336
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1338
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1337
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            # parameter_1340
            paddle.static.InputSpec(shape=[17, 40, 1, 1], dtype='float16'),
            # parameter_1341
            paddle.static.InputSpec(shape=[17], dtype='float16'),
            # feed_0
            paddle.static.InputSpec(shape=[None, 3, 128, 96], dtype='float32'),
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