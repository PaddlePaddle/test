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
    return [1209][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_2747_0_0(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_114, parameter_111, parameter_113, parameter_112, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_154, parameter_151, parameter_153, parameter_152, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_169, parameter_166, parameter_168, parameter_167, parameter_170, parameter_174, parameter_171, parameter_173, parameter_172, parameter_175, parameter_179, parameter_176, parameter_178, parameter_177, parameter_180, parameter_184, parameter_181, parameter_183, parameter_182, parameter_185, parameter_189, parameter_186, parameter_188, parameter_187, parameter_190, parameter_194, parameter_191, parameter_193, parameter_192, parameter_195, parameter_199, parameter_196, parameter_198, parameter_197, parameter_200, parameter_204, parameter_201, parameter_203, parameter_202, parameter_205, parameter_209, parameter_206, parameter_208, parameter_207, parameter_210, parameter_214, parameter_211, parameter_213, parameter_212, parameter_215, parameter_219, parameter_216, parameter_218, parameter_217, parameter_220, parameter_224, parameter_221, parameter_223, parameter_222, parameter_225, parameter_229, parameter_226, parameter_228, parameter_227, parameter_230, parameter_234, parameter_231, parameter_233, parameter_232, parameter_235, parameter_239, parameter_236, parameter_238, parameter_237, parameter_240, parameter_244, parameter_241, parameter_243, parameter_242, parameter_245, parameter_249, parameter_246, parameter_248, parameter_247, parameter_250, parameter_254, parameter_251, parameter_253, parameter_252, parameter_255, parameter_259, parameter_256, parameter_258, parameter_257, parameter_260, parameter_264, parameter_261, parameter_263, parameter_262, parameter_265, parameter_269, parameter_266, parameter_268, parameter_267, parameter_270, parameter_274, parameter_271, parameter_273, parameter_272, parameter_275, parameter_279, parameter_276, parameter_278, parameter_277, parameter_280, parameter_284, parameter_281, parameter_283, parameter_282, parameter_285, parameter_289, parameter_286, parameter_288, parameter_287, parameter_290, parameter_294, parameter_291, parameter_293, parameter_292, parameter_295, parameter_299, parameter_296, parameter_298, parameter_297, parameter_300, parameter_304, parameter_301, parameter_303, parameter_302, parameter_305, parameter_309, parameter_306, parameter_308, parameter_307, parameter_310, parameter_314, parameter_311, parameter_313, parameter_312, parameter_315, parameter_319, parameter_316, parameter_318, parameter_317, parameter_320, parameter_324, parameter_321, parameter_323, parameter_322, parameter_325, parameter_329, parameter_326, parameter_328, parameter_327, parameter_330, parameter_334, parameter_331, parameter_333, parameter_332, parameter_335, parameter_339, parameter_336, parameter_338, parameter_337, parameter_340, parameter_344, parameter_341, parameter_343, parameter_342, parameter_345, parameter_349, parameter_346, parameter_348, parameter_347, parameter_350, parameter_354, parameter_351, parameter_353, parameter_352, parameter_355, parameter_359, parameter_356, parameter_358, parameter_357, parameter_360, parameter_364, parameter_361, parameter_363, parameter_362, parameter_365, parameter_369, parameter_366, parameter_368, parameter_367, parameter_370, parameter_374, parameter_371, parameter_373, parameter_372, parameter_375, parameter_379, parameter_376, parameter_378, parameter_377, parameter_380, parameter_384, parameter_381, parameter_383, parameter_382, parameter_385, parameter_389, parameter_386, parameter_388, parameter_387, parameter_390, parameter_394, parameter_391, parameter_393, parameter_392, parameter_395, parameter_399, parameter_396, parameter_398, parameter_397, parameter_400, parameter_404, parameter_401, parameter_403, parameter_402, parameter_405, parameter_409, parameter_406, parameter_408, parameter_407, parameter_410, parameter_414, parameter_411, parameter_413, parameter_412, parameter_415, parameter_419, parameter_416, parameter_418, parameter_417, parameter_420, parameter_424, parameter_421, parameter_423, parameter_422, parameter_425, parameter_429, parameter_426, parameter_428, parameter_427, parameter_430, parameter_434, parameter_431, parameter_433, parameter_432, parameter_435, parameter_439, parameter_436, parameter_438, parameter_437, parameter_440, parameter_444, parameter_441, parameter_443, parameter_442, parameter_445, parameter_449, parameter_446, parameter_448, parameter_447, parameter_450, parameter_454, parameter_451, parameter_453, parameter_452, parameter_455, parameter_459, parameter_456, parameter_458, parameter_457, parameter_460, parameter_464, parameter_461, parameter_463, parameter_462, parameter_465, parameter_469, parameter_466, parameter_468, parameter_467, parameter_470, parameter_474, parameter_471, parameter_473, parameter_472, parameter_475, parameter_479, parameter_476, parameter_478, parameter_477, parameter_480, parameter_484, parameter_481, parameter_483, parameter_482, parameter_485, parameter_489, parameter_486, parameter_488, parameter_487, parameter_490, parameter_494, parameter_491, parameter_493, parameter_492, parameter_495, parameter_499, parameter_496, parameter_498, parameter_497, parameter_500, parameter_504, parameter_501, parameter_503, parameter_502, parameter_505, parameter_509, parameter_506, parameter_508, parameter_507, parameter_510, parameter_514, parameter_511, parameter_513, parameter_512, parameter_515, parameter_519, parameter_516, parameter_518, parameter_517, parameter_520, parameter_524, parameter_521, parameter_523, parameter_522, parameter_525, parameter_529, parameter_526, parameter_528, parameter_527, parameter_530, parameter_534, parameter_531, parameter_533, parameter_532, parameter_535, parameter_539, parameter_536, parameter_538, parameter_537, parameter_540, parameter_544, parameter_541, parameter_543, parameter_542, parameter_545, parameter_549, parameter_546, parameter_548, parameter_547, parameter_550, parameter_554, parameter_551, parameter_553, parameter_552, parameter_555, parameter_559, parameter_556, parameter_558, parameter_557, parameter_560, parameter_564, parameter_561, parameter_563, parameter_562, parameter_565, parameter_569, parameter_566, parameter_568, parameter_567, parameter_570, parameter_574, parameter_571, parameter_573, parameter_572, parameter_575, parameter_579, parameter_576, parameter_578, parameter_577, parameter_580, parameter_584, parameter_581, parameter_583, parameter_582, parameter_585, parameter_589, parameter_586, parameter_588, parameter_587, parameter_590, parameter_594, parameter_591, parameter_593, parameter_592, parameter_595, parameter_599, parameter_596, parameter_598, parameter_597, parameter_600, parameter_604, parameter_601, parameter_603, parameter_602, parameter_605, parameter_609, parameter_606, parameter_608, parameter_607, parameter_610, parameter_614, parameter_611, parameter_613, parameter_612, parameter_615, parameter_619, parameter_616, parameter_618, parameter_617, parameter_620, parameter_624, parameter_621, parameter_623, parameter_622, parameter_625, parameter_629, parameter_626, parameter_628, parameter_627, parameter_630, parameter_634, parameter_631, parameter_633, parameter_632, parameter_635, parameter_639, parameter_636, parameter_638, parameter_637, parameter_640, parameter_644, parameter_641, parameter_643, parameter_642, parameter_645, parameter_649, parameter_646, parameter_648, parameter_647, parameter_650, parameter_654, parameter_651, parameter_653, parameter_652, parameter_655, parameter_659, parameter_656, parameter_658, parameter_657, parameter_660, parameter_664, parameter_661, parameter_663, parameter_662, parameter_665, parameter_669, parameter_666, parameter_668, parameter_667, parameter_670, parameter_674, parameter_671, parameter_673, parameter_672, parameter_675, parameter_679, parameter_676, parameter_678, parameter_677, parameter_680, parameter_684, parameter_681, parameter_683, parameter_682, parameter_685, parameter_689, parameter_686, parameter_688, parameter_687, parameter_690, parameter_694, parameter_691, parameter_693, parameter_692, parameter_695, parameter_699, parameter_696, parameter_698, parameter_697, parameter_700, parameter_704, parameter_701, parameter_703, parameter_702, parameter_705, parameter_709, parameter_706, parameter_708, parameter_707, parameter_710, parameter_714, parameter_711, parameter_713, parameter_712, parameter_715, parameter_719, parameter_716, parameter_718, parameter_717, parameter_720, parameter_724, parameter_721, parameter_723, parameter_722, parameter_725, parameter_729, parameter_726, parameter_728, parameter_727, parameter_730, parameter_734, parameter_731, parameter_733, parameter_732, parameter_735, parameter_739, parameter_736, parameter_738, parameter_737, parameter_740, parameter_744, parameter_741, parameter_743, parameter_742, parameter_745, parameter_749, parameter_746, parameter_748, parameter_747, parameter_750, parameter_754, parameter_751, parameter_753, parameter_752, parameter_755, parameter_759, parameter_756, parameter_758, parameter_757, parameter_760, parameter_764, parameter_761, parameter_763, parameter_762, parameter_765, parameter_769, parameter_766, parameter_768, parameter_767, parameter_770, parameter_774, parameter_771, parameter_773, parameter_772, parameter_775, parameter_779, parameter_776, parameter_778, parameter_777, parameter_780, parameter_784, parameter_781, parameter_783, parameter_782, parameter_785, parameter_789, parameter_786, parameter_788, parameter_787, parameter_790, parameter_794, parameter_791, parameter_793, parameter_792, parameter_795, parameter_799, parameter_796, parameter_798, parameter_797, parameter_800, parameter_804, parameter_801, parameter_803, parameter_802, parameter_805, parameter_809, parameter_806, parameter_808, parameter_807, parameter_810, parameter_814, parameter_811, parameter_813, parameter_812, parameter_815, parameter_819, parameter_816, parameter_818, parameter_817, parameter_820, parameter_824, parameter_821, parameter_823, parameter_822, parameter_825, parameter_829, parameter_826, parameter_828, parameter_827, parameter_830, parameter_834, parameter_831, parameter_833, parameter_832, parameter_835, parameter_839, parameter_836, parameter_838, parameter_837, parameter_840, parameter_844, parameter_841, parameter_843, parameter_842, parameter_845, parameter_849, parameter_846, parameter_848, parameter_847, parameter_850, parameter_854, parameter_851, parameter_853, parameter_852, parameter_855, parameter_859, parameter_856, parameter_858, parameter_857, parameter_860, parameter_864, parameter_861, parameter_863, parameter_862, parameter_865, parameter_869, parameter_866, parameter_868, parameter_867, parameter_870, parameter_874, parameter_871, parameter_873, parameter_872, parameter_875, parameter_879, parameter_876, parameter_878, parameter_877, parameter_880, parameter_884, parameter_881, parameter_883, parameter_882, parameter_885, parameter_889, parameter_886, parameter_888, parameter_887, parameter_890, parameter_894, parameter_891, parameter_893, parameter_892, parameter_895, parameter_899, parameter_896, parameter_898, parameter_897, parameter_900, parameter_904, parameter_901, parameter_903, parameter_902, parameter_905, parameter_909, parameter_906, parameter_908, parameter_907, parameter_910, parameter_914, parameter_911, parameter_913, parameter_912, parameter_915, parameter_919, parameter_916, parameter_918, parameter_917, parameter_920, parameter_924, parameter_921, parameter_923, parameter_922, parameter_925, parameter_929, parameter_926, parameter_928, parameter_927, parameter_930, parameter_934, parameter_931, parameter_933, parameter_932, parameter_935, parameter_939, parameter_936, parameter_938, parameter_937, parameter_940, parameter_944, parameter_941, parameter_943, parameter_942, parameter_945, parameter_949, parameter_946, parameter_948, parameter_947, parameter_950, parameter_954, parameter_951, parameter_953, parameter_952, parameter_955, parameter_959, parameter_956, parameter_958, parameter_957, parameter_960, parameter_964, parameter_961, parameter_963, parameter_962, parameter_965, parameter_969, parameter_966, parameter_968, parameter_967, parameter_970, parameter_974, parameter_971, parameter_973, parameter_972, parameter_975, parameter_979, parameter_976, parameter_978, parameter_977, parameter_980, parameter_984, parameter_981, parameter_983, parameter_982, parameter_985, parameter_989, parameter_986, parameter_988, parameter_987, parameter_990, parameter_994, parameter_991, parameter_993, parameter_992, parameter_995, parameter_999, parameter_996, parameter_998, parameter_997, parameter_1000, parameter_1004, parameter_1001, parameter_1003, parameter_1002, parameter_1005, parameter_1009, parameter_1006, parameter_1008, parameter_1007, parameter_1010, parameter_1014, parameter_1011, parameter_1013, parameter_1012, parameter_1015, parameter_1019, parameter_1016, parameter_1018, parameter_1017, parameter_1020, parameter_1024, parameter_1021, parameter_1023, parameter_1022, parameter_1025, parameter_1029, parameter_1026, parameter_1028, parameter_1027, parameter_1030, parameter_1034, parameter_1031, parameter_1033, parameter_1032, parameter_1035, parameter_1039, parameter_1036, parameter_1038, parameter_1037, parameter_1040, parameter_1044, parameter_1041, parameter_1043, parameter_1042, parameter_1045, parameter_1049, parameter_1046, parameter_1048, parameter_1047, parameter_1050, parameter_1054, parameter_1051, parameter_1053, parameter_1052, parameter_1055, parameter_1059, parameter_1056, parameter_1058, parameter_1057, parameter_1060, parameter_1064, parameter_1061, parameter_1063, parameter_1062, parameter_1065, parameter_1069, parameter_1066, parameter_1068, parameter_1067, parameter_1070, parameter_1074, parameter_1071, parameter_1073, parameter_1072, parameter_1075, parameter_1079, parameter_1076, parameter_1078, parameter_1077, parameter_1080, parameter_1084, parameter_1081, parameter_1083, parameter_1082, parameter_1085, parameter_1089, parameter_1086, parameter_1088, parameter_1087, parameter_1090, parameter_1094, parameter_1091, parameter_1093, parameter_1092, parameter_1095, parameter_1099, parameter_1096, parameter_1098, parameter_1097, parameter_1100, parameter_1104, parameter_1101, parameter_1103, parameter_1102, parameter_1105, parameter_1109, parameter_1106, parameter_1108, parameter_1107, parameter_1110, parameter_1114, parameter_1111, parameter_1113, parameter_1112, parameter_1115, parameter_1119, parameter_1116, parameter_1118, parameter_1117, parameter_1120, parameter_1124, parameter_1121, parameter_1123, parameter_1122, parameter_1125, parameter_1129, parameter_1126, parameter_1128, parameter_1127, parameter_1130, parameter_1134, parameter_1131, parameter_1133, parameter_1132, parameter_1135, parameter_1139, parameter_1136, parameter_1138, parameter_1137, parameter_1140, parameter_1144, parameter_1141, parameter_1143, parameter_1142, parameter_1145, parameter_1149, parameter_1146, parameter_1148, parameter_1147, parameter_1150, parameter_1154, parameter_1151, parameter_1153, parameter_1152, parameter_1155, parameter_1159, parameter_1156, parameter_1158, parameter_1157, parameter_1160, parameter_1164, parameter_1161, parameter_1163, parameter_1162, parameter_1165, parameter_1169, parameter_1166, parameter_1168, parameter_1167, parameter_1170, parameter_1174, parameter_1171, parameter_1173, parameter_1172, parameter_1175, parameter_1179, parameter_1176, parameter_1178, parameter_1177, parameter_1180, parameter_1184, parameter_1181, parameter_1183, parameter_1182, parameter_1185, parameter_1189, parameter_1186, parameter_1188, parameter_1187, parameter_1190, parameter_1194, parameter_1191, parameter_1193, parameter_1192, parameter_1195, parameter_1199, parameter_1196, parameter_1198, parameter_1197, parameter_1200, parameter_1204, parameter_1201, parameter_1203, parameter_1202, parameter_1205, parameter_1209, parameter_1206, parameter_1208, parameter_1207, parameter_1210, parameter_1214, parameter_1211, parameter_1213, parameter_1212, parameter_1215, parameter_1219, parameter_1216, parameter_1218, parameter_1217, parameter_1220, parameter_1224, parameter_1221, parameter_1223, parameter_1222, parameter_1225, parameter_1229, parameter_1226, parameter_1228, parameter_1227, parameter_1230, parameter_1234, parameter_1231, parameter_1233, parameter_1232, parameter_1235, parameter_1239, parameter_1236, parameter_1238, parameter_1237, parameter_1240, parameter_1244, parameter_1241, parameter_1243, parameter_1242, parameter_1245, parameter_1249, parameter_1246, parameter_1248, parameter_1247, parameter_1250, parameter_1254, parameter_1251, parameter_1253, parameter_1252, parameter_1255, parameter_1259, parameter_1256, parameter_1258, parameter_1257, parameter_1260, parameter_1264, parameter_1261, parameter_1263, parameter_1262, parameter_1265, parameter_1269, parameter_1266, parameter_1268, parameter_1267, parameter_1270, parameter_1274, parameter_1271, parameter_1273, parameter_1272, parameter_1275, parameter_1279, parameter_1276, parameter_1278, parameter_1277, parameter_1280, parameter_1284, parameter_1281, parameter_1283, parameter_1282, parameter_1285, parameter_1289, parameter_1286, parameter_1288, parameter_1287, parameter_1290, parameter_1294, parameter_1291, parameter_1293, parameter_1292, parameter_1295, parameter_1299, parameter_1296, parameter_1298, parameter_1297, parameter_1300, parameter_1304, parameter_1301, parameter_1303, parameter_1302, parameter_1305, parameter_1309, parameter_1306, parameter_1308, parameter_1307, parameter_1310, parameter_1314, parameter_1311, parameter_1313, parameter_1312, parameter_1315, parameter_1319, parameter_1316, parameter_1318, parameter_1317, parameter_1320, parameter_1324, parameter_1321, parameter_1323, parameter_1322, parameter_1325, parameter_1329, parameter_1326, parameter_1328, parameter_1327, parameter_1330, parameter_1334, parameter_1331, parameter_1333, parameter_1332, parameter_1335, parameter_1339, parameter_1336, parameter_1338, parameter_1337, parameter_1340, parameter_1344, parameter_1341, parameter_1343, parameter_1342, parameter_1345, parameter_1349, parameter_1346, parameter_1348, parameter_1347, parameter_1350, parameter_1354, parameter_1351, parameter_1353, parameter_1352, parameter_1355, parameter_1359, parameter_1356, parameter_1358, parameter_1357, parameter_1360, parameter_1364, parameter_1361, parameter_1363, parameter_1362, parameter_1365, parameter_1369, parameter_1366, parameter_1368, parameter_1367, parameter_1370, parameter_1374, parameter_1371, parameter_1373, parameter_1372, parameter_1375, parameter_1379, parameter_1376, parameter_1378, parameter_1377, parameter_1380, parameter_1384, parameter_1381, parameter_1383, parameter_1382, parameter_1385, parameter_1389, parameter_1386, parameter_1388, parameter_1387, parameter_1390, parameter_1394, parameter_1391, parameter_1393, parameter_1392, parameter_1395, parameter_1399, parameter_1396, parameter_1398, parameter_1397, parameter_1400, parameter_1404, parameter_1401, parameter_1403, parameter_1402, parameter_1405, parameter_1409, parameter_1406, parameter_1408, parameter_1407, parameter_1410, parameter_1414, parameter_1411, parameter_1413, parameter_1412, parameter_1415, parameter_1419, parameter_1416, parameter_1418, parameter_1417, parameter_1420, parameter_1424, parameter_1421, parameter_1423, parameter_1422, parameter_1425, parameter_1429, parameter_1426, parameter_1428, parameter_1427, parameter_1430, parameter_1434, parameter_1431, parameter_1433, parameter_1432, parameter_1435, parameter_1439, parameter_1436, parameter_1438, parameter_1437, parameter_1440, parameter_1444, parameter_1441, parameter_1443, parameter_1442, parameter_1445, parameter_1449, parameter_1446, parameter_1448, parameter_1447, parameter_1450, parameter_1454, parameter_1451, parameter_1453, parameter_1452, parameter_1455, parameter_1459, parameter_1456, parameter_1458, parameter_1457, parameter_1460, parameter_1464, parameter_1461, parameter_1463, parameter_1462, parameter_1465, parameter_1469, parameter_1466, parameter_1468, parameter_1467, parameter_1470, parameter_1474, parameter_1471, parameter_1473, parameter_1472, parameter_1475, parameter_1479, parameter_1476, parameter_1478, parameter_1477, parameter_1480, parameter_1484, parameter_1481, parameter_1483, parameter_1482, parameter_1485, parameter_1489, parameter_1486, parameter_1488, parameter_1487, parameter_1490, parameter_1494, parameter_1491, parameter_1493, parameter_1492, parameter_1495, parameter_1499, parameter_1496, parameter_1498, parameter_1497, parameter_1500, parameter_1504, parameter_1501, parameter_1503, parameter_1502, parameter_1505, parameter_1509, parameter_1506, parameter_1508, parameter_1507, parameter_1510, parameter_1514, parameter_1511, parameter_1513, parameter_1512, parameter_1515, parameter_1519, parameter_1516, parameter_1518, parameter_1517, parameter_1520, parameter_1524, parameter_1521, parameter_1523, parameter_1522, parameter_1525, parameter_1529, parameter_1526, parameter_1528, parameter_1527, parameter_1530, feed_0):

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x3x-1x-1xf32, 64x3x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(feed_0, parameter_0, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_0, parameter_1, parameter_2, parameter_3, parameter_4, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_0 = paddle._C_ops.relu(batch_norm__0)

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(relu_0, parameter_5, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_1, parameter_6, parameter_7, parameter_8, parameter_9, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_1 = paddle._C_ops.relu(batch_norm__6)

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(relu_1, parameter_10, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_2, parameter_11, parameter_12, parameter_13, parameter_14, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_2 = paddle._C_ops.relu(batch_norm__12)

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(relu_2, parameter_15, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_3, parameter_16, parameter_17, parameter_18, parameter_19, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_3 = paddle._C_ops.relu(batch_norm__18)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x64x-1x-1xf32, 256x64x1x1xf32)
        conv2d_4 = paddle._C_ops.conv2d(relu_3, parameter_20, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_4, parameter_21, parameter_22, parameter_23, parameter_24, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x64x-1x-1xf32, 256x64x1x1xf32)
        conv2d_5 = paddle._C_ops.conv2d(relu_1, parameter_25, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_5, parameter_26, parameter_27, parameter_28, parameter_29, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, -1x256x-1x-1xf32)
        add_0 = paddle._C_ops.add(batch_norm__24, batch_norm__30)

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_4 = paddle._C_ops.relu(add_0)

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x256x-1x-1xf32, 64x256x1x1xf32)
        conv2d_6 = paddle._C_ops.conv2d(relu_4, parameter_30, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_6, parameter_31, parameter_32, parameter_33, parameter_34, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_5 = paddle._C_ops.relu(batch_norm__36)

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x3x3xf32)
        conv2d_7 = paddle._C_ops.conv2d(relu_5, parameter_35, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_7, parameter_36, parameter_37, parameter_38, parameter_39, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_6 = paddle._C_ops.relu(batch_norm__42)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x64x-1x-1xf32, 256x64x1x1xf32)
        conv2d_8 = paddle._C_ops.conv2d(relu_6, parameter_40, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_8, parameter_41, parameter_42, parameter_43, parameter_44, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, -1x256x-1x-1xf32)
        add_1 = paddle._C_ops.add(batch_norm__48, relu_4)

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_7 = paddle._C_ops.relu(add_1)

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x256x-1x-1xf32, 64x256x1x1xf32)
        conv2d_9 = paddle._C_ops.conv2d(relu_7, parameter_45, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_9, parameter_46, parameter_47, parameter_48, parameter_49, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_8 = paddle._C_ops.relu(batch_norm__54)

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x3x3xf32)
        conv2d_10 = paddle._C_ops.conv2d(relu_8, parameter_50, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_10, parameter_51, parameter_52, parameter_53, parameter_54, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_9 = paddle._C_ops.relu(batch_norm__60)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x64x-1x-1xf32, 256x64x1x1xf32)
        conv2d_11 = paddle._C_ops.conv2d(relu_9, parameter_55, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_11, parameter_56, parameter_57, parameter_58, parameter_59, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, -1x256x-1x-1xf32)
        add_2 = paddle._C_ops.add(batch_norm__66, relu_7)

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_10 = paddle._C_ops.relu(add_2)

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x256x-1x-1xf32, 64x256x1x1xf32)
        conv2d_12 = paddle._C_ops.conv2d(relu_10, parameter_60, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_12, parameter_61, parameter_62, parameter_63, parameter_64, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_11 = paddle._C_ops.relu(batch_norm__72)

        # pd_op.conv2d: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32, 64x64x3x3xf32)
        conv2d_13 = paddle._C_ops.conv2d(relu_11, parameter_65, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x-1x-1xf32, 64xf32, 64xf32, xf32, xf32, None) <- (-1x64x-1x-1xf32, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_13, parameter_66, parameter_67, parameter_68, parameter_69, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x64x-1x-1xf32) <- (-1x64x-1x-1xf32)
        relu_12 = paddle._C_ops.relu(batch_norm__78)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x64x-1x-1xf32, 256x64x1x1xf32)
        conv2d_14 = paddle._C_ops.conv2d(relu_12, parameter_70, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x-1x-1xf32, 256xf32, 256xf32, xf32, xf32, None) <- (-1x256x-1x-1xf32, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_14, parameter_71, parameter_72, parameter_73, parameter_74, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, -1x256x-1x-1xf32)
        add_3 = paddle._C_ops.add(batch_norm__84, relu_10)

        # pd_op.relu: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32)
        relu_13 = paddle._C_ops.relu(add_3)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x256x-1x-1xf32, 18x256x3x3xf32)
        conv2d_15 = paddle._C_ops.conv2d(relu_13, parameter_75, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_15, parameter_76, parameter_77, parameter_78, parameter_79, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_14 = paddle._C_ops.relu(batch_norm__90)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x256x-1x-1xf32, 36x256x3x3xf32)
        conv2d_16 = paddle._C_ops.conv2d(relu_13, parameter_80, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_16, parameter_81, parameter_82, parameter_83, parameter_84, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_15 = paddle._C_ops.relu(batch_norm__96)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_17 = paddle._C_ops.conv2d(relu_14, parameter_85, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_17, parameter_86, parameter_87, parameter_88, parameter_89, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_16 = paddle._C_ops.relu(batch_norm__102)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_18 = paddle._C_ops.conv2d(relu_16, parameter_90, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_18, parameter_91, parameter_92, parameter_93, parameter_94, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_4 = paddle._C_ops.add(batch_norm__108, relu_14)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_17 = paddle._C_ops.relu(add_4)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_19 = paddle._C_ops.conv2d(relu_17, parameter_95, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_19, parameter_96, parameter_97, parameter_98, parameter_99, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_18 = paddle._C_ops.relu(batch_norm__114)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_20 = paddle._C_ops.conv2d(relu_18, parameter_100, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_20, parameter_101, parameter_102, parameter_103, parameter_104, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_5 = paddle._C_ops.add(batch_norm__120, relu_17)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_19 = paddle._C_ops.relu(add_5)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_21 = paddle._C_ops.conv2d(relu_19, parameter_105, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_21, parameter_106, parameter_107, parameter_108, parameter_109, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_20 = paddle._C_ops.relu(batch_norm__126)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_22 = paddle._C_ops.conv2d(relu_20, parameter_110, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_22, parameter_111, parameter_112, parameter_113, parameter_114, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_6 = paddle._C_ops.add(batch_norm__132, relu_19)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_21 = paddle._C_ops.relu(add_6)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_23 = paddle._C_ops.conv2d(relu_21, parameter_115, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_23, parameter_116, parameter_117, parameter_118, parameter_119, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_22 = paddle._C_ops.relu(batch_norm__138)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_24 = paddle._C_ops.conv2d(relu_22, parameter_120, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_24, parameter_121, parameter_122, parameter_123, parameter_124, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_7 = paddle._C_ops.add(batch_norm__144, relu_21)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_23 = paddle._C_ops.relu(add_7)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_25 = paddle._C_ops.conv2d(relu_15, parameter_125, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_25, parameter_126, parameter_127, parameter_128, parameter_129, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_24 = paddle._C_ops.relu(batch_norm__150)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_26 = paddle._C_ops.conv2d(relu_24, parameter_130, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_26, parameter_131, parameter_132, parameter_133, parameter_134, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_8 = paddle._C_ops.add(batch_norm__156, relu_15)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_25 = paddle._C_ops.relu(add_8)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_27 = paddle._C_ops.conv2d(relu_25, parameter_135, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_27, parameter_136, parameter_137, parameter_138, parameter_139, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_26 = paddle._C_ops.relu(batch_norm__162)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_28 = paddle._C_ops.conv2d(relu_26, parameter_140, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_28, parameter_141, parameter_142, parameter_143, parameter_144, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_9 = paddle._C_ops.add(batch_norm__168, relu_25)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_27 = paddle._C_ops.relu(add_9)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_29 = paddle._C_ops.conv2d(relu_27, parameter_145, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_29, parameter_146, parameter_147, parameter_148, parameter_149, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_28 = paddle._C_ops.relu(batch_norm__174)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_30 = paddle._C_ops.conv2d(relu_28, parameter_150, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__180, batch_norm__181, batch_norm__182, batch_norm__183, batch_norm__184, batch_norm__185 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_30, parameter_151, parameter_152, parameter_153, parameter_154, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_10 = paddle._C_ops.add(batch_norm__180, relu_27)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_29 = paddle._C_ops.relu(add_10)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_31 = paddle._C_ops.conv2d(relu_29, parameter_155, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__186, batch_norm__187, batch_norm__188, batch_norm__189, batch_norm__190, batch_norm__191 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_31, parameter_156, parameter_157, parameter_158, parameter_159, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_30 = paddle._C_ops.relu(batch_norm__186)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_32 = paddle._C_ops.conv2d(relu_30, parameter_160, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__192, batch_norm__193, batch_norm__194, batch_norm__195, batch_norm__196, batch_norm__197 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_32, parameter_161, parameter_162, parameter_163, parameter_164, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_11 = paddle._C_ops.add(batch_norm__192, relu_29)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_31 = paddle._C_ops.relu(add_11)

        # pd_op.shape: (4xi32) <- (-1x18x-1x-1xf32)
        shape_0 = paddle._C_ops.shape(relu_23)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [-2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [2147483647]

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], full_int_array_0, full_int_array_1, [1], [])

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x36x-1x-1xf32, 18x36x1x1xf32)
        conv2d_33 = paddle._C_ops.conv2d(relu_31, parameter_165, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__198, batch_norm__199, batch_norm__200, batch_norm__201, batch_norm__202, batch_norm__203 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_33, parameter_166, parameter_167, parameter_168, parameter_169, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.cast_: (2xi32) <- (2xi32)
        cast__0 = paddle._C_ops.cast_(slice_0, paddle.int32)

        # pd_op.bilinear_interp: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_0 = paddle._C_ops.bilinear_interp(batch_norm__198, cast__0, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_12 = paddle._C_ops.add(relu_23, bilinear_interp_0)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_32 = paddle._C_ops.relu(add_12)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x18x-1x-1xf32, 36x18x3x3xf32)
        conv2d_34 = paddle._C_ops.conv2d(relu_23, parameter_170, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__204, batch_norm__205, batch_norm__206, batch_norm__207, batch_norm__208, batch_norm__209 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_34, parameter_171, parameter_172, parameter_173, parameter_174, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_13 = paddle._C_ops.add(relu_31, batch_norm__204)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_33 = paddle._C_ops.relu(add_13)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x36x-1x-1xf32, 72x36x3x3xf32)
        conv2d_35 = paddle._C_ops.conv2d(relu_33, parameter_175, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__210, batch_norm__211, batch_norm__212, batch_norm__213, batch_norm__214, batch_norm__215 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_35, parameter_176, parameter_177, parameter_178, parameter_179, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_34 = paddle._C_ops.relu(batch_norm__210)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_36 = paddle._C_ops.conv2d(relu_32, parameter_180, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__216, batch_norm__217, batch_norm__218, batch_norm__219, batch_norm__220, batch_norm__221 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_36, parameter_181, parameter_182, parameter_183, parameter_184, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_35 = paddle._C_ops.relu(batch_norm__216)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_37 = paddle._C_ops.conv2d(relu_35, parameter_185, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__222, batch_norm__223, batch_norm__224, batch_norm__225, batch_norm__226, batch_norm__227 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_37, parameter_186, parameter_187, parameter_188, parameter_189, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_14 = paddle._C_ops.add(batch_norm__222, relu_32)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_36 = paddle._C_ops.relu(add_14)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_38 = paddle._C_ops.conv2d(relu_36, parameter_190, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__228, batch_norm__229, batch_norm__230, batch_norm__231, batch_norm__232, batch_norm__233 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_38, parameter_191, parameter_192, parameter_193, parameter_194, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_37 = paddle._C_ops.relu(batch_norm__228)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_39 = paddle._C_ops.conv2d(relu_37, parameter_195, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__234, batch_norm__235, batch_norm__236, batch_norm__237, batch_norm__238, batch_norm__239 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_39, parameter_196, parameter_197, parameter_198, parameter_199, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_15 = paddle._C_ops.add(batch_norm__234, relu_36)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_38 = paddle._C_ops.relu(add_15)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_40 = paddle._C_ops.conv2d(relu_38, parameter_200, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__240, batch_norm__241, batch_norm__242, batch_norm__243, batch_norm__244, batch_norm__245 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_40, parameter_201, parameter_202, parameter_203, parameter_204, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_39 = paddle._C_ops.relu(batch_norm__240)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_41 = paddle._C_ops.conv2d(relu_39, parameter_205, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__246, batch_norm__247, batch_norm__248, batch_norm__249, batch_norm__250, batch_norm__251 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_41, parameter_206, parameter_207, parameter_208, parameter_209, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_16 = paddle._C_ops.add(batch_norm__246, relu_38)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_40 = paddle._C_ops.relu(add_16)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_42 = paddle._C_ops.conv2d(relu_40, parameter_210, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__252, batch_norm__253, batch_norm__254, batch_norm__255, batch_norm__256, batch_norm__257 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_42, parameter_211, parameter_212, parameter_213, parameter_214, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_41 = paddle._C_ops.relu(batch_norm__252)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_43 = paddle._C_ops.conv2d(relu_41, parameter_215, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__258, batch_norm__259, batch_norm__260, batch_norm__261, batch_norm__262, batch_norm__263 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_43, parameter_216, parameter_217, parameter_218, parameter_219, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_17 = paddle._C_ops.add(batch_norm__258, relu_40)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_42 = paddle._C_ops.relu(add_17)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_44 = paddle._C_ops.conv2d(relu_33, parameter_220, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__264, batch_norm__265, batch_norm__266, batch_norm__267, batch_norm__268, batch_norm__269 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_44, parameter_221, parameter_222, parameter_223, parameter_224, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_43 = paddle._C_ops.relu(batch_norm__264)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_45 = paddle._C_ops.conv2d(relu_43, parameter_225, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__270, batch_norm__271, batch_norm__272, batch_norm__273, batch_norm__274, batch_norm__275 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_45, parameter_226, parameter_227, parameter_228, parameter_229, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_18 = paddle._C_ops.add(batch_norm__270, relu_33)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_44 = paddle._C_ops.relu(add_18)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_46 = paddle._C_ops.conv2d(relu_44, parameter_230, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__276, batch_norm__277, batch_norm__278, batch_norm__279, batch_norm__280, batch_norm__281 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_46, parameter_231, parameter_232, parameter_233, parameter_234, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_45 = paddle._C_ops.relu(batch_norm__276)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_47 = paddle._C_ops.conv2d(relu_45, parameter_235, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__282, batch_norm__283, batch_norm__284, batch_norm__285, batch_norm__286, batch_norm__287 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_47, parameter_236, parameter_237, parameter_238, parameter_239, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_19 = paddle._C_ops.add(batch_norm__282, relu_44)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_46 = paddle._C_ops.relu(add_19)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_48 = paddle._C_ops.conv2d(relu_46, parameter_240, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__288, batch_norm__289, batch_norm__290, batch_norm__291, batch_norm__292, batch_norm__293 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_48, parameter_241, parameter_242, parameter_243, parameter_244, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_47 = paddle._C_ops.relu(batch_norm__288)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_49 = paddle._C_ops.conv2d(relu_47, parameter_245, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__294, batch_norm__295, batch_norm__296, batch_norm__297, batch_norm__298, batch_norm__299 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_49, parameter_246, parameter_247, parameter_248, parameter_249, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_20 = paddle._C_ops.add(batch_norm__294, relu_46)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_48 = paddle._C_ops.relu(add_20)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_50 = paddle._C_ops.conv2d(relu_48, parameter_250, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__300, batch_norm__301, batch_norm__302, batch_norm__303, batch_norm__304, batch_norm__305 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_50, parameter_251, parameter_252, parameter_253, parameter_254, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_49 = paddle._C_ops.relu(batch_norm__300)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_51 = paddle._C_ops.conv2d(relu_49, parameter_255, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__306, batch_norm__307, batch_norm__308, batch_norm__309, batch_norm__310, batch_norm__311 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_51, parameter_256, parameter_257, parameter_258, parameter_259, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_21 = paddle._C_ops.add(batch_norm__306, relu_48)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_50 = paddle._C_ops.relu(add_21)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_52 = paddle._C_ops.conv2d(relu_34, parameter_260, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__312, batch_norm__313, batch_norm__314, batch_norm__315, batch_norm__316, batch_norm__317 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_52, parameter_261, parameter_262, parameter_263, parameter_264, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_51 = paddle._C_ops.relu(batch_norm__312)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_53 = paddle._C_ops.conv2d(relu_51, parameter_265, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__318, batch_norm__319, batch_norm__320, batch_norm__321, batch_norm__322, batch_norm__323 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_53, parameter_266, parameter_267, parameter_268, parameter_269, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_22 = paddle._C_ops.add(batch_norm__318, relu_34)

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_52 = paddle._C_ops.relu(add_22)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_54 = paddle._C_ops.conv2d(relu_52, parameter_270, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__324, batch_norm__325, batch_norm__326, batch_norm__327, batch_norm__328, batch_norm__329 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_54, parameter_271, parameter_272, parameter_273, parameter_274, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_53 = paddle._C_ops.relu(batch_norm__324)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_55 = paddle._C_ops.conv2d(relu_53, parameter_275, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__330, batch_norm__331, batch_norm__332, batch_norm__333, batch_norm__334, batch_norm__335 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_55, parameter_276, parameter_277, parameter_278, parameter_279, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_23 = paddle._C_ops.add(batch_norm__330, relu_52)

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_54 = paddle._C_ops.relu(add_23)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_56 = paddle._C_ops.conv2d(relu_54, parameter_280, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__336, batch_norm__337, batch_norm__338, batch_norm__339, batch_norm__340, batch_norm__341 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_56, parameter_281, parameter_282, parameter_283, parameter_284, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_55 = paddle._C_ops.relu(batch_norm__336)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_57 = paddle._C_ops.conv2d(relu_55, parameter_285, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__342, batch_norm__343, batch_norm__344, batch_norm__345, batch_norm__346, batch_norm__347 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_57, parameter_286, parameter_287, parameter_288, parameter_289, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_24 = paddle._C_ops.add(batch_norm__342, relu_54)

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_56 = paddle._C_ops.relu(add_24)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_58 = paddle._C_ops.conv2d(relu_56, parameter_290, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__348, batch_norm__349, batch_norm__350, batch_norm__351, batch_norm__352, batch_norm__353 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_58, parameter_291, parameter_292, parameter_293, parameter_294, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_57 = paddle._C_ops.relu(batch_norm__348)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_59 = paddle._C_ops.conv2d(relu_57, parameter_295, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__354, batch_norm__355, batch_norm__356, batch_norm__357, batch_norm__358, batch_norm__359 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_59, parameter_296, parameter_297, parameter_298, parameter_299, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_25 = paddle._C_ops.add(batch_norm__354, relu_56)

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_58 = paddle._C_ops.relu(add_25)

        # pd_op.shape: (4xi32) <- (-1x18x-1x-1xf32)
        shape_1 = paddle._C_ops.shape(relu_42)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [-2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [2147483647]

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(shape_1, [0], full_int_array_2, full_int_array_3, [1], [])

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x36x-1x-1xf32, 18x36x1x1xf32)
        conv2d_60 = paddle._C_ops.conv2d(relu_50, parameter_300, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__360, batch_norm__361, batch_norm__362, batch_norm__363, batch_norm__364, batch_norm__365 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_60, parameter_301, parameter_302, parameter_303, parameter_304, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.cast: (2xi32) <- (2xi32)
        cast_0 = paddle._C_ops.cast(slice_1, paddle.int32)

        # pd_op.bilinear_interp: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_1 = paddle._C_ops.bilinear_interp(batch_norm__360, cast_0, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_26 = paddle._C_ops.add(relu_42, bilinear_interp_1)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x72x-1x-1xf32, 18x72x1x1xf32)
        conv2d_61 = paddle._C_ops.conv2d(relu_58, parameter_305, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__366, batch_norm__367, batch_norm__368, batch_norm__369, batch_norm__370, batch_norm__371 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_61, parameter_306, parameter_307, parameter_308, parameter_309, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.cast_: (2xi32) <- (2xi32)
        cast__1 = paddle._C_ops.cast_(slice_1, paddle.int32)

        # pd_op.bilinear_interp: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_2 = paddle._C_ops.bilinear_interp(batch_norm__366, cast__1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_27 = paddle._C_ops.add(add_26, bilinear_interp_2)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_59 = paddle._C_ops.relu(add_27)

        # pd_op.shape: (4xi32) <- (-1x36x-1x-1xf32)
        shape_2 = paddle._C_ops.shape(relu_50)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [-2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [2147483647]

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(shape_2, [0], full_int_array_4, full_int_array_5, [1], [])

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x18x-1x-1xf32, 36x18x3x3xf32)
        conv2d_62 = paddle._C_ops.conv2d(relu_42, parameter_310, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__372, batch_norm__373, batch_norm__374, batch_norm__375, batch_norm__376, batch_norm__377 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_62, parameter_311, parameter_312, parameter_313, parameter_314, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_28 = paddle._C_ops.add(relu_50, batch_norm__372)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x72x-1x-1xf32, 36x72x1x1xf32)
        conv2d_63 = paddle._C_ops.conv2d(relu_58, parameter_315, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__378, batch_norm__379, batch_norm__380, batch_norm__381, batch_norm__382, batch_norm__383 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_63, parameter_316, parameter_317, parameter_318, parameter_319, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.cast_: (2xi32) <- (2xi32)
        cast__2 = paddle._C_ops.cast_(slice_2, paddle.int32)

        # pd_op.bilinear_interp: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_3 = paddle._C_ops.bilinear_interp(batch_norm__378, cast__2, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_29 = paddle._C_ops.add(add_28, bilinear_interp_3)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_60 = paddle._C_ops.relu(add_29)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_64 = paddle._C_ops.conv2d(relu_42, parameter_320, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__384, batch_norm__385, batch_norm__386, batch_norm__387, batch_norm__388, batch_norm__389 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_64, parameter_321, parameter_322, parameter_323, parameter_324, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_61 = paddle._C_ops.relu(batch_norm__384)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x18x-1x-1xf32, 72x18x3x3xf32)
        conv2d_65 = paddle._C_ops.conv2d(relu_61, parameter_325, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__390, batch_norm__391, batch_norm__392, batch_norm__393, batch_norm__394, batch_norm__395 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_65, parameter_326, parameter_327, parameter_328, parameter_329, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_30 = paddle._C_ops.add(relu_58, batch_norm__390)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x36x-1x-1xf32, 72x36x3x3xf32)
        conv2d_66 = paddle._C_ops.conv2d(relu_50, parameter_330, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__396, batch_norm__397, batch_norm__398, batch_norm__399, batch_norm__400, batch_norm__401 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_66, parameter_331, parameter_332, parameter_333, parameter_334, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_31 = paddle._C_ops.add(add_30, batch_norm__396)

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_62 = paddle._C_ops.relu(add_31)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_67 = paddle._C_ops.conv2d(relu_59, parameter_335, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__402, batch_norm__403, batch_norm__404, batch_norm__405, batch_norm__406, batch_norm__407 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_67, parameter_336, parameter_337, parameter_338, parameter_339, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_63 = paddle._C_ops.relu(batch_norm__402)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_68 = paddle._C_ops.conv2d(relu_63, parameter_340, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__408, batch_norm__409, batch_norm__410, batch_norm__411, batch_norm__412, batch_norm__413 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_68, parameter_341, parameter_342, parameter_343, parameter_344, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_32 = paddle._C_ops.add(batch_norm__408, relu_59)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_64 = paddle._C_ops.relu(add_32)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_69 = paddle._C_ops.conv2d(relu_64, parameter_345, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__414, batch_norm__415, batch_norm__416, batch_norm__417, batch_norm__418, batch_norm__419 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_69, parameter_346, parameter_347, parameter_348, parameter_349, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_65 = paddle._C_ops.relu(batch_norm__414)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_70 = paddle._C_ops.conv2d(relu_65, parameter_350, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__420, batch_norm__421, batch_norm__422, batch_norm__423, batch_norm__424, batch_norm__425 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_70, parameter_351, parameter_352, parameter_353, parameter_354, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_33 = paddle._C_ops.add(batch_norm__420, relu_64)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_66 = paddle._C_ops.relu(add_33)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_71 = paddle._C_ops.conv2d(relu_66, parameter_355, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__426, batch_norm__427, batch_norm__428, batch_norm__429, batch_norm__430, batch_norm__431 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_71, parameter_356, parameter_357, parameter_358, parameter_359, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_67 = paddle._C_ops.relu(batch_norm__426)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_72 = paddle._C_ops.conv2d(relu_67, parameter_360, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__432, batch_norm__433, batch_norm__434, batch_norm__435, batch_norm__436, batch_norm__437 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_72, parameter_361, parameter_362, parameter_363, parameter_364, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_34 = paddle._C_ops.add(batch_norm__432, relu_66)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_68 = paddle._C_ops.relu(add_34)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_73 = paddle._C_ops.conv2d(relu_68, parameter_365, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__438, batch_norm__439, batch_norm__440, batch_norm__441, batch_norm__442, batch_norm__443 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_73, parameter_366, parameter_367, parameter_368, parameter_369, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_69 = paddle._C_ops.relu(batch_norm__438)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_74 = paddle._C_ops.conv2d(relu_69, parameter_370, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__444, batch_norm__445, batch_norm__446, batch_norm__447, batch_norm__448, batch_norm__449 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_74, parameter_371, parameter_372, parameter_373, parameter_374, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_35 = paddle._C_ops.add(batch_norm__444, relu_68)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_70 = paddle._C_ops.relu(add_35)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_75 = paddle._C_ops.conv2d(relu_60, parameter_375, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__450, batch_norm__451, batch_norm__452, batch_norm__453, batch_norm__454, batch_norm__455 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_75, parameter_376, parameter_377, parameter_378, parameter_379, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_71 = paddle._C_ops.relu(batch_norm__450)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_76 = paddle._C_ops.conv2d(relu_71, parameter_380, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__456, batch_norm__457, batch_norm__458, batch_norm__459, batch_norm__460, batch_norm__461 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_76, parameter_381, parameter_382, parameter_383, parameter_384, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_36 = paddle._C_ops.add(batch_norm__456, relu_60)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_72 = paddle._C_ops.relu(add_36)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_77 = paddle._C_ops.conv2d(relu_72, parameter_385, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__462, batch_norm__463, batch_norm__464, batch_norm__465, batch_norm__466, batch_norm__467 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_77, parameter_386, parameter_387, parameter_388, parameter_389, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_73 = paddle._C_ops.relu(batch_norm__462)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_78 = paddle._C_ops.conv2d(relu_73, parameter_390, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__468, batch_norm__469, batch_norm__470, batch_norm__471, batch_norm__472, batch_norm__473 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_78, parameter_391, parameter_392, parameter_393, parameter_394, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_37 = paddle._C_ops.add(batch_norm__468, relu_72)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_74 = paddle._C_ops.relu(add_37)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_79 = paddle._C_ops.conv2d(relu_74, parameter_395, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__474, batch_norm__475, batch_norm__476, batch_norm__477, batch_norm__478, batch_norm__479 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_79, parameter_396, parameter_397, parameter_398, parameter_399, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_75 = paddle._C_ops.relu(batch_norm__474)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_80 = paddle._C_ops.conv2d(relu_75, parameter_400, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__480, batch_norm__481, batch_norm__482, batch_norm__483, batch_norm__484, batch_norm__485 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_80, parameter_401, parameter_402, parameter_403, parameter_404, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_38 = paddle._C_ops.add(batch_norm__480, relu_74)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_76 = paddle._C_ops.relu(add_38)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_81 = paddle._C_ops.conv2d(relu_76, parameter_405, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__486, batch_norm__487, batch_norm__488, batch_norm__489, batch_norm__490, batch_norm__491 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_81, parameter_406, parameter_407, parameter_408, parameter_409, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_77 = paddle._C_ops.relu(batch_norm__486)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_82 = paddle._C_ops.conv2d(relu_77, parameter_410, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__492, batch_norm__493, batch_norm__494, batch_norm__495, batch_norm__496, batch_norm__497 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_82, parameter_411, parameter_412, parameter_413, parameter_414, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_39 = paddle._C_ops.add(batch_norm__492, relu_76)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_78 = paddle._C_ops.relu(add_39)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_83 = paddle._C_ops.conv2d(relu_62, parameter_415, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__498, batch_norm__499, batch_norm__500, batch_norm__501, batch_norm__502, batch_norm__503 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_83, parameter_416, parameter_417, parameter_418, parameter_419, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_79 = paddle._C_ops.relu(batch_norm__498)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_84 = paddle._C_ops.conv2d(relu_79, parameter_420, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__504, batch_norm__505, batch_norm__506, batch_norm__507, batch_norm__508, batch_norm__509 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_84, parameter_421, parameter_422, parameter_423, parameter_424, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_40 = paddle._C_ops.add(batch_norm__504, relu_62)

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_80 = paddle._C_ops.relu(add_40)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_85 = paddle._C_ops.conv2d(relu_80, parameter_425, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__510, batch_norm__511, batch_norm__512, batch_norm__513, batch_norm__514, batch_norm__515 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_85, parameter_426, parameter_427, parameter_428, parameter_429, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_81 = paddle._C_ops.relu(batch_norm__510)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_86 = paddle._C_ops.conv2d(relu_81, parameter_430, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__516, batch_norm__517, batch_norm__518, batch_norm__519, batch_norm__520, batch_norm__521 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_86, parameter_431, parameter_432, parameter_433, parameter_434, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_41 = paddle._C_ops.add(batch_norm__516, relu_80)

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_82 = paddle._C_ops.relu(add_41)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_87 = paddle._C_ops.conv2d(relu_82, parameter_435, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__522, batch_norm__523, batch_norm__524, batch_norm__525, batch_norm__526, batch_norm__527 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_87, parameter_436, parameter_437, parameter_438, parameter_439, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_83 = paddle._C_ops.relu(batch_norm__522)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_88 = paddle._C_ops.conv2d(relu_83, parameter_440, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__528, batch_norm__529, batch_norm__530, batch_norm__531, batch_norm__532, batch_norm__533 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_88, parameter_441, parameter_442, parameter_443, parameter_444, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_42 = paddle._C_ops.add(batch_norm__528, relu_82)

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_84 = paddle._C_ops.relu(add_42)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_89 = paddle._C_ops.conv2d(relu_84, parameter_445, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__534, batch_norm__535, batch_norm__536, batch_norm__537, batch_norm__538, batch_norm__539 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_89, parameter_446, parameter_447, parameter_448, parameter_449, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_85 = paddle._C_ops.relu(batch_norm__534)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_90 = paddle._C_ops.conv2d(relu_85, parameter_450, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__540, batch_norm__541, batch_norm__542, batch_norm__543, batch_norm__544, batch_norm__545 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_90, parameter_451, parameter_452, parameter_453, parameter_454, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_43 = paddle._C_ops.add(batch_norm__540, relu_84)

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_86 = paddle._C_ops.relu(add_43)

        # pd_op.shape: (4xi32) <- (-1x18x-1x-1xf32)
        shape_3 = paddle._C_ops.shape(relu_70)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [-2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [2147483647]

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(shape_3, [0], full_int_array_6, full_int_array_7, [1], [])

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x36x-1x-1xf32, 18x36x1x1xf32)
        conv2d_91 = paddle._C_ops.conv2d(relu_78, parameter_455, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__546, batch_norm__547, batch_norm__548, batch_norm__549, batch_norm__550, batch_norm__551 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_91, parameter_456, parameter_457, parameter_458, parameter_459, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.cast: (2xi32) <- (2xi32)
        cast_1 = paddle._C_ops.cast(slice_3, paddle.int32)

        # pd_op.bilinear_interp: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_4 = paddle._C_ops.bilinear_interp(batch_norm__546, cast_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_44 = paddle._C_ops.add(relu_70, bilinear_interp_4)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x72x-1x-1xf32, 18x72x1x1xf32)
        conv2d_92 = paddle._C_ops.conv2d(relu_86, parameter_460, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__552, batch_norm__553, batch_norm__554, batch_norm__555, batch_norm__556, batch_norm__557 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_92, parameter_461, parameter_462, parameter_463, parameter_464, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.cast_: (2xi32) <- (2xi32)
        cast__3 = paddle._C_ops.cast_(slice_3, paddle.int32)

        # pd_op.bilinear_interp: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_5 = paddle._C_ops.bilinear_interp(batch_norm__552, cast__3, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_45 = paddle._C_ops.add(add_44, bilinear_interp_5)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_87 = paddle._C_ops.relu(add_45)

        # pd_op.shape: (4xi32) <- (-1x36x-1x-1xf32)
        shape_4 = paddle._C_ops.shape(relu_78)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [-2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [2147483647]

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(shape_4, [0], full_int_array_8, full_int_array_9, [1], [])

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x18x-1x-1xf32, 36x18x3x3xf32)
        conv2d_93 = paddle._C_ops.conv2d(relu_70, parameter_465, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__558, batch_norm__559, batch_norm__560, batch_norm__561, batch_norm__562, batch_norm__563 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_93, parameter_466, parameter_467, parameter_468, parameter_469, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_46 = paddle._C_ops.add(relu_78, batch_norm__558)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x72x-1x-1xf32, 36x72x1x1xf32)
        conv2d_94 = paddle._C_ops.conv2d(relu_86, parameter_470, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__564, batch_norm__565, batch_norm__566, batch_norm__567, batch_norm__568, batch_norm__569 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_94, parameter_471, parameter_472, parameter_473, parameter_474, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.cast_: (2xi32) <- (2xi32)
        cast__4 = paddle._C_ops.cast_(slice_4, paddle.int32)

        # pd_op.bilinear_interp: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_6 = paddle._C_ops.bilinear_interp(batch_norm__564, cast__4, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_47 = paddle._C_ops.add(add_46, bilinear_interp_6)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_88 = paddle._C_ops.relu(add_47)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_95 = paddle._C_ops.conv2d(relu_70, parameter_475, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__570, batch_norm__571, batch_norm__572, batch_norm__573, batch_norm__574, batch_norm__575 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_95, parameter_476, parameter_477, parameter_478, parameter_479, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_89 = paddle._C_ops.relu(batch_norm__570)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x18x-1x-1xf32, 72x18x3x3xf32)
        conv2d_96 = paddle._C_ops.conv2d(relu_89, parameter_480, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__576, batch_norm__577, batch_norm__578, batch_norm__579, batch_norm__580, batch_norm__581 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_96, parameter_481, parameter_482, parameter_483, parameter_484, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_48 = paddle._C_ops.add(relu_86, batch_norm__576)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x36x-1x-1xf32, 72x36x3x3xf32)
        conv2d_97 = paddle._C_ops.conv2d(relu_78, parameter_485, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__582, batch_norm__583, batch_norm__584, batch_norm__585, batch_norm__586, batch_norm__587 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_97, parameter_486, parameter_487, parameter_488, parameter_489, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_49 = paddle._C_ops.add(add_48, batch_norm__582)

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_90 = paddle._C_ops.relu(add_49)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_98 = paddle._C_ops.conv2d(relu_87, parameter_490, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__588, batch_norm__589, batch_norm__590, batch_norm__591, batch_norm__592, batch_norm__593 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_98, parameter_491, parameter_492, parameter_493, parameter_494, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_91 = paddle._C_ops.relu(batch_norm__588)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_99 = paddle._C_ops.conv2d(relu_91, parameter_495, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__594, batch_norm__595, batch_norm__596, batch_norm__597, batch_norm__598, batch_norm__599 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_99, parameter_496, parameter_497, parameter_498, parameter_499, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_50 = paddle._C_ops.add(batch_norm__594, relu_87)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_92 = paddle._C_ops.relu(add_50)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_100 = paddle._C_ops.conv2d(relu_92, parameter_500, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__600, batch_norm__601, batch_norm__602, batch_norm__603, batch_norm__604, batch_norm__605 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_100, parameter_501, parameter_502, parameter_503, parameter_504, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_93 = paddle._C_ops.relu(batch_norm__600)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_101 = paddle._C_ops.conv2d(relu_93, parameter_505, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__606, batch_norm__607, batch_norm__608, batch_norm__609, batch_norm__610, batch_norm__611 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_101, parameter_506, parameter_507, parameter_508, parameter_509, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_51 = paddle._C_ops.add(batch_norm__606, relu_92)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_94 = paddle._C_ops.relu(add_51)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_102 = paddle._C_ops.conv2d(relu_94, parameter_510, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__612, batch_norm__613, batch_norm__614, batch_norm__615, batch_norm__616, batch_norm__617 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_102, parameter_511, parameter_512, parameter_513, parameter_514, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_95 = paddle._C_ops.relu(batch_norm__612)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_103 = paddle._C_ops.conv2d(relu_95, parameter_515, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__618, batch_norm__619, batch_norm__620, batch_norm__621, batch_norm__622, batch_norm__623 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_103, parameter_516, parameter_517, parameter_518, parameter_519, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_52 = paddle._C_ops.add(batch_norm__618, relu_94)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_96 = paddle._C_ops.relu(add_52)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_104 = paddle._C_ops.conv2d(relu_96, parameter_520, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__624, batch_norm__625, batch_norm__626, batch_norm__627, batch_norm__628, batch_norm__629 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_104, parameter_521, parameter_522, parameter_523, parameter_524, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_97 = paddle._C_ops.relu(batch_norm__624)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_105 = paddle._C_ops.conv2d(relu_97, parameter_525, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__630, batch_norm__631, batch_norm__632, batch_norm__633, batch_norm__634, batch_norm__635 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_105, parameter_526, parameter_527, parameter_528, parameter_529, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_53 = paddle._C_ops.add(batch_norm__630, relu_96)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_98 = paddle._C_ops.relu(add_53)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_106 = paddle._C_ops.conv2d(relu_88, parameter_530, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__636, batch_norm__637, batch_norm__638, batch_norm__639, batch_norm__640, batch_norm__641 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_106, parameter_531, parameter_532, parameter_533, parameter_534, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_99 = paddle._C_ops.relu(batch_norm__636)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_107 = paddle._C_ops.conv2d(relu_99, parameter_535, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__642, batch_norm__643, batch_norm__644, batch_norm__645, batch_norm__646, batch_norm__647 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_107, parameter_536, parameter_537, parameter_538, parameter_539, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_54 = paddle._C_ops.add(batch_norm__642, relu_88)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_100 = paddle._C_ops.relu(add_54)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_108 = paddle._C_ops.conv2d(relu_100, parameter_540, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__648, batch_norm__649, batch_norm__650, batch_norm__651, batch_norm__652, batch_norm__653 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_108, parameter_541, parameter_542, parameter_543, parameter_544, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_101 = paddle._C_ops.relu(batch_norm__648)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_109 = paddle._C_ops.conv2d(relu_101, parameter_545, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__654, batch_norm__655, batch_norm__656, batch_norm__657, batch_norm__658, batch_norm__659 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_109, parameter_546, parameter_547, parameter_548, parameter_549, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_55 = paddle._C_ops.add(batch_norm__654, relu_100)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_102 = paddle._C_ops.relu(add_55)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_110 = paddle._C_ops.conv2d(relu_102, parameter_550, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__660, batch_norm__661, batch_norm__662, batch_norm__663, batch_norm__664, batch_norm__665 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_110, parameter_551, parameter_552, parameter_553, parameter_554, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_103 = paddle._C_ops.relu(batch_norm__660)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_111 = paddle._C_ops.conv2d(relu_103, parameter_555, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__666, batch_norm__667, batch_norm__668, batch_norm__669, batch_norm__670, batch_norm__671 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_111, parameter_556, parameter_557, parameter_558, parameter_559, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_56 = paddle._C_ops.add(batch_norm__666, relu_102)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_104 = paddle._C_ops.relu(add_56)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_112 = paddle._C_ops.conv2d(relu_104, parameter_560, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__672, batch_norm__673, batch_norm__674, batch_norm__675, batch_norm__676, batch_norm__677 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_112, parameter_561, parameter_562, parameter_563, parameter_564, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_105 = paddle._C_ops.relu(batch_norm__672)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_113 = paddle._C_ops.conv2d(relu_105, parameter_565, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__678, batch_norm__679, batch_norm__680, batch_norm__681, batch_norm__682, batch_norm__683 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_113, parameter_566, parameter_567, parameter_568, parameter_569, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_57 = paddle._C_ops.add(batch_norm__678, relu_104)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_106 = paddle._C_ops.relu(add_57)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_114 = paddle._C_ops.conv2d(relu_90, parameter_570, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__684, batch_norm__685, batch_norm__686, batch_norm__687, batch_norm__688, batch_norm__689 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_114, parameter_571, parameter_572, parameter_573, parameter_574, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_107 = paddle._C_ops.relu(batch_norm__684)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_115 = paddle._C_ops.conv2d(relu_107, parameter_575, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__690, batch_norm__691, batch_norm__692, batch_norm__693, batch_norm__694, batch_norm__695 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_115, parameter_576, parameter_577, parameter_578, parameter_579, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_58 = paddle._C_ops.add(batch_norm__690, relu_90)

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_108 = paddle._C_ops.relu(add_58)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_116 = paddle._C_ops.conv2d(relu_108, parameter_580, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__696, batch_norm__697, batch_norm__698, batch_norm__699, batch_norm__700, batch_norm__701 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_116, parameter_581, parameter_582, parameter_583, parameter_584, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_109 = paddle._C_ops.relu(batch_norm__696)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_117 = paddle._C_ops.conv2d(relu_109, parameter_585, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__702, batch_norm__703, batch_norm__704, batch_norm__705, batch_norm__706, batch_norm__707 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_117, parameter_586, parameter_587, parameter_588, parameter_589, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_59 = paddle._C_ops.add(batch_norm__702, relu_108)

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_110 = paddle._C_ops.relu(add_59)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_118 = paddle._C_ops.conv2d(relu_110, parameter_590, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__708, batch_norm__709, batch_norm__710, batch_norm__711, batch_norm__712, batch_norm__713 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_118, parameter_591, parameter_592, parameter_593, parameter_594, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_111 = paddle._C_ops.relu(batch_norm__708)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_119 = paddle._C_ops.conv2d(relu_111, parameter_595, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__714, batch_norm__715, batch_norm__716, batch_norm__717, batch_norm__718, batch_norm__719 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_119, parameter_596, parameter_597, parameter_598, parameter_599, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_60 = paddle._C_ops.add(batch_norm__714, relu_110)

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_112 = paddle._C_ops.relu(add_60)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_120 = paddle._C_ops.conv2d(relu_112, parameter_600, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__720, batch_norm__721, batch_norm__722, batch_norm__723, batch_norm__724, batch_norm__725 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_120, parameter_601, parameter_602, parameter_603, parameter_604, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_113 = paddle._C_ops.relu(batch_norm__720)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_121 = paddle._C_ops.conv2d(relu_113, parameter_605, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__726, batch_norm__727, batch_norm__728, batch_norm__729, batch_norm__730, batch_norm__731 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_121, parameter_606, parameter_607, parameter_608, parameter_609, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_61 = paddle._C_ops.add(batch_norm__726, relu_112)

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_114 = paddle._C_ops.relu(add_61)

        # pd_op.shape: (4xi32) <- (-1x18x-1x-1xf32)
        shape_5 = paddle._C_ops.shape(relu_98)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_10 = [-2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_11 = [2147483647]

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(shape_5, [0], full_int_array_10, full_int_array_11, [1], [])

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x36x-1x-1xf32, 18x36x1x1xf32)
        conv2d_122 = paddle._C_ops.conv2d(relu_106, parameter_610, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__732, batch_norm__733, batch_norm__734, batch_norm__735, batch_norm__736, batch_norm__737 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_122, parameter_611, parameter_612, parameter_613, parameter_614, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.cast: (2xi32) <- (2xi32)
        cast_2 = paddle._C_ops.cast(slice_5, paddle.int32)

        # pd_op.bilinear_interp: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_7 = paddle._C_ops.bilinear_interp(batch_norm__732, cast_2, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_62 = paddle._C_ops.add(relu_98, bilinear_interp_7)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x72x-1x-1xf32, 18x72x1x1xf32)
        conv2d_123 = paddle._C_ops.conv2d(relu_114, parameter_615, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__738, batch_norm__739, batch_norm__740, batch_norm__741, batch_norm__742, batch_norm__743 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_123, parameter_616, parameter_617, parameter_618, parameter_619, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.cast_: (2xi32) <- (2xi32)
        cast__5 = paddle._C_ops.cast_(slice_5, paddle.int32)

        # pd_op.bilinear_interp: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_8 = paddle._C_ops.bilinear_interp(batch_norm__738, cast__5, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_63 = paddle._C_ops.add(add_62, bilinear_interp_8)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_115 = paddle._C_ops.relu(add_63)

        # pd_op.shape: (4xi32) <- (-1x36x-1x-1xf32)
        shape_6 = paddle._C_ops.shape(relu_106)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_12 = [-2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_13 = [2147483647]

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(shape_6, [0], full_int_array_12, full_int_array_13, [1], [])

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x18x-1x-1xf32, 36x18x3x3xf32)
        conv2d_124 = paddle._C_ops.conv2d(relu_98, parameter_620, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__744, batch_norm__745, batch_norm__746, batch_norm__747, batch_norm__748, batch_norm__749 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_124, parameter_621, parameter_622, parameter_623, parameter_624, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_64 = paddle._C_ops.add(relu_106, batch_norm__744)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x72x-1x-1xf32, 36x72x1x1xf32)
        conv2d_125 = paddle._C_ops.conv2d(relu_114, parameter_625, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__750, batch_norm__751, batch_norm__752, batch_norm__753, batch_norm__754, batch_norm__755 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_125, parameter_626, parameter_627, parameter_628, parameter_629, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.cast_: (2xi32) <- (2xi32)
        cast__6 = paddle._C_ops.cast_(slice_6, paddle.int32)

        # pd_op.bilinear_interp: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_9 = paddle._C_ops.bilinear_interp(batch_norm__750, cast__6, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_65 = paddle._C_ops.add(add_64, bilinear_interp_9)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_116 = paddle._C_ops.relu(add_65)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_126 = paddle._C_ops.conv2d(relu_98, parameter_630, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__756, batch_norm__757, batch_norm__758, batch_norm__759, batch_norm__760, batch_norm__761 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_126, parameter_631, parameter_632, parameter_633, parameter_634, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_117 = paddle._C_ops.relu(batch_norm__756)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x18x-1x-1xf32, 72x18x3x3xf32)
        conv2d_127 = paddle._C_ops.conv2d(relu_117, parameter_635, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__762, batch_norm__763, batch_norm__764, batch_norm__765, batch_norm__766, batch_norm__767 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_127, parameter_636, parameter_637, parameter_638, parameter_639, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_66 = paddle._C_ops.add(relu_114, batch_norm__762)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x36x-1x-1xf32, 72x36x3x3xf32)
        conv2d_128 = paddle._C_ops.conv2d(relu_106, parameter_640, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__768, batch_norm__769, batch_norm__770, batch_norm__771, batch_norm__772, batch_norm__773 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_128, parameter_641, parameter_642, parameter_643, parameter_644, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_67 = paddle._C_ops.add(add_66, batch_norm__768)

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_118 = paddle._C_ops.relu(add_67)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_129 = paddle._C_ops.conv2d(relu_115, parameter_645, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__774, batch_norm__775, batch_norm__776, batch_norm__777, batch_norm__778, batch_norm__779 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_129, parameter_646, parameter_647, parameter_648, parameter_649, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_119 = paddle._C_ops.relu(batch_norm__774)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_130 = paddle._C_ops.conv2d(relu_119, parameter_650, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__780, batch_norm__781, batch_norm__782, batch_norm__783, batch_norm__784, batch_norm__785 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_130, parameter_651, parameter_652, parameter_653, parameter_654, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_68 = paddle._C_ops.add(batch_norm__780, relu_115)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_120 = paddle._C_ops.relu(add_68)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_131 = paddle._C_ops.conv2d(relu_120, parameter_655, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__786, batch_norm__787, batch_norm__788, batch_norm__789, batch_norm__790, batch_norm__791 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_131, parameter_656, parameter_657, parameter_658, parameter_659, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_121 = paddle._C_ops.relu(batch_norm__786)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_132 = paddle._C_ops.conv2d(relu_121, parameter_660, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__792, batch_norm__793, batch_norm__794, batch_norm__795, batch_norm__796, batch_norm__797 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_132, parameter_661, parameter_662, parameter_663, parameter_664, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_69 = paddle._C_ops.add(batch_norm__792, relu_120)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_122 = paddle._C_ops.relu(add_69)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_133 = paddle._C_ops.conv2d(relu_122, parameter_665, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__798, batch_norm__799, batch_norm__800, batch_norm__801, batch_norm__802, batch_norm__803 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_133, parameter_666, parameter_667, parameter_668, parameter_669, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_123 = paddle._C_ops.relu(batch_norm__798)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_134 = paddle._C_ops.conv2d(relu_123, parameter_670, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__804, batch_norm__805, batch_norm__806, batch_norm__807, batch_norm__808, batch_norm__809 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_134, parameter_671, parameter_672, parameter_673, parameter_674, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_70 = paddle._C_ops.add(batch_norm__804, relu_122)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_124 = paddle._C_ops.relu(add_70)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_135 = paddle._C_ops.conv2d(relu_124, parameter_675, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__810, batch_norm__811, batch_norm__812, batch_norm__813, batch_norm__814, batch_norm__815 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_135, parameter_676, parameter_677, parameter_678, parameter_679, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_125 = paddle._C_ops.relu(batch_norm__810)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_136 = paddle._C_ops.conv2d(relu_125, parameter_680, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__816, batch_norm__817, batch_norm__818, batch_norm__819, batch_norm__820, batch_norm__821 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_136, parameter_681, parameter_682, parameter_683, parameter_684, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_71 = paddle._C_ops.add(batch_norm__816, relu_124)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_126 = paddle._C_ops.relu(add_71)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_137 = paddle._C_ops.conv2d(relu_116, parameter_685, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__822, batch_norm__823, batch_norm__824, batch_norm__825, batch_norm__826, batch_norm__827 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_137, parameter_686, parameter_687, parameter_688, parameter_689, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_127 = paddle._C_ops.relu(batch_norm__822)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_138 = paddle._C_ops.conv2d(relu_127, parameter_690, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__828, batch_norm__829, batch_norm__830, batch_norm__831, batch_norm__832, batch_norm__833 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_138, parameter_691, parameter_692, parameter_693, parameter_694, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_72 = paddle._C_ops.add(batch_norm__828, relu_116)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_128 = paddle._C_ops.relu(add_72)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_139 = paddle._C_ops.conv2d(relu_128, parameter_695, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__834, batch_norm__835, batch_norm__836, batch_norm__837, batch_norm__838, batch_norm__839 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_139, parameter_696, parameter_697, parameter_698, parameter_699, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_129 = paddle._C_ops.relu(batch_norm__834)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_140 = paddle._C_ops.conv2d(relu_129, parameter_700, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__840, batch_norm__841, batch_norm__842, batch_norm__843, batch_norm__844, batch_norm__845 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_140, parameter_701, parameter_702, parameter_703, parameter_704, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_73 = paddle._C_ops.add(batch_norm__840, relu_128)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_130 = paddle._C_ops.relu(add_73)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_141 = paddle._C_ops.conv2d(relu_130, parameter_705, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__846, batch_norm__847, batch_norm__848, batch_norm__849, batch_norm__850, batch_norm__851 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_141, parameter_706, parameter_707, parameter_708, parameter_709, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_131 = paddle._C_ops.relu(batch_norm__846)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_142 = paddle._C_ops.conv2d(relu_131, parameter_710, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__852, batch_norm__853, batch_norm__854, batch_norm__855, batch_norm__856, batch_norm__857 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_142, parameter_711, parameter_712, parameter_713, parameter_714, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_74 = paddle._C_ops.add(batch_norm__852, relu_130)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_132 = paddle._C_ops.relu(add_74)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_143 = paddle._C_ops.conv2d(relu_132, parameter_715, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__858, batch_norm__859, batch_norm__860, batch_norm__861, batch_norm__862, batch_norm__863 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_143, parameter_716, parameter_717, parameter_718, parameter_719, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_133 = paddle._C_ops.relu(batch_norm__858)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_144 = paddle._C_ops.conv2d(relu_133, parameter_720, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__864, batch_norm__865, batch_norm__866, batch_norm__867, batch_norm__868, batch_norm__869 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_144, parameter_721, parameter_722, parameter_723, parameter_724, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_75 = paddle._C_ops.add(batch_norm__864, relu_132)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_134 = paddle._C_ops.relu(add_75)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_145 = paddle._C_ops.conv2d(relu_118, parameter_725, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__870, batch_norm__871, batch_norm__872, batch_norm__873, batch_norm__874, batch_norm__875 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_145, parameter_726, parameter_727, parameter_728, parameter_729, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_135 = paddle._C_ops.relu(batch_norm__870)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_146 = paddle._C_ops.conv2d(relu_135, parameter_730, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__876, batch_norm__877, batch_norm__878, batch_norm__879, batch_norm__880, batch_norm__881 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_146, parameter_731, parameter_732, parameter_733, parameter_734, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_76 = paddle._C_ops.add(batch_norm__876, relu_118)

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_136 = paddle._C_ops.relu(add_76)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_147 = paddle._C_ops.conv2d(relu_136, parameter_735, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__882, batch_norm__883, batch_norm__884, batch_norm__885, batch_norm__886, batch_norm__887 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_147, parameter_736, parameter_737, parameter_738, parameter_739, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_137 = paddle._C_ops.relu(batch_norm__882)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_148 = paddle._C_ops.conv2d(relu_137, parameter_740, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__888, batch_norm__889, batch_norm__890, batch_norm__891, batch_norm__892, batch_norm__893 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_148, parameter_741, parameter_742, parameter_743, parameter_744, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_77 = paddle._C_ops.add(batch_norm__888, relu_136)

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_138 = paddle._C_ops.relu(add_77)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_149 = paddle._C_ops.conv2d(relu_138, parameter_745, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__894, batch_norm__895, batch_norm__896, batch_norm__897, batch_norm__898, batch_norm__899 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_149, parameter_746, parameter_747, parameter_748, parameter_749, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_139 = paddle._C_ops.relu(batch_norm__894)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_150 = paddle._C_ops.conv2d(relu_139, parameter_750, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__900, batch_norm__901, batch_norm__902, batch_norm__903, batch_norm__904, batch_norm__905 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_150, parameter_751, parameter_752, parameter_753, parameter_754, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_78 = paddle._C_ops.add(batch_norm__900, relu_138)

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_140 = paddle._C_ops.relu(add_78)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_151 = paddle._C_ops.conv2d(relu_140, parameter_755, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__906, batch_norm__907, batch_norm__908, batch_norm__909, batch_norm__910, batch_norm__911 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_151, parameter_756, parameter_757, parameter_758, parameter_759, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_141 = paddle._C_ops.relu(batch_norm__906)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_152 = paddle._C_ops.conv2d(relu_141, parameter_760, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__912, batch_norm__913, batch_norm__914, batch_norm__915, batch_norm__916, batch_norm__917 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_152, parameter_761, parameter_762, parameter_763, parameter_764, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_79 = paddle._C_ops.add(batch_norm__912, relu_140)

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_142 = paddle._C_ops.relu(add_79)

        # pd_op.shape: (4xi32) <- (-1x18x-1x-1xf32)
        shape_7 = paddle._C_ops.shape(relu_126)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_14 = [-2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_15 = [2147483647]

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(shape_7, [0], full_int_array_14, full_int_array_15, [1], [])

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x36x-1x-1xf32, 18x36x1x1xf32)
        conv2d_153 = paddle._C_ops.conv2d(relu_134, parameter_765, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__918, batch_norm__919, batch_norm__920, batch_norm__921, batch_norm__922, batch_norm__923 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_153, parameter_766, parameter_767, parameter_768, parameter_769, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.cast: (2xi32) <- (2xi32)
        cast_3 = paddle._C_ops.cast(slice_7, paddle.int32)

        # pd_op.bilinear_interp: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_10 = paddle._C_ops.bilinear_interp(batch_norm__918, cast_3, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_80 = paddle._C_ops.add(relu_126, bilinear_interp_10)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x72x-1x-1xf32, 18x72x1x1xf32)
        conv2d_154 = paddle._C_ops.conv2d(relu_142, parameter_770, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__924, batch_norm__925, batch_norm__926, batch_norm__927, batch_norm__928, batch_norm__929 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_154, parameter_771, parameter_772, parameter_773, parameter_774, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.cast_: (2xi32) <- (2xi32)
        cast__7 = paddle._C_ops.cast_(slice_7, paddle.int32)

        # pd_op.bilinear_interp: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_11 = paddle._C_ops.bilinear_interp(batch_norm__924, cast__7, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_81 = paddle._C_ops.add(add_80, bilinear_interp_11)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_143 = paddle._C_ops.relu(add_81)

        # pd_op.shape: (4xi32) <- (-1x36x-1x-1xf32)
        shape_8 = paddle._C_ops.shape(relu_134)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_16 = [-2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_17 = [2147483647]

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(shape_8, [0], full_int_array_16, full_int_array_17, [1], [])

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x18x-1x-1xf32, 36x18x3x3xf32)
        conv2d_155 = paddle._C_ops.conv2d(relu_126, parameter_775, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__930, batch_norm__931, batch_norm__932, batch_norm__933, batch_norm__934, batch_norm__935 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_155, parameter_776, parameter_777, parameter_778, parameter_779, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_82 = paddle._C_ops.add(relu_134, batch_norm__930)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x72x-1x-1xf32, 36x72x1x1xf32)
        conv2d_156 = paddle._C_ops.conv2d(relu_142, parameter_780, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__936, batch_norm__937, batch_norm__938, batch_norm__939, batch_norm__940, batch_norm__941 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_156, parameter_781, parameter_782, parameter_783, parameter_784, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.cast_: (2xi32) <- (2xi32)
        cast__8 = paddle._C_ops.cast_(slice_8, paddle.int32)

        # pd_op.bilinear_interp: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_12 = paddle._C_ops.bilinear_interp(batch_norm__936, cast__8, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_83 = paddle._C_ops.add(add_82, bilinear_interp_12)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_144 = paddle._C_ops.relu(add_83)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_157 = paddle._C_ops.conv2d(relu_126, parameter_785, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__942, batch_norm__943, batch_norm__944, batch_norm__945, batch_norm__946, batch_norm__947 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_157, parameter_786, parameter_787, parameter_788, parameter_789, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_145 = paddle._C_ops.relu(batch_norm__942)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x18x-1x-1xf32, 72x18x3x3xf32)
        conv2d_158 = paddle._C_ops.conv2d(relu_145, parameter_790, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__948, batch_norm__949, batch_norm__950, batch_norm__951, batch_norm__952, batch_norm__953 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_158, parameter_791, parameter_792, parameter_793, parameter_794, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_84 = paddle._C_ops.add(relu_142, batch_norm__948)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x36x-1x-1xf32, 72x36x3x3xf32)
        conv2d_159 = paddle._C_ops.conv2d(relu_134, parameter_795, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__954, batch_norm__955, batch_norm__956, batch_norm__957, batch_norm__958, batch_norm__959 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_159, parameter_796, parameter_797, parameter_798, parameter_799, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_85 = paddle._C_ops.add(add_84, batch_norm__954)

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_146 = paddle._C_ops.relu(add_85)

        # pd_op.conv2d: (-1x144x-1x-1xf32) <- (-1x72x-1x-1xf32, 144x72x3x3xf32)
        conv2d_160 = paddle._C_ops.conv2d(relu_146, parameter_800, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x-1x-1xf32, 144xf32, 144xf32, xf32, xf32, None) <- (-1x144x-1x-1xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__960, batch_norm__961, batch_norm__962, batch_norm__963, batch_norm__964, batch_norm__965 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_160, parameter_801, parameter_802, parameter_803, parameter_804, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32)
        relu_147 = paddle._C_ops.relu(batch_norm__960)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_161 = paddle._C_ops.conv2d(relu_143, parameter_805, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__966, batch_norm__967, batch_norm__968, batch_norm__969, batch_norm__970, batch_norm__971 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_161, parameter_806, parameter_807, parameter_808, parameter_809, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_148 = paddle._C_ops.relu(batch_norm__966)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_162 = paddle._C_ops.conv2d(relu_148, parameter_810, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__972, batch_norm__973, batch_norm__974, batch_norm__975, batch_norm__976, batch_norm__977 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_162, parameter_811, parameter_812, parameter_813, parameter_814, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_86 = paddle._C_ops.add(batch_norm__972, relu_143)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_149 = paddle._C_ops.relu(add_86)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_163 = paddle._C_ops.conv2d(relu_149, parameter_815, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__978, batch_norm__979, batch_norm__980, batch_norm__981, batch_norm__982, batch_norm__983 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_163, parameter_816, parameter_817, parameter_818, parameter_819, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_150 = paddle._C_ops.relu(batch_norm__978)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_164 = paddle._C_ops.conv2d(relu_150, parameter_820, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__984, batch_norm__985, batch_norm__986, batch_norm__987, batch_norm__988, batch_norm__989 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_164, parameter_821, parameter_822, parameter_823, parameter_824, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_87 = paddle._C_ops.add(batch_norm__984, relu_149)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_151 = paddle._C_ops.relu(add_87)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_165 = paddle._C_ops.conv2d(relu_151, parameter_825, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__990, batch_norm__991, batch_norm__992, batch_norm__993, batch_norm__994, batch_norm__995 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_165, parameter_826, parameter_827, parameter_828, parameter_829, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_152 = paddle._C_ops.relu(batch_norm__990)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_166 = paddle._C_ops.conv2d(relu_152, parameter_830, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__996, batch_norm__997, batch_norm__998, batch_norm__999, batch_norm__1000, batch_norm__1001 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_166, parameter_831, parameter_832, parameter_833, parameter_834, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_88 = paddle._C_ops.add(batch_norm__996, relu_151)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_153 = paddle._C_ops.relu(add_88)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_167 = paddle._C_ops.conv2d(relu_153, parameter_835, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__1002, batch_norm__1003, batch_norm__1004, batch_norm__1005, batch_norm__1006, batch_norm__1007 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_167, parameter_836, parameter_837, parameter_838, parameter_839, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_154 = paddle._C_ops.relu(batch_norm__1002)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_168 = paddle._C_ops.conv2d(relu_154, parameter_840, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__1008, batch_norm__1009, batch_norm__1010, batch_norm__1011, batch_norm__1012, batch_norm__1013 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_168, parameter_841, parameter_842, parameter_843, parameter_844, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_89 = paddle._C_ops.add(batch_norm__1008, relu_153)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_155 = paddle._C_ops.relu(add_89)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_169 = paddle._C_ops.conv2d(relu_144, parameter_845, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__1014, batch_norm__1015, batch_norm__1016, batch_norm__1017, batch_norm__1018, batch_norm__1019 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_169, parameter_846, parameter_847, parameter_848, parameter_849, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_156 = paddle._C_ops.relu(batch_norm__1014)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_170 = paddle._C_ops.conv2d(relu_156, parameter_850, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__1020, batch_norm__1021, batch_norm__1022, batch_norm__1023, batch_norm__1024, batch_norm__1025 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_170, parameter_851, parameter_852, parameter_853, parameter_854, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_90 = paddle._C_ops.add(batch_norm__1020, relu_144)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_157 = paddle._C_ops.relu(add_90)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_171 = paddle._C_ops.conv2d(relu_157, parameter_855, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__1026, batch_norm__1027, batch_norm__1028, batch_norm__1029, batch_norm__1030, batch_norm__1031 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_171, parameter_856, parameter_857, parameter_858, parameter_859, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_158 = paddle._C_ops.relu(batch_norm__1026)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_172 = paddle._C_ops.conv2d(relu_158, parameter_860, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__1032, batch_norm__1033, batch_norm__1034, batch_norm__1035, batch_norm__1036, batch_norm__1037 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_172, parameter_861, parameter_862, parameter_863, parameter_864, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_91 = paddle._C_ops.add(batch_norm__1032, relu_157)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_159 = paddle._C_ops.relu(add_91)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_173 = paddle._C_ops.conv2d(relu_159, parameter_865, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__1038, batch_norm__1039, batch_norm__1040, batch_norm__1041, batch_norm__1042, batch_norm__1043 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_173, parameter_866, parameter_867, parameter_868, parameter_869, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_160 = paddle._C_ops.relu(batch_norm__1038)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_174 = paddle._C_ops.conv2d(relu_160, parameter_870, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__1044, batch_norm__1045, batch_norm__1046, batch_norm__1047, batch_norm__1048, batch_norm__1049 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_174, parameter_871, parameter_872, parameter_873, parameter_874, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_92 = paddle._C_ops.add(batch_norm__1044, relu_159)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_161 = paddle._C_ops.relu(add_92)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_175 = paddle._C_ops.conv2d(relu_161, parameter_875, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__1050, batch_norm__1051, batch_norm__1052, batch_norm__1053, batch_norm__1054, batch_norm__1055 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_175, parameter_876, parameter_877, parameter_878, parameter_879, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_162 = paddle._C_ops.relu(batch_norm__1050)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_176 = paddle._C_ops.conv2d(relu_162, parameter_880, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__1056, batch_norm__1057, batch_norm__1058, batch_norm__1059, batch_norm__1060, batch_norm__1061 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_176, parameter_881, parameter_882, parameter_883, parameter_884, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_93 = paddle._C_ops.add(batch_norm__1056, relu_161)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_163 = paddle._C_ops.relu(add_93)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_177 = paddle._C_ops.conv2d(relu_146, parameter_885, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__1062, batch_norm__1063, batch_norm__1064, batch_norm__1065, batch_norm__1066, batch_norm__1067 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_177, parameter_886, parameter_887, parameter_888, parameter_889, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_164 = paddle._C_ops.relu(batch_norm__1062)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_178 = paddle._C_ops.conv2d(relu_164, parameter_890, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__1068, batch_norm__1069, batch_norm__1070, batch_norm__1071, batch_norm__1072, batch_norm__1073 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_178, parameter_891, parameter_892, parameter_893, parameter_894, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_94 = paddle._C_ops.add(batch_norm__1068, relu_146)

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_165 = paddle._C_ops.relu(add_94)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_179 = paddle._C_ops.conv2d(relu_165, parameter_895, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__1074, batch_norm__1075, batch_norm__1076, batch_norm__1077, batch_norm__1078, batch_norm__1079 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_179, parameter_896, parameter_897, parameter_898, parameter_899, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_166 = paddle._C_ops.relu(batch_norm__1074)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_180 = paddle._C_ops.conv2d(relu_166, parameter_900, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__1080, batch_norm__1081, batch_norm__1082, batch_norm__1083, batch_norm__1084, batch_norm__1085 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_180, parameter_901, parameter_902, parameter_903, parameter_904, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_95 = paddle._C_ops.add(batch_norm__1080, relu_165)

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_167 = paddle._C_ops.relu(add_95)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_181 = paddle._C_ops.conv2d(relu_167, parameter_905, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__1086, batch_norm__1087, batch_norm__1088, batch_norm__1089, batch_norm__1090, batch_norm__1091 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_181, parameter_906, parameter_907, parameter_908, parameter_909, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_168 = paddle._C_ops.relu(batch_norm__1086)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_182 = paddle._C_ops.conv2d(relu_168, parameter_910, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__1092, batch_norm__1093, batch_norm__1094, batch_norm__1095, batch_norm__1096, batch_norm__1097 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_182, parameter_911, parameter_912, parameter_913, parameter_914, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_96 = paddle._C_ops.add(batch_norm__1092, relu_167)

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_169 = paddle._C_ops.relu(add_96)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_183 = paddle._C_ops.conv2d(relu_169, parameter_915, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__1098, batch_norm__1099, batch_norm__1100, batch_norm__1101, batch_norm__1102, batch_norm__1103 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_183, parameter_916, parameter_917, parameter_918, parameter_919, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_170 = paddle._C_ops.relu(batch_norm__1098)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_184 = paddle._C_ops.conv2d(relu_170, parameter_920, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__1104, batch_norm__1105, batch_norm__1106, batch_norm__1107, batch_norm__1108, batch_norm__1109 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_184, parameter_921, parameter_922, parameter_923, parameter_924, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_97 = paddle._C_ops.add(batch_norm__1104, relu_169)

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_171 = paddle._C_ops.relu(add_97)

        # pd_op.conv2d: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, 144x144x3x3xf32)
        conv2d_185 = paddle._C_ops.conv2d(relu_147, parameter_925, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x-1x-1xf32, 144xf32, 144xf32, xf32, xf32, None) <- (-1x144x-1x-1xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__1110, batch_norm__1111, batch_norm__1112, batch_norm__1113, batch_norm__1114, batch_norm__1115 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_185, parameter_926, parameter_927, parameter_928, parameter_929, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32)
        relu_172 = paddle._C_ops.relu(batch_norm__1110)

        # pd_op.conv2d: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, 144x144x3x3xf32)
        conv2d_186 = paddle._C_ops.conv2d(relu_172, parameter_930, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x-1x-1xf32, 144xf32, 144xf32, xf32, xf32, None) <- (-1x144x-1x-1xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__1116, batch_norm__1117, batch_norm__1118, batch_norm__1119, batch_norm__1120, batch_norm__1121 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_186, parameter_931, parameter_932, parameter_933, parameter_934, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, -1x144x-1x-1xf32)
        add_98 = paddle._C_ops.add(batch_norm__1116, relu_147)

        # pd_op.relu: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32)
        relu_173 = paddle._C_ops.relu(add_98)

        # pd_op.conv2d: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, 144x144x3x3xf32)
        conv2d_187 = paddle._C_ops.conv2d(relu_173, parameter_935, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x-1x-1xf32, 144xf32, 144xf32, xf32, xf32, None) <- (-1x144x-1x-1xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__1122, batch_norm__1123, batch_norm__1124, batch_norm__1125, batch_norm__1126, batch_norm__1127 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_187, parameter_936, parameter_937, parameter_938, parameter_939, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32)
        relu_174 = paddle._C_ops.relu(batch_norm__1122)

        # pd_op.conv2d: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, 144x144x3x3xf32)
        conv2d_188 = paddle._C_ops.conv2d(relu_174, parameter_940, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x-1x-1xf32, 144xf32, 144xf32, xf32, xf32, None) <- (-1x144x-1x-1xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__1128, batch_norm__1129, batch_norm__1130, batch_norm__1131, batch_norm__1132, batch_norm__1133 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_188, parameter_941, parameter_942, parameter_943, parameter_944, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, -1x144x-1x-1xf32)
        add_99 = paddle._C_ops.add(batch_norm__1128, relu_173)

        # pd_op.relu: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32)
        relu_175 = paddle._C_ops.relu(add_99)

        # pd_op.conv2d: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, 144x144x3x3xf32)
        conv2d_189 = paddle._C_ops.conv2d(relu_175, parameter_945, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x-1x-1xf32, 144xf32, 144xf32, xf32, xf32, None) <- (-1x144x-1x-1xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__1134, batch_norm__1135, batch_norm__1136, batch_norm__1137, batch_norm__1138, batch_norm__1139 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_189, parameter_946, parameter_947, parameter_948, parameter_949, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32)
        relu_176 = paddle._C_ops.relu(batch_norm__1134)

        # pd_op.conv2d: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, 144x144x3x3xf32)
        conv2d_190 = paddle._C_ops.conv2d(relu_176, parameter_950, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x-1x-1xf32, 144xf32, 144xf32, xf32, xf32, None) <- (-1x144x-1x-1xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__1140, batch_norm__1141, batch_norm__1142, batch_norm__1143, batch_norm__1144, batch_norm__1145 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_190, parameter_951, parameter_952, parameter_953, parameter_954, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, -1x144x-1x-1xf32)
        add_100 = paddle._C_ops.add(batch_norm__1140, relu_175)

        # pd_op.relu: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32)
        relu_177 = paddle._C_ops.relu(add_100)

        # pd_op.conv2d: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, 144x144x3x3xf32)
        conv2d_191 = paddle._C_ops.conv2d(relu_177, parameter_955, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x-1x-1xf32, 144xf32, 144xf32, xf32, xf32, None) <- (-1x144x-1x-1xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__1146, batch_norm__1147, batch_norm__1148, batch_norm__1149, batch_norm__1150, batch_norm__1151 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_191, parameter_956, parameter_957, parameter_958, parameter_959, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32)
        relu_178 = paddle._C_ops.relu(batch_norm__1146)

        # pd_op.conv2d: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, 144x144x3x3xf32)
        conv2d_192 = paddle._C_ops.conv2d(relu_178, parameter_960, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x-1x-1xf32, 144xf32, 144xf32, xf32, xf32, None) <- (-1x144x-1x-1xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__1152, batch_norm__1153, batch_norm__1154, batch_norm__1155, batch_norm__1156, batch_norm__1157 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_192, parameter_961, parameter_962, parameter_963, parameter_964, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, -1x144x-1x-1xf32)
        add_101 = paddle._C_ops.add(batch_norm__1152, relu_177)

        # pd_op.relu: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32)
        relu_179 = paddle._C_ops.relu(add_101)

        # pd_op.shape: (4xi32) <- (-1x18x-1x-1xf32)
        shape_9 = paddle._C_ops.shape(relu_155)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_18 = [-2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_19 = [2147483647]

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(shape_9, [0], full_int_array_18, full_int_array_19, [1], [])

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x36x-1x-1xf32, 18x36x1x1xf32)
        conv2d_193 = paddle._C_ops.conv2d(relu_163, parameter_965, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__1158, batch_norm__1159, batch_norm__1160, batch_norm__1161, batch_norm__1162, batch_norm__1163 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_193, parameter_966, parameter_967, parameter_968, parameter_969, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.cast: (2xi32) <- (2xi32)
        cast_4 = paddle._C_ops.cast(slice_9, paddle.int32)

        # pd_op.bilinear_interp: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_13 = paddle._C_ops.bilinear_interp(batch_norm__1158, cast_4, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_102 = paddle._C_ops.add(relu_155, bilinear_interp_13)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x72x-1x-1xf32, 18x72x1x1xf32)
        conv2d_194 = paddle._C_ops.conv2d(relu_171, parameter_970, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__1164, batch_norm__1165, batch_norm__1166, batch_norm__1167, batch_norm__1168, batch_norm__1169 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_194, parameter_971, parameter_972, parameter_973, parameter_974, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.cast: (2xi32) <- (2xi32)
        cast_5 = paddle._C_ops.cast(slice_9, paddle.int32)

        # pd_op.bilinear_interp: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_14 = paddle._C_ops.bilinear_interp(batch_norm__1164, cast_5, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_103 = paddle._C_ops.add(add_102, bilinear_interp_14)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x144x-1x-1xf32, 18x144x1x1xf32)
        conv2d_195 = paddle._C_ops.conv2d(relu_179, parameter_975, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__1170, batch_norm__1171, batch_norm__1172, batch_norm__1173, batch_norm__1174, batch_norm__1175 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_195, parameter_976, parameter_977, parameter_978, parameter_979, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.cast_: (2xi32) <- (2xi32)
        cast__9 = paddle._C_ops.cast_(slice_9, paddle.int32)

        # pd_op.bilinear_interp: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_15 = paddle._C_ops.bilinear_interp(batch_norm__1170, cast__9, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_104 = paddle._C_ops.add(add_103, bilinear_interp_15)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_180 = paddle._C_ops.relu(add_104)

        # pd_op.shape: (4xi32) <- (-1x36x-1x-1xf32)
        shape_10 = paddle._C_ops.shape(relu_163)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_20 = [-2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_21 = [2147483647]

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(shape_10, [0], full_int_array_20, full_int_array_21, [1], [])

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x18x-1x-1xf32, 36x18x3x3xf32)
        conv2d_196 = paddle._C_ops.conv2d(relu_155, parameter_980, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__1176, batch_norm__1177, batch_norm__1178, batch_norm__1179, batch_norm__1180, batch_norm__1181 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_196, parameter_981, parameter_982, parameter_983, parameter_984, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_105 = paddle._C_ops.add(relu_163, batch_norm__1176)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x72x-1x-1xf32, 36x72x1x1xf32)
        conv2d_197 = paddle._C_ops.conv2d(relu_171, parameter_985, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__1182, batch_norm__1183, batch_norm__1184, batch_norm__1185, batch_norm__1186, batch_norm__1187 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_197, parameter_986, parameter_987, parameter_988, parameter_989, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.cast: (2xi32) <- (2xi32)
        cast_6 = paddle._C_ops.cast(slice_10, paddle.int32)

        # pd_op.bilinear_interp: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_16 = paddle._C_ops.bilinear_interp(batch_norm__1182, cast_6, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_106 = paddle._C_ops.add(add_105, bilinear_interp_16)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x144x-1x-1xf32, 36x144x1x1xf32)
        conv2d_198 = paddle._C_ops.conv2d(relu_179, parameter_990, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__1188, batch_norm__1189, batch_norm__1190, batch_norm__1191, batch_norm__1192, batch_norm__1193 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_198, parameter_991, parameter_992, parameter_993, parameter_994, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.cast_: (2xi32) <- (2xi32)
        cast__10 = paddle._C_ops.cast_(slice_10, paddle.int32)

        # pd_op.bilinear_interp: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_17 = paddle._C_ops.bilinear_interp(batch_norm__1188, cast__10, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_107 = paddle._C_ops.add(add_106, bilinear_interp_17)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_181 = paddle._C_ops.relu(add_107)

        # pd_op.shape: (4xi32) <- (-1x72x-1x-1xf32)
        shape_11 = paddle._C_ops.shape(relu_171)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_22 = [-2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_23 = [2147483647]

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(shape_11, [0], full_int_array_22, full_int_array_23, [1], [])

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_199 = paddle._C_ops.conv2d(relu_155, parameter_995, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__1194, batch_norm__1195, batch_norm__1196, batch_norm__1197, batch_norm__1198, batch_norm__1199 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_199, parameter_996, parameter_997, parameter_998, parameter_999, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_182 = paddle._C_ops.relu(batch_norm__1194)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x18x-1x-1xf32, 72x18x3x3xf32)
        conv2d_200 = paddle._C_ops.conv2d(relu_182, parameter_1000, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__1200, batch_norm__1201, batch_norm__1202, batch_norm__1203, batch_norm__1204, batch_norm__1205 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_200, parameter_1001, parameter_1002, parameter_1003, parameter_1004, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_108 = paddle._C_ops.add(relu_171, batch_norm__1200)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x36x-1x-1xf32, 72x36x3x3xf32)
        conv2d_201 = paddle._C_ops.conv2d(relu_163, parameter_1005, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__1206, batch_norm__1207, batch_norm__1208, batch_norm__1209, batch_norm__1210, batch_norm__1211 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_201, parameter_1006, parameter_1007, parameter_1008, parameter_1009, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_109 = paddle._C_ops.add(add_108, batch_norm__1206)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x144x-1x-1xf32, 72x144x1x1xf32)
        conv2d_202 = paddle._C_ops.conv2d(relu_179, parameter_1010, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__1212, batch_norm__1213, batch_norm__1214, batch_norm__1215, batch_norm__1216, batch_norm__1217 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_202, parameter_1011, parameter_1012, parameter_1013, parameter_1014, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.cast_: (2xi32) <- (2xi32)
        cast__11 = paddle._C_ops.cast_(slice_11, paddle.int32)

        # pd_op.bilinear_interp: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_18 = paddle._C_ops.bilinear_interp(batch_norm__1212, cast__11, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_110 = paddle._C_ops.add(add_109, bilinear_interp_18)

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_183 = paddle._C_ops.relu(add_110)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_203 = paddle._C_ops.conv2d(relu_155, parameter_1015, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__1218, batch_norm__1219, batch_norm__1220, batch_norm__1221, batch_norm__1222, batch_norm__1223 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_203, parameter_1016, parameter_1017, parameter_1018, parameter_1019, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_184 = paddle._C_ops.relu(batch_norm__1218)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_204 = paddle._C_ops.conv2d(relu_184, parameter_1020, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__1224, batch_norm__1225, batch_norm__1226, batch_norm__1227, batch_norm__1228, batch_norm__1229 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_204, parameter_1021, parameter_1022, parameter_1023, parameter_1024, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_185 = paddle._C_ops.relu(batch_norm__1224)

        # pd_op.conv2d: (-1x144x-1x-1xf32) <- (-1x18x-1x-1xf32, 144x18x3x3xf32)
        conv2d_205 = paddle._C_ops.conv2d(relu_185, parameter_1025, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x-1x-1xf32, 144xf32, 144xf32, xf32, xf32, None) <- (-1x144x-1x-1xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__1230, batch_norm__1231, batch_norm__1232, batch_norm__1233, batch_norm__1234, batch_norm__1235 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_205, parameter_1026, parameter_1027, parameter_1028, parameter_1029, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, -1x144x-1x-1xf32)
        add_111 = paddle._C_ops.add(relu_179, batch_norm__1230)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_206 = paddle._C_ops.conv2d(relu_163, parameter_1030, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__1236, batch_norm__1237, batch_norm__1238, batch_norm__1239, batch_norm__1240, batch_norm__1241 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_206, parameter_1031, parameter_1032, parameter_1033, parameter_1034, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_186 = paddle._C_ops.relu(batch_norm__1236)

        # pd_op.conv2d: (-1x144x-1x-1xf32) <- (-1x36x-1x-1xf32, 144x36x3x3xf32)
        conv2d_207 = paddle._C_ops.conv2d(relu_186, parameter_1035, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x-1x-1xf32, 144xf32, 144xf32, xf32, xf32, None) <- (-1x144x-1x-1xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__1242, batch_norm__1243, batch_norm__1244, batch_norm__1245, batch_norm__1246, batch_norm__1247 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_207, parameter_1036, parameter_1037, parameter_1038, parameter_1039, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, -1x144x-1x-1xf32)
        add_112 = paddle._C_ops.add(add_111, batch_norm__1242)

        # pd_op.conv2d: (-1x144x-1x-1xf32) <- (-1x72x-1x-1xf32, 144x72x3x3xf32)
        conv2d_208 = paddle._C_ops.conv2d(relu_171, parameter_1040, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x-1x-1xf32, 144xf32, 144xf32, xf32, xf32, None) <- (-1x144x-1x-1xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__1248, batch_norm__1249, batch_norm__1250, batch_norm__1251, batch_norm__1252, batch_norm__1253 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_208, parameter_1041, parameter_1042, parameter_1043, parameter_1044, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, -1x144x-1x-1xf32)
        add_113 = paddle._C_ops.add(add_112, batch_norm__1248)

        # pd_op.relu: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32)
        relu_187 = paddle._C_ops.relu(add_113)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_209 = paddle._C_ops.conv2d(relu_180, parameter_1045, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__1254, batch_norm__1255, batch_norm__1256, batch_norm__1257, batch_norm__1258, batch_norm__1259 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_209, parameter_1046, parameter_1047, parameter_1048, parameter_1049, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_188 = paddle._C_ops.relu(batch_norm__1254)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_210 = paddle._C_ops.conv2d(relu_188, parameter_1050, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__1260, batch_norm__1261, batch_norm__1262, batch_norm__1263, batch_norm__1264, batch_norm__1265 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_210, parameter_1051, parameter_1052, parameter_1053, parameter_1054, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_114 = paddle._C_ops.add(batch_norm__1260, relu_180)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_189 = paddle._C_ops.relu(add_114)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_211 = paddle._C_ops.conv2d(relu_189, parameter_1055, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__1266, batch_norm__1267, batch_norm__1268, batch_norm__1269, batch_norm__1270, batch_norm__1271 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_211, parameter_1056, parameter_1057, parameter_1058, parameter_1059, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_190 = paddle._C_ops.relu(batch_norm__1266)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_212 = paddle._C_ops.conv2d(relu_190, parameter_1060, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__1272, batch_norm__1273, batch_norm__1274, batch_norm__1275, batch_norm__1276, batch_norm__1277 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_212, parameter_1061, parameter_1062, parameter_1063, parameter_1064, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_115 = paddle._C_ops.add(batch_norm__1272, relu_189)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_191 = paddle._C_ops.relu(add_115)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_213 = paddle._C_ops.conv2d(relu_191, parameter_1065, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__1278, batch_norm__1279, batch_norm__1280, batch_norm__1281, batch_norm__1282, batch_norm__1283 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_213, parameter_1066, parameter_1067, parameter_1068, parameter_1069, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_192 = paddle._C_ops.relu(batch_norm__1278)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_214 = paddle._C_ops.conv2d(relu_192, parameter_1070, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__1284, batch_norm__1285, batch_norm__1286, batch_norm__1287, batch_norm__1288, batch_norm__1289 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_214, parameter_1071, parameter_1072, parameter_1073, parameter_1074, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_116 = paddle._C_ops.add(batch_norm__1284, relu_191)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_193 = paddle._C_ops.relu(add_116)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_215 = paddle._C_ops.conv2d(relu_193, parameter_1075, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__1290, batch_norm__1291, batch_norm__1292, batch_norm__1293, batch_norm__1294, batch_norm__1295 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_215, parameter_1076, parameter_1077, parameter_1078, parameter_1079, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_194 = paddle._C_ops.relu(batch_norm__1290)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_216 = paddle._C_ops.conv2d(relu_194, parameter_1080, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__1296, batch_norm__1297, batch_norm__1298, batch_norm__1299, batch_norm__1300, batch_norm__1301 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_216, parameter_1081, parameter_1082, parameter_1083, parameter_1084, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_117 = paddle._C_ops.add(batch_norm__1296, relu_193)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_195 = paddle._C_ops.relu(add_117)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_217 = paddle._C_ops.conv2d(relu_181, parameter_1085, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__1302, batch_norm__1303, batch_norm__1304, batch_norm__1305, batch_norm__1306, batch_norm__1307 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_217, parameter_1086, parameter_1087, parameter_1088, parameter_1089, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_196 = paddle._C_ops.relu(batch_norm__1302)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_218 = paddle._C_ops.conv2d(relu_196, parameter_1090, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__1308, batch_norm__1309, batch_norm__1310, batch_norm__1311, batch_norm__1312, batch_norm__1313 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_218, parameter_1091, parameter_1092, parameter_1093, parameter_1094, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_118 = paddle._C_ops.add(batch_norm__1308, relu_181)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_197 = paddle._C_ops.relu(add_118)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_219 = paddle._C_ops.conv2d(relu_197, parameter_1095, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__1314, batch_norm__1315, batch_norm__1316, batch_norm__1317, batch_norm__1318, batch_norm__1319 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_219, parameter_1096, parameter_1097, parameter_1098, parameter_1099, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_198 = paddle._C_ops.relu(batch_norm__1314)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_220 = paddle._C_ops.conv2d(relu_198, parameter_1100, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__1320, batch_norm__1321, batch_norm__1322, batch_norm__1323, batch_norm__1324, batch_norm__1325 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_220, parameter_1101, parameter_1102, parameter_1103, parameter_1104, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_119 = paddle._C_ops.add(batch_norm__1320, relu_197)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_199 = paddle._C_ops.relu(add_119)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_221 = paddle._C_ops.conv2d(relu_199, parameter_1105, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__1326, batch_norm__1327, batch_norm__1328, batch_norm__1329, batch_norm__1330, batch_norm__1331 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_221, parameter_1106, parameter_1107, parameter_1108, parameter_1109, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_200 = paddle._C_ops.relu(batch_norm__1326)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_222 = paddle._C_ops.conv2d(relu_200, parameter_1110, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__1332, batch_norm__1333, batch_norm__1334, batch_norm__1335, batch_norm__1336, batch_norm__1337 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_222, parameter_1111, parameter_1112, parameter_1113, parameter_1114, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_120 = paddle._C_ops.add(batch_norm__1332, relu_199)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_201 = paddle._C_ops.relu(add_120)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_223 = paddle._C_ops.conv2d(relu_201, parameter_1115, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__1338, batch_norm__1339, batch_norm__1340, batch_norm__1341, batch_norm__1342, batch_norm__1343 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_223, parameter_1116, parameter_1117, parameter_1118, parameter_1119, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_202 = paddle._C_ops.relu(batch_norm__1338)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_224 = paddle._C_ops.conv2d(relu_202, parameter_1120, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__1344, batch_norm__1345, batch_norm__1346, batch_norm__1347, batch_norm__1348, batch_norm__1349 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_224, parameter_1121, parameter_1122, parameter_1123, parameter_1124, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_121 = paddle._C_ops.add(batch_norm__1344, relu_201)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_203 = paddle._C_ops.relu(add_121)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_225 = paddle._C_ops.conv2d(relu_183, parameter_1125, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__1350, batch_norm__1351, batch_norm__1352, batch_norm__1353, batch_norm__1354, batch_norm__1355 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_225, parameter_1126, parameter_1127, parameter_1128, parameter_1129, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_204 = paddle._C_ops.relu(batch_norm__1350)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_226 = paddle._C_ops.conv2d(relu_204, parameter_1130, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__1356, batch_norm__1357, batch_norm__1358, batch_norm__1359, batch_norm__1360, batch_norm__1361 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_226, parameter_1131, parameter_1132, parameter_1133, parameter_1134, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_122 = paddle._C_ops.add(batch_norm__1356, relu_183)

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_205 = paddle._C_ops.relu(add_122)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_227 = paddle._C_ops.conv2d(relu_205, parameter_1135, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__1362, batch_norm__1363, batch_norm__1364, batch_norm__1365, batch_norm__1366, batch_norm__1367 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_227, parameter_1136, parameter_1137, parameter_1138, parameter_1139, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_206 = paddle._C_ops.relu(batch_norm__1362)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_228 = paddle._C_ops.conv2d(relu_206, parameter_1140, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__1368, batch_norm__1369, batch_norm__1370, batch_norm__1371, batch_norm__1372, batch_norm__1373 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_228, parameter_1141, parameter_1142, parameter_1143, parameter_1144, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_123 = paddle._C_ops.add(batch_norm__1368, relu_205)

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_207 = paddle._C_ops.relu(add_123)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_229 = paddle._C_ops.conv2d(relu_207, parameter_1145, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__1374, batch_norm__1375, batch_norm__1376, batch_norm__1377, batch_norm__1378, batch_norm__1379 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_229, parameter_1146, parameter_1147, parameter_1148, parameter_1149, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_208 = paddle._C_ops.relu(batch_norm__1374)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_230 = paddle._C_ops.conv2d(relu_208, parameter_1150, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__1380, batch_norm__1381, batch_norm__1382, batch_norm__1383, batch_norm__1384, batch_norm__1385 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_230, parameter_1151, parameter_1152, parameter_1153, parameter_1154, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_124 = paddle._C_ops.add(batch_norm__1380, relu_207)

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_209 = paddle._C_ops.relu(add_124)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_231 = paddle._C_ops.conv2d(relu_209, parameter_1155, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__1386, batch_norm__1387, batch_norm__1388, batch_norm__1389, batch_norm__1390, batch_norm__1391 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_231, parameter_1156, parameter_1157, parameter_1158, parameter_1159, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_210 = paddle._C_ops.relu(batch_norm__1386)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_232 = paddle._C_ops.conv2d(relu_210, parameter_1160, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__1392, batch_norm__1393, batch_norm__1394, batch_norm__1395, batch_norm__1396, batch_norm__1397 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_232, parameter_1161, parameter_1162, parameter_1163, parameter_1164, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_125 = paddle._C_ops.add(batch_norm__1392, relu_209)

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_211 = paddle._C_ops.relu(add_125)

        # pd_op.conv2d: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, 144x144x3x3xf32)
        conv2d_233 = paddle._C_ops.conv2d(relu_187, parameter_1165, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x-1x-1xf32, 144xf32, 144xf32, xf32, xf32, None) <- (-1x144x-1x-1xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__1398, batch_norm__1399, batch_norm__1400, batch_norm__1401, batch_norm__1402, batch_norm__1403 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_233, parameter_1166, parameter_1167, parameter_1168, parameter_1169, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32)
        relu_212 = paddle._C_ops.relu(batch_norm__1398)

        # pd_op.conv2d: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, 144x144x3x3xf32)
        conv2d_234 = paddle._C_ops.conv2d(relu_212, parameter_1170, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x-1x-1xf32, 144xf32, 144xf32, xf32, xf32, None) <- (-1x144x-1x-1xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__1404, batch_norm__1405, batch_norm__1406, batch_norm__1407, batch_norm__1408, batch_norm__1409 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_234, parameter_1171, parameter_1172, parameter_1173, parameter_1174, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, -1x144x-1x-1xf32)
        add_126 = paddle._C_ops.add(batch_norm__1404, relu_187)

        # pd_op.relu: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32)
        relu_213 = paddle._C_ops.relu(add_126)

        # pd_op.conv2d: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, 144x144x3x3xf32)
        conv2d_235 = paddle._C_ops.conv2d(relu_213, parameter_1175, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x-1x-1xf32, 144xf32, 144xf32, xf32, xf32, None) <- (-1x144x-1x-1xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__1410, batch_norm__1411, batch_norm__1412, batch_norm__1413, batch_norm__1414, batch_norm__1415 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_235, parameter_1176, parameter_1177, parameter_1178, parameter_1179, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32)
        relu_214 = paddle._C_ops.relu(batch_norm__1410)

        # pd_op.conv2d: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, 144x144x3x3xf32)
        conv2d_236 = paddle._C_ops.conv2d(relu_214, parameter_1180, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x-1x-1xf32, 144xf32, 144xf32, xf32, xf32, None) <- (-1x144x-1x-1xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__1416, batch_norm__1417, batch_norm__1418, batch_norm__1419, batch_norm__1420, batch_norm__1421 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_236, parameter_1181, parameter_1182, parameter_1183, parameter_1184, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, -1x144x-1x-1xf32)
        add_127 = paddle._C_ops.add(batch_norm__1416, relu_213)

        # pd_op.relu: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32)
        relu_215 = paddle._C_ops.relu(add_127)

        # pd_op.conv2d: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, 144x144x3x3xf32)
        conv2d_237 = paddle._C_ops.conv2d(relu_215, parameter_1185, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x-1x-1xf32, 144xf32, 144xf32, xf32, xf32, None) <- (-1x144x-1x-1xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__1422, batch_norm__1423, batch_norm__1424, batch_norm__1425, batch_norm__1426, batch_norm__1427 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_237, parameter_1186, parameter_1187, parameter_1188, parameter_1189, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32)
        relu_216 = paddle._C_ops.relu(batch_norm__1422)

        # pd_op.conv2d: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, 144x144x3x3xf32)
        conv2d_238 = paddle._C_ops.conv2d(relu_216, parameter_1190, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x-1x-1xf32, 144xf32, 144xf32, xf32, xf32, None) <- (-1x144x-1x-1xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__1428, batch_norm__1429, batch_norm__1430, batch_norm__1431, batch_norm__1432, batch_norm__1433 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_238, parameter_1191, parameter_1192, parameter_1193, parameter_1194, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, -1x144x-1x-1xf32)
        add_128 = paddle._C_ops.add(batch_norm__1428, relu_215)

        # pd_op.relu: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32)
        relu_217 = paddle._C_ops.relu(add_128)

        # pd_op.conv2d: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, 144x144x3x3xf32)
        conv2d_239 = paddle._C_ops.conv2d(relu_217, parameter_1195, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x-1x-1xf32, 144xf32, 144xf32, xf32, xf32, None) <- (-1x144x-1x-1xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__1434, batch_norm__1435, batch_norm__1436, batch_norm__1437, batch_norm__1438, batch_norm__1439 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_239, parameter_1196, parameter_1197, parameter_1198, parameter_1199, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32)
        relu_218 = paddle._C_ops.relu(batch_norm__1434)

        # pd_op.conv2d: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, 144x144x3x3xf32)
        conv2d_240 = paddle._C_ops.conv2d(relu_218, parameter_1200, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x-1x-1xf32, 144xf32, 144xf32, xf32, xf32, None) <- (-1x144x-1x-1xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__1440, batch_norm__1441, batch_norm__1442, batch_norm__1443, batch_norm__1444, batch_norm__1445 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_240, parameter_1201, parameter_1202, parameter_1203, parameter_1204, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, -1x144x-1x-1xf32)
        add_129 = paddle._C_ops.add(batch_norm__1440, relu_217)

        # pd_op.relu: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32)
        relu_219 = paddle._C_ops.relu(add_129)

        # pd_op.shape: (4xi32) <- (-1x18x-1x-1xf32)
        shape_12 = paddle._C_ops.shape(relu_195)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_24 = [-2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_25 = [2147483647]

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(shape_12, [0], full_int_array_24, full_int_array_25, [1], [])

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x36x-1x-1xf32, 18x36x1x1xf32)
        conv2d_241 = paddle._C_ops.conv2d(relu_203, parameter_1205, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__1446, batch_norm__1447, batch_norm__1448, batch_norm__1449, batch_norm__1450, batch_norm__1451 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_241, parameter_1206, parameter_1207, parameter_1208, parameter_1209, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.cast: (2xi32) <- (2xi32)
        cast_7 = paddle._C_ops.cast(slice_12, paddle.int32)

        # pd_op.bilinear_interp: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_19 = paddle._C_ops.bilinear_interp(batch_norm__1446, cast_7, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_130 = paddle._C_ops.add(relu_195, bilinear_interp_19)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x72x-1x-1xf32, 18x72x1x1xf32)
        conv2d_242 = paddle._C_ops.conv2d(relu_211, parameter_1210, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__1452, batch_norm__1453, batch_norm__1454, batch_norm__1455, batch_norm__1456, batch_norm__1457 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_242, parameter_1211, parameter_1212, parameter_1213, parameter_1214, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.cast: (2xi32) <- (2xi32)
        cast_8 = paddle._C_ops.cast(slice_12, paddle.int32)

        # pd_op.bilinear_interp: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_20 = paddle._C_ops.bilinear_interp(batch_norm__1452, cast_8, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_131 = paddle._C_ops.add(add_130, bilinear_interp_20)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x144x-1x-1xf32, 18x144x1x1xf32)
        conv2d_243 = paddle._C_ops.conv2d(relu_219, parameter_1215, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__1458, batch_norm__1459, batch_norm__1460, batch_norm__1461, batch_norm__1462, batch_norm__1463 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_243, parameter_1216, parameter_1217, parameter_1218, parameter_1219, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.cast_: (2xi32) <- (2xi32)
        cast__12 = paddle._C_ops.cast_(slice_12, paddle.int32)

        # pd_op.bilinear_interp: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_21 = paddle._C_ops.bilinear_interp(batch_norm__1458, cast__12, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_132 = paddle._C_ops.add(add_131, bilinear_interp_21)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_220 = paddle._C_ops.relu(add_132)

        # pd_op.shape: (4xi32) <- (-1x36x-1x-1xf32)
        shape_13 = paddle._C_ops.shape(relu_203)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_26 = [-2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_27 = [2147483647]

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(shape_13, [0], full_int_array_26, full_int_array_27, [1], [])

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x18x-1x-1xf32, 36x18x3x3xf32)
        conv2d_244 = paddle._C_ops.conv2d(relu_195, parameter_1220, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__1464, batch_norm__1465, batch_norm__1466, batch_norm__1467, batch_norm__1468, batch_norm__1469 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_244, parameter_1221, parameter_1222, parameter_1223, parameter_1224, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_133 = paddle._C_ops.add(relu_203, batch_norm__1464)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x72x-1x-1xf32, 36x72x1x1xf32)
        conv2d_245 = paddle._C_ops.conv2d(relu_211, parameter_1225, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__1470, batch_norm__1471, batch_norm__1472, batch_norm__1473, batch_norm__1474, batch_norm__1475 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_245, parameter_1226, parameter_1227, parameter_1228, parameter_1229, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.cast: (2xi32) <- (2xi32)
        cast_9 = paddle._C_ops.cast(slice_13, paddle.int32)

        # pd_op.bilinear_interp: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_22 = paddle._C_ops.bilinear_interp(batch_norm__1470, cast_9, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_134 = paddle._C_ops.add(add_133, bilinear_interp_22)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x144x-1x-1xf32, 36x144x1x1xf32)
        conv2d_246 = paddle._C_ops.conv2d(relu_219, parameter_1230, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__1476, batch_norm__1477, batch_norm__1478, batch_norm__1479, batch_norm__1480, batch_norm__1481 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_246, parameter_1231, parameter_1232, parameter_1233, parameter_1234, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.cast_: (2xi32) <- (2xi32)
        cast__13 = paddle._C_ops.cast_(slice_13, paddle.int32)

        # pd_op.bilinear_interp: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_23 = paddle._C_ops.bilinear_interp(batch_norm__1476, cast__13, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_135 = paddle._C_ops.add(add_134, bilinear_interp_23)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_221 = paddle._C_ops.relu(add_135)

        # pd_op.shape: (4xi32) <- (-1x72x-1x-1xf32)
        shape_14 = paddle._C_ops.shape(relu_211)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_28 = [-2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_29 = [2147483647]

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(shape_14, [0], full_int_array_28, full_int_array_29, [1], [])

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_247 = paddle._C_ops.conv2d(relu_195, parameter_1235, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__1482, batch_norm__1483, batch_norm__1484, batch_norm__1485, batch_norm__1486, batch_norm__1487 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_247, parameter_1236, parameter_1237, parameter_1238, parameter_1239, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_222 = paddle._C_ops.relu(batch_norm__1482)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x18x-1x-1xf32, 72x18x3x3xf32)
        conv2d_248 = paddle._C_ops.conv2d(relu_222, parameter_1240, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__1488, batch_norm__1489, batch_norm__1490, batch_norm__1491, batch_norm__1492, batch_norm__1493 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_248, parameter_1241, parameter_1242, parameter_1243, parameter_1244, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_136 = paddle._C_ops.add(relu_211, batch_norm__1488)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x36x-1x-1xf32, 72x36x3x3xf32)
        conv2d_249 = paddle._C_ops.conv2d(relu_203, parameter_1245, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__1494, batch_norm__1495, batch_norm__1496, batch_norm__1497, batch_norm__1498, batch_norm__1499 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_249, parameter_1246, parameter_1247, parameter_1248, parameter_1249, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_137 = paddle._C_ops.add(add_136, batch_norm__1494)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x144x-1x-1xf32, 72x144x1x1xf32)
        conv2d_250 = paddle._C_ops.conv2d(relu_219, parameter_1250, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__1500, batch_norm__1501, batch_norm__1502, batch_norm__1503, batch_norm__1504, batch_norm__1505 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_250, parameter_1251, parameter_1252, parameter_1253, parameter_1254, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.cast_: (2xi32) <- (2xi32)
        cast__14 = paddle._C_ops.cast_(slice_14, paddle.int32)

        # pd_op.bilinear_interp: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_24 = paddle._C_ops.bilinear_interp(batch_norm__1500, cast__14, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_138 = paddle._C_ops.add(add_137, bilinear_interp_24)

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_223 = paddle._C_ops.relu(add_138)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_251 = paddle._C_ops.conv2d(relu_195, parameter_1255, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__1506, batch_norm__1507, batch_norm__1508, batch_norm__1509, batch_norm__1510, batch_norm__1511 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_251, parameter_1256, parameter_1257, parameter_1258, parameter_1259, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_224 = paddle._C_ops.relu(batch_norm__1506)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_252 = paddle._C_ops.conv2d(relu_224, parameter_1260, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__1512, batch_norm__1513, batch_norm__1514, batch_norm__1515, batch_norm__1516, batch_norm__1517 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_252, parameter_1261, parameter_1262, parameter_1263, parameter_1264, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_225 = paddle._C_ops.relu(batch_norm__1512)

        # pd_op.conv2d: (-1x144x-1x-1xf32) <- (-1x18x-1x-1xf32, 144x18x3x3xf32)
        conv2d_253 = paddle._C_ops.conv2d(relu_225, parameter_1265, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x-1x-1xf32, 144xf32, 144xf32, xf32, xf32, None) <- (-1x144x-1x-1xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__1518, batch_norm__1519, batch_norm__1520, batch_norm__1521, batch_norm__1522, batch_norm__1523 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_253, parameter_1266, parameter_1267, parameter_1268, parameter_1269, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, -1x144x-1x-1xf32)
        add_139 = paddle._C_ops.add(relu_219, batch_norm__1518)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_254 = paddle._C_ops.conv2d(relu_203, parameter_1270, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__1524, batch_norm__1525, batch_norm__1526, batch_norm__1527, batch_norm__1528, batch_norm__1529 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_254, parameter_1271, parameter_1272, parameter_1273, parameter_1274, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_226 = paddle._C_ops.relu(batch_norm__1524)

        # pd_op.conv2d: (-1x144x-1x-1xf32) <- (-1x36x-1x-1xf32, 144x36x3x3xf32)
        conv2d_255 = paddle._C_ops.conv2d(relu_226, parameter_1275, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x-1x-1xf32, 144xf32, 144xf32, xf32, xf32, None) <- (-1x144x-1x-1xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__1530, batch_norm__1531, batch_norm__1532, batch_norm__1533, batch_norm__1534, batch_norm__1535 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_255, parameter_1276, parameter_1277, parameter_1278, parameter_1279, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, -1x144x-1x-1xf32)
        add_140 = paddle._C_ops.add(add_139, batch_norm__1530)

        # pd_op.conv2d: (-1x144x-1x-1xf32) <- (-1x72x-1x-1xf32, 144x72x3x3xf32)
        conv2d_256 = paddle._C_ops.conv2d(relu_211, parameter_1280, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x-1x-1xf32, 144xf32, 144xf32, xf32, xf32, None) <- (-1x144x-1x-1xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__1536, batch_norm__1537, batch_norm__1538, batch_norm__1539, batch_norm__1540, batch_norm__1541 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_256, parameter_1281, parameter_1282, parameter_1283, parameter_1284, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, -1x144x-1x-1xf32)
        add_141 = paddle._C_ops.add(add_140, batch_norm__1536)

        # pd_op.relu: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32)
        relu_227 = paddle._C_ops.relu(add_141)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_257 = paddle._C_ops.conv2d(relu_220, parameter_1285, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__1542, batch_norm__1543, batch_norm__1544, batch_norm__1545, batch_norm__1546, batch_norm__1547 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_257, parameter_1286, parameter_1287, parameter_1288, parameter_1289, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_228 = paddle._C_ops.relu(batch_norm__1542)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_258 = paddle._C_ops.conv2d(relu_228, parameter_1290, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__1548, batch_norm__1549, batch_norm__1550, batch_norm__1551, batch_norm__1552, batch_norm__1553 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_258, parameter_1291, parameter_1292, parameter_1293, parameter_1294, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_142 = paddle._C_ops.add(batch_norm__1548, relu_220)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_229 = paddle._C_ops.relu(add_142)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_259 = paddle._C_ops.conv2d(relu_229, parameter_1295, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__1554, batch_norm__1555, batch_norm__1556, batch_norm__1557, batch_norm__1558, batch_norm__1559 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_259, parameter_1296, parameter_1297, parameter_1298, parameter_1299, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_230 = paddle._C_ops.relu(batch_norm__1554)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_260 = paddle._C_ops.conv2d(relu_230, parameter_1300, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__1560, batch_norm__1561, batch_norm__1562, batch_norm__1563, batch_norm__1564, batch_norm__1565 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_260, parameter_1301, parameter_1302, parameter_1303, parameter_1304, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_143 = paddle._C_ops.add(batch_norm__1560, relu_229)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_231 = paddle._C_ops.relu(add_143)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_261 = paddle._C_ops.conv2d(relu_231, parameter_1305, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__1566, batch_norm__1567, batch_norm__1568, batch_norm__1569, batch_norm__1570, batch_norm__1571 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_261, parameter_1306, parameter_1307, parameter_1308, parameter_1309, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_232 = paddle._C_ops.relu(batch_norm__1566)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_262 = paddle._C_ops.conv2d(relu_232, parameter_1310, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__1572, batch_norm__1573, batch_norm__1574, batch_norm__1575, batch_norm__1576, batch_norm__1577 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_262, parameter_1311, parameter_1312, parameter_1313, parameter_1314, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_144 = paddle._C_ops.add(batch_norm__1572, relu_231)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_233 = paddle._C_ops.relu(add_144)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_263 = paddle._C_ops.conv2d(relu_233, parameter_1315, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__1578, batch_norm__1579, batch_norm__1580, batch_norm__1581, batch_norm__1582, batch_norm__1583 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_263, parameter_1316, parameter_1317, parameter_1318, parameter_1319, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_234 = paddle._C_ops.relu(batch_norm__1578)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_264 = paddle._C_ops.conv2d(relu_234, parameter_1320, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__1584, batch_norm__1585, batch_norm__1586, batch_norm__1587, batch_norm__1588, batch_norm__1589 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_264, parameter_1321, parameter_1322, parameter_1323, parameter_1324, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_145 = paddle._C_ops.add(batch_norm__1584, relu_233)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_235 = paddle._C_ops.relu(add_145)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_265 = paddle._C_ops.conv2d(relu_221, parameter_1325, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__1590, batch_norm__1591, batch_norm__1592, batch_norm__1593, batch_norm__1594, batch_norm__1595 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_265, parameter_1326, parameter_1327, parameter_1328, parameter_1329, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_236 = paddle._C_ops.relu(batch_norm__1590)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_266 = paddle._C_ops.conv2d(relu_236, parameter_1330, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__1596, batch_norm__1597, batch_norm__1598, batch_norm__1599, batch_norm__1600, batch_norm__1601 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_266, parameter_1331, parameter_1332, parameter_1333, parameter_1334, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_146 = paddle._C_ops.add(batch_norm__1596, relu_221)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_237 = paddle._C_ops.relu(add_146)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_267 = paddle._C_ops.conv2d(relu_237, parameter_1335, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__1602, batch_norm__1603, batch_norm__1604, batch_norm__1605, batch_norm__1606, batch_norm__1607 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_267, parameter_1336, parameter_1337, parameter_1338, parameter_1339, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_238 = paddle._C_ops.relu(batch_norm__1602)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_268 = paddle._C_ops.conv2d(relu_238, parameter_1340, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__1608, batch_norm__1609, batch_norm__1610, batch_norm__1611, batch_norm__1612, batch_norm__1613 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_268, parameter_1341, parameter_1342, parameter_1343, parameter_1344, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_147 = paddle._C_ops.add(batch_norm__1608, relu_237)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_239 = paddle._C_ops.relu(add_147)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_269 = paddle._C_ops.conv2d(relu_239, parameter_1345, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__1614, batch_norm__1615, batch_norm__1616, batch_norm__1617, batch_norm__1618, batch_norm__1619 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_269, parameter_1346, parameter_1347, parameter_1348, parameter_1349, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_240 = paddle._C_ops.relu(batch_norm__1614)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_270 = paddle._C_ops.conv2d(relu_240, parameter_1350, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__1620, batch_norm__1621, batch_norm__1622, batch_norm__1623, batch_norm__1624, batch_norm__1625 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_270, parameter_1351, parameter_1352, parameter_1353, parameter_1354, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_148 = paddle._C_ops.add(batch_norm__1620, relu_239)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_241 = paddle._C_ops.relu(add_148)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_271 = paddle._C_ops.conv2d(relu_241, parameter_1355, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__1626, batch_norm__1627, batch_norm__1628, batch_norm__1629, batch_norm__1630, batch_norm__1631 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_271, parameter_1356, parameter_1357, parameter_1358, parameter_1359, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_242 = paddle._C_ops.relu(batch_norm__1626)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_272 = paddle._C_ops.conv2d(relu_242, parameter_1360, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__1632, batch_norm__1633, batch_norm__1634, batch_norm__1635, batch_norm__1636, batch_norm__1637 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_272, parameter_1361, parameter_1362, parameter_1363, parameter_1364, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_149 = paddle._C_ops.add(batch_norm__1632, relu_241)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_243 = paddle._C_ops.relu(add_149)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_273 = paddle._C_ops.conv2d(relu_223, parameter_1365, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__1638, batch_norm__1639, batch_norm__1640, batch_norm__1641, batch_norm__1642, batch_norm__1643 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_273, parameter_1366, parameter_1367, parameter_1368, parameter_1369, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_244 = paddle._C_ops.relu(batch_norm__1638)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_274 = paddle._C_ops.conv2d(relu_244, parameter_1370, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__1644, batch_norm__1645, batch_norm__1646, batch_norm__1647, batch_norm__1648, batch_norm__1649 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_274, parameter_1371, parameter_1372, parameter_1373, parameter_1374, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_150 = paddle._C_ops.add(batch_norm__1644, relu_223)

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_245 = paddle._C_ops.relu(add_150)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_275 = paddle._C_ops.conv2d(relu_245, parameter_1375, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__1650, batch_norm__1651, batch_norm__1652, batch_norm__1653, batch_norm__1654, batch_norm__1655 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_275, parameter_1376, parameter_1377, parameter_1378, parameter_1379, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_246 = paddle._C_ops.relu(batch_norm__1650)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_276 = paddle._C_ops.conv2d(relu_246, parameter_1380, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__1656, batch_norm__1657, batch_norm__1658, batch_norm__1659, batch_norm__1660, batch_norm__1661 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_276, parameter_1381, parameter_1382, parameter_1383, parameter_1384, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_151 = paddle._C_ops.add(batch_norm__1656, relu_245)

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_247 = paddle._C_ops.relu(add_151)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_277 = paddle._C_ops.conv2d(relu_247, parameter_1385, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__1662, batch_norm__1663, batch_norm__1664, batch_norm__1665, batch_norm__1666, batch_norm__1667 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_277, parameter_1386, parameter_1387, parameter_1388, parameter_1389, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_248 = paddle._C_ops.relu(batch_norm__1662)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_278 = paddle._C_ops.conv2d(relu_248, parameter_1390, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__1668, batch_norm__1669, batch_norm__1670, batch_norm__1671, batch_norm__1672, batch_norm__1673 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_278, parameter_1391, parameter_1392, parameter_1393, parameter_1394, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_152 = paddle._C_ops.add(batch_norm__1668, relu_247)

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_249 = paddle._C_ops.relu(add_152)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_279 = paddle._C_ops.conv2d(relu_249, parameter_1395, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__1674, batch_norm__1675, batch_norm__1676, batch_norm__1677, batch_norm__1678, batch_norm__1679 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_279, parameter_1396, parameter_1397, parameter_1398, parameter_1399, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_250 = paddle._C_ops.relu(batch_norm__1674)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 72x72x3x3xf32)
        conv2d_280 = paddle._C_ops.conv2d(relu_250, parameter_1400, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__1680, batch_norm__1681, batch_norm__1682, batch_norm__1683, batch_norm__1684, batch_norm__1685 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_280, parameter_1401, parameter_1402, parameter_1403, parameter_1404, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_153 = paddle._C_ops.add(batch_norm__1680, relu_249)

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_251 = paddle._C_ops.relu(add_153)

        # pd_op.conv2d: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, 144x144x3x3xf32)
        conv2d_281 = paddle._C_ops.conv2d(relu_227, parameter_1405, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x-1x-1xf32, 144xf32, 144xf32, xf32, xf32, None) <- (-1x144x-1x-1xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__1686, batch_norm__1687, batch_norm__1688, batch_norm__1689, batch_norm__1690, batch_norm__1691 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_281, parameter_1406, parameter_1407, parameter_1408, parameter_1409, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32)
        relu_252 = paddle._C_ops.relu(batch_norm__1686)

        # pd_op.conv2d: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, 144x144x3x3xf32)
        conv2d_282 = paddle._C_ops.conv2d(relu_252, parameter_1410, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x-1x-1xf32, 144xf32, 144xf32, xf32, xf32, None) <- (-1x144x-1x-1xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__1692, batch_norm__1693, batch_norm__1694, batch_norm__1695, batch_norm__1696, batch_norm__1697 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_282, parameter_1411, parameter_1412, parameter_1413, parameter_1414, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, -1x144x-1x-1xf32)
        add_154 = paddle._C_ops.add(batch_norm__1692, relu_227)

        # pd_op.relu: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32)
        relu_253 = paddle._C_ops.relu(add_154)

        # pd_op.conv2d: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, 144x144x3x3xf32)
        conv2d_283 = paddle._C_ops.conv2d(relu_253, parameter_1415, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x-1x-1xf32, 144xf32, 144xf32, xf32, xf32, None) <- (-1x144x-1x-1xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__1698, batch_norm__1699, batch_norm__1700, batch_norm__1701, batch_norm__1702, batch_norm__1703 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_283, parameter_1416, parameter_1417, parameter_1418, parameter_1419, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32)
        relu_254 = paddle._C_ops.relu(batch_norm__1698)

        # pd_op.conv2d: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, 144x144x3x3xf32)
        conv2d_284 = paddle._C_ops.conv2d(relu_254, parameter_1420, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x-1x-1xf32, 144xf32, 144xf32, xf32, xf32, None) <- (-1x144x-1x-1xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__1704, batch_norm__1705, batch_norm__1706, batch_norm__1707, batch_norm__1708, batch_norm__1709 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_284, parameter_1421, parameter_1422, parameter_1423, parameter_1424, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, -1x144x-1x-1xf32)
        add_155 = paddle._C_ops.add(batch_norm__1704, relu_253)

        # pd_op.relu: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32)
        relu_255 = paddle._C_ops.relu(add_155)

        # pd_op.conv2d: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, 144x144x3x3xf32)
        conv2d_285 = paddle._C_ops.conv2d(relu_255, parameter_1425, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x-1x-1xf32, 144xf32, 144xf32, xf32, xf32, None) <- (-1x144x-1x-1xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__1710, batch_norm__1711, batch_norm__1712, batch_norm__1713, batch_norm__1714, batch_norm__1715 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_285, parameter_1426, parameter_1427, parameter_1428, parameter_1429, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32)
        relu_256 = paddle._C_ops.relu(batch_norm__1710)

        # pd_op.conv2d: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, 144x144x3x3xf32)
        conv2d_286 = paddle._C_ops.conv2d(relu_256, parameter_1430, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x-1x-1xf32, 144xf32, 144xf32, xf32, xf32, None) <- (-1x144x-1x-1xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__1716, batch_norm__1717, batch_norm__1718, batch_norm__1719, batch_norm__1720, batch_norm__1721 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_286, parameter_1431, parameter_1432, parameter_1433, parameter_1434, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, -1x144x-1x-1xf32)
        add_156 = paddle._C_ops.add(batch_norm__1716, relu_255)

        # pd_op.relu: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32)
        relu_257 = paddle._C_ops.relu(add_156)

        # pd_op.conv2d: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, 144x144x3x3xf32)
        conv2d_287 = paddle._C_ops.conv2d(relu_257, parameter_1435, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x-1x-1xf32, 144xf32, 144xf32, xf32, xf32, None) <- (-1x144x-1x-1xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__1722, batch_norm__1723, batch_norm__1724, batch_norm__1725, batch_norm__1726, batch_norm__1727 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_287, parameter_1436, parameter_1437, parameter_1438, parameter_1439, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32)
        relu_258 = paddle._C_ops.relu(batch_norm__1722)

        # pd_op.conv2d: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, 144x144x3x3xf32)
        conv2d_288 = paddle._C_ops.conv2d(relu_258, parameter_1440, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x-1x-1xf32, 144xf32, 144xf32, xf32, xf32, None) <- (-1x144x-1x-1xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__1728, batch_norm__1729, batch_norm__1730, batch_norm__1731, batch_norm__1732, batch_norm__1733 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_288, parameter_1441, parameter_1442, parameter_1443, parameter_1444, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, -1x144x-1x-1xf32)
        add_157 = paddle._C_ops.add(batch_norm__1728, relu_257)

        # pd_op.relu: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32)
        relu_259 = paddle._C_ops.relu(add_157)

        # pd_op.shape: (4xi32) <- (-1x18x-1x-1xf32)
        shape_15 = paddle._C_ops.shape(relu_235)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_30 = [-2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_31 = [2147483647]

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(shape_15, [0], full_int_array_30, full_int_array_31, [1], [])

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x36x-1x-1xf32, 18x36x1x1xf32)
        conv2d_289 = paddle._C_ops.conv2d(relu_243, parameter_1445, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__1734, batch_norm__1735, batch_norm__1736, batch_norm__1737, batch_norm__1738, batch_norm__1739 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_289, parameter_1446, parameter_1447, parameter_1448, parameter_1449, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.cast: (2xi32) <- (2xi32)
        cast_10 = paddle._C_ops.cast(slice_15, paddle.int32)

        # pd_op.bilinear_interp: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_25 = paddle._C_ops.bilinear_interp(batch_norm__1734, cast_10, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_158 = paddle._C_ops.add(relu_235, bilinear_interp_25)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x72x-1x-1xf32, 18x72x1x1xf32)
        conv2d_290 = paddle._C_ops.conv2d(relu_251, parameter_1450, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__1740, batch_norm__1741, batch_norm__1742, batch_norm__1743, batch_norm__1744, batch_norm__1745 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_290, parameter_1451, parameter_1452, parameter_1453, parameter_1454, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.cast: (2xi32) <- (2xi32)
        cast_11 = paddle._C_ops.cast(slice_15, paddle.int32)

        # pd_op.bilinear_interp: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_26 = paddle._C_ops.bilinear_interp(batch_norm__1740, cast_11, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_159 = paddle._C_ops.add(add_158, bilinear_interp_26)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x144x-1x-1xf32, 18x144x1x1xf32)
        conv2d_291 = paddle._C_ops.conv2d(relu_259, parameter_1455, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__1746, batch_norm__1747, batch_norm__1748, batch_norm__1749, batch_norm__1750, batch_norm__1751 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_291, parameter_1456, parameter_1457, parameter_1458, parameter_1459, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.cast_: (2xi32) <- (2xi32)
        cast__15 = paddle._C_ops.cast_(slice_15, paddle.int32)

        # pd_op.bilinear_interp: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_27 = paddle._C_ops.bilinear_interp(batch_norm__1746, cast__15, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, -1x18x-1x-1xf32)
        add_160 = paddle._C_ops.add(add_159, bilinear_interp_27)

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_260 = paddle._C_ops.relu(add_160)

        # pd_op.shape: (4xi32) <- (-1x36x-1x-1xf32)
        shape_16 = paddle._C_ops.shape(relu_243)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_32 = [-2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_33 = [2147483647]

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(shape_16, [0], full_int_array_32, full_int_array_33, [1], [])

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x18x-1x-1xf32, 36x18x3x3xf32)
        conv2d_292 = paddle._C_ops.conv2d(relu_235, parameter_1460, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__1752, batch_norm__1753, batch_norm__1754, batch_norm__1755, batch_norm__1756, batch_norm__1757 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_292, parameter_1461, parameter_1462, parameter_1463, parameter_1464, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_161 = paddle._C_ops.add(relu_243, batch_norm__1752)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x72x-1x-1xf32, 36x72x1x1xf32)
        conv2d_293 = paddle._C_ops.conv2d(relu_251, parameter_1465, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__1758, batch_norm__1759, batch_norm__1760, batch_norm__1761, batch_norm__1762, batch_norm__1763 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_293, parameter_1466, parameter_1467, parameter_1468, parameter_1469, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.cast: (2xi32) <- (2xi32)
        cast_12 = paddle._C_ops.cast(slice_16, paddle.int32)

        # pd_op.bilinear_interp: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_28 = paddle._C_ops.bilinear_interp(batch_norm__1758, cast_12, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_162 = paddle._C_ops.add(add_161, bilinear_interp_28)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x144x-1x-1xf32, 36x144x1x1xf32)
        conv2d_294 = paddle._C_ops.conv2d(relu_259, parameter_1470, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__1764, batch_norm__1765, batch_norm__1766, batch_norm__1767, batch_norm__1768, batch_norm__1769 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_294, parameter_1471, parameter_1472, parameter_1473, parameter_1474, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.cast_: (2xi32) <- (2xi32)
        cast__16 = paddle._C_ops.cast_(slice_16, paddle.int32)

        # pd_op.bilinear_interp: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_29 = paddle._C_ops.bilinear_interp(batch_norm__1764, cast__16, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, -1x36x-1x-1xf32)
        add_163 = paddle._C_ops.add(add_162, bilinear_interp_29)

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_261 = paddle._C_ops.relu(add_163)

        # pd_op.shape: (4xi32) <- (-1x72x-1x-1xf32)
        shape_17 = paddle._C_ops.shape(relu_251)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_34 = [-2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_35 = [2147483647]

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(shape_17, [0], full_int_array_34, full_int_array_35, [1], [])

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_295 = paddle._C_ops.conv2d(relu_235, parameter_1475, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__1770, batch_norm__1771, batch_norm__1772, batch_norm__1773, batch_norm__1774, batch_norm__1775 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_295, parameter_1476, parameter_1477, parameter_1478, parameter_1479, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_262 = paddle._C_ops.relu(batch_norm__1770)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x18x-1x-1xf32, 72x18x3x3xf32)
        conv2d_296 = paddle._C_ops.conv2d(relu_262, parameter_1480, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__1776, batch_norm__1777, batch_norm__1778, batch_norm__1779, batch_norm__1780, batch_norm__1781 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_296, parameter_1481, parameter_1482, parameter_1483, parameter_1484, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_164 = paddle._C_ops.add(relu_251, batch_norm__1776)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x36x-1x-1xf32, 72x36x3x3xf32)
        conv2d_297 = paddle._C_ops.conv2d(relu_243, parameter_1485, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__1782, batch_norm__1783, batch_norm__1784, batch_norm__1785, batch_norm__1786, batch_norm__1787 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_297, parameter_1486, parameter_1487, parameter_1488, parameter_1489, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_165 = paddle._C_ops.add(add_164, batch_norm__1782)

        # pd_op.conv2d: (-1x72x-1x-1xf32) <- (-1x144x-1x-1xf32, 72x144x1x1xf32)
        conv2d_298 = paddle._C_ops.conv2d(relu_259, parameter_1490, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x72x-1x-1xf32, 72xf32, 72xf32, xf32, xf32, None) <- (-1x72x-1x-1xf32, 72xf32, 72xf32, 72xf32, 72xf32)
        batch_norm__1788, batch_norm__1789, batch_norm__1790, batch_norm__1791, batch_norm__1792, batch_norm__1793 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_298, parameter_1491, parameter_1492, parameter_1493, parameter_1494, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.cast_: (2xi32) <- (2xi32)
        cast__17 = paddle._C_ops.cast_(slice_17, paddle.int32)

        # pd_op.bilinear_interp: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_30 = paddle._C_ops.bilinear_interp(batch_norm__1788, cast__17, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.add: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, -1x72x-1x-1xf32)
        add_166 = paddle._C_ops.add(add_165, bilinear_interp_30)

        # pd_op.relu: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32)
        relu_263 = paddle._C_ops.relu(add_166)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_299 = paddle._C_ops.conv2d(relu_235, parameter_1495, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__1794, batch_norm__1795, batch_norm__1796, batch_norm__1797, batch_norm__1798, batch_norm__1799 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_299, parameter_1496, parameter_1497, parameter_1498, parameter_1499, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_264 = paddle._C_ops.relu(batch_norm__1794)

        # pd_op.conv2d: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32, 18x18x3x3xf32)
        conv2d_300 = paddle._C_ops.conv2d(relu_264, parameter_1500, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x18x-1x-1xf32, 18xf32, 18xf32, xf32, xf32, None) <- (-1x18x-1x-1xf32, 18xf32, 18xf32, 18xf32, 18xf32)
        batch_norm__1800, batch_norm__1801, batch_norm__1802, batch_norm__1803, batch_norm__1804, batch_norm__1805 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_300, parameter_1501, parameter_1502, parameter_1503, parameter_1504, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x18x-1x-1xf32) <- (-1x18x-1x-1xf32)
        relu_265 = paddle._C_ops.relu(batch_norm__1800)

        # pd_op.conv2d: (-1x144x-1x-1xf32) <- (-1x18x-1x-1xf32, 144x18x3x3xf32)
        conv2d_301 = paddle._C_ops.conv2d(relu_265, parameter_1505, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x-1x-1xf32, 144xf32, 144xf32, xf32, xf32, None) <- (-1x144x-1x-1xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__1806, batch_norm__1807, batch_norm__1808, batch_norm__1809, batch_norm__1810, batch_norm__1811 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_301, parameter_1506, parameter_1507, parameter_1508, parameter_1509, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, -1x144x-1x-1xf32)
        add_167 = paddle._C_ops.add(relu_259, batch_norm__1806)

        # pd_op.conv2d: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 36x36x3x3xf32)
        conv2d_302 = paddle._C_ops.conv2d(relu_243, parameter_1510, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x36x-1x-1xf32, 36xf32, 36xf32, xf32, xf32, None) <- (-1x36x-1x-1xf32, 36xf32, 36xf32, 36xf32, 36xf32)
        batch_norm__1812, batch_norm__1813, batch_norm__1814, batch_norm__1815, batch_norm__1816, batch_norm__1817 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_302, parameter_1511, parameter_1512, parameter_1513, parameter_1514, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32)
        relu_266 = paddle._C_ops.relu(batch_norm__1812)

        # pd_op.conv2d: (-1x144x-1x-1xf32) <- (-1x36x-1x-1xf32, 144x36x3x3xf32)
        conv2d_303 = paddle._C_ops.conv2d(relu_266, parameter_1515, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x-1x-1xf32, 144xf32, 144xf32, xf32, xf32, None) <- (-1x144x-1x-1xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__1818, batch_norm__1819, batch_norm__1820, batch_norm__1821, batch_norm__1822, batch_norm__1823 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_303, parameter_1516, parameter_1517, parameter_1518, parameter_1519, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, -1x144x-1x-1xf32)
        add_168 = paddle._C_ops.add(add_167, batch_norm__1818)

        # pd_op.conv2d: (-1x144x-1x-1xf32) <- (-1x72x-1x-1xf32, 144x72x3x3xf32)
        conv2d_304 = paddle._C_ops.conv2d(relu_251, parameter_1520, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x144x-1x-1xf32, 144xf32, 144xf32, xf32, xf32, None) <- (-1x144x-1x-1xf32, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__1824, batch_norm__1825, batch_norm__1826, batch_norm__1827, batch_norm__1828, batch_norm__1829 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_304, parameter_1521, parameter_1522, parameter_1523, parameter_1524, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, -1x144x-1x-1xf32)
        add_169 = paddle._C_ops.add(add_168, batch_norm__1824)

        # pd_op.relu: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32)
        relu_267 = paddle._C_ops.relu(add_169)

        # pd_op.shape: (4xi32) <- (-1x18x-1x-1xf32)
        shape_18 = paddle._C_ops.shape(relu_260)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_36 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_37 = [2147483647]

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(shape_18, [0], full_int_array_36, full_int_array_37, [1], [])

        # pd_op.cast: (2xi32) <- (2xi32)
        cast_13 = paddle._C_ops.cast(slice_18, paddle.int32)

        # pd_op.bilinear_interp: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_31 = paddle._C_ops.bilinear_interp(relu_261, cast_13, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.cast: (2xi32) <- (2xi32)
        cast_14 = paddle._C_ops.cast(slice_18, paddle.int32)

        # pd_op.bilinear_interp: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_32 = paddle._C_ops.bilinear_interp(relu_263, cast_14, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.cast_: (2xi32) <- (2xi32)
        cast__18 = paddle._C_ops.cast_(slice_18, paddle.int32)

        # pd_op.bilinear_interp: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_33 = paddle._C_ops.bilinear_interp(relu_267, cast__18, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # builtin.combine: ([-1x18x-1x-1xf32, -1x36x-1x-1xf32, -1x72x-1x-1xf32, -1x144x-1x-1xf32]) <- (-1x18x-1x-1xf32, -1x36x-1x-1xf32, -1x72x-1x-1xf32, -1x144x-1x-1xf32)
        combine_0 = [relu_260, bilinear_interp_31, bilinear_interp_32, bilinear_interp_33]

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x270x-1x-1xf32) <- ([-1x18x-1x-1xf32, -1x36x-1x-1xf32, -1x72x-1x-1xf32, -1x144x-1x-1xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        # pd_op.conv2d: (-1x270x-1x-1xf32) <- (-1x270x-1x-1xf32, 270x270x1x1xf32)
        conv2d_305 = paddle._C_ops.conv2d(concat_0, parameter_1525, [1, 1], [0, 0], 'SAME', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x270x-1x-1xf32, 270xf32, 270xf32, xf32, xf32, None) <- (-1x270x-1x-1xf32, 270xf32, 270xf32, 270xf32, 270xf32)
        batch_norm__1830, batch_norm__1831, batch_norm__1832, batch_norm__1833, batch_norm__1834, batch_norm__1835 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_305, parameter_1526, parameter_1527, parameter_1528, parameter_1529, True, float('0.9'), float('1e-05'), 'NCHW', True, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x270x-1x-1xf32) <- (-1x270x-1x-1xf32)
        relu_268 = paddle._C_ops.relu(batch_norm__1830)

        # pd_op.conv2d: (-1x19x-1x-1xf32) <- (-1x270x-1x-1xf32, 19x270x1x1xf32)
        conv2d_306 = paddle._C_ops.conv2d(relu_268, parameter_1530, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.shape: (4xi32) <- (-1x3x-1x-1xf32)
        shape_19 = paddle._C_ops.shape(feed_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_38 = [2]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_39 = [2147483647]

        # pd_op.slice: (2xi32) <- (4xi32, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(shape_19, [0], full_int_array_38, full_int_array_39, [1], [])

        # pd_op.cast_: (2xi32) <- (2xi32)
        cast__19 = paddle._C_ops.cast_(slice_19, paddle.int32)

        # pd_op.bilinear_interp: (-1x19x-1x-1xf32) <- (-1x19x-1x-1xf32, 2xi32, None, None)
        bilinear_interp_34 = paddle._C_ops.bilinear_interp(conv2d_306, cast__19, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        # pd_op.full: (1xi64) <- ()
        full_1 = paddle._C_ops.full([1], float('1'), paddle.int64, paddle.core.CPUPlace())

        # pd_op.argmax: (-1x-1x-1xi32) <- (-1x19x-1x-1xf32, 1xi64)
        argmax_0 = paddle._C_ops.argmax(bilinear_interp_34, full_1, False, False, paddle.int32)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x-1x-1xi32) <- (-1x-1x-1xi32, 1xf32)
        scale_0 = paddle._C_ops.scale(argmax_0, full_2, float('0'), True)
        return scale_0



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

    def forward(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_114, parameter_111, parameter_113, parameter_112, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_154, parameter_151, parameter_153, parameter_152, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_169, parameter_166, parameter_168, parameter_167, parameter_170, parameter_174, parameter_171, parameter_173, parameter_172, parameter_175, parameter_179, parameter_176, parameter_178, parameter_177, parameter_180, parameter_184, parameter_181, parameter_183, parameter_182, parameter_185, parameter_189, parameter_186, parameter_188, parameter_187, parameter_190, parameter_194, parameter_191, parameter_193, parameter_192, parameter_195, parameter_199, parameter_196, parameter_198, parameter_197, parameter_200, parameter_204, parameter_201, parameter_203, parameter_202, parameter_205, parameter_209, parameter_206, parameter_208, parameter_207, parameter_210, parameter_214, parameter_211, parameter_213, parameter_212, parameter_215, parameter_219, parameter_216, parameter_218, parameter_217, parameter_220, parameter_224, parameter_221, parameter_223, parameter_222, parameter_225, parameter_229, parameter_226, parameter_228, parameter_227, parameter_230, parameter_234, parameter_231, parameter_233, parameter_232, parameter_235, parameter_239, parameter_236, parameter_238, parameter_237, parameter_240, parameter_244, parameter_241, parameter_243, parameter_242, parameter_245, parameter_249, parameter_246, parameter_248, parameter_247, parameter_250, parameter_254, parameter_251, parameter_253, parameter_252, parameter_255, parameter_259, parameter_256, parameter_258, parameter_257, parameter_260, parameter_264, parameter_261, parameter_263, parameter_262, parameter_265, parameter_269, parameter_266, parameter_268, parameter_267, parameter_270, parameter_274, parameter_271, parameter_273, parameter_272, parameter_275, parameter_279, parameter_276, parameter_278, parameter_277, parameter_280, parameter_284, parameter_281, parameter_283, parameter_282, parameter_285, parameter_289, parameter_286, parameter_288, parameter_287, parameter_290, parameter_294, parameter_291, parameter_293, parameter_292, parameter_295, parameter_299, parameter_296, parameter_298, parameter_297, parameter_300, parameter_304, parameter_301, parameter_303, parameter_302, parameter_305, parameter_309, parameter_306, parameter_308, parameter_307, parameter_310, parameter_314, parameter_311, parameter_313, parameter_312, parameter_315, parameter_319, parameter_316, parameter_318, parameter_317, parameter_320, parameter_324, parameter_321, parameter_323, parameter_322, parameter_325, parameter_329, parameter_326, parameter_328, parameter_327, parameter_330, parameter_334, parameter_331, parameter_333, parameter_332, parameter_335, parameter_339, parameter_336, parameter_338, parameter_337, parameter_340, parameter_344, parameter_341, parameter_343, parameter_342, parameter_345, parameter_349, parameter_346, parameter_348, parameter_347, parameter_350, parameter_354, parameter_351, parameter_353, parameter_352, parameter_355, parameter_359, parameter_356, parameter_358, parameter_357, parameter_360, parameter_364, parameter_361, parameter_363, parameter_362, parameter_365, parameter_369, parameter_366, parameter_368, parameter_367, parameter_370, parameter_374, parameter_371, parameter_373, parameter_372, parameter_375, parameter_379, parameter_376, parameter_378, parameter_377, parameter_380, parameter_384, parameter_381, parameter_383, parameter_382, parameter_385, parameter_389, parameter_386, parameter_388, parameter_387, parameter_390, parameter_394, parameter_391, parameter_393, parameter_392, parameter_395, parameter_399, parameter_396, parameter_398, parameter_397, parameter_400, parameter_404, parameter_401, parameter_403, parameter_402, parameter_405, parameter_409, parameter_406, parameter_408, parameter_407, parameter_410, parameter_414, parameter_411, parameter_413, parameter_412, parameter_415, parameter_419, parameter_416, parameter_418, parameter_417, parameter_420, parameter_424, parameter_421, parameter_423, parameter_422, parameter_425, parameter_429, parameter_426, parameter_428, parameter_427, parameter_430, parameter_434, parameter_431, parameter_433, parameter_432, parameter_435, parameter_439, parameter_436, parameter_438, parameter_437, parameter_440, parameter_444, parameter_441, parameter_443, parameter_442, parameter_445, parameter_449, parameter_446, parameter_448, parameter_447, parameter_450, parameter_454, parameter_451, parameter_453, parameter_452, parameter_455, parameter_459, parameter_456, parameter_458, parameter_457, parameter_460, parameter_464, parameter_461, parameter_463, parameter_462, parameter_465, parameter_469, parameter_466, parameter_468, parameter_467, parameter_470, parameter_474, parameter_471, parameter_473, parameter_472, parameter_475, parameter_479, parameter_476, parameter_478, parameter_477, parameter_480, parameter_484, parameter_481, parameter_483, parameter_482, parameter_485, parameter_489, parameter_486, parameter_488, parameter_487, parameter_490, parameter_494, parameter_491, parameter_493, parameter_492, parameter_495, parameter_499, parameter_496, parameter_498, parameter_497, parameter_500, parameter_504, parameter_501, parameter_503, parameter_502, parameter_505, parameter_509, parameter_506, parameter_508, parameter_507, parameter_510, parameter_514, parameter_511, parameter_513, parameter_512, parameter_515, parameter_519, parameter_516, parameter_518, parameter_517, parameter_520, parameter_524, parameter_521, parameter_523, parameter_522, parameter_525, parameter_529, parameter_526, parameter_528, parameter_527, parameter_530, parameter_534, parameter_531, parameter_533, parameter_532, parameter_535, parameter_539, parameter_536, parameter_538, parameter_537, parameter_540, parameter_544, parameter_541, parameter_543, parameter_542, parameter_545, parameter_549, parameter_546, parameter_548, parameter_547, parameter_550, parameter_554, parameter_551, parameter_553, parameter_552, parameter_555, parameter_559, parameter_556, parameter_558, parameter_557, parameter_560, parameter_564, parameter_561, parameter_563, parameter_562, parameter_565, parameter_569, parameter_566, parameter_568, parameter_567, parameter_570, parameter_574, parameter_571, parameter_573, parameter_572, parameter_575, parameter_579, parameter_576, parameter_578, parameter_577, parameter_580, parameter_584, parameter_581, parameter_583, parameter_582, parameter_585, parameter_589, parameter_586, parameter_588, parameter_587, parameter_590, parameter_594, parameter_591, parameter_593, parameter_592, parameter_595, parameter_599, parameter_596, parameter_598, parameter_597, parameter_600, parameter_604, parameter_601, parameter_603, parameter_602, parameter_605, parameter_609, parameter_606, parameter_608, parameter_607, parameter_610, parameter_614, parameter_611, parameter_613, parameter_612, parameter_615, parameter_619, parameter_616, parameter_618, parameter_617, parameter_620, parameter_624, parameter_621, parameter_623, parameter_622, parameter_625, parameter_629, parameter_626, parameter_628, parameter_627, parameter_630, parameter_634, parameter_631, parameter_633, parameter_632, parameter_635, parameter_639, parameter_636, parameter_638, parameter_637, parameter_640, parameter_644, parameter_641, parameter_643, parameter_642, parameter_645, parameter_649, parameter_646, parameter_648, parameter_647, parameter_650, parameter_654, parameter_651, parameter_653, parameter_652, parameter_655, parameter_659, parameter_656, parameter_658, parameter_657, parameter_660, parameter_664, parameter_661, parameter_663, parameter_662, parameter_665, parameter_669, parameter_666, parameter_668, parameter_667, parameter_670, parameter_674, parameter_671, parameter_673, parameter_672, parameter_675, parameter_679, parameter_676, parameter_678, parameter_677, parameter_680, parameter_684, parameter_681, parameter_683, parameter_682, parameter_685, parameter_689, parameter_686, parameter_688, parameter_687, parameter_690, parameter_694, parameter_691, parameter_693, parameter_692, parameter_695, parameter_699, parameter_696, parameter_698, parameter_697, parameter_700, parameter_704, parameter_701, parameter_703, parameter_702, parameter_705, parameter_709, parameter_706, parameter_708, parameter_707, parameter_710, parameter_714, parameter_711, parameter_713, parameter_712, parameter_715, parameter_719, parameter_716, parameter_718, parameter_717, parameter_720, parameter_724, parameter_721, parameter_723, parameter_722, parameter_725, parameter_729, parameter_726, parameter_728, parameter_727, parameter_730, parameter_734, parameter_731, parameter_733, parameter_732, parameter_735, parameter_739, parameter_736, parameter_738, parameter_737, parameter_740, parameter_744, parameter_741, parameter_743, parameter_742, parameter_745, parameter_749, parameter_746, parameter_748, parameter_747, parameter_750, parameter_754, parameter_751, parameter_753, parameter_752, parameter_755, parameter_759, parameter_756, parameter_758, parameter_757, parameter_760, parameter_764, parameter_761, parameter_763, parameter_762, parameter_765, parameter_769, parameter_766, parameter_768, parameter_767, parameter_770, parameter_774, parameter_771, parameter_773, parameter_772, parameter_775, parameter_779, parameter_776, parameter_778, parameter_777, parameter_780, parameter_784, parameter_781, parameter_783, parameter_782, parameter_785, parameter_789, parameter_786, parameter_788, parameter_787, parameter_790, parameter_794, parameter_791, parameter_793, parameter_792, parameter_795, parameter_799, parameter_796, parameter_798, parameter_797, parameter_800, parameter_804, parameter_801, parameter_803, parameter_802, parameter_805, parameter_809, parameter_806, parameter_808, parameter_807, parameter_810, parameter_814, parameter_811, parameter_813, parameter_812, parameter_815, parameter_819, parameter_816, parameter_818, parameter_817, parameter_820, parameter_824, parameter_821, parameter_823, parameter_822, parameter_825, parameter_829, parameter_826, parameter_828, parameter_827, parameter_830, parameter_834, parameter_831, parameter_833, parameter_832, parameter_835, parameter_839, parameter_836, parameter_838, parameter_837, parameter_840, parameter_844, parameter_841, parameter_843, parameter_842, parameter_845, parameter_849, parameter_846, parameter_848, parameter_847, parameter_850, parameter_854, parameter_851, parameter_853, parameter_852, parameter_855, parameter_859, parameter_856, parameter_858, parameter_857, parameter_860, parameter_864, parameter_861, parameter_863, parameter_862, parameter_865, parameter_869, parameter_866, parameter_868, parameter_867, parameter_870, parameter_874, parameter_871, parameter_873, parameter_872, parameter_875, parameter_879, parameter_876, parameter_878, parameter_877, parameter_880, parameter_884, parameter_881, parameter_883, parameter_882, parameter_885, parameter_889, parameter_886, parameter_888, parameter_887, parameter_890, parameter_894, parameter_891, parameter_893, parameter_892, parameter_895, parameter_899, parameter_896, parameter_898, parameter_897, parameter_900, parameter_904, parameter_901, parameter_903, parameter_902, parameter_905, parameter_909, parameter_906, parameter_908, parameter_907, parameter_910, parameter_914, parameter_911, parameter_913, parameter_912, parameter_915, parameter_919, parameter_916, parameter_918, parameter_917, parameter_920, parameter_924, parameter_921, parameter_923, parameter_922, parameter_925, parameter_929, parameter_926, parameter_928, parameter_927, parameter_930, parameter_934, parameter_931, parameter_933, parameter_932, parameter_935, parameter_939, parameter_936, parameter_938, parameter_937, parameter_940, parameter_944, parameter_941, parameter_943, parameter_942, parameter_945, parameter_949, parameter_946, parameter_948, parameter_947, parameter_950, parameter_954, parameter_951, parameter_953, parameter_952, parameter_955, parameter_959, parameter_956, parameter_958, parameter_957, parameter_960, parameter_964, parameter_961, parameter_963, parameter_962, parameter_965, parameter_969, parameter_966, parameter_968, parameter_967, parameter_970, parameter_974, parameter_971, parameter_973, parameter_972, parameter_975, parameter_979, parameter_976, parameter_978, parameter_977, parameter_980, parameter_984, parameter_981, parameter_983, parameter_982, parameter_985, parameter_989, parameter_986, parameter_988, parameter_987, parameter_990, parameter_994, parameter_991, parameter_993, parameter_992, parameter_995, parameter_999, parameter_996, parameter_998, parameter_997, parameter_1000, parameter_1004, parameter_1001, parameter_1003, parameter_1002, parameter_1005, parameter_1009, parameter_1006, parameter_1008, parameter_1007, parameter_1010, parameter_1014, parameter_1011, parameter_1013, parameter_1012, parameter_1015, parameter_1019, parameter_1016, parameter_1018, parameter_1017, parameter_1020, parameter_1024, parameter_1021, parameter_1023, parameter_1022, parameter_1025, parameter_1029, parameter_1026, parameter_1028, parameter_1027, parameter_1030, parameter_1034, parameter_1031, parameter_1033, parameter_1032, parameter_1035, parameter_1039, parameter_1036, parameter_1038, parameter_1037, parameter_1040, parameter_1044, parameter_1041, parameter_1043, parameter_1042, parameter_1045, parameter_1049, parameter_1046, parameter_1048, parameter_1047, parameter_1050, parameter_1054, parameter_1051, parameter_1053, parameter_1052, parameter_1055, parameter_1059, parameter_1056, parameter_1058, parameter_1057, parameter_1060, parameter_1064, parameter_1061, parameter_1063, parameter_1062, parameter_1065, parameter_1069, parameter_1066, parameter_1068, parameter_1067, parameter_1070, parameter_1074, parameter_1071, parameter_1073, parameter_1072, parameter_1075, parameter_1079, parameter_1076, parameter_1078, parameter_1077, parameter_1080, parameter_1084, parameter_1081, parameter_1083, parameter_1082, parameter_1085, parameter_1089, parameter_1086, parameter_1088, parameter_1087, parameter_1090, parameter_1094, parameter_1091, parameter_1093, parameter_1092, parameter_1095, parameter_1099, parameter_1096, parameter_1098, parameter_1097, parameter_1100, parameter_1104, parameter_1101, parameter_1103, parameter_1102, parameter_1105, parameter_1109, parameter_1106, parameter_1108, parameter_1107, parameter_1110, parameter_1114, parameter_1111, parameter_1113, parameter_1112, parameter_1115, parameter_1119, parameter_1116, parameter_1118, parameter_1117, parameter_1120, parameter_1124, parameter_1121, parameter_1123, parameter_1122, parameter_1125, parameter_1129, parameter_1126, parameter_1128, parameter_1127, parameter_1130, parameter_1134, parameter_1131, parameter_1133, parameter_1132, parameter_1135, parameter_1139, parameter_1136, parameter_1138, parameter_1137, parameter_1140, parameter_1144, parameter_1141, parameter_1143, parameter_1142, parameter_1145, parameter_1149, parameter_1146, parameter_1148, parameter_1147, parameter_1150, parameter_1154, parameter_1151, parameter_1153, parameter_1152, parameter_1155, parameter_1159, parameter_1156, parameter_1158, parameter_1157, parameter_1160, parameter_1164, parameter_1161, parameter_1163, parameter_1162, parameter_1165, parameter_1169, parameter_1166, parameter_1168, parameter_1167, parameter_1170, parameter_1174, parameter_1171, parameter_1173, parameter_1172, parameter_1175, parameter_1179, parameter_1176, parameter_1178, parameter_1177, parameter_1180, parameter_1184, parameter_1181, parameter_1183, parameter_1182, parameter_1185, parameter_1189, parameter_1186, parameter_1188, parameter_1187, parameter_1190, parameter_1194, parameter_1191, parameter_1193, parameter_1192, parameter_1195, parameter_1199, parameter_1196, parameter_1198, parameter_1197, parameter_1200, parameter_1204, parameter_1201, parameter_1203, parameter_1202, parameter_1205, parameter_1209, parameter_1206, parameter_1208, parameter_1207, parameter_1210, parameter_1214, parameter_1211, parameter_1213, parameter_1212, parameter_1215, parameter_1219, parameter_1216, parameter_1218, parameter_1217, parameter_1220, parameter_1224, parameter_1221, parameter_1223, parameter_1222, parameter_1225, parameter_1229, parameter_1226, parameter_1228, parameter_1227, parameter_1230, parameter_1234, parameter_1231, parameter_1233, parameter_1232, parameter_1235, parameter_1239, parameter_1236, parameter_1238, parameter_1237, parameter_1240, parameter_1244, parameter_1241, parameter_1243, parameter_1242, parameter_1245, parameter_1249, parameter_1246, parameter_1248, parameter_1247, parameter_1250, parameter_1254, parameter_1251, parameter_1253, parameter_1252, parameter_1255, parameter_1259, parameter_1256, parameter_1258, parameter_1257, parameter_1260, parameter_1264, parameter_1261, parameter_1263, parameter_1262, parameter_1265, parameter_1269, parameter_1266, parameter_1268, parameter_1267, parameter_1270, parameter_1274, parameter_1271, parameter_1273, parameter_1272, parameter_1275, parameter_1279, parameter_1276, parameter_1278, parameter_1277, parameter_1280, parameter_1284, parameter_1281, parameter_1283, parameter_1282, parameter_1285, parameter_1289, parameter_1286, parameter_1288, parameter_1287, parameter_1290, parameter_1294, parameter_1291, parameter_1293, parameter_1292, parameter_1295, parameter_1299, parameter_1296, parameter_1298, parameter_1297, parameter_1300, parameter_1304, parameter_1301, parameter_1303, parameter_1302, parameter_1305, parameter_1309, parameter_1306, parameter_1308, parameter_1307, parameter_1310, parameter_1314, parameter_1311, parameter_1313, parameter_1312, parameter_1315, parameter_1319, parameter_1316, parameter_1318, parameter_1317, parameter_1320, parameter_1324, parameter_1321, parameter_1323, parameter_1322, parameter_1325, parameter_1329, parameter_1326, parameter_1328, parameter_1327, parameter_1330, parameter_1334, parameter_1331, parameter_1333, parameter_1332, parameter_1335, parameter_1339, parameter_1336, parameter_1338, parameter_1337, parameter_1340, parameter_1344, parameter_1341, parameter_1343, parameter_1342, parameter_1345, parameter_1349, parameter_1346, parameter_1348, parameter_1347, parameter_1350, parameter_1354, parameter_1351, parameter_1353, parameter_1352, parameter_1355, parameter_1359, parameter_1356, parameter_1358, parameter_1357, parameter_1360, parameter_1364, parameter_1361, parameter_1363, parameter_1362, parameter_1365, parameter_1369, parameter_1366, parameter_1368, parameter_1367, parameter_1370, parameter_1374, parameter_1371, parameter_1373, parameter_1372, parameter_1375, parameter_1379, parameter_1376, parameter_1378, parameter_1377, parameter_1380, parameter_1384, parameter_1381, parameter_1383, parameter_1382, parameter_1385, parameter_1389, parameter_1386, parameter_1388, parameter_1387, parameter_1390, parameter_1394, parameter_1391, parameter_1393, parameter_1392, parameter_1395, parameter_1399, parameter_1396, parameter_1398, parameter_1397, parameter_1400, parameter_1404, parameter_1401, parameter_1403, parameter_1402, parameter_1405, parameter_1409, parameter_1406, parameter_1408, parameter_1407, parameter_1410, parameter_1414, parameter_1411, parameter_1413, parameter_1412, parameter_1415, parameter_1419, parameter_1416, parameter_1418, parameter_1417, parameter_1420, parameter_1424, parameter_1421, parameter_1423, parameter_1422, parameter_1425, parameter_1429, parameter_1426, parameter_1428, parameter_1427, parameter_1430, parameter_1434, parameter_1431, parameter_1433, parameter_1432, parameter_1435, parameter_1439, parameter_1436, parameter_1438, parameter_1437, parameter_1440, parameter_1444, parameter_1441, parameter_1443, parameter_1442, parameter_1445, parameter_1449, parameter_1446, parameter_1448, parameter_1447, parameter_1450, parameter_1454, parameter_1451, parameter_1453, parameter_1452, parameter_1455, parameter_1459, parameter_1456, parameter_1458, parameter_1457, parameter_1460, parameter_1464, parameter_1461, parameter_1463, parameter_1462, parameter_1465, parameter_1469, parameter_1466, parameter_1468, parameter_1467, parameter_1470, parameter_1474, parameter_1471, parameter_1473, parameter_1472, parameter_1475, parameter_1479, parameter_1476, parameter_1478, parameter_1477, parameter_1480, parameter_1484, parameter_1481, parameter_1483, parameter_1482, parameter_1485, parameter_1489, parameter_1486, parameter_1488, parameter_1487, parameter_1490, parameter_1494, parameter_1491, parameter_1493, parameter_1492, parameter_1495, parameter_1499, parameter_1496, parameter_1498, parameter_1497, parameter_1500, parameter_1504, parameter_1501, parameter_1503, parameter_1502, parameter_1505, parameter_1509, parameter_1506, parameter_1508, parameter_1507, parameter_1510, parameter_1514, parameter_1511, parameter_1513, parameter_1512, parameter_1515, parameter_1519, parameter_1516, parameter_1518, parameter_1517, parameter_1520, parameter_1524, parameter_1521, parameter_1523, parameter_1522, parameter_1525, parameter_1529, parameter_1526, parameter_1528, parameter_1527, parameter_1530, feed_0):
        return self.builtin_module_2747_0_0(parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_114, parameter_111, parameter_113, parameter_112, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_154, parameter_151, parameter_153, parameter_152, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_169, parameter_166, parameter_168, parameter_167, parameter_170, parameter_174, parameter_171, parameter_173, parameter_172, parameter_175, parameter_179, parameter_176, parameter_178, parameter_177, parameter_180, parameter_184, parameter_181, parameter_183, parameter_182, parameter_185, parameter_189, parameter_186, parameter_188, parameter_187, parameter_190, parameter_194, parameter_191, parameter_193, parameter_192, parameter_195, parameter_199, parameter_196, parameter_198, parameter_197, parameter_200, parameter_204, parameter_201, parameter_203, parameter_202, parameter_205, parameter_209, parameter_206, parameter_208, parameter_207, parameter_210, parameter_214, parameter_211, parameter_213, parameter_212, parameter_215, parameter_219, parameter_216, parameter_218, parameter_217, parameter_220, parameter_224, parameter_221, parameter_223, parameter_222, parameter_225, parameter_229, parameter_226, parameter_228, parameter_227, parameter_230, parameter_234, parameter_231, parameter_233, parameter_232, parameter_235, parameter_239, parameter_236, parameter_238, parameter_237, parameter_240, parameter_244, parameter_241, parameter_243, parameter_242, parameter_245, parameter_249, parameter_246, parameter_248, parameter_247, parameter_250, parameter_254, parameter_251, parameter_253, parameter_252, parameter_255, parameter_259, parameter_256, parameter_258, parameter_257, parameter_260, parameter_264, parameter_261, parameter_263, parameter_262, parameter_265, parameter_269, parameter_266, parameter_268, parameter_267, parameter_270, parameter_274, parameter_271, parameter_273, parameter_272, parameter_275, parameter_279, parameter_276, parameter_278, parameter_277, parameter_280, parameter_284, parameter_281, parameter_283, parameter_282, parameter_285, parameter_289, parameter_286, parameter_288, parameter_287, parameter_290, parameter_294, parameter_291, parameter_293, parameter_292, parameter_295, parameter_299, parameter_296, parameter_298, parameter_297, parameter_300, parameter_304, parameter_301, parameter_303, parameter_302, parameter_305, parameter_309, parameter_306, parameter_308, parameter_307, parameter_310, parameter_314, parameter_311, parameter_313, parameter_312, parameter_315, parameter_319, parameter_316, parameter_318, parameter_317, parameter_320, parameter_324, parameter_321, parameter_323, parameter_322, parameter_325, parameter_329, parameter_326, parameter_328, parameter_327, parameter_330, parameter_334, parameter_331, parameter_333, parameter_332, parameter_335, parameter_339, parameter_336, parameter_338, parameter_337, parameter_340, parameter_344, parameter_341, parameter_343, parameter_342, parameter_345, parameter_349, parameter_346, parameter_348, parameter_347, parameter_350, parameter_354, parameter_351, parameter_353, parameter_352, parameter_355, parameter_359, parameter_356, parameter_358, parameter_357, parameter_360, parameter_364, parameter_361, parameter_363, parameter_362, parameter_365, parameter_369, parameter_366, parameter_368, parameter_367, parameter_370, parameter_374, parameter_371, parameter_373, parameter_372, parameter_375, parameter_379, parameter_376, parameter_378, parameter_377, parameter_380, parameter_384, parameter_381, parameter_383, parameter_382, parameter_385, parameter_389, parameter_386, parameter_388, parameter_387, parameter_390, parameter_394, parameter_391, parameter_393, parameter_392, parameter_395, parameter_399, parameter_396, parameter_398, parameter_397, parameter_400, parameter_404, parameter_401, parameter_403, parameter_402, parameter_405, parameter_409, parameter_406, parameter_408, parameter_407, parameter_410, parameter_414, parameter_411, parameter_413, parameter_412, parameter_415, parameter_419, parameter_416, parameter_418, parameter_417, parameter_420, parameter_424, parameter_421, parameter_423, parameter_422, parameter_425, parameter_429, parameter_426, parameter_428, parameter_427, parameter_430, parameter_434, parameter_431, parameter_433, parameter_432, parameter_435, parameter_439, parameter_436, parameter_438, parameter_437, parameter_440, parameter_444, parameter_441, parameter_443, parameter_442, parameter_445, parameter_449, parameter_446, parameter_448, parameter_447, parameter_450, parameter_454, parameter_451, parameter_453, parameter_452, parameter_455, parameter_459, parameter_456, parameter_458, parameter_457, parameter_460, parameter_464, parameter_461, parameter_463, parameter_462, parameter_465, parameter_469, parameter_466, parameter_468, parameter_467, parameter_470, parameter_474, parameter_471, parameter_473, parameter_472, parameter_475, parameter_479, parameter_476, parameter_478, parameter_477, parameter_480, parameter_484, parameter_481, parameter_483, parameter_482, parameter_485, parameter_489, parameter_486, parameter_488, parameter_487, parameter_490, parameter_494, parameter_491, parameter_493, parameter_492, parameter_495, parameter_499, parameter_496, parameter_498, parameter_497, parameter_500, parameter_504, parameter_501, parameter_503, parameter_502, parameter_505, parameter_509, parameter_506, parameter_508, parameter_507, parameter_510, parameter_514, parameter_511, parameter_513, parameter_512, parameter_515, parameter_519, parameter_516, parameter_518, parameter_517, parameter_520, parameter_524, parameter_521, parameter_523, parameter_522, parameter_525, parameter_529, parameter_526, parameter_528, parameter_527, parameter_530, parameter_534, parameter_531, parameter_533, parameter_532, parameter_535, parameter_539, parameter_536, parameter_538, parameter_537, parameter_540, parameter_544, parameter_541, parameter_543, parameter_542, parameter_545, parameter_549, parameter_546, parameter_548, parameter_547, parameter_550, parameter_554, parameter_551, parameter_553, parameter_552, parameter_555, parameter_559, parameter_556, parameter_558, parameter_557, parameter_560, parameter_564, parameter_561, parameter_563, parameter_562, parameter_565, parameter_569, parameter_566, parameter_568, parameter_567, parameter_570, parameter_574, parameter_571, parameter_573, parameter_572, parameter_575, parameter_579, parameter_576, parameter_578, parameter_577, parameter_580, parameter_584, parameter_581, parameter_583, parameter_582, parameter_585, parameter_589, parameter_586, parameter_588, parameter_587, parameter_590, parameter_594, parameter_591, parameter_593, parameter_592, parameter_595, parameter_599, parameter_596, parameter_598, parameter_597, parameter_600, parameter_604, parameter_601, parameter_603, parameter_602, parameter_605, parameter_609, parameter_606, parameter_608, parameter_607, parameter_610, parameter_614, parameter_611, parameter_613, parameter_612, parameter_615, parameter_619, parameter_616, parameter_618, parameter_617, parameter_620, parameter_624, parameter_621, parameter_623, parameter_622, parameter_625, parameter_629, parameter_626, parameter_628, parameter_627, parameter_630, parameter_634, parameter_631, parameter_633, parameter_632, parameter_635, parameter_639, parameter_636, parameter_638, parameter_637, parameter_640, parameter_644, parameter_641, parameter_643, parameter_642, parameter_645, parameter_649, parameter_646, parameter_648, parameter_647, parameter_650, parameter_654, parameter_651, parameter_653, parameter_652, parameter_655, parameter_659, parameter_656, parameter_658, parameter_657, parameter_660, parameter_664, parameter_661, parameter_663, parameter_662, parameter_665, parameter_669, parameter_666, parameter_668, parameter_667, parameter_670, parameter_674, parameter_671, parameter_673, parameter_672, parameter_675, parameter_679, parameter_676, parameter_678, parameter_677, parameter_680, parameter_684, parameter_681, parameter_683, parameter_682, parameter_685, parameter_689, parameter_686, parameter_688, parameter_687, parameter_690, parameter_694, parameter_691, parameter_693, parameter_692, parameter_695, parameter_699, parameter_696, parameter_698, parameter_697, parameter_700, parameter_704, parameter_701, parameter_703, parameter_702, parameter_705, parameter_709, parameter_706, parameter_708, parameter_707, parameter_710, parameter_714, parameter_711, parameter_713, parameter_712, parameter_715, parameter_719, parameter_716, parameter_718, parameter_717, parameter_720, parameter_724, parameter_721, parameter_723, parameter_722, parameter_725, parameter_729, parameter_726, parameter_728, parameter_727, parameter_730, parameter_734, parameter_731, parameter_733, parameter_732, parameter_735, parameter_739, parameter_736, parameter_738, parameter_737, parameter_740, parameter_744, parameter_741, parameter_743, parameter_742, parameter_745, parameter_749, parameter_746, parameter_748, parameter_747, parameter_750, parameter_754, parameter_751, parameter_753, parameter_752, parameter_755, parameter_759, parameter_756, parameter_758, parameter_757, parameter_760, parameter_764, parameter_761, parameter_763, parameter_762, parameter_765, parameter_769, parameter_766, parameter_768, parameter_767, parameter_770, parameter_774, parameter_771, parameter_773, parameter_772, parameter_775, parameter_779, parameter_776, parameter_778, parameter_777, parameter_780, parameter_784, parameter_781, parameter_783, parameter_782, parameter_785, parameter_789, parameter_786, parameter_788, parameter_787, parameter_790, parameter_794, parameter_791, parameter_793, parameter_792, parameter_795, parameter_799, parameter_796, parameter_798, parameter_797, parameter_800, parameter_804, parameter_801, parameter_803, parameter_802, parameter_805, parameter_809, parameter_806, parameter_808, parameter_807, parameter_810, parameter_814, parameter_811, parameter_813, parameter_812, parameter_815, parameter_819, parameter_816, parameter_818, parameter_817, parameter_820, parameter_824, parameter_821, parameter_823, parameter_822, parameter_825, parameter_829, parameter_826, parameter_828, parameter_827, parameter_830, parameter_834, parameter_831, parameter_833, parameter_832, parameter_835, parameter_839, parameter_836, parameter_838, parameter_837, parameter_840, parameter_844, parameter_841, parameter_843, parameter_842, parameter_845, parameter_849, parameter_846, parameter_848, parameter_847, parameter_850, parameter_854, parameter_851, parameter_853, parameter_852, parameter_855, parameter_859, parameter_856, parameter_858, parameter_857, parameter_860, parameter_864, parameter_861, parameter_863, parameter_862, parameter_865, parameter_869, parameter_866, parameter_868, parameter_867, parameter_870, parameter_874, parameter_871, parameter_873, parameter_872, parameter_875, parameter_879, parameter_876, parameter_878, parameter_877, parameter_880, parameter_884, parameter_881, parameter_883, parameter_882, parameter_885, parameter_889, parameter_886, parameter_888, parameter_887, parameter_890, parameter_894, parameter_891, parameter_893, parameter_892, parameter_895, parameter_899, parameter_896, parameter_898, parameter_897, parameter_900, parameter_904, parameter_901, parameter_903, parameter_902, parameter_905, parameter_909, parameter_906, parameter_908, parameter_907, parameter_910, parameter_914, parameter_911, parameter_913, parameter_912, parameter_915, parameter_919, parameter_916, parameter_918, parameter_917, parameter_920, parameter_924, parameter_921, parameter_923, parameter_922, parameter_925, parameter_929, parameter_926, parameter_928, parameter_927, parameter_930, parameter_934, parameter_931, parameter_933, parameter_932, parameter_935, parameter_939, parameter_936, parameter_938, parameter_937, parameter_940, parameter_944, parameter_941, parameter_943, parameter_942, parameter_945, parameter_949, parameter_946, parameter_948, parameter_947, parameter_950, parameter_954, parameter_951, parameter_953, parameter_952, parameter_955, parameter_959, parameter_956, parameter_958, parameter_957, parameter_960, parameter_964, parameter_961, parameter_963, parameter_962, parameter_965, parameter_969, parameter_966, parameter_968, parameter_967, parameter_970, parameter_974, parameter_971, parameter_973, parameter_972, parameter_975, parameter_979, parameter_976, parameter_978, parameter_977, parameter_980, parameter_984, parameter_981, parameter_983, parameter_982, parameter_985, parameter_989, parameter_986, parameter_988, parameter_987, parameter_990, parameter_994, parameter_991, parameter_993, parameter_992, parameter_995, parameter_999, parameter_996, parameter_998, parameter_997, parameter_1000, parameter_1004, parameter_1001, parameter_1003, parameter_1002, parameter_1005, parameter_1009, parameter_1006, parameter_1008, parameter_1007, parameter_1010, parameter_1014, parameter_1011, parameter_1013, parameter_1012, parameter_1015, parameter_1019, parameter_1016, parameter_1018, parameter_1017, parameter_1020, parameter_1024, parameter_1021, parameter_1023, parameter_1022, parameter_1025, parameter_1029, parameter_1026, parameter_1028, parameter_1027, parameter_1030, parameter_1034, parameter_1031, parameter_1033, parameter_1032, parameter_1035, parameter_1039, parameter_1036, parameter_1038, parameter_1037, parameter_1040, parameter_1044, parameter_1041, parameter_1043, parameter_1042, parameter_1045, parameter_1049, parameter_1046, parameter_1048, parameter_1047, parameter_1050, parameter_1054, parameter_1051, parameter_1053, parameter_1052, parameter_1055, parameter_1059, parameter_1056, parameter_1058, parameter_1057, parameter_1060, parameter_1064, parameter_1061, parameter_1063, parameter_1062, parameter_1065, parameter_1069, parameter_1066, parameter_1068, parameter_1067, parameter_1070, parameter_1074, parameter_1071, parameter_1073, parameter_1072, parameter_1075, parameter_1079, parameter_1076, parameter_1078, parameter_1077, parameter_1080, parameter_1084, parameter_1081, parameter_1083, parameter_1082, parameter_1085, parameter_1089, parameter_1086, parameter_1088, parameter_1087, parameter_1090, parameter_1094, parameter_1091, parameter_1093, parameter_1092, parameter_1095, parameter_1099, parameter_1096, parameter_1098, parameter_1097, parameter_1100, parameter_1104, parameter_1101, parameter_1103, parameter_1102, parameter_1105, parameter_1109, parameter_1106, parameter_1108, parameter_1107, parameter_1110, parameter_1114, parameter_1111, parameter_1113, parameter_1112, parameter_1115, parameter_1119, parameter_1116, parameter_1118, parameter_1117, parameter_1120, parameter_1124, parameter_1121, parameter_1123, parameter_1122, parameter_1125, parameter_1129, parameter_1126, parameter_1128, parameter_1127, parameter_1130, parameter_1134, parameter_1131, parameter_1133, parameter_1132, parameter_1135, parameter_1139, parameter_1136, parameter_1138, parameter_1137, parameter_1140, parameter_1144, parameter_1141, parameter_1143, parameter_1142, parameter_1145, parameter_1149, parameter_1146, parameter_1148, parameter_1147, parameter_1150, parameter_1154, parameter_1151, parameter_1153, parameter_1152, parameter_1155, parameter_1159, parameter_1156, parameter_1158, parameter_1157, parameter_1160, parameter_1164, parameter_1161, parameter_1163, parameter_1162, parameter_1165, parameter_1169, parameter_1166, parameter_1168, parameter_1167, parameter_1170, parameter_1174, parameter_1171, parameter_1173, parameter_1172, parameter_1175, parameter_1179, parameter_1176, parameter_1178, parameter_1177, parameter_1180, parameter_1184, parameter_1181, parameter_1183, parameter_1182, parameter_1185, parameter_1189, parameter_1186, parameter_1188, parameter_1187, parameter_1190, parameter_1194, parameter_1191, parameter_1193, parameter_1192, parameter_1195, parameter_1199, parameter_1196, parameter_1198, parameter_1197, parameter_1200, parameter_1204, parameter_1201, parameter_1203, parameter_1202, parameter_1205, parameter_1209, parameter_1206, parameter_1208, parameter_1207, parameter_1210, parameter_1214, parameter_1211, parameter_1213, parameter_1212, parameter_1215, parameter_1219, parameter_1216, parameter_1218, parameter_1217, parameter_1220, parameter_1224, parameter_1221, parameter_1223, parameter_1222, parameter_1225, parameter_1229, parameter_1226, parameter_1228, parameter_1227, parameter_1230, parameter_1234, parameter_1231, parameter_1233, parameter_1232, parameter_1235, parameter_1239, parameter_1236, parameter_1238, parameter_1237, parameter_1240, parameter_1244, parameter_1241, parameter_1243, parameter_1242, parameter_1245, parameter_1249, parameter_1246, parameter_1248, parameter_1247, parameter_1250, parameter_1254, parameter_1251, parameter_1253, parameter_1252, parameter_1255, parameter_1259, parameter_1256, parameter_1258, parameter_1257, parameter_1260, parameter_1264, parameter_1261, parameter_1263, parameter_1262, parameter_1265, parameter_1269, parameter_1266, parameter_1268, parameter_1267, parameter_1270, parameter_1274, parameter_1271, parameter_1273, parameter_1272, parameter_1275, parameter_1279, parameter_1276, parameter_1278, parameter_1277, parameter_1280, parameter_1284, parameter_1281, parameter_1283, parameter_1282, parameter_1285, parameter_1289, parameter_1286, parameter_1288, parameter_1287, parameter_1290, parameter_1294, parameter_1291, parameter_1293, parameter_1292, parameter_1295, parameter_1299, parameter_1296, parameter_1298, parameter_1297, parameter_1300, parameter_1304, parameter_1301, parameter_1303, parameter_1302, parameter_1305, parameter_1309, parameter_1306, parameter_1308, parameter_1307, parameter_1310, parameter_1314, parameter_1311, parameter_1313, parameter_1312, parameter_1315, parameter_1319, parameter_1316, parameter_1318, parameter_1317, parameter_1320, parameter_1324, parameter_1321, parameter_1323, parameter_1322, parameter_1325, parameter_1329, parameter_1326, parameter_1328, parameter_1327, parameter_1330, parameter_1334, parameter_1331, parameter_1333, parameter_1332, parameter_1335, parameter_1339, parameter_1336, parameter_1338, parameter_1337, parameter_1340, parameter_1344, parameter_1341, parameter_1343, parameter_1342, parameter_1345, parameter_1349, parameter_1346, parameter_1348, parameter_1347, parameter_1350, parameter_1354, parameter_1351, parameter_1353, parameter_1352, parameter_1355, parameter_1359, parameter_1356, parameter_1358, parameter_1357, parameter_1360, parameter_1364, parameter_1361, parameter_1363, parameter_1362, parameter_1365, parameter_1369, parameter_1366, parameter_1368, parameter_1367, parameter_1370, parameter_1374, parameter_1371, parameter_1373, parameter_1372, parameter_1375, parameter_1379, parameter_1376, parameter_1378, parameter_1377, parameter_1380, parameter_1384, parameter_1381, parameter_1383, parameter_1382, parameter_1385, parameter_1389, parameter_1386, parameter_1388, parameter_1387, parameter_1390, parameter_1394, parameter_1391, parameter_1393, parameter_1392, parameter_1395, parameter_1399, parameter_1396, parameter_1398, parameter_1397, parameter_1400, parameter_1404, parameter_1401, parameter_1403, parameter_1402, parameter_1405, parameter_1409, parameter_1406, parameter_1408, parameter_1407, parameter_1410, parameter_1414, parameter_1411, parameter_1413, parameter_1412, parameter_1415, parameter_1419, parameter_1416, parameter_1418, parameter_1417, parameter_1420, parameter_1424, parameter_1421, parameter_1423, parameter_1422, parameter_1425, parameter_1429, parameter_1426, parameter_1428, parameter_1427, parameter_1430, parameter_1434, parameter_1431, parameter_1433, parameter_1432, parameter_1435, parameter_1439, parameter_1436, parameter_1438, parameter_1437, parameter_1440, parameter_1444, parameter_1441, parameter_1443, parameter_1442, parameter_1445, parameter_1449, parameter_1446, parameter_1448, parameter_1447, parameter_1450, parameter_1454, parameter_1451, parameter_1453, parameter_1452, parameter_1455, parameter_1459, parameter_1456, parameter_1458, parameter_1457, parameter_1460, parameter_1464, parameter_1461, parameter_1463, parameter_1462, parameter_1465, parameter_1469, parameter_1466, parameter_1468, parameter_1467, parameter_1470, parameter_1474, parameter_1471, parameter_1473, parameter_1472, parameter_1475, parameter_1479, parameter_1476, parameter_1478, parameter_1477, parameter_1480, parameter_1484, parameter_1481, parameter_1483, parameter_1482, parameter_1485, parameter_1489, parameter_1486, parameter_1488, parameter_1487, parameter_1490, parameter_1494, parameter_1491, parameter_1493, parameter_1492, parameter_1495, parameter_1499, parameter_1496, parameter_1498, parameter_1497, parameter_1500, parameter_1504, parameter_1501, parameter_1503, parameter_1502, parameter_1505, parameter_1509, parameter_1506, parameter_1508, parameter_1507, parameter_1510, parameter_1514, parameter_1511, parameter_1513, parameter_1512, parameter_1515, parameter_1519, parameter_1516, parameter_1518, parameter_1517, parameter_1520, parameter_1524, parameter_1521, parameter_1523, parameter_1522, parameter_1525, parameter_1529, parameter_1526, parameter_1528, parameter_1527, parameter_1530, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_2747_0_0(CinnTestBase, unittest.TestCase):
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
            paddle.uniform([64, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([64, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
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
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
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
            paddle.uniform([64, 64, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([256, 64, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([18, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([36, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_139
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_149
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_154
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([18, 36, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([36, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([72, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_189
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_209
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_214
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_213
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_215
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_219
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_218
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_217
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_224
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_221
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_223
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_225
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_229
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_227
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_234
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_233
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_239
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_238
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_244
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_241
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_243
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_245
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_249
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_246
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_248
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_250
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_254
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_251
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_253
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_252
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_255
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_259
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_256
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_258
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_257
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_264
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_261
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_263
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_262
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_265
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_269
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_266
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_268
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_267
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_270
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_274
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_271
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_273
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_272
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_275
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_279
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_276
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_278
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_277
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_280
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_284
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_281
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_283
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_282
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_285
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_289
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_286
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_288
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_287
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_290
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_294
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_291
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_293
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_292
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_295
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_299
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_296
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_298
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_297
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_300
            paddle.uniform([18, 36, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_304
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_301
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_303
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_302
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_305
            paddle.uniform([18, 72, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_309
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_306
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_308
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_307
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_310
            paddle.uniform([36, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_314
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_311
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_313
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_312
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_315
            paddle.uniform([36, 72, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_319
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_316
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_318
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_317
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_320
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_324
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_321
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_323
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_322
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_325
            paddle.uniform([72, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_329
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_326
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_328
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_327
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_330
            paddle.uniform([72, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_334
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_331
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_333
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_332
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_335
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_339
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_336
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_338
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_337
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_340
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_344
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_341
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_343
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_342
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_345
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_349
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_346
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_348
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_347
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_350
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_354
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_351
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_353
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_352
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_355
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_359
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_356
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_358
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_357
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_360
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_364
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_361
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_363
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_362
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_365
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_369
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_366
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_368
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_367
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_370
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_374
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_371
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_373
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_372
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_375
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_379
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_376
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_378
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_377
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_380
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_384
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_381
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_383
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_382
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_385
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_389
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_386
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_388
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_387
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_390
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_394
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_391
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_393
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_392
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_395
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_399
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_396
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_398
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_397
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_400
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_404
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_401
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_403
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_402
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_405
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_409
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_406
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_408
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_407
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_410
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_414
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_411
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_413
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_412
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_415
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_419
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_416
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_418
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_417
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_420
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_424
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_421
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_423
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_422
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_425
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_429
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_426
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_428
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_427
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_430
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_434
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_431
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_433
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_432
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_435
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_439
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_436
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_438
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_437
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_440
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_444
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_441
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_443
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_442
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_445
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_449
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_446
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_448
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_447
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_450
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_454
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_451
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_453
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_452
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_455
            paddle.uniform([18, 36, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_459
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_456
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_458
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_457
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_460
            paddle.uniform([18, 72, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_464
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_461
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_463
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_462
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_465
            paddle.uniform([36, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_469
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_466
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_468
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_467
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_470
            paddle.uniform([36, 72, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_474
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_471
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_473
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_472
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_475
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_479
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_476
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_478
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_477
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_480
            paddle.uniform([72, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_484
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_481
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_483
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_482
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_485
            paddle.uniform([72, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_489
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_486
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_488
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_487
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_490
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_494
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_491
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_493
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_492
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_495
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_499
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_496
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_498
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_497
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_500
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_504
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_501
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_503
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_502
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_505
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_509
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_506
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_508
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_507
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_510
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_514
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_511
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_513
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_512
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_515
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_519
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_516
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_518
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_517
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_520
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_524
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_521
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_523
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_522
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_525
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_529
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_526
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_528
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_527
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_530
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_534
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_531
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_533
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_532
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_535
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_539
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_536
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_538
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_537
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_540
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_544
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_541
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_543
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_542
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_545
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_549
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_546
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_548
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_547
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_550
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_554
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_551
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_553
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_552
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_555
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_559
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_556
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_558
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_557
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_560
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_564
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_561
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_563
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_562
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_565
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_569
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_566
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_568
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_567
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_570
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_574
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_571
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_573
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_572
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_575
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_579
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_576
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_578
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_577
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_580
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_584
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_581
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_583
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_582
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_585
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_589
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_586
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_588
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_587
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_590
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_594
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_591
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_593
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_592
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_595
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_599
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_596
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_598
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_597
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_600
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_604
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_601
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_603
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_602
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_605
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_609
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_606
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_608
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_607
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_610
            paddle.uniform([18, 36, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_614
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_611
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_613
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_612
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_615
            paddle.uniform([18, 72, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_619
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_616
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_618
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_617
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_620
            paddle.uniform([36, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_624
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_621
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_623
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_622
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_625
            paddle.uniform([36, 72, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_629
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_626
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_628
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_627
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_630
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_634
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_631
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_633
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_632
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_635
            paddle.uniform([72, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_639
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_636
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_638
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_637
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_640
            paddle.uniform([72, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_644
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_641
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_643
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_642
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_645
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_649
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_646
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_648
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_647
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_650
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_654
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_651
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_653
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_652
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_655
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_659
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_656
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_658
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_657
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_660
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_664
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_661
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_663
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_662
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_665
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_669
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_666
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_668
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_667
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_670
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_674
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_671
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_673
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_672
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_675
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_679
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_676
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_678
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_677
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_680
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_684
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_681
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_683
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_682
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_685
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_689
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_686
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_688
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_687
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_690
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_694
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_691
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_693
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_692
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_695
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_699
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_696
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_698
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_697
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_700
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_704
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_701
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_703
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_702
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_705
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_709
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_706
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_708
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_707
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_710
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_714
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_711
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_713
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_712
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_715
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_719
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_716
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_718
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_717
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_720
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_724
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_721
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_723
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_722
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_725
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_729
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_726
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_728
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_727
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_730
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_734
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_731
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_733
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_732
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_735
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_739
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_736
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_738
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_737
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_740
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_744
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_741
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_743
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_742
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_745
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_749
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_746
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_748
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_747
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_750
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_754
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_751
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_753
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_752
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_755
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_759
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_756
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_758
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_757
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_760
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_764
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_761
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_763
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_762
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_765
            paddle.uniform([18, 36, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_769
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_766
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_768
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_767
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_770
            paddle.uniform([18, 72, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_774
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_771
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_773
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_772
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_775
            paddle.uniform([36, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_779
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_776
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_778
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_777
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_780
            paddle.uniform([36, 72, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_784
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_781
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_783
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_782
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_785
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_789
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_786
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_788
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_787
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_790
            paddle.uniform([72, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_794
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_791
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_793
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_792
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_795
            paddle.uniform([72, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_799
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_796
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_798
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_797
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_800
            paddle.uniform([144, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_804
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_801
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_803
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_802
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_805
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_809
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_806
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_808
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_807
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_810
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_814
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_811
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_813
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_812
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_815
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_819
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_816
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_818
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_817
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_820
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_824
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_821
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_823
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_822
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_825
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_829
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_826
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_828
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_827
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_830
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_834
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_831
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_833
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_832
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_835
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_839
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_836
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_838
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_837
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_840
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_844
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_841
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_843
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_842
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_845
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_849
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_846
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_848
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_847
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_850
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_854
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_851
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_853
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_852
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_855
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_859
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_856
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_858
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_857
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_860
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_864
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_861
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_863
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_862
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_865
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_869
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_866
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_868
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_867
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_870
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_874
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_871
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_873
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_872
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_875
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_879
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_876
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_878
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_877
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_880
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_884
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_881
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_883
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_882
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_885
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_889
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_886
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_888
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_887
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_890
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_894
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_891
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_893
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_892
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_895
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_899
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_896
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_898
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_897
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_900
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_904
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_901
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_903
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_902
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_905
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_909
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_906
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_908
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_907
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_910
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_914
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_911
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_913
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_912
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_915
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_919
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_916
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_918
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_917
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_920
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_924
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_921
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_923
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_922
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_925
            paddle.uniform([144, 144, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_929
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_926
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_928
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_927
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_930
            paddle.uniform([144, 144, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_934
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_931
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_933
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_932
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_935
            paddle.uniform([144, 144, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_939
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_936
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_938
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_937
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_940
            paddle.uniform([144, 144, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_944
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_941
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_943
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_942
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_945
            paddle.uniform([144, 144, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_949
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_946
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_948
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_947
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_950
            paddle.uniform([144, 144, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_954
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_951
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_953
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_952
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_955
            paddle.uniform([144, 144, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_959
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_956
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_958
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_957
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_960
            paddle.uniform([144, 144, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_964
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_961
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_963
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_962
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_965
            paddle.uniform([18, 36, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_969
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_966
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_968
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_967
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_970
            paddle.uniform([18, 72, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_974
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_971
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_973
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_972
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_975
            paddle.uniform([18, 144, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_979
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_976
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_978
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_977
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_980
            paddle.uniform([36, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_984
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_981
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_983
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_982
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_985
            paddle.uniform([36, 72, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_989
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_986
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_988
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_987
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_990
            paddle.uniform([36, 144, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_994
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_991
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_993
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_992
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_995
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_999
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_996
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_998
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_997
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1000
            paddle.uniform([72, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1004
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1001
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1003
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1002
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1005
            paddle.uniform([72, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1009
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1006
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1008
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1007
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1010
            paddle.uniform([72, 144, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_1014
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1011
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1013
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1012
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1015
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1019
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1016
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1018
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1017
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1020
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1024
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1021
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1023
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1022
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1025
            paddle.uniform([144, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1029
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1026
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1028
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1027
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1030
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1034
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1031
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1033
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1032
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1035
            paddle.uniform([144, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1039
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1036
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1038
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1037
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1040
            paddle.uniform([144, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1044
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1041
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1043
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1042
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1045
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1049
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1046
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1048
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1047
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1050
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1054
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1051
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1053
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1052
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1055
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1059
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1056
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1058
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1057
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1060
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1064
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1061
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1063
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1062
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1065
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1069
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1066
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1068
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1067
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1070
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1074
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1071
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1073
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1072
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1075
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1079
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1076
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1078
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1077
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1080
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1084
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1081
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1083
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1082
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1085
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1089
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1086
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1088
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1087
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1090
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1094
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1091
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1093
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1092
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1095
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1099
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1096
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1098
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1097
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1100
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1104
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1101
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1103
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1102
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1105
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1109
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1106
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1108
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1107
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1110
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1114
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1111
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1113
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1112
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1115
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1119
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1116
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1118
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1117
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1120
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1124
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1121
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1123
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1122
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1125
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1129
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1126
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1128
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1127
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1130
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1134
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1131
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1133
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1132
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1135
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1139
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1136
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1138
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1137
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1140
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1144
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1141
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1143
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1142
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1145
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1149
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1146
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1148
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1147
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1150
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1154
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1151
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1153
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1152
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1155
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1159
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1156
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1158
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1157
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1160
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1164
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1161
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1163
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1162
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1165
            paddle.uniform([144, 144, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1169
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1166
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1168
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1167
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1170
            paddle.uniform([144, 144, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1174
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1171
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1173
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1172
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1175
            paddle.uniform([144, 144, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1179
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1176
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1178
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1177
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1180
            paddle.uniform([144, 144, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1184
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1181
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1183
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1182
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1185
            paddle.uniform([144, 144, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1189
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1186
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1188
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1187
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1190
            paddle.uniform([144, 144, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1194
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1191
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1193
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1192
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1195
            paddle.uniform([144, 144, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1199
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1196
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1198
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1197
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1200
            paddle.uniform([144, 144, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1204
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1201
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1203
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1202
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1205
            paddle.uniform([18, 36, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_1209
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1206
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1208
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1207
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1210
            paddle.uniform([18, 72, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_1214
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1211
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1213
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1212
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1215
            paddle.uniform([18, 144, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_1219
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1216
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1218
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1217
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1220
            paddle.uniform([36, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1224
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1221
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1223
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1222
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1225
            paddle.uniform([36, 72, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_1229
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1226
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1228
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1227
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1230
            paddle.uniform([36, 144, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_1234
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1231
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1233
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1232
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1235
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1239
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1236
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1238
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1237
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1240
            paddle.uniform([72, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1244
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1241
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1243
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1242
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1245
            paddle.uniform([72, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1249
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1246
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1248
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1247
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1250
            paddle.uniform([72, 144, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_1254
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1251
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1253
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1252
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1255
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1259
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1256
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1258
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1257
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1260
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1264
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1261
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1263
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1262
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1265
            paddle.uniform([144, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1269
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1266
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1268
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1267
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1270
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1274
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1271
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1273
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1272
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1275
            paddle.uniform([144, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1279
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1276
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1278
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1277
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1280
            paddle.uniform([144, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1284
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1281
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1283
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1282
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1285
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1289
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1286
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1288
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1287
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1290
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1294
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1291
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1293
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1292
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1295
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1299
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1296
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1298
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1297
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1300
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1304
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1301
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1303
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1302
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1305
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1309
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1306
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1308
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1307
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1310
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1314
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1311
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1313
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1312
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1315
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1319
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1316
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1318
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1317
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1320
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1324
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1321
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1323
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1322
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1325
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1329
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1326
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1328
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1327
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1330
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1334
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1331
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1333
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1332
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1335
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1339
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1336
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1338
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1337
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1340
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1344
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1341
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1343
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1342
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1345
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1349
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1346
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1348
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1347
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1350
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1354
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1351
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1353
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1352
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1355
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1359
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1356
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1358
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1357
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1360
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1364
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1361
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1363
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1362
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1365
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1369
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1366
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1368
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1367
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1370
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1374
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1371
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1373
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1372
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1375
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1379
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1376
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1378
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1377
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1380
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1384
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1381
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1383
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1382
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1385
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1389
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1386
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1388
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1387
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1390
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1394
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1391
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1393
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1392
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1395
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1399
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1396
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1398
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1397
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1400
            paddle.uniform([72, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1404
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1401
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1403
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1402
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1405
            paddle.uniform([144, 144, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1409
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1406
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1408
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1407
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1410
            paddle.uniform([144, 144, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1414
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1411
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1413
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1412
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1415
            paddle.uniform([144, 144, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1419
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1416
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1418
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1417
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1420
            paddle.uniform([144, 144, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1424
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1421
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1423
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1422
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1425
            paddle.uniform([144, 144, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1429
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1426
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1428
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1427
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1430
            paddle.uniform([144, 144, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1434
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1431
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1433
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1432
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1435
            paddle.uniform([144, 144, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1439
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1436
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1438
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1437
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1440
            paddle.uniform([144, 144, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1444
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1441
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1443
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1442
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1445
            paddle.uniform([18, 36, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_1449
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1446
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1448
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1447
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1450
            paddle.uniform([18, 72, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_1454
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1451
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1453
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1452
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1455
            paddle.uniform([18, 144, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_1459
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1456
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1458
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1457
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1460
            paddle.uniform([36, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1464
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1461
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1463
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1462
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1465
            paddle.uniform([36, 72, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_1469
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1466
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1468
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1467
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1470
            paddle.uniform([36, 144, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_1474
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1471
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1473
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1472
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1475
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1479
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1476
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1478
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1477
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1480
            paddle.uniform([72, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1484
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1481
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1483
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1482
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1485
            paddle.uniform([72, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1489
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1486
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1488
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1487
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1490
            paddle.uniform([72, 144, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_1494
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1491
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1493
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1492
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            # parameter_1495
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1499
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1496
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1498
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1497
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1500
            paddle.uniform([18, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1504
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1501
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1503
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1502
            paddle.uniform([18], dtype='float32', min=0, max=0.5),
            # parameter_1505
            paddle.uniform([144, 18, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1509
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1506
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1508
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1507
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1510
            paddle.uniform([36, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1514
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1511
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1513
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1512
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            # parameter_1515
            paddle.uniform([144, 36, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1519
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1516
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1518
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1517
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1520
            paddle.uniform([144, 72, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1524
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1521
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1523
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1522
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_1525
            paddle.uniform([270, 270, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_1529
            paddle.uniform([270], dtype='float32', min=0, max=0.5),
            # parameter_1526
            paddle.uniform([270], dtype='float32', min=0, max=0.5),
            # parameter_1528
            paddle.uniform([270], dtype='float32', min=0, max=0.5),
            # parameter_1527
            paddle.uniform([270], dtype='float32', min=0, max=0.5),
            # parameter_1530
            paddle.uniform([19, 270, 1, 1], dtype='float32', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 3, 512, 1024], dtype='float32', min=0, max=0.5),
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
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[64, 256, 1, 1], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float32'),
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
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float32'),
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
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[256, 64, 1, 1], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[18, 256, 3, 3], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[36, 256, 3, 3], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_139
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_149
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_154
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[18, 36, 1, 1], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[36, 18, 3, 3], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[72, 36, 3, 3], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_189
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_209
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_214
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_213
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_215
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_219
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_218
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_217
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_224
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_221
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_223
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_225
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_229
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_227
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_234
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_233
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_239
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_238
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_244
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_241
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_243
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_245
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_249
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_246
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_248
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_250
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_254
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_251
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_253
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_252
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_255
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_259
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_256
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_258
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_257
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_264
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_261
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_263
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_262
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_265
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_269
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_266
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_268
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_267
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_270
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_274
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_271
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_273
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_272
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_275
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_279
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_276
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_278
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_277
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_280
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_284
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_281
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_283
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_282
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_285
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_289
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_286
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_288
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_287
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_290
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_294
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_291
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_293
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_292
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_295
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_299
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_296
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_298
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_297
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_300
            paddle.static.InputSpec(shape=[18, 36, 1, 1], dtype='float32'),
            # parameter_304
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_301
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_303
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_302
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_305
            paddle.static.InputSpec(shape=[18, 72, 1, 1], dtype='float32'),
            # parameter_309
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_306
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_308
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_307
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_310
            paddle.static.InputSpec(shape=[36, 18, 3, 3], dtype='float32'),
            # parameter_314
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_311
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_313
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_312
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_315
            paddle.static.InputSpec(shape=[36, 72, 1, 1], dtype='float32'),
            # parameter_319
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_316
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_318
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_317
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_320
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_324
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_321
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_323
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_322
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_325
            paddle.static.InputSpec(shape=[72, 18, 3, 3], dtype='float32'),
            # parameter_329
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_326
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_328
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_327
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_330
            paddle.static.InputSpec(shape=[72, 36, 3, 3], dtype='float32'),
            # parameter_334
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_331
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_333
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_332
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_335
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_339
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_336
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_338
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_337
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_340
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_344
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_341
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_343
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_342
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_345
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_349
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_346
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_348
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_347
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_350
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_354
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_351
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_353
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_352
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_355
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_359
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_356
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_358
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_357
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_360
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_364
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_361
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_363
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_362
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_365
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_369
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_366
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_368
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_367
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_370
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_374
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_371
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_373
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_372
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_375
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_379
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_376
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_378
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_377
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_380
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_384
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_381
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_383
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_382
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_385
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_389
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_386
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_388
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_387
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_390
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_394
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_391
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_393
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_392
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_395
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_399
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_396
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_398
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_397
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_400
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_404
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_401
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_403
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_402
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_405
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_409
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_406
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_408
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_407
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_410
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_414
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_411
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_413
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_412
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_415
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_419
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_416
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_418
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_417
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_420
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_424
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_421
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_423
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_422
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_425
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_429
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_426
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_428
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_427
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_430
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_434
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_431
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_433
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_432
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_435
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_439
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_436
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_438
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_437
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_440
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_444
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_441
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_443
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_442
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_445
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_449
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_446
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_448
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_447
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_450
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_454
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_451
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_453
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_452
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_455
            paddle.static.InputSpec(shape=[18, 36, 1, 1], dtype='float32'),
            # parameter_459
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_456
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_458
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_457
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_460
            paddle.static.InputSpec(shape=[18, 72, 1, 1], dtype='float32'),
            # parameter_464
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_461
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_463
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_462
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_465
            paddle.static.InputSpec(shape=[36, 18, 3, 3], dtype='float32'),
            # parameter_469
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_466
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_468
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_467
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_470
            paddle.static.InputSpec(shape=[36, 72, 1, 1], dtype='float32'),
            # parameter_474
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_471
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_473
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_472
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_475
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_479
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_476
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_478
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_477
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_480
            paddle.static.InputSpec(shape=[72, 18, 3, 3], dtype='float32'),
            # parameter_484
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_481
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_483
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_482
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_485
            paddle.static.InputSpec(shape=[72, 36, 3, 3], dtype='float32'),
            # parameter_489
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_486
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_488
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_487
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_490
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_494
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_491
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_493
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_492
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_495
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_499
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_496
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_498
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_497
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_500
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_504
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_501
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_503
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_502
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_505
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_509
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_506
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_508
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_507
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_510
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_514
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_511
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_513
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_512
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_515
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_519
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_516
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_518
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_517
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_520
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_524
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_521
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_523
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_522
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_525
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_529
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_526
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_528
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_527
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_530
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_534
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_531
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_533
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_532
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_535
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_539
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_536
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_538
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_537
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_540
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_544
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_541
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_543
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_542
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_545
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_549
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_546
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_548
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_547
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_550
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_554
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_551
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_553
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_552
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_555
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_559
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_556
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_558
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_557
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_560
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_564
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_561
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_563
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_562
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_565
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_569
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_566
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_568
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_567
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_570
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_574
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_571
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_573
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_572
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_575
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_579
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_576
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_578
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_577
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_580
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_584
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_581
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_583
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_582
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_585
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_589
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_586
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_588
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_587
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_590
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_594
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_591
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_593
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_592
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_595
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_599
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_596
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_598
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_597
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_600
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_604
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_601
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_603
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_602
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_605
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_609
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_606
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_608
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_607
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_610
            paddle.static.InputSpec(shape=[18, 36, 1, 1], dtype='float32'),
            # parameter_614
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_611
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_613
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_612
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_615
            paddle.static.InputSpec(shape=[18, 72, 1, 1], dtype='float32'),
            # parameter_619
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_616
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_618
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_617
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_620
            paddle.static.InputSpec(shape=[36, 18, 3, 3], dtype='float32'),
            # parameter_624
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_621
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_623
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_622
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_625
            paddle.static.InputSpec(shape=[36, 72, 1, 1], dtype='float32'),
            # parameter_629
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_626
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_628
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_627
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_630
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_634
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_631
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_633
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_632
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_635
            paddle.static.InputSpec(shape=[72, 18, 3, 3], dtype='float32'),
            # parameter_639
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_636
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_638
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_637
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_640
            paddle.static.InputSpec(shape=[72, 36, 3, 3], dtype='float32'),
            # parameter_644
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_641
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_643
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_642
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_645
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_649
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_646
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_648
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_647
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_650
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_654
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_651
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_653
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_652
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_655
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_659
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_656
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_658
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_657
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_660
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_664
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_661
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_663
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_662
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_665
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_669
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_666
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_668
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_667
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_670
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_674
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_671
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_673
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_672
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_675
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_679
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_676
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_678
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_677
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_680
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_684
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_681
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_683
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_682
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_685
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_689
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_686
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_688
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_687
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_690
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_694
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_691
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_693
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_692
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_695
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_699
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_696
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_698
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_697
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_700
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_704
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_701
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_703
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_702
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_705
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_709
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_706
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_708
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_707
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_710
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_714
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_711
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_713
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_712
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_715
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_719
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_716
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_718
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_717
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_720
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_724
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_721
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_723
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_722
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_725
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_729
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_726
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_728
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_727
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_730
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_734
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_731
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_733
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_732
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_735
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_739
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_736
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_738
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_737
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_740
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_744
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_741
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_743
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_742
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_745
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_749
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_746
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_748
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_747
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_750
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_754
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_751
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_753
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_752
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_755
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_759
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_756
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_758
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_757
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_760
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_764
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_761
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_763
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_762
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_765
            paddle.static.InputSpec(shape=[18, 36, 1, 1], dtype='float32'),
            # parameter_769
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_766
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_768
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_767
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_770
            paddle.static.InputSpec(shape=[18, 72, 1, 1], dtype='float32'),
            # parameter_774
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_771
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_773
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_772
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_775
            paddle.static.InputSpec(shape=[36, 18, 3, 3], dtype='float32'),
            # parameter_779
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_776
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_778
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_777
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_780
            paddle.static.InputSpec(shape=[36, 72, 1, 1], dtype='float32'),
            # parameter_784
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_781
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_783
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_782
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_785
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_789
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_786
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_788
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_787
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_790
            paddle.static.InputSpec(shape=[72, 18, 3, 3], dtype='float32'),
            # parameter_794
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_791
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_793
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_792
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_795
            paddle.static.InputSpec(shape=[72, 36, 3, 3], dtype='float32'),
            # parameter_799
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_796
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_798
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_797
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_800
            paddle.static.InputSpec(shape=[144, 72, 3, 3], dtype='float32'),
            # parameter_804
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_801
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_803
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_802
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_805
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_809
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_806
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_808
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_807
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_810
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_814
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_811
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_813
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_812
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_815
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_819
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_816
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_818
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_817
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_820
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_824
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_821
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_823
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_822
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_825
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_829
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_826
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_828
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_827
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_830
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_834
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_831
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_833
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_832
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_835
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_839
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_836
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_838
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_837
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_840
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_844
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_841
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_843
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_842
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_845
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_849
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_846
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_848
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_847
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_850
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_854
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_851
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_853
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_852
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_855
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_859
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_856
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_858
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_857
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_860
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_864
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_861
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_863
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_862
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_865
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_869
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_866
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_868
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_867
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_870
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_874
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_871
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_873
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_872
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_875
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_879
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_876
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_878
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_877
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_880
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_884
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_881
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_883
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_882
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_885
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_889
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_886
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_888
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_887
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_890
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_894
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_891
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_893
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_892
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_895
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_899
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_896
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_898
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_897
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_900
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_904
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_901
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_903
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_902
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_905
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_909
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_906
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_908
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_907
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_910
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_914
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_911
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_913
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_912
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_915
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_919
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_916
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_918
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_917
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_920
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_924
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_921
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_923
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_922
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_925
            paddle.static.InputSpec(shape=[144, 144, 3, 3], dtype='float32'),
            # parameter_929
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_926
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_928
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_927
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_930
            paddle.static.InputSpec(shape=[144, 144, 3, 3], dtype='float32'),
            # parameter_934
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_931
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_933
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_932
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_935
            paddle.static.InputSpec(shape=[144, 144, 3, 3], dtype='float32'),
            # parameter_939
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_936
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_938
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_937
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_940
            paddle.static.InputSpec(shape=[144, 144, 3, 3], dtype='float32'),
            # parameter_944
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_941
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_943
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_942
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_945
            paddle.static.InputSpec(shape=[144, 144, 3, 3], dtype='float32'),
            # parameter_949
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_946
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_948
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_947
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_950
            paddle.static.InputSpec(shape=[144, 144, 3, 3], dtype='float32'),
            # parameter_954
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_951
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_953
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_952
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_955
            paddle.static.InputSpec(shape=[144, 144, 3, 3], dtype='float32'),
            # parameter_959
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_956
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_958
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_957
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_960
            paddle.static.InputSpec(shape=[144, 144, 3, 3], dtype='float32'),
            # parameter_964
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_961
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_963
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_962
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_965
            paddle.static.InputSpec(shape=[18, 36, 1, 1], dtype='float32'),
            # parameter_969
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_966
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_968
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_967
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_970
            paddle.static.InputSpec(shape=[18, 72, 1, 1], dtype='float32'),
            # parameter_974
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_971
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_973
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_972
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_975
            paddle.static.InputSpec(shape=[18, 144, 1, 1], dtype='float32'),
            # parameter_979
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_976
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_978
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_977
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_980
            paddle.static.InputSpec(shape=[36, 18, 3, 3], dtype='float32'),
            # parameter_984
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_981
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_983
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_982
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_985
            paddle.static.InputSpec(shape=[36, 72, 1, 1], dtype='float32'),
            # parameter_989
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_986
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_988
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_987
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_990
            paddle.static.InputSpec(shape=[36, 144, 1, 1], dtype='float32'),
            # parameter_994
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_991
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_993
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_992
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_995
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_999
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_996
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_998
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_997
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1000
            paddle.static.InputSpec(shape=[72, 18, 3, 3], dtype='float32'),
            # parameter_1004
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1001
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1003
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1002
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1005
            paddle.static.InputSpec(shape=[72, 36, 3, 3], dtype='float32'),
            # parameter_1009
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1006
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1008
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1007
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1010
            paddle.static.InputSpec(shape=[72, 144, 1, 1], dtype='float32'),
            # parameter_1014
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1011
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1013
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1012
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1015
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_1019
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1016
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1018
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1017
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1020
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_1024
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1021
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1023
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1022
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1025
            paddle.static.InputSpec(shape=[144, 18, 3, 3], dtype='float32'),
            # parameter_1029
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1026
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1028
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1027
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1030
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_1034
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1031
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1033
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1032
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1035
            paddle.static.InputSpec(shape=[144, 36, 3, 3], dtype='float32'),
            # parameter_1039
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1036
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1038
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1037
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1040
            paddle.static.InputSpec(shape=[144, 72, 3, 3], dtype='float32'),
            # parameter_1044
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1041
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1043
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1042
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1045
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_1049
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1046
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1048
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1047
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1050
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_1054
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1051
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1053
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1052
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1055
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_1059
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1056
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1058
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1057
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1060
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_1064
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1061
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1063
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1062
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1065
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_1069
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1066
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1068
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1067
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1070
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_1074
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1071
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1073
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1072
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1075
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_1079
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1076
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1078
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1077
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1080
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_1084
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1081
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1083
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1082
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1085
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_1089
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1086
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1088
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1087
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1090
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_1094
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1091
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1093
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1092
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1095
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_1099
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1096
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1098
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1097
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1100
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_1104
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1101
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1103
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1102
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1105
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_1109
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1106
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1108
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1107
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1110
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_1114
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1111
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1113
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1112
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1115
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_1119
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1116
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1118
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1117
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1120
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_1124
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1121
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1123
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1122
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1125
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_1129
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1126
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1128
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1127
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1130
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_1134
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1131
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1133
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1132
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1135
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_1139
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1136
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1138
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1137
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1140
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_1144
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1141
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1143
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1142
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1145
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_1149
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1146
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1148
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1147
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1150
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_1154
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1151
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1153
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1152
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1155
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_1159
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1156
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1158
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1157
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1160
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_1164
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1161
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1163
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1162
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1165
            paddle.static.InputSpec(shape=[144, 144, 3, 3], dtype='float32'),
            # parameter_1169
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1166
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1168
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1167
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1170
            paddle.static.InputSpec(shape=[144, 144, 3, 3], dtype='float32'),
            # parameter_1174
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1171
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1173
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1172
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1175
            paddle.static.InputSpec(shape=[144, 144, 3, 3], dtype='float32'),
            # parameter_1179
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1176
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1178
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1177
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1180
            paddle.static.InputSpec(shape=[144, 144, 3, 3], dtype='float32'),
            # parameter_1184
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1181
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1183
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1182
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1185
            paddle.static.InputSpec(shape=[144, 144, 3, 3], dtype='float32'),
            # parameter_1189
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1186
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1188
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1187
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1190
            paddle.static.InputSpec(shape=[144, 144, 3, 3], dtype='float32'),
            # parameter_1194
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1191
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1193
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1192
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1195
            paddle.static.InputSpec(shape=[144, 144, 3, 3], dtype='float32'),
            # parameter_1199
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1196
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1198
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1197
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1200
            paddle.static.InputSpec(shape=[144, 144, 3, 3], dtype='float32'),
            # parameter_1204
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1201
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1203
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1202
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1205
            paddle.static.InputSpec(shape=[18, 36, 1, 1], dtype='float32'),
            # parameter_1209
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1206
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1208
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1207
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1210
            paddle.static.InputSpec(shape=[18, 72, 1, 1], dtype='float32'),
            # parameter_1214
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1211
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1213
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1212
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1215
            paddle.static.InputSpec(shape=[18, 144, 1, 1], dtype='float32'),
            # parameter_1219
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1216
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1218
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1217
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1220
            paddle.static.InputSpec(shape=[36, 18, 3, 3], dtype='float32'),
            # parameter_1224
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1221
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1223
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1222
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1225
            paddle.static.InputSpec(shape=[36, 72, 1, 1], dtype='float32'),
            # parameter_1229
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1226
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1228
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1227
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1230
            paddle.static.InputSpec(shape=[36, 144, 1, 1], dtype='float32'),
            # parameter_1234
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1231
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1233
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1232
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1235
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_1239
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1236
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1238
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1237
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1240
            paddle.static.InputSpec(shape=[72, 18, 3, 3], dtype='float32'),
            # parameter_1244
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1241
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1243
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1242
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1245
            paddle.static.InputSpec(shape=[72, 36, 3, 3], dtype='float32'),
            # parameter_1249
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1246
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1248
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1247
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1250
            paddle.static.InputSpec(shape=[72, 144, 1, 1], dtype='float32'),
            # parameter_1254
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1251
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1253
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1252
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1255
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_1259
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1256
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1258
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1257
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1260
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_1264
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1261
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1263
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1262
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1265
            paddle.static.InputSpec(shape=[144, 18, 3, 3], dtype='float32'),
            # parameter_1269
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1266
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1268
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1267
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1270
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_1274
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1271
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1273
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1272
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1275
            paddle.static.InputSpec(shape=[144, 36, 3, 3], dtype='float32'),
            # parameter_1279
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1276
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1278
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1277
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1280
            paddle.static.InputSpec(shape=[144, 72, 3, 3], dtype='float32'),
            # parameter_1284
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1281
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1283
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1282
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1285
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_1289
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1286
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1288
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1287
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1290
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_1294
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1291
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1293
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1292
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1295
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_1299
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1296
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1298
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1297
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1300
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_1304
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1301
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1303
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1302
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1305
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_1309
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1306
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1308
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1307
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1310
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_1314
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1311
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1313
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1312
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1315
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_1319
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1316
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1318
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1317
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1320
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_1324
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1321
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1323
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1322
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1325
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_1329
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1326
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1328
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1327
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1330
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_1334
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1331
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1333
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1332
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1335
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_1339
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1336
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1338
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1337
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1340
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_1344
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1341
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1343
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1342
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1345
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_1349
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1346
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1348
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1347
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1350
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_1354
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1351
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1353
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1352
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1355
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_1359
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1356
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1358
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1357
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1360
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_1364
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1361
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1363
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1362
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1365
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_1369
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1366
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1368
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1367
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1370
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_1374
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1371
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1373
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1372
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1375
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_1379
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1376
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1378
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1377
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1380
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_1384
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1381
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1383
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1382
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1385
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_1389
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1386
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1388
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1387
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1390
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_1394
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1391
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1393
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1392
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1395
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_1399
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1396
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1398
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1397
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1400
            paddle.static.InputSpec(shape=[72, 72, 3, 3], dtype='float32'),
            # parameter_1404
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1401
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1403
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1402
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1405
            paddle.static.InputSpec(shape=[144, 144, 3, 3], dtype='float32'),
            # parameter_1409
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1406
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1408
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1407
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1410
            paddle.static.InputSpec(shape=[144, 144, 3, 3], dtype='float32'),
            # parameter_1414
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1411
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1413
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1412
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1415
            paddle.static.InputSpec(shape=[144, 144, 3, 3], dtype='float32'),
            # parameter_1419
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1416
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1418
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1417
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1420
            paddle.static.InputSpec(shape=[144, 144, 3, 3], dtype='float32'),
            # parameter_1424
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1421
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1423
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1422
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1425
            paddle.static.InputSpec(shape=[144, 144, 3, 3], dtype='float32'),
            # parameter_1429
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1426
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1428
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1427
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1430
            paddle.static.InputSpec(shape=[144, 144, 3, 3], dtype='float32'),
            # parameter_1434
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1431
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1433
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1432
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1435
            paddle.static.InputSpec(shape=[144, 144, 3, 3], dtype='float32'),
            # parameter_1439
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1436
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1438
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1437
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1440
            paddle.static.InputSpec(shape=[144, 144, 3, 3], dtype='float32'),
            # parameter_1444
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1441
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1443
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1442
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1445
            paddle.static.InputSpec(shape=[18, 36, 1, 1], dtype='float32'),
            # parameter_1449
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1446
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1448
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1447
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1450
            paddle.static.InputSpec(shape=[18, 72, 1, 1], dtype='float32'),
            # parameter_1454
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1451
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1453
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1452
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1455
            paddle.static.InputSpec(shape=[18, 144, 1, 1], dtype='float32'),
            # parameter_1459
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1456
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1458
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1457
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1460
            paddle.static.InputSpec(shape=[36, 18, 3, 3], dtype='float32'),
            # parameter_1464
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1461
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1463
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1462
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1465
            paddle.static.InputSpec(shape=[36, 72, 1, 1], dtype='float32'),
            # parameter_1469
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1466
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1468
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1467
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1470
            paddle.static.InputSpec(shape=[36, 144, 1, 1], dtype='float32'),
            # parameter_1474
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1471
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1473
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1472
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1475
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_1479
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1476
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1478
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1477
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1480
            paddle.static.InputSpec(shape=[72, 18, 3, 3], dtype='float32'),
            # parameter_1484
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1481
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1483
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1482
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1485
            paddle.static.InputSpec(shape=[72, 36, 3, 3], dtype='float32'),
            # parameter_1489
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1486
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1488
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1487
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1490
            paddle.static.InputSpec(shape=[72, 144, 1, 1], dtype='float32'),
            # parameter_1494
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1491
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1493
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1492
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            # parameter_1495
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_1499
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1496
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1498
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1497
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1500
            paddle.static.InputSpec(shape=[18, 18, 3, 3], dtype='float32'),
            # parameter_1504
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1501
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1503
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1502
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            # parameter_1505
            paddle.static.InputSpec(shape=[144, 18, 3, 3], dtype='float32'),
            # parameter_1509
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1506
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1508
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1507
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1510
            paddle.static.InputSpec(shape=[36, 36, 3, 3], dtype='float32'),
            # parameter_1514
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1511
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1513
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1512
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            # parameter_1515
            paddle.static.InputSpec(shape=[144, 36, 3, 3], dtype='float32'),
            # parameter_1519
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1516
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1518
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1517
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1520
            paddle.static.InputSpec(shape=[144, 72, 3, 3], dtype='float32'),
            # parameter_1524
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1521
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1523
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1522
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_1525
            paddle.static.InputSpec(shape=[270, 270, 1, 1], dtype='float32'),
            # parameter_1529
            paddle.static.InputSpec(shape=[270], dtype='float32'),
            # parameter_1526
            paddle.static.InputSpec(shape=[270], dtype='float32'),
            # parameter_1528
            paddle.static.InputSpec(shape=[270], dtype='float32'),
            # parameter_1527
            paddle.static.InputSpec(shape=[270], dtype='float32'),
            # parameter_1530
            paddle.static.InputSpec(shape=[19, 270, 1, 1], dtype='float32'),
            # feed_0
            paddle.static.InputSpec(shape=[None, 3, None, None], dtype='float32'),
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