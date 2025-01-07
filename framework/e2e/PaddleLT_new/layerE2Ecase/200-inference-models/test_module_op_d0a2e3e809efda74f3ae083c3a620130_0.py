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
    return [1394][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_1764_0_0(self, parameter_0, parameter_1, parameter_2, parameter_4, parameter_3, parameter_5, parameter_6, parameter_8, parameter_7, parameter_9, parameter_10, parameter_12, parameter_11, parameter_13, parameter_14, parameter_15, parameter_16, parameter_18, parameter_17, parameter_19, parameter_20, parameter_21, parameter_22, parameter_24, parameter_23, parameter_25, parameter_27, parameter_26, parameter_29, parameter_28, parameter_30, parameter_31, parameter_32, parameter_33, parameter_35, parameter_34, parameter_36, parameter_37, parameter_38, parameter_39, parameter_41, parameter_40, parameter_42, parameter_43, parameter_44, parameter_45, parameter_47, parameter_46, parameter_48, parameter_49, parameter_50, parameter_51, parameter_53, parameter_52, parameter_54, parameter_56, parameter_55, parameter_58, parameter_57, parameter_59, parameter_60, parameter_61, parameter_62, parameter_64, parameter_63, parameter_65, parameter_66, parameter_67, parameter_68, parameter_70, parameter_69, parameter_71, parameter_72, parameter_73, parameter_74, parameter_76, parameter_75, parameter_77, parameter_78, parameter_79, parameter_80, parameter_82, parameter_81, parameter_83, parameter_85, parameter_84, parameter_87, parameter_86, parameter_88, parameter_89, parameter_90, parameter_91, parameter_93, parameter_92, parameter_94, parameter_95, parameter_96, parameter_97, parameter_99, parameter_98, parameter_100, parameter_101, parameter_102, parameter_103, parameter_105, parameter_104, parameter_106, parameter_107, parameter_108, parameter_109, parameter_111, parameter_110, parameter_112, parameter_114, parameter_113, parameter_116, parameter_115, parameter_117, parameter_118, parameter_119, parameter_120, parameter_122, parameter_121, parameter_123, parameter_124, parameter_125, parameter_126, parameter_128, parameter_127, parameter_129, parameter_130, parameter_131, parameter_132, parameter_134, parameter_133, parameter_135, parameter_136, parameter_137, parameter_138, parameter_140, parameter_139, parameter_141, parameter_143, parameter_142, parameter_145, parameter_144, parameter_146, parameter_147, parameter_148, parameter_149, parameter_151, parameter_150, parameter_152, parameter_153, parameter_154, parameter_155, parameter_157, parameter_156, parameter_158, parameter_159, parameter_160, parameter_161, parameter_163, parameter_162, parameter_164, parameter_165, parameter_166, parameter_167, parameter_169, parameter_168, parameter_170, parameter_172, parameter_171, parameter_174, parameter_173, parameter_175, parameter_176, parameter_177, parameter_178, parameter_180, parameter_179, parameter_181, parameter_182, parameter_183, parameter_184, parameter_186, parameter_185, parameter_187, parameter_188, parameter_189, parameter_190, parameter_192, parameter_191, parameter_193, parameter_194, parameter_195, parameter_196, parameter_198, parameter_197, parameter_199, parameter_201, parameter_200, parameter_203, parameter_202, parameter_204, parameter_205, parameter_206, parameter_207, parameter_209, parameter_208, parameter_210, parameter_211, parameter_212, parameter_213, parameter_215, parameter_214, parameter_216, parameter_217, parameter_218, parameter_219, parameter_221, parameter_220, parameter_222, parameter_223, parameter_224, parameter_225, parameter_227, parameter_226, parameter_228, parameter_230, parameter_229, parameter_232, parameter_231, parameter_233, parameter_234, parameter_235, parameter_236, parameter_238, parameter_237, parameter_239, parameter_240, parameter_241, parameter_242, parameter_244, parameter_243, parameter_245, parameter_246, parameter_247, parameter_248, parameter_250, parameter_249, parameter_251, parameter_252, parameter_253, parameter_254, parameter_256, parameter_255, parameter_257, parameter_259, parameter_258, parameter_261, parameter_260, parameter_262, parameter_263, parameter_264, parameter_265, parameter_267, parameter_266, parameter_268, parameter_269, parameter_270, parameter_271, parameter_273, parameter_272, parameter_274, parameter_275, parameter_276, parameter_277, parameter_279, parameter_278, parameter_280, parameter_281, parameter_282, parameter_283, parameter_285, parameter_284, parameter_286, parameter_288, parameter_287, parameter_290, parameter_289, parameter_291, parameter_292, parameter_293, parameter_294, parameter_296, parameter_295, parameter_297, parameter_298, parameter_299, parameter_300, parameter_302, parameter_301, parameter_303, parameter_304, parameter_305, parameter_306, parameter_308, parameter_307, parameter_309, parameter_310, parameter_311, parameter_312, parameter_314, parameter_313, parameter_315, parameter_317, parameter_316, parameter_319, parameter_318, parameter_320, parameter_321, parameter_322, parameter_323, parameter_325, parameter_324, parameter_326, parameter_327, parameter_328, parameter_329, parameter_331, parameter_330, parameter_332, parameter_333, parameter_334, parameter_335, parameter_337, parameter_336, parameter_338, parameter_339, parameter_340, parameter_341, parameter_343, parameter_342, parameter_344, parameter_346, parameter_345, parameter_348, parameter_347, parameter_349, parameter_350, parameter_351, parameter_352, parameter_354, parameter_353, parameter_355, parameter_356, parameter_357, parameter_358, parameter_360, parameter_359, parameter_361, parameter_362, feed_0):

        # pd_op.shape: (4xi32) <- (-1x3x224x224xf32)
        shape_0 = paddle._C_ops.shape(feed_0)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(shape_0, [0], full_int_array_0, full_int_array_1, [1], [0])

        # pd_op.unfold: (-1x768x196xf32) <- (-1x3x224x224xf32)
        unfold_0 = paddle._C_ops.unfold(feed_0, [16, 16], [16, 16], [0, 0, 0, 0], [1, 1])

        # pd_op.transpose: (-1x196x768xf32) <- (-1x768x196xf32)
        transpose_0 = paddle._C_ops.transpose(unfold_0, [0, 2, 1])

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_2 = [-1, 3, 16, 16]

        # pd_op.reshape_: (-1x3x16x16xf32, 0x-1x196x768xf32) <- (-1x196x768xf32, 4xi64)
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_0, full_int_array_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x24x4x4xf32) <- (-1x3x16x16xf32, 24x3x7x7xf32)
        conv2d_0 = paddle._C_ops.conv2d(reshape__0, parameter_0, [4, 4], [3, 3], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [1, 24, 1, 1]

        # pd_op.reshape: (1x24x1x1xf32, 0x24xf32) <- (24xf32, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_1, full_int_array_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x24x4x4xf32) <- (-1x24x4x4xf32, 1x24x1x1xf32)
        add__0 = paddle._C_ops.add_(conv2d_0, reshape_0)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_4 = [-1, 24, 16]

        # pd_op.reshape_: (-1x24x16xf32, 0x-1x24x4x4xf32) <- (-1x24x4x4xf32, 3xi64)
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__0, full_int_array_4), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x16x24xf32) <- (-1x24x16xf32)
        transpose_1 = paddle._C_ops.transpose(reshape__2, [0, 2, 1])

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, 1x16x24xf32)
        add__1 = paddle._C_ops.add_(transpose_1, parameter_2)

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_5 = [-1, 196, 384]

        # pd_op.reshape: (-1x196x384xf32, 0x-1x16x24xf32) <- (-1x16x24xf32, 3xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__1, full_int_array_5), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.layer_norm: (-1x196x384xf32, -196xf32, -196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_0, layer_norm_1, layer_norm_2 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(reshape_2, parameter_3, parameter_4, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x196x384xf32) <- (-1x196x384xf32, 384x384xf32)
        matmul_0 = paddle._C_ops.matmul(layer_norm_0, parameter_5, False, False)

        # pd_op.add_: (-1x196x384xf32) <- (-1x196x384xf32, 384xf32)
        add__2 = paddle._C_ops.add_(matmul_0, parameter_6)

        # pd_op.layer_norm: (-1x196x384xf32, -196xf32, -196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_3, layer_norm_4, layer_norm_5 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__2, parameter_7, parameter_8, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('-1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_0 = [slice_0, full_0, full_1]

        # pd_op.expand: (-1x1x384xf32) <- (1x1x384xf32, [1xi32, 1xi32, 1xi32])
        expand_0 = paddle._C_ops.expand(parameter_9, [x.reshape([]) for x in combine_0])

        # pd_op.cast_: (-1x1x384xf32) <- (-1x1x384xf32)
        cast__0 = paddle._C_ops.cast_(expand_0, paddle.float32)

        # builtin.combine: ([-1x1x384xf32, -1x196x384xf32]) <- (-1x1x384xf32, -1x196x384xf32)
        combine_1 = [cast__0, layer_norm_3]

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x197x384xf32) <- ([-1x1x384xf32, -1x196x384xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_1, full_2)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, 1x197x384xf32)
        add__3 = paddle._C_ops.add_(concat_0, parameter_10)

        # pd_op.layer_norm: (-1x16x24xf32, -16xf32, -16xf32) <- (-1x16x24xf32, 24xf32, 24xf32)
        layer_norm_6, layer_norm_7, layer_norm_8 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__1, parameter_11, parameter_12, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x16x24xf32)
        shape_1 = paddle._C_ops.shape(layer_norm_6)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(shape_1, [0], full_int_array_6, full_int_array_7, [1], [0])

        # pd_op.matmul: (-1x16x48xf32) <- (-1x16x24xf32, 24x48xf32)
        matmul_1 = paddle._C_ops.matmul(layer_norm_6, parameter_13, False, False)

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_2 = [slice_1, full_3, full_4, full_5, full_6]

        # pd_op.reshape_: (-1x16x2x4x6xf32, 0x-1x16x48xf32) <- (-1x16x48xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_1, [x.reshape([]) for x in combine_2]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x4x16x6xf32) <- (-1x16x2x4x6xf32)
        transpose_2 = paddle._C_ops.transpose(reshape__4, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [1]

        # pd_op.slice: (-1x4x16x6xf32) <- (2x-1x4x16x6xf32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(transpose_2, [0], full_int_array_8, full_int_array_9, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_10 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_11 = [2]

        # pd_op.slice: (-1x4x16x6xf32) <- (2x-1x4x16x6xf32, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(transpose_2, [0], full_int_array_10, full_int_array_11, [1], [0])

        # pd_op.matmul: (-1x16x24xf32) <- (-1x16x24xf32, 24x24xf32)
        matmul_2 = paddle._C_ops.matmul(layer_norm_6, parameter_14, False, False)

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_8 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_9 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_3 = [slice_1, full_7, full_8, full_9]

        # pd_op.reshape_: (-1x16x4x6xf32, 0x-1x16x24xf32) <- (-1x16x24xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_2, [x.reshape([]) for x in combine_3]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x16x6xf32) <- (-1x16x4x6xf32)
        transpose_3 = paddle._C_ops.transpose(reshape__6, [0, 2, 1, 3])

        # pd_op.transpose: (-1x4x6x16xf32) <- (-1x4x16x6xf32)
        transpose_4 = paddle._C_ops.transpose(slice_3, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x16x16xf32) <- (-1x4x16x6xf32, -1x4x6x16xf32)
        matmul_3 = paddle._C_ops.matmul(slice_2, transpose_4, False, False)

        # pd_op.full: (1xf32) <- ()
        full_10 = paddle._C_ops.full([1], float('0.408248'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x4x16x16xf32) <- (-1x4x16x16xf32, 1xf32)
        scale__0 = paddle._C_ops.scale_(matmul_3, full_10, float('0'), True)

        # pd_op.softmax_: (-1x4x16x16xf32) <- (-1x4x16x16xf32)
        softmax__0 = paddle._C_ops.softmax_(scale__0, -1)

        # pd_op.matmul: (-1x4x16x6xf32) <- (-1x4x16x16xf32, -1x4x16x6xf32)
        matmul_4 = paddle._C_ops.matmul(softmax__0, transpose_3, False, False)

        # pd_op.transpose: (-1x16x4x6xf32) <- (-1x4x16x6xf32)
        transpose_5 = paddle._C_ops.transpose(matmul_4, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_11 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_12 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_4 = [slice_1, full_11, full_12]

        # pd_op.reshape_: (-1x16x24xf32, 0x-1x16x4x6xf32) <- (-1x16x4x6xf32, [1xi32, 1xi32, 1xi32])
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_5, [x.reshape([]) for x in combine_4]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16x24xf32) <- (-1x16x24xf32, 24x24xf32)
        matmul_5 = paddle._C_ops.matmul(reshape__8, parameter_15, False, False)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, 24xf32)
        add__4 = paddle._C_ops.add_(matmul_5, parameter_16)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, -1x16x24xf32)
        add__5 = paddle._C_ops.add_(add__1, add__4)

        # pd_op.layer_norm: (-1x16x24xf32, -16xf32, -16xf32) <- (-1x16x24xf32, 24xf32, 24xf32)
        layer_norm_9, layer_norm_10, layer_norm_11 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__5, parameter_17, parameter_18, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x16x96xf32) <- (-1x16x24xf32, 24x96xf32)
        matmul_6 = paddle._C_ops.matmul(layer_norm_9, parameter_19, False, False)

        # pd_op.add_: (-1x16x96xf32) <- (-1x16x96xf32, 96xf32)
        add__6 = paddle._C_ops.add_(matmul_6, parameter_20)

        # pd_op.gelu: (-1x16x96xf32) <- (-1x16x96xf32)
        gelu_0 = paddle._C_ops.gelu(add__6, False)

        # pd_op.matmul: (-1x16x24xf32) <- (-1x16x96xf32, 96x24xf32)
        matmul_7 = paddle._C_ops.matmul(gelu_0, parameter_21, False, False)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, 24xf32)
        add__7 = paddle._C_ops.add_(matmul_7, parameter_22)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, -1x16x24xf32)
        add__8 = paddle._C_ops.add_(add__5, add__7)

        # pd_op.shape: (3xi32) <- (-1x197x384xf32)
        shape_2 = paddle._C_ops.shape(add__3)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_12 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_13 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(shape_2, [0], full_int_array_12, full_int_array_13, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_13 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_14 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_5 = [slice_4, full_13, full_14]

        # pd_op.reshape: (-1x196x384xf32, 0x-1x16x24xf32) <- (-1x16x24xf32, [1xi32, 1xi32, 1xi32])
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__8, [x.reshape([]) for x in combine_5]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.layer_norm: (-1x196x384xf32, -196xf32, -196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_12, layer_norm_13, layer_norm_14 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(reshape_4, parameter_23, parameter_24, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_14 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_15 = [197]

        # pd_op.slice: (-1x196x384xf32) <- (-1x197x384xf32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(add__3, [1], full_int_array_14, full_int_array_15, [1], [])

        # pd_op.matmul: (-1x196x384xf32) <- (-1x196x384xf32, 384x384xf32)
        matmul_8 = paddle._C_ops.matmul(layer_norm_12, parameter_25, False, False)

        # pd_op.layer_norm: (-1x196x384xf32, -196xf32, -196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_15, layer_norm_16, layer_norm_17 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(matmul_8, parameter_26, parameter_27, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.add_: (-1x196x384xf32) <- (-1x196x384xf32, -1x196x384xf32)
        add__9 = paddle._C_ops.add_(slice_5, layer_norm_15)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_16 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_17 = [2147483647]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_18 = [1]

        # pd_op.set_value_with_tensor_: (-1x197x384xf32) <- (-1x197x384xf32, -1x196x384xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__0 = paddle._C_ops.set_value_with_tensor_(add__3, add__9, full_int_array_16, full_int_array_17, full_int_array_18, [1], [], [])

        # pd_op.layer_norm: (-1x197x384xf32, -197xf32, -197xf32) <- (-1x197x384xf32, 384xf32, 384xf32)
        layer_norm_18, layer_norm_19, layer_norm_20 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(set_value_with_tensor__0, parameter_28, parameter_29, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x197x384xf32)
        shape_3 = paddle._C_ops.shape(layer_norm_18)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_19 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_20 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(shape_3, [0], full_int_array_19, full_int_array_20, [1], [0])

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x384xf32, 384x768xf32)
        matmul_9 = paddle._C_ops.matmul(layer_norm_18, parameter_30, False, False)

        # pd_op.full: (1xi32) <- ()
        full_15 = paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_16 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_17 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_18 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_6 = [slice_6, full_15, full_16, full_17, full_18]

        # pd_op.reshape_: (-1x197x2x6x64xf32, 0x-1x197x768xf32) <- (-1x197x768xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_9, [x.reshape([]) for x in combine_6]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x6x197x64xf32) <- (-1x197x2x6x64xf32)
        transpose_6 = paddle._C_ops.transpose(reshape__10, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_21 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_22 = [1]

        # pd_op.slice: (-1x6x197x64xf32) <- (2x-1x6x197x64xf32, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(transpose_6, [0], full_int_array_21, full_int_array_22, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_23 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_24 = [2]

        # pd_op.slice: (-1x6x197x64xf32) <- (2x-1x6x197x64xf32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(transpose_6, [0], full_int_array_23, full_int_array_24, [1], [0])

        # pd_op.matmul: (-1x197x384xf32) <- (-1x197x384xf32, 384x384xf32)
        matmul_10 = paddle._C_ops.matmul(layer_norm_18, parameter_31, False, False)

        # pd_op.full: (1xi32) <- ()
        full_19 = paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_20 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_21 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_7 = [slice_6, full_19, full_20, full_21]

        # pd_op.reshape_: (-1x197x6x64xf32, 0x-1x197x384xf32) <- (-1x197x384xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_10, [x.reshape([]) for x in combine_7]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x6x197x64xf32) <- (-1x197x6x64xf32)
        transpose_7 = paddle._C_ops.transpose(reshape__12, [0, 2, 1, 3])

        # pd_op.transpose: (-1x6x64x197xf32) <- (-1x6x197x64xf32)
        transpose_8 = paddle._C_ops.transpose(slice_8, [0, 1, 3, 2])

        # pd_op.matmul: (-1x6x197x197xf32) <- (-1x6x197x64xf32, -1x6x64x197xf32)
        matmul_11 = paddle._C_ops.matmul(slice_7, transpose_8, False, False)

        # pd_op.full: (1xf32) <- ()
        full_22 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x6x197x197xf32) <- (-1x6x197x197xf32, 1xf32)
        scale__1 = paddle._C_ops.scale_(matmul_11, full_22, float('0'), True)

        # pd_op.softmax_: (-1x6x197x197xf32) <- (-1x6x197x197xf32)
        softmax__1 = paddle._C_ops.softmax_(scale__1, -1)

        # pd_op.matmul: (-1x6x197x64xf32) <- (-1x6x197x197xf32, -1x6x197x64xf32)
        matmul_12 = paddle._C_ops.matmul(softmax__1, transpose_7, False, False)

        # pd_op.transpose: (-1x197x6x64xf32) <- (-1x6x197x64xf32)
        transpose_9 = paddle._C_ops.transpose(matmul_12, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_23 = paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_24 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_8 = [slice_6, full_23, full_24]

        # pd_op.reshape_: (-1x197x384xf32, 0x-1x197x6x64xf32) <- (-1x197x6x64xf32, [1xi32, 1xi32, 1xi32])
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_9, [x.reshape([]) for x in combine_8]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x197x384xf32) <- (-1x197x384xf32, 384x384xf32)
        matmul_13 = paddle._C_ops.matmul(reshape__14, parameter_32, False, False)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, 384xf32)
        add__10 = paddle._C_ops.add_(matmul_13, parameter_33)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, -1x197x384xf32)
        add__11 = paddle._C_ops.add_(set_value_with_tensor__0, add__10)

        # pd_op.layer_norm: (-1x197x384xf32, -197xf32, -197xf32) <- (-1x197x384xf32, 384xf32, 384xf32)
        layer_norm_21, layer_norm_22, layer_norm_23 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__11, parameter_34, parameter_35, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x1536xf32) <- (-1x197x384xf32, 384x1536xf32)
        matmul_14 = paddle._C_ops.matmul(layer_norm_21, parameter_36, False, False)

        # pd_op.add_: (-1x197x1536xf32) <- (-1x197x1536xf32, 1536xf32)
        add__12 = paddle._C_ops.add_(matmul_14, parameter_37)

        # pd_op.gelu: (-1x197x1536xf32) <- (-1x197x1536xf32)
        gelu_1 = paddle._C_ops.gelu(add__12, False)

        # pd_op.matmul: (-1x197x384xf32) <- (-1x197x1536xf32, 1536x384xf32)
        matmul_15 = paddle._C_ops.matmul(gelu_1, parameter_38, False, False)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, 384xf32)
        add__13 = paddle._C_ops.add_(matmul_15, parameter_39)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, -1x197x384xf32)
        add__14 = paddle._C_ops.add_(add__11, add__13)

        # pd_op.layer_norm: (-1x16x24xf32, -16xf32, -16xf32) <- (-1x16x24xf32, 24xf32, 24xf32)
        layer_norm_24, layer_norm_25, layer_norm_26 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__8, parameter_40, parameter_41, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x16x24xf32)
        shape_4 = paddle._C_ops.shape(layer_norm_24)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_25 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_26 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(shape_4, [0], full_int_array_25, full_int_array_26, [1], [0])

        # pd_op.matmul: (-1x16x48xf32) <- (-1x16x24xf32, 24x48xf32)
        matmul_16 = paddle._C_ops.matmul(layer_norm_24, parameter_42, False, False)

        # pd_op.full: (1xi32) <- ()
        full_25 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_26 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_27 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_28 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_9 = [slice_9, full_25, full_26, full_27, full_28]

        # pd_op.reshape_: (-1x16x2x4x6xf32, 0x-1x16x48xf32) <- (-1x16x48xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__16, reshape__17 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_16, [x.reshape([]) for x in combine_9]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x4x16x6xf32) <- (-1x16x2x4x6xf32)
        transpose_10 = paddle._C_ops.transpose(reshape__16, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_27 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_28 = [1]

        # pd_op.slice: (-1x4x16x6xf32) <- (2x-1x4x16x6xf32, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(transpose_10, [0], full_int_array_27, full_int_array_28, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_29 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_30 = [2]

        # pd_op.slice: (-1x4x16x6xf32) <- (2x-1x4x16x6xf32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(transpose_10, [0], full_int_array_29, full_int_array_30, [1], [0])

        # pd_op.matmul: (-1x16x24xf32) <- (-1x16x24xf32, 24x24xf32)
        matmul_17 = paddle._C_ops.matmul(layer_norm_24, parameter_43, False, False)

        # pd_op.full: (1xi32) <- ()
        full_29 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_30 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_31 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_10 = [slice_9, full_29, full_30, full_31]

        # pd_op.reshape_: (-1x16x4x6xf32, 0x-1x16x24xf32) <- (-1x16x24xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__18, reshape__19 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_17, [x.reshape([]) for x in combine_10]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x16x6xf32) <- (-1x16x4x6xf32)
        transpose_11 = paddle._C_ops.transpose(reshape__18, [0, 2, 1, 3])

        # pd_op.transpose: (-1x4x6x16xf32) <- (-1x4x16x6xf32)
        transpose_12 = paddle._C_ops.transpose(slice_11, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x16x16xf32) <- (-1x4x16x6xf32, -1x4x6x16xf32)
        matmul_18 = paddle._C_ops.matmul(slice_10, transpose_12, False, False)

        # pd_op.full: (1xf32) <- ()
        full_32 = paddle._C_ops.full([1], float('0.408248'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x4x16x16xf32) <- (-1x4x16x16xf32, 1xf32)
        scale__2 = paddle._C_ops.scale_(matmul_18, full_32, float('0'), True)

        # pd_op.softmax_: (-1x4x16x16xf32) <- (-1x4x16x16xf32)
        softmax__2 = paddle._C_ops.softmax_(scale__2, -1)

        # pd_op.matmul: (-1x4x16x6xf32) <- (-1x4x16x16xf32, -1x4x16x6xf32)
        matmul_19 = paddle._C_ops.matmul(softmax__2, transpose_11, False, False)

        # pd_op.transpose: (-1x16x4x6xf32) <- (-1x4x16x6xf32)
        transpose_13 = paddle._C_ops.transpose(matmul_19, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_33 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_34 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_11 = [slice_9, full_33, full_34]

        # pd_op.reshape_: (-1x16x24xf32, 0x-1x16x4x6xf32) <- (-1x16x4x6xf32, [1xi32, 1xi32, 1xi32])
        reshape__20, reshape__21 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_13, [x.reshape([]) for x in combine_11]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16x24xf32) <- (-1x16x24xf32, 24x24xf32)
        matmul_20 = paddle._C_ops.matmul(reshape__20, parameter_44, False, False)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, 24xf32)
        add__15 = paddle._C_ops.add_(matmul_20, parameter_45)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, -1x16x24xf32)
        add__16 = paddle._C_ops.add_(add__8, add__15)

        # pd_op.layer_norm: (-1x16x24xf32, -16xf32, -16xf32) <- (-1x16x24xf32, 24xf32, 24xf32)
        layer_norm_27, layer_norm_28, layer_norm_29 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__16, parameter_46, parameter_47, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x16x96xf32) <- (-1x16x24xf32, 24x96xf32)
        matmul_21 = paddle._C_ops.matmul(layer_norm_27, parameter_48, False, False)

        # pd_op.add_: (-1x16x96xf32) <- (-1x16x96xf32, 96xf32)
        add__17 = paddle._C_ops.add_(matmul_21, parameter_49)

        # pd_op.gelu: (-1x16x96xf32) <- (-1x16x96xf32)
        gelu_2 = paddle._C_ops.gelu(add__17, False)

        # pd_op.matmul: (-1x16x24xf32) <- (-1x16x96xf32, 96x24xf32)
        matmul_22 = paddle._C_ops.matmul(gelu_2, parameter_50, False, False)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, 24xf32)
        add__18 = paddle._C_ops.add_(matmul_22, parameter_51)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, -1x16x24xf32)
        add__19 = paddle._C_ops.add_(add__16, add__18)

        # pd_op.shape: (3xi32) <- (-1x197x384xf32)
        shape_5 = paddle._C_ops.shape(add__14)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_31 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_32 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(shape_5, [0], full_int_array_31, full_int_array_32, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_35 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_36 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_12 = [slice_12, full_35, full_36]

        # pd_op.reshape: (-1x196x384xf32, 0x-1x16x24xf32) <- (-1x16x24xf32, [1xi32, 1xi32, 1xi32])
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__19, [x.reshape([]) for x in combine_12]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.layer_norm: (-1x196x384xf32, -196xf32, -196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_30, layer_norm_31, layer_norm_32 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(reshape_6, parameter_52, parameter_53, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_33 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_34 = [197]

        # pd_op.slice: (-1x196x384xf32) <- (-1x197x384xf32, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(add__14, [1], full_int_array_33, full_int_array_34, [1], [])

        # pd_op.matmul: (-1x196x384xf32) <- (-1x196x384xf32, 384x384xf32)
        matmul_23 = paddle._C_ops.matmul(layer_norm_30, parameter_54, False, False)

        # pd_op.layer_norm: (-1x196x384xf32, -196xf32, -196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_33, layer_norm_34, layer_norm_35 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(matmul_23, parameter_55, parameter_56, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.add_: (-1x196x384xf32) <- (-1x196x384xf32, -1x196x384xf32)
        add__20 = paddle._C_ops.add_(slice_13, layer_norm_33)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_35 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_36 = [2147483647]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_37 = [1]

        # pd_op.set_value_with_tensor_: (-1x197x384xf32) <- (-1x197x384xf32, -1x196x384xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__1 = paddle._C_ops.set_value_with_tensor_(add__14, add__20, full_int_array_35, full_int_array_36, full_int_array_37, [1], [], [])

        # pd_op.layer_norm: (-1x197x384xf32, -197xf32, -197xf32) <- (-1x197x384xf32, 384xf32, 384xf32)
        layer_norm_36, layer_norm_37, layer_norm_38 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(set_value_with_tensor__1, parameter_57, parameter_58, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x197x384xf32)
        shape_6 = paddle._C_ops.shape(layer_norm_36)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_38 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_39 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(shape_6, [0], full_int_array_38, full_int_array_39, [1], [0])

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x384xf32, 384x768xf32)
        matmul_24 = paddle._C_ops.matmul(layer_norm_36, parameter_59, False, False)

        # pd_op.full: (1xi32) <- ()
        full_37 = paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_38 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_39 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_40 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_13 = [slice_14, full_37, full_38, full_39, full_40]

        # pd_op.reshape_: (-1x197x2x6x64xf32, 0x-1x197x768xf32) <- (-1x197x768xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__22, reshape__23 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_24, [x.reshape([]) for x in combine_13]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x6x197x64xf32) <- (-1x197x2x6x64xf32)
        transpose_14 = paddle._C_ops.transpose(reshape__22, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_40 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_41 = [1]

        # pd_op.slice: (-1x6x197x64xf32) <- (2x-1x6x197x64xf32, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(transpose_14, [0], full_int_array_40, full_int_array_41, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_42 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_43 = [2]

        # pd_op.slice: (-1x6x197x64xf32) <- (2x-1x6x197x64xf32, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(transpose_14, [0], full_int_array_42, full_int_array_43, [1], [0])

        # pd_op.matmul: (-1x197x384xf32) <- (-1x197x384xf32, 384x384xf32)
        matmul_25 = paddle._C_ops.matmul(layer_norm_36, parameter_60, False, False)

        # pd_op.full: (1xi32) <- ()
        full_41 = paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_42 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_43 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_14 = [slice_14, full_41, full_42, full_43]

        # pd_op.reshape_: (-1x197x6x64xf32, 0x-1x197x384xf32) <- (-1x197x384xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__24, reshape__25 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_25, [x.reshape([]) for x in combine_14]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x6x197x64xf32) <- (-1x197x6x64xf32)
        transpose_15 = paddle._C_ops.transpose(reshape__24, [0, 2, 1, 3])

        # pd_op.transpose: (-1x6x64x197xf32) <- (-1x6x197x64xf32)
        transpose_16 = paddle._C_ops.transpose(slice_16, [0, 1, 3, 2])

        # pd_op.matmul: (-1x6x197x197xf32) <- (-1x6x197x64xf32, -1x6x64x197xf32)
        matmul_26 = paddle._C_ops.matmul(slice_15, transpose_16, False, False)

        # pd_op.full: (1xf32) <- ()
        full_44 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x6x197x197xf32) <- (-1x6x197x197xf32, 1xf32)
        scale__3 = paddle._C_ops.scale_(matmul_26, full_44, float('0'), True)

        # pd_op.softmax_: (-1x6x197x197xf32) <- (-1x6x197x197xf32)
        softmax__3 = paddle._C_ops.softmax_(scale__3, -1)

        # pd_op.matmul: (-1x6x197x64xf32) <- (-1x6x197x197xf32, -1x6x197x64xf32)
        matmul_27 = paddle._C_ops.matmul(softmax__3, transpose_15, False, False)

        # pd_op.transpose: (-1x197x6x64xf32) <- (-1x6x197x64xf32)
        transpose_17 = paddle._C_ops.transpose(matmul_27, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_45 = paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_46 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_15 = [slice_14, full_45, full_46]

        # pd_op.reshape_: (-1x197x384xf32, 0x-1x197x6x64xf32) <- (-1x197x6x64xf32, [1xi32, 1xi32, 1xi32])
        reshape__26, reshape__27 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_17, [x.reshape([]) for x in combine_15]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x197x384xf32) <- (-1x197x384xf32, 384x384xf32)
        matmul_28 = paddle._C_ops.matmul(reshape__26, parameter_61, False, False)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, 384xf32)
        add__21 = paddle._C_ops.add_(matmul_28, parameter_62)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, -1x197x384xf32)
        add__22 = paddle._C_ops.add_(set_value_with_tensor__1, add__21)

        # pd_op.layer_norm: (-1x197x384xf32, -197xf32, -197xf32) <- (-1x197x384xf32, 384xf32, 384xf32)
        layer_norm_39, layer_norm_40, layer_norm_41 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__22, parameter_63, parameter_64, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x1536xf32) <- (-1x197x384xf32, 384x1536xf32)
        matmul_29 = paddle._C_ops.matmul(layer_norm_39, parameter_65, False, False)

        # pd_op.add_: (-1x197x1536xf32) <- (-1x197x1536xf32, 1536xf32)
        add__23 = paddle._C_ops.add_(matmul_29, parameter_66)

        # pd_op.gelu: (-1x197x1536xf32) <- (-1x197x1536xf32)
        gelu_3 = paddle._C_ops.gelu(add__23, False)

        # pd_op.matmul: (-1x197x384xf32) <- (-1x197x1536xf32, 1536x384xf32)
        matmul_30 = paddle._C_ops.matmul(gelu_3, parameter_67, False, False)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, 384xf32)
        add__24 = paddle._C_ops.add_(matmul_30, parameter_68)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, -1x197x384xf32)
        add__25 = paddle._C_ops.add_(add__22, add__24)

        # pd_op.layer_norm: (-1x16x24xf32, -16xf32, -16xf32) <- (-1x16x24xf32, 24xf32, 24xf32)
        layer_norm_42, layer_norm_43, layer_norm_44 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__19, parameter_69, parameter_70, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x16x24xf32)
        shape_7 = paddle._C_ops.shape(layer_norm_42)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_44 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_45 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(shape_7, [0], full_int_array_44, full_int_array_45, [1], [0])

        # pd_op.matmul: (-1x16x48xf32) <- (-1x16x24xf32, 24x48xf32)
        matmul_31 = paddle._C_ops.matmul(layer_norm_42, parameter_71, False, False)

        # pd_op.full: (1xi32) <- ()
        full_47 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_48 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_49 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_50 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_16 = [slice_17, full_47, full_48, full_49, full_50]

        # pd_op.reshape_: (-1x16x2x4x6xf32, 0x-1x16x48xf32) <- (-1x16x48xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__28, reshape__29 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_31, [x.reshape([]) for x in combine_16]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x4x16x6xf32) <- (-1x16x2x4x6xf32)
        transpose_18 = paddle._C_ops.transpose(reshape__28, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_46 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_47 = [1]

        # pd_op.slice: (-1x4x16x6xf32) <- (2x-1x4x16x6xf32, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(transpose_18, [0], full_int_array_46, full_int_array_47, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_48 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_49 = [2]

        # pd_op.slice: (-1x4x16x6xf32) <- (2x-1x4x16x6xf32, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(transpose_18, [0], full_int_array_48, full_int_array_49, [1], [0])

        # pd_op.matmul: (-1x16x24xf32) <- (-1x16x24xf32, 24x24xf32)
        matmul_32 = paddle._C_ops.matmul(layer_norm_42, parameter_72, False, False)

        # pd_op.full: (1xi32) <- ()
        full_51 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_52 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_53 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_17 = [slice_17, full_51, full_52, full_53]

        # pd_op.reshape_: (-1x16x4x6xf32, 0x-1x16x24xf32) <- (-1x16x24xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__30, reshape__31 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_32, [x.reshape([]) for x in combine_17]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x16x6xf32) <- (-1x16x4x6xf32)
        transpose_19 = paddle._C_ops.transpose(reshape__30, [0, 2, 1, 3])

        # pd_op.transpose: (-1x4x6x16xf32) <- (-1x4x16x6xf32)
        transpose_20 = paddle._C_ops.transpose(slice_19, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x16x16xf32) <- (-1x4x16x6xf32, -1x4x6x16xf32)
        matmul_33 = paddle._C_ops.matmul(slice_18, transpose_20, False, False)

        # pd_op.full: (1xf32) <- ()
        full_54 = paddle._C_ops.full([1], float('0.408248'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x4x16x16xf32) <- (-1x4x16x16xf32, 1xf32)
        scale__4 = paddle._C_ops.scale_(matmul_33, full_54, float('0'), True)

        # pd_op.softmax_: (-1x4x16x16xf32) <- (-1x4x16x16xf32)
        softmax__4 = paddle._C_ops.softmax_(scale__4, -1)

        # pd_op.matmul: (-1x4x16x6xf32) <- (-1x4x16x16xf32, -1x4x16x6xf32)
        matmul_34 = paddle._C_ops.matmul(softmax__4, transpose_19, False, False)

        # pd_op.transpose: (-1x16x4x6xf32) <- (-1x4x16x6xf32)
        transpose_21 = paddle._C_ops.transpose(matmul_34, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_55 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_56 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_18 = [slice_17, full_55, full_56]

        # pd_op.reshape_: (-1x16x24xf32, 0x-1x16x4x6xf32) <- (-1x16x4x6xf32, [1xi32, 1xi32, 1xi32])
        reshape__32, reshape__33 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_21, [x.reshape([]) for x in combine_18]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16x24xf32) <- (-1x16x24xf32, 24x24xf32)
        matmul_35 = paddle._C_ops.matmul(reshape__32, parameter_73, False, False)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, 24xf32)
        add__26 = paddle._C_ops.add_(matmul_35, parameter_74)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, -1x16x24xf32)
        add__27 = paddle._C_ops.add_(add__19, add__26)

        # pd_op.layer_norm: (-1x16x24xf32, -16xf32, -16xf32) <- (-1x16x24xf32, 24xf32, 24xf32)
        layer_norm_45, layer_norm_46, layer_norm_47 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__27, parameter_75, parameter_76, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x16x96xf32) <- (-1x16x24xf32, 24x96xf32)
        matmul_36 = paddle._C_ops.matmul(layer_norm_45, parameter_77, False, False)

        # pd_op.add_: (-1x16x96xf32) <- (-1x16x96xf32, 96xf32)
        add__28 = paddle._C_ops.add_(matmul_36, parameter_78)

        # pd_op.gelu: (-1x16x96xf32) <- (-1x16x96xf32)
        gelu_4 = paddle._C_ops.gelu(add__28, False)

        # pd_op.matmul: (-1x16x24xf32) <- (-1x16x96xf32, 96x24xf32)
        matmul_37 = paddle._C_ops.matmul(gelu_4, parameter_79, False, False)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, 24xf32)
        add__29 = paddle._C_ops.add_(matmul_37, parameter_80)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, -1x16x24xf32)
        add__30 = paddle._C_ops.add_(add__27, add__29)

        # pd_op.shape: (3xi32) <- (-1x197x384xf32)
        shape_8 = paddle._C_ops.shape(add__25)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_50 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_51 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(shape_8, [0], full_int_array_50, full_int_array_51, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_57 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_58 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_19 = [slice_20, full_57, full_58]

        # pd_op.reshape: (-1x196x384xf32, 0x-1x16x24xf32) <- (-1x16x24xf32, [1xi32, 1xi32, 1xi32])
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__30, [x.reshape([]) for x in combine_19]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.layer_norm: (-1x196x384xf32, -196xf32, -196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_48, layer_norm_49, layer_norm_50 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(reshape_8, parameter_81, parameter_82, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_52 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_53 = [197]

        # pd_op.slice: (-1x196x384xf32) <- (-1x197x384xf32, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(add__25, [1], full_int_array_52, full_int_array_53, [1], [])

        # pd_op.matmul: (-1x196x384xf32) <- (-1x196x384xf32, 384x384xf32)
        matmul_38 = paddle._C_ops.matmul(layer_norm_48, parameter_83, False, False)

        # pd_op.layer_norm: (-1x196x384xf32, -196xf32, -196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_51, layer_norm_52, layer_norm_53 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(matmul_38, parameter_84, parameter_85, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.add_: (-1x196x384xf32) <- (-1x196x384xf32, -1x196x384xf32)
        add__31 = paddle._C_ops.add_(slice_21, layer_norm_51)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_54 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_55 = [2147483647]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_56 = [1]

        # pd_op.set_value_with_tensor_: (-1x197x384xf32) <- (-1x197x384xf32, -1x196x384xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__2 = paddle._C_ops.set_value_with_tensor_(add__25, add__31, full_int_array_54, full_int_array_55, full_int_array_56, [1], [], [])

        # pd_op.layer_norm: (-1x197x384xf32, -197xf32, -197xf32) <- (-1x197x384xf32, 384xf32, 384xf32)
        layer_norm_54, layer_norm_55, layer_norm_56 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(set_value_with_tensor__2, parameter_86, parameter_87, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x197x384xf32)
        shape_9 = paddle._C_ops.shape(layer_norm_54)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_57 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_58 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(shape_9, [0], full_int_array_57, full_int_array_58, [1], [0])

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x384xf32, 384x768xf32)
        matmul_39 = paddle._C_ops.matmul(layer_norm_54, parameter_88, False, False)

        # pd_op.full: (1xi32) <- ()
        full_59 = paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_60 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_61 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_62 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_20 = [slice_22, full_59, full_60, full_61, full_62]

        # pd_op.reshape_: (-1x197x2x6x64xf32, 0x-1x197x768xf32) <- (-1x197x768xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__34, reshape__35 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_39, [x.reshape([]) for x in combine_20]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x6x197x64xf32) <- (-1x197x2x6x64xf32)
        transpose_22 = paddle._C_ops.transpose(reshape__34, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_59 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_60 = [1]

        # pd_op.slice: (-1x6x197x64xf32) <- (2x-1x6x197x64xf32, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(transpose_22, [0], full_int_array_59, full_int_array_60, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_61 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_62 = [2]

        # pd_op.slice: (-1x6x197x64xf32) <- (2x-1x6x197x64xf32, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(transpose_22, [0], full_int_array_61, full_int_array_62, [1], [0])

        # pd_op.matmul: (-1x197x384xf32) <- (-1x197x384xf32, 384x384xf32)
        matmul_40 = paddle._C_ops.matmul(layer_norm_54, parameter_89, False, False)

        # pd_op.full: (1xi32) <- ()
        full_63 = paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_64 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_65 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_21 = [slice_22, full_63, full_64, full_65]

        # pd_op.reshape_: (-1x197x6x64xf32, 0x-1x197x384xf32) <- (-1x197x384xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__36, reshape__37 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_40, [x.reshape([]) for x in combine_21]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x6x197x64xf32) <- (-1x197x6x64xf32)
        transpose_23 = paddle._C_ops.transpose(reshape__36, [0, 2, 1, 3])

        # pd_op.transpose: (-1x6x64x197xf32) <- (-1x6x197x64xf32)
        transpose_24 = paddle._C_ops.transpose(slice_24, [0, 1, 3, 2])

        # pd_op.matmul: (-1x6x197x197xf32) <- (-1x6x197x64xf32, -1x6x64x197xf32)
        matmul_41 = paddle._C_ops.matmul(slice_23, transpose_24, False, False)

        # pd_op.full: (1xf32) <- ()
        full_66 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x6x197x197xf32) <- (-1x6x197x197xf32, 1xf32)
        scale__5 = paddle._C_ops.scale_(matmul_41, full_66, float('0'), True)

        # pd_op.softmax_: (-1x6x197x197xf32) <- (-1x6x197x197xf32)
        softmax__5 = paddle._C_ops.softmax_(scale__5, -1)

        # pd_op.matmul: (-1x6x197x64xf32) <- (-1x6x197x197xf32, -1x6x197x64xf32)
        matmul_42 = paddle._C_ops.matmul(softmax__5, transpose_23, False, False)

        # pd_op.transpose: (-1x197x6x64xf32) <- (-1x6x197x64xf32)
        transpose_25 = paddle._C_ops.transpose(matmul_42, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_67 = paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_68 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_22 = [slice_22, full_67, full_68]

        # pd_op.reshape_: (-1x197x384xf32, 0x-1x197x6x64xf32) <- (-1x197x6x64xf32, [1xi32, 1xi32, 1xi32])
        reshape__38, reshape__39 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_25, [x.reshape([]) for x in combine_22]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x197x384xf32) <- (-1x197x384xf32, 384x384xf32)
        matmul_43 = paddle._C_ops.matmul(reshape__38, parameter_90, False, False)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, 384xf32)
        add__32 = paddle._C_ops.add_(matmul_43, parameter_91)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, -1x197x384xf32)
        add__33 = paddle._C_ops.add_(set_value_with_tensor__2, add__32)

        # pd_op.layer_norm: (-1x197x384xf32, -197xf32, -197xf32) <- (-1x197x384xf32, 384xf32, 384xf32)
        layer_norm_57, layer_norm_58, layer_norm_59 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__33, parameter_92, parameter_93, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x1536xf32) <- (-1x197x384xf32, 384x1536xf32)
        matmul_44 = paddle._C_ops.matmul(layer_norm_57, parameter_94, False, False)

        # pd_op.add_: (-1x197x1536xf32) <- (-1x197x1536xf32, 1536xf32)
        add__34 = paddle._C_ops.add_(matmul_44, parameter_95)

        # pd_op.gelu: (-1x197x1536xf32) <- (-1x197x1536xf32)
        gelu_5 = paddle._C_ops.gelu(add__34, False)

        # pd_op.matmul: (-1x197x384xf32) <- (-1x197x1536xf32, 1536x384xf32)
        matmul_45 = paddle._C_ops.matmul(gelu_5, parameter_96, False, False)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, 384xf32)
        add__35 = paddle._C_ops.add_(matmul_45, parameter_97)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, -1x197x384xf32)
        add__36 = paddle._C_ops.add_(add__33, add__35)

        # pd_op.layer_norm: (-1x16x24xf32, -16xf32, -16xf32) <- (-1x16x24xf32, 24xf32, 24xf32)
        layer_norm_60, layer_norm_61, layer_norm_62 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__30, parameter_98, parameter_99, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x16x24xf32)
        shape_10 = paddle._C_ops.shape(layer_norm_60)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_63 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_64 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(shape_10, [0], full_int_array_63, full_int_array_64, [1], [0])

        # pd_op.matmul: (-1x16x48xf32) <- (-1x16x24xf32, 24x48xf32)
        matmul_46 = paddle._C_ops.matmul(layer_norm_60, parameter_100, False, False)

        # pd_op.full: (1xi32) <- ()
        full_69 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_70 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_71 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_72 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_23 = [slice_25, full_69, full_70, full_71, full_72]

        # pd_op.reshape_: (-1x16x2x4x6xf32, 0x-1x16x48xf32) <- (-1x16x48xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__40, reshape__41 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_46, [x.reshape([]) for x in combine_23]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x4x16x6xf32) <- (-1x16x2x4x6xf32)
        transpose_26 = paddle._C_ops.transpose(reshape__40, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_65 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_66 = [1]

        # pd_op.slice: (-1x4x16x6xf32) <- (2x-1x4x16x6xf32, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(transpose_26, [0], full_int_array_65, full_int_array_66, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_67 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_68 = [2]

        # pd_op.slice: (-1x4x16x6xf32) <- (2x-1x4x16x6xf32, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(transpose_26, [0], full_int_array_67, full_int_array_68, [1], [0])

        # pd_op.matmul: (-1x16x24xf32) <- (-1x16x24xf32, 24x24xf32)
        matmul_47 = paddle._C_ops.matmul(layer_norm_60, parameter_101, False, False)

        # pd_op.full: (1xi32) <- ()
        full_73 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_74 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_75 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_24 = [slice_25, full_73, full_74, full_75]

        # pd_op.reshape_: (-1x16x4x6xf32, 0x-1x16x24xf32) <- (-1x16x24xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__42, reshape__43 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_47, [x.reshape([]) for x in combine_24]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x16x6xf32) <- (-1x16x4x6xf32)
        transpose_27 = paddle._C_ops.transpose(reshape__42, [0, 2, 1, 3])

        # pd_op.transpose: (-1x4x6x16xf32) <- (-1x4x16x6xf32)
        transpose_28 = paddle._C_ops.transpose(slice_27, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x16x16xf32) <- (-1x4x16x6xf32, -1x4x6x16xf32)
        matmul_48 = paddle._C_ops.matmul(slice_26, transpose_28, False, False)

        # pd_op.full: (1xf32) <- ()
        full_76 = paddle._C_ops.full([1], float('0.408248'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x4x16x16xf32) <- (-1x4x16x16xf32, 1xf32)
        scale__6 = paddle._C_ops.scale_(matmul_48, full_76, float('0'), True)

        # pd_op.softmax_: (-1x4x16x16xf32) <- (-1x4x16x16xf32)
        softmax__6 = paddle._C_ops.softmax_(scale__6, -1)

        # pd_op.matmul: (-1x4x16x6xf32) <- (-1x4x16x16xf32, -1x4x16x6xf32)
        matmul_49 = paddle._C_ops.matmul(softmax__6, transpose_27, False, False)

        # pd_op.transpose: (-1x16x4x6xf32) <- (-1x4x16x6xf32)
        transpose_29 = paddle._C_ops.transpose(matmul_49, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_77 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_78 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_25 = [slice_25, full_77, full_78]

        # pd_op.reshape_: (-1x16x24xf32, 0x-1x16x4x6xf32) <- (-1x16x4x6xf32, [1xi32, 1xi32, 1xi32])
        reshape__44, reshape__45 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_29, [x.reshape([]) for x in combine_25]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16x24xf32) <- (-1x16x24xf32, 24x24xf32)
        matmul_50 = paddle._C_ops.matmul(reshape__44, parameter_102, False, False)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, 24xf32)
        add__37 = paddle._C_ops.add_(matmul_50, parameter_103)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, -1x16x24xf32)
        add__38 = paddle._C_ops.add_(add__30, add__37)

        # pd_op.layer_norm: (-1x16x24xf32, -16xf32, -16xf32) <- (-1x16x24xf32, 24xf32, 24xf32)
        layer_norm_63, layer_norm_64, layer_norm_65 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__38, parameter_104, parameter_105, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x16x96xf32) <- (-1x16x24xf32, 24x96xf32)
        matmul_51 = paddle._C_ops.matmul(layer_norm_63, parameter_106, False, False)

        # pd_op.add_: (-1x16x96xf32) <- (-1x16x96xf32, 96xf32)
        add__39 = paddle._C_ops.add_(matmul_51, parameter_107)

        # pd_op.gelu: (-1x16x96xf32) <- (-1x16x96xf32)
        gelu_6 = paddle._C_ops.gelu(add__39, False)

        # pd_op.matmul: (-1x16x24xf32) <- (-1x16x96xf32, 96x24xf32)
        matmul_52 = paddle._C_ops.matmul(gelu_6, parameter_108, False, False)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, 24xf32)
        add__40 = paddle._C_ops.add_(matmul_52, parameter_109)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, -1x16x24xf32)
        add__41 = paddle._C_ops.add_(add__38, add__40)

        # pd_op.shape: (3xi32) <- (-1x197x384xf32)
        shape_11 = paddle._C_ops.shape(add__36)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_69 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_70 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(shape_11, [0], full_int_array_69, full_int_array_70, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_79 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_80 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_26 = [slice_28, full_79, full_80]

        # pd_op.reshape: (-1x196x384xf32, 0x-1x16x24xf32) <- (-1x16x24xf32, [1xi32, 1xi32, 1xi32])
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__41, [x.reshape([]) for x in combine_26]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.layer_norm: (-1x196x384xf32, -196xf32, -196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_66, layer_norm_67, layer_norm_68 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(reshape_10, parameter_110, parameter_111, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_71 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_72 = [197]

        # pd_op.slice: (-1x196x384xf32) <- (-1x197x384xf32, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(add__36, [1], full_int_array_71, full_int_array_72, [1], [])

        # pd_op.matmul: (-1x196x384xf32) <- (-1x196x384xf32, 384x384xf32)
        matmul_53 = paddle._C_ops.matmul(layer_norm_66, parameter_112, False, False)

        # pd_op.layer_norm: (-1x196x384xf32, -196xf32, -196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_69, layer_norm_70, layer_norm_71 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(matmul_53, parameter_113, parameter_114, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.add_: (-1x196x384xf32) <- (-1x196x384xf32, -1x196x384xf32)
        add__42 = paddle._C_ops.add_(slice_29, layer_norm_69)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_73 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_74 = [2147483647]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_75 = [1]

        # pd_op.set_value_with_tensor_: (-1x197x384xf32) <- (-1x197x384xf32, -1x196x384xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__3 = paddle._C_ops.set_value_with_tensor_(add__36, add__42, full_int_array_73, full_int_array_74, full_int_array_75, [1], [], [])

        # pd_op.layer_norm: (-1x197x384xf32, -197xf32, -197xf32) <- (-1x197x384xf32, 384xf32, 384xf32)
        layer_norm_72, layer_norm_73, layer_norm_74 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(set_value_with_tensor__3, parameter_115, parameter_116, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x197x384xf32)
        shape_12 = paddle._C_ops.shape(layer_norm_72)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_76 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_77 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(shape_12, [0], full_int_array_76, full_int_array_77, [1], [0])

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x384xf32, 384x768xf32)
        matmul_54 = paddle._C_ops.matmul(layer_norm_72, parameter_117, False, False)

        # pd_op.full: (1xi32) <- ()
        full_81 = paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_82 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_83 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_84 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_27 = [slice_30, full_81, full_82, full_83, full_84]

        # pd_op.reshape_: (-1x197x2x6x64xf32, 0x-1x197x768xf32) <- (-1x197x768xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__46, reshape__47 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_54, [x.reshape([]) for x in combine_27]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x6x197x64xf32) <- (-1x197x2x6x64xf32)
        transpose_30 = paddle._C_ops.transpose(reshape__46, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_78 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_79 = [1]

        # pd_op.slice: (-1x6x197x64xf32) <- (2x-1x6x197x64xf32, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(transpose_30, [0], full_int_array_78, full_int_array_79, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_80 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_81 = [2]

        # pd_op.slice: (-1x6x197x64xf32) <- (2x-1x6x197x64xf32, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(transpose_30, [0], full_int_array_80, full_int_array_81, [1], [0])

        # pd_op.matmul: (-1x197x384xf32) <- (-1x197x384xf32, 384x384xf32)
        matmul_55 = paddle._C_ops.matmul(layer_norm_72, parameter_118, False, False)

        # pd_op.full: (1xi32) <- ()
        full_85 = paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_86 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_87 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_28 = [slice_30, full_85, full_86, full_87]

        # pd_op.reshape_: (-1x197x6x64xf32, 0x-1x197x384xf32) <- (-1x197x384xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__48, reshape__49 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_55, [x.reshape([]) for x in combine_28]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x6x197x64xf32) <- (-1x197x6x64xf32)
        transpose_31 = paddle._C_ops.transpose(reshape__48, [0, 2, 1, 3])

        # pd_op.transpose: (-1x6x64x197xf32) <- (-1x6x197x64xf32)
        transpose_32 = paddle._C_ops.transpose(slice_32, [0, 1, 3, 2])

        # pd_op.matmul: (-1x6x197x197xf32) <- (-1x6x197x64xf32, -1x6x64x197xf32)
        matmul_56 = paddle._C_ops.matmul(slice_31, transpose_32, False, False)

        # pd_op.full: (1xf32) <- ()
        full_88 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x6x197x197xf32) <- (-1x6x197x197xf32, 1xf32)
        scale__7 = paddle._C_ops.scale_(matmul_56, full_88, float('0'), True)

        # pd_op.softmax_: (-1x6x197x197xf32) <- (-1x6x197x197xf32)
        softmax__7 = paddle._C_ops.softmax_(scale__7, -1)

        # pd_op.matmul: (-1x6x197x64xf32) <- (-1x6x197x197xf32, -1x6x197x64xf32)
        matmul_57 = paddle._C_ops.matmul(softmax__7, transpose_31, False, False)

        # pd_op.transpose: (-1x197x6x64xf32) <- (-1x6x197x64xf32)
        transpose_33 = paddle._C_ops.transpose(matmul_57, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_89 = paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_90 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_29 = [slice_30, full_89, full_90]

        # pd_op.reshape_: (-1x197x384xf32, 0x-1x197x6x64xf32) <- (-1x197x6x64xf32, [1xi32, 1xi32, 1xi32])
        reshape__50, reshape__51 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_33, [x.reshape([]) for x in combine_29]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x197x384xf32) <- (-1x197x384xf32, 384x384xf32)
        matmul_58 = paddle._C_ops.matmul(reshape__50, parameter_119, False, False)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, 384xf32)
        add__43 = paddle._C_ops.add_(matmul_58, parameter_120)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, -1x197x384xf32)
        add__44 = paddle._C_ops.add_(set_value_with_tensor__3, add__43)

        # pd_op.layer_norm: (-1x197x384xf32, -197xf32, -197xf32) <- (-1x197x384xf32, 384xf32, 384xf32)
        layer_norm_75, layer_norm_76, layer_norm_77 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__44, parameter_121, parameter_122, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x1536xf32) <- (-1x197x384xf32, 384x1536xf32)
        matmul_59 = paddle._C_ops.matmul(layer_norm_75, parameter_123, False, False)

        # pd_op.add_: (-1x197x1536xf32) <- (-1x197x1536xf32, 1536xf32)
        add__45 = paddle._C_ops.add_(matmul_59, parameter_124)

        # pd_op.gelu: (-1x197x1536xf32) <- (-1x197x1536xf32)
        gelu_7 = paddle._C_ops.gelu(add__45, False)

        # pd_op.matmul: (-1x197x384xf32) <- (-1x197x1536xf32, 1536x384xf32)
        matmul_60 = paddle._C_ops.matmul(gelu_7, parameter_125, False, False)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, 384xf32)
        add__46 = paddle._C_ops.add_(matmul_60, parameter_126)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, -1x197x384xf32)
        add__47 = paddle._C_ops.add_(add__44, add__46)

        # pd_op.layer_norm: (-1x16x24xf32, -16xf32, -16xf32) <- (-1x16x24xf32, 24xf32, 24xf32)
        layer_norm_78, layer_norm_79, layer_norm_80 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__41, parameter_127, parameter_128, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x16x24xf32)
        shape_13 = paddle._C_ops.shape(layer_norm_78)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_82 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_83 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_33 = paddle._C_ops.slice(shape_13, [0], full_int_array_82, full_int_array_83, [1], [0])

        # pd_op.matmul: (-1x16x48xf32) <- (-1x16x24xf32, 24x48xf32)
        matmul_61 = paddle._C_ops.matmul(layer_norm_78, parameter_129, False, False)

        # pd_op.full: (1xi32) <- ()
        full_91 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_92 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_93 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_94 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_30 = [slice_33, full_91, full_92, full_93, full_94]

        # pd_op.reshape_: (-1x16x2x4x6xf32, 0x-1x16x48xf32) <- (-1x16x48xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__52, reshape__53 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_61, [x.reshape([]) for x in combine_30]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x4x16x6xf32) <- (-1x16x2x4x6xf32)
        transpose_34 = paddle._C_ops.transpose(reshape__52, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_84 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_85 = [1]

        # pd_op.slice: (-1x4x16x6xf32) <- (2x-1x4x16x6xf32, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(transpose_34, [0], full_int_array_84, full_int_array_85, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_86 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_87 = [2]

        # pd_op.slice: (-1x4x16x6xf32) <- (2x-1x4x16x6xf32, 1xi64, 1xi64)
        slice_35 = paddle._C_ops.slice(transpose_34, [0], full_int_array_86, full_int_array_87, [1], [0])

        # pd_op.matmul: (-1x16x24xf32) <- (-1x16x24xf32, 24x24xf32)
        matmul_62 = paddle._C_ops.matmul(layer_norm_78, parameter_130, False, False)

        # pd_op.full: (1xi32) <- ()
        full_95 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_96 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_97 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_31 = [slice_33, full_95, full_96, full_97]

        # pd_op.reshape_: (-1x16x4x6xf32, 0x-1x16x24xf32) <- (-1x16x24xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__54, reshape__55 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_62, [x.reshape([]) for x in combine_31]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x16x6xf32) <- (-1x16x4x6xf32)
        transpose_35 = paddle._C_ops.transpose(reshape__54, [0, 2, 1, 3])

        # pd_op.transpose: (-1x4x6x16xf32) <- (-1x4x16x6xf32)
        transpose_36 = paddle._C_ops.transpose(slice_35, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x16x16xf32) <- (-1x4x16x6xf32, -1x4x6x16xf32)
        matmul_63 = paddle._C_ops.matmul(slice_34, transpose_36, False, False)

        # pd_op.full: (1xf32) <- ()
        full_98 = paddle._C_ops.full([1], float('0.408248'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x4x16x16xf32) <- (-1x4x16x16xf32, 1xf32)
        scale__8 = paddle._C_ops.scale_(matmul_63, full_98, float('0'), True)

        # pd_op.softmax_: (-1x4x16x16xf32) <- (-1x4x16x16xf32)
        softmax__8 = paddle._C_ops.softmax_(scale__8, -1)

        # pd_op.matmul: (-1x4x16x6xf32) <- (-1x4x16x16xf32, -1x4x16x6xf32)
        matmul_64 = paddle._C_ops.matmul(softmax__8, transpose_35, False, False)

        # pd_op.transpose: (-1x16x4x6xf32) <- (-1x4x16x6xf32)
        transpose_37 = paddle._C_ops.transpose(matmul_64, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_99 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_100 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_32 = [slice_33, full_99, full_100]

        # pd_op.reshape_: (-1x16x24xf32, 0x-1x16x4x6xf32) <- (-1x16x4x6xf32, [1xi32, 1xi32, 1xi32])
        reshape__56, reshape__57 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_37, [x.reshape([]) for x in combine_32]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16x24xf32) <- (-1x16x24xf32, 24x24xf32)
        matmul_65 = paddle._C_ops.matmul(reshape__56, parameter_131, False, False)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, 24xf32)
        add__48 = paddle._C_ops.add_(matmul_65, parameter_132)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, -1x16x24xf32)
        add__49 = paddle._C_ops.add_(add__41, add__48)

        # pd_op.layer_norm: (-1x16x24xf32, -16xf32, -16xf32) <- (-1x16x24xf32, 24xf32, 24xf32)
        layer_norm_81, layer_norm_82, layer_norm_83 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__49, parameter_133, parameter_134, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x16x96xf32) <- (-1x16x24xf32, 24x96xf32)
        matmul_66 = paddle._C_ops.matmul(layer_norm_81, parameter_135, False, False)

        # pd_op.add_: (-1x16x96xf32) <- (-1x16x96xf32, 96xf32)
        add__50 = paddle._C_ops.add_(matmul_66, parameter_136)

        # pd_op.gelu: (-1x16x96xf32) <- (-1x16x96xf32)
        gelu_8 = paddle._C_ops.gelu(add__50, False)

        # pd_op.matmul: (-1x16x24xf32) <- (-1x16x96xf32, 96x24xf32)
        matmul_67 = paddle._C_ops.matmul(gelu_8, parameter_137, False, False)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, 24xf32)
        add__51 = paddle._C_ops.add_(matmul_67, parameter_138)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, -1x16x24xf32)
        add__52 = paddle._C_ops.add_(add__49, add__51)

        # pd_op.shape: (3xi32) <- (-1x197x384xf32)
        shape_14 = paddle._C_ops.shape(add__47)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_88 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_89 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(shape_14, [0], full_int_array_88, full_int_array_89, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_101 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_102 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_33 = [slice_36, full_101, full_102]

        # pd_op.reshape: (-1x196x384xf32, 0x-1x16x24xf32) <- (-1x16x24xf32, [1xi32, 1xi32, 1xi32])
        reshape_12, reshape_13 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__52, [x.reshape([]) for x in combine_33]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.layer_norm: (-1x196x384xf32, -196xf32, -196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_84, layer_norm_85, layer_norm_86 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(reshape_12, parameter_139, parameter_140, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_90 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_91 = [197]

        # pd_op.slice: (-1x196x384xf32) <- (-1x197x384xf32, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(add__47, [1], full_int_array_90, full_int_array_91, [1], [])

        # pd_op.matmul: (-1x196x384xf32) <- (-1x196x384xf32, 384x384xf32)
        matmul_68 = paddle._C_ops.matmul(layer_norm_84, parameter_141, False, False)

        # pd_op.layer_norm: (-1x196x384xf32, -196xf32, -196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_87, layer_norm_88, layer_norm_89 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(matmul_68, parameter_142, parameter_143, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.add_: (-1x196x384xf32) <- (-1x196x384xf32, -1x196x384xf32)
        add__53 = paddle._C_ops.add_(slice_37, layer_norm_87)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_92 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_93 = [2147483647]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_94 = [1]

        # pd_op.set_value_with_tensor_: (-1x197x384xf32) <- (-1x197x384xf32, -1x196x384xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__4 = paddle._C_ops.set_value_with_tensor_(add__47, add__53, full_int_array_92, full_int_array_93, full_int_array_94, [1], [], [])

        # pd_op.layer_norm: (-1x197x384xf32, -197xf32, -197xf32) <- (-1x197x384xf32, 384xf32, 384xf32)
        layer_norm_90, layer_norm_91, layer_norm_92 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(set_value_with_tensor__4, parameter_144, parameter_145, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x197x384xf32)
        shape_15 = paddle._C_ops.shape(layer_norm_90)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_95 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_96 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_38 = paddle._C_ops.slice(shape_15, [0], full_int_array_95, full_int_array_96, [1], [0])

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x384xf32, 384x768xf32)
        matmul_69 = paddle._C_ops.matmul(layer_norm_90, parameter_146, False, False)

        # pd_op.full: (1xi32) <- ()
        full_103 = paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_104 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_105 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_106 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_34 = [slice_38, full_103, full_104, full_105, full_106]

        # pd_op.reshape_: (-1x197x2x6x64xf32, 0x-1x197x768xf32) <- (-1x197x768xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__58, reshape__59 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_69, [x.reshape([]) for x in combine_34]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x6x197x64xf32) <- (-1x197x2x6x64xf32)
        transpose_38 = paddle._C_ops.transpose(reshape__58, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_97 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_98 = [1]

        # pd_op.slice: (-1x6x197x64xf32) <- (2x-1x6x197x64xf32, 1xi64, 1xi64)
        slice_39 = paddle._C_ops.slice(transpose_38, [0], full_int_array_97, full_int_array_98, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_99 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_100 = [2]

        # pd_op.slice: (-1x6x197x64xf32) <- (2x-1x6x197x64xf32, 1xi64, 1xi64)
        slice_40 = paddle._C_ops.slice(transpose_38, [0], full_int_array_99, full_int_array_100, [1], [0])

        # pd_op.matmul: (-1x197x384xf32) <- (-1x197x384xf32, 384x384xf32)
        matmul_70 = paddle._C_ops.matmul(layer_norm_90, parameter_147, False, False)

        # pd_op.full: (1xi32) <- ()
        full_107 = paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_108 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_109 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_35 = [slice_38, full_107, full_108, full_109]

        # pd_op.reshape_: (-1x197x6x64xf32, 0x-1x197x384xf32) <- (-1x197x384xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__60, reshape__61 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_70, [x.reshape([]) for x in combine_35]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x6x197x64xf32) <- (-1x197x6x64xf32)
        transpose_39 = paddle._C_ops.transpose(reshape__60, [0, 2, 1, 3])

        # pd_op.transpose: (-1x6x64x197xf32) <- (-1x6x197x64xf32)
        transpose_40 = paddle._C_ops.transpose(slice_40, [0, 1, 3, 2])

        # pd_op.matmul: (-1x6x197x197xf32) <- (-1x6x197x64xf32, -1x6x64x197xf32)
        matmul_71 = paddle._C_ops.matmul(slice_39, transpose_40, False, False)

        # pd_op.full: (1xf32) <- ()
        full_110 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x6x197x197xf32) <- (-1x6x197x197xf32, 1xf32)
        scale__9 = paddle._C_ops.scale_(matmul_71, full_110, float('0'), True)

        # pd_op.softmax_: (-1x6x197x197xf32) <- (-1x6x197x197xf32)
        softmax__9 = paddle._C_ops.softmax_(scale__9, -1)

        # pd_op.matmul: (-1x6x197x64xf32) <- (-1x6x197x197xf32, -1x6x197x64xf32)
        matmul_72 = paddle._C_ops.matmul(softmax__9, transpose_39, False, False)

        # pd_op.transpose: (-1x197x6x64xf32) <- (-1x6x197x64xf32)
        transpose_41 = paddle._C_ops.transpose(matmul_72, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_111 = paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_112 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_36 = [slice_38, full_111, full_112]

        # pd_op.reshape_: (-1x197x384xf32, 0x-1x197x6x64xf32) <- (-1x197x6x64xf32, [1xi32, 1xi32, 1xi32])
        reshape__62, reshape__63 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_41, [x.reshape([]) for x in combine_36]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x197x384xf32) <- (-1x197x384xf32, 384x384xf32)
        matmul_73 = paddle._C_ops.matmul(reshape__62, parameter_148, False, False)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, 384xf32)
        add__54 = paddle._C_ops.add_(matmul_73, parameter_149)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, -1x197x384xf32)
        add__55 = paddle._C_ops.add_(set_value_with_tensor__4, add__54)

        # pd_op.layer_norm: (-1x197x384xf32, -197xf32, -197xf32) <- (-1x197x384xf32, 384xf32, 384xf32)
        layer_norm_93, layer_norm_94, layer_norm_95 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__55, parameter_150, parameter_151, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x1536xf32) <- (-1x197x384xf32, 384x1536xf32)
        matmul_74 = paddle._C_ops.matmul(layer_norm_93, parameter_152, False, False)

        # pd_op.add_: (-1x197x1536xf32) <- (-1x197x1536xf32, 1536xf32)
        add__56 = paddle._C_ops.add_(matmul_74, parameter_153)

        # pd_op.gelu: (-1x197x1536xf32) <- (-1x197x1536xf32)
        gelu_9 = paddle._C_ops.gelu(add__56, False)

        # pd_op.matmul: (-1x197x384xf32) <- (-1x197x1536xf32, 1536x384xf32)
        matmul_75 = paddle._C_ops.matmul(gelu_9, parameter_154, False, False)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, 384xf32)
        add__57 = paddle._C_ops.add_(matmul_75, parameter_155)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, -1x197x384xf32)
        add__58 = paddle._C_ops.add_(add__55, add__57)

        # pd_op.layer_norm: (-1x16x24xf32, -16xf32, -16xf32) <- (-1x16x24xf32, 24xf32, 24xf32)
        layer_norm_96, layer_norm_97, layer_norm_98 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__52, parameter_156, parameter_157, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x16x24xf32)
        shape_16 = paddle._C_ops.shape(layer_norm_96)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_101 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_102 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_41 = paddle._C_ops.slice(shape_16, [0], full_int_array_101, full_int_array_102, [1], [0])

        # pd_op.matmul: (-1x16x48xf32) <- (-1x16x24xf32, 24x48xf32)
        matmul_76 = paddle._C_ops.matmul(layer_norm_96, parameter_158, False, False)

        # pd_op.full: (1xi32) <- ()
        full_113 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_114 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_115 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_116 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_37 = [slice_41, full_113, full_114, full_115, full_116]

        # pd_op.reshape_: (-1x16x2x4x6xf32, 0x-1x16x48xf32) <- (-1x16x48xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__64, reshape__65 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_76, [x.reshape([]) for x in combine_37]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x4x16x6xf32) <- (-1x16x2x4x6xf32)
        transpose_42 = paddle._C_ops.transpose(reshape__64, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_103 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_104 = [1]

        # pd_op.slice: (-1x4x16x6xf32) <- (2x-1x4x16x6xf32, 1xi64, 1xi64)
        slice_42 = paddle._C_ops.slice(transpose_42, [0], full_int_array_103, full_int_array_104, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_105 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_106 = [2]

        # pd_op.slice: (-1x4x16x6xf32) <- (2x-1x4x16x6xf32, 1xi64, 1xi64)
        slice_43 = paddle._C_ops.slice(transpose_42, [0], full_int_array_105, full_int_array_106, [1], [0])

        # pd_op.matmul: (-1x16x24xf32) <- (-1x16x24xf32, 24x24xf32)
        matmul_77 = paddle._C_ops.matmul(layer_norm_96, parameter_159, False, False)

        # pd_op.full: (1xi32) <- ()
        full_117 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_118 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_119 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_38 = [slice_41, full_117, full_118, full_119]

        # pd_op.reshape_: (-1x16x4x6xf32, 0x-1x16x24xf32) <- (-1x16x24xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__66, reshape__67 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_77, [x.reshape([]) for x in combine_38]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x16x6xf32) <- (-1x16x4x6xf32)
        transpose_43 = paddle._C_ops.transpose(reshape__66, [0, 2, 1, 3])

        # pd_op.transpose: (-1x4x6x16xf32) <- (-1x4x16x6xf32)
        transpose_44 = paddle._C_ops.transpose(slice_43, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x16x16xf32) <- (-1x4x16x6xf32, -1x4x6x16xf32)
        matmul_78 = paddle._C_ops.matmul(slice_42, transpose_44, False, False)

        # pd_op.full: (1xf32) <- ()
        full_120 = paddle._C_ops.full([1], float('0.408248'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x4x16x16xf32) <- (-1x4x16x16xf32, 1xf32)
        scale__10 = paddle._C_ops.scale_(matmul_78, full_120, float('0'), True)

        # pd_op.softmax_: (-1x4x16x16xf32) <- (-1x4x16x16xf32)
        softmax__10 = paddle._C_ops.softmax_(scale__10, -1)

        # pd_op.matmul: (-1x4x16x6xf32) <- (-1x4x16x16xf32, -1x4x16x6xf32)
        matmul_79 = paddle._C_ops.matmul(softmax__10, transpose_43, False, False)

        # pd_op.transpose: (-1x16x4x6xf32) <- (-1x4x16x6xf32)
        transpose_45 = paddle._C_ops.transpose(matmul_79, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_121 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_122 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_39 = [slice_41, full_121, full_122]

        # pd_op.reshape_: (-1x16x24xf32, 0x-1x16x4x6xf32) <- (-1x16x4x6xf32, [1xi32, 1xi32, 1xi32])
        reshape__68, reshape__69 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_45, [x.reshape([]) for x in combine_39]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16x24xf32) <- (-1x16x24xf32, 24x24xf32)
        matmul_80 = paddle._C_ops.matmul(reshape__68, parameter_160, False, False)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, 24xf32)
        add__59 = paddle._C_ops.add_(matmul_80, parameter_161)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, -1x16x24xf32)
        add__60 = paddle._C_ops.add_(add__52, add__59)

        # pd_op.layer_norm: (-1x16x24xf32, -16xf32, -16xf32) <- (-1x16x24xf32, 24xf32, 24xf32)
        layer_norm_99, layer_norm_100, layer_norm_101 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__60, parameter_162, parameter_163, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x16x96xf32) <- (-1x16x24xf32, 24x96xf32)
        matmul_81 = paddle._C_ops.matmul(layer_norm_99, parameter_164, False, False)

        # pd_op.add_: (-1x16x96xf32) <- (-1x16x96xf32, 96xf32)
        add__61 = paddle._C_ops.add_(matmul_81, parameter_165)

        # pd_op.gelu: (-1x16x96xf32) <- (-1x16x96xf32)
        gelu_10 = paddle._C_ops.gelu(add__61, False)

        # pd_op.matmul: (-1x16x24xf32) <- (-1x16x96xf32, 96x24xf32)
        matmul_82 = paddle._C_ops.matmul(gelu_10, parameter_166, False, False)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, 24xf32)
        add__62 = paddle._C_ops.add_(matmul_82, parameter_167)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, -1x16x24xf32)
        add__63 = paddle._C_ops.add_(add__60, add__62)

        # pd_op.shape: (3xi32) <- (-1x197x384xf32)
        shape_17 = paddle._C_ops.shape(add__58)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_107 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_108 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_44 = paddle._C_ops.slice(shape_17, [0], full_int_array_107, full_int_array_108, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_123 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_124 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_40 = [slice_44, full_123, full_124]

        # pd_op.reshape: (-1x196x384xf32, 0x-1x16x24xf32) <- (-1x16x24xf32, [1xi32, 1xi32, 1xi32])
        reshape_14, reshape_15 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__63, [x.reshape([]) for x in combine_40]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.layer_norm: (-1x196x384xf32, -196xf32, -196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_102, layer_norm_103, layer_norm_104 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(reshape_14, parameter_168, parameter_169, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_109 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_110 = [197]

        # pd_op.slice: (-1x196x384xf32) <- (-1x197x384xf32, 1xi64, 1xi64)
        slice_45 = paddle._C_ops.slice(add__58, [1], full_int_array_109, full_int_array_110, [1], [])

        # pd_op.matmul: (-1x196x384xf32) <- (-1x196x384xf32, 384x384xf32)
        matmul_83 = paddle._C_ops.matmul(layer_norm_102, parameter_170, False, False)

        # pd_op.layer_norm: (-1x196x384xf32, -196xf32, -196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_105, layer_norm_106, layer_norm_107 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(matmul_83, parameter_171, parameter_172, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.add_: (-1x196x384xf32) <- (-1x196x384xf32, -1x196x384xf32)
        add__64 = paddle._C_ops.add_(slice_45, layer_norm_105)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_111 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_112 = [2147483647]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_113 = [1]

        # pd_op.set_value_with_tensor_: (-1x197x384xf32) <- (-1x197x384xf32, -1x196x384xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__5 = paddle._C_ops.set_value_with_tensor_(add__58, add__64, full_int_array_111, full_int_array_112, full_int_array_113, [1], [], [])

        # pd_op.layer_norm: (-1x197x384xf32, -197xf32, -197xf32) <- (-1x197x384xf32, 384xf32, 384xf32)
        layer_norm_108, layer_norm_109, layer_norm_110 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(set_value_with_tensor__5, parameter_173, parameter_174, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x197x384xf32)
        shape_18 = paddle._C_ops.shape(layer_norm_108)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_114 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_115 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_46 = paddle._C_ops.slice(shape_18, [0], full_int_array_114, full_int_array_115, [1], [0])

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x384xf32, 384x768xf32)
        matmul_84 = paddle._C_ops.matmul(layer_norm_108, parameter_175, False, False)

        # pd_op.full: (1xi32) <- ()
        full_125 = paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_126 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_127 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_128 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_41 = [slice_46, full_125, full_126, full_127, full_128]

        # pd_op.reshape_: (-1x197x2x6x64xf32, 0x-1x197x768xf32) <- (-1x197x768xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__70, reshape__71 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_84, [x.reshape([]) for x in combine_41]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x6x197x64xf32) <- (-1x197x2x6x64xf32)
        transpose_46 = paddle._C_ops.transpose(reshape__70, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_116 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_117 = [1]

        # pd_op.slice: (-1x6x197x64xf32) <- (2x-1x6x197x64xf32, 1xi64, 1xi64)
        slice_47 = paddle._C_ops.slice(transpose_46, [0], full_int_array_116, full_int_array_117, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_118 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_119 = [2]

        # pd_op.slice: (-1x6x197x64xf32) <- (2x-1x6x197x64xf32, 1xi64, 1xi64)
        slice_48 = paddle._C_ops.slice(transpose_46, [0], full_int_array_118, full_int_array_119, [1], [0])

        # pd_op.matmul: (-1x197x384xf32) <- (-1x197x384xf32, 384x384xf32)
        matmul_85 = paddle._C_ops.matmul(layer_norm_108, parameter_176, False, False)

        # pd_op.full: (1xi32) <- ()
        full_129 = paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_130 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_131 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_42 = [slice_46, full_129, full_130, full_131]

        # pd_op.reshape_: (-1x197x6x64xf32, 0x-1x197x384xf32) <- (-1x197x384xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__72, reshape__73 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_85, [x.reshape([]) for x in combine_42]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x6x197x64xf32) <- (-1x197x6x64xf32)
        transpose_47 = paddle._C_ops.transpose(reshape__72, [0, 2, 1, 3])

        # pd_op.transpose: (-1x6x64x197xf32) <- (-1x6x197x64xf32)
        transpose_48 = paddle._C_ops.transpose(slice_48, [0, 1, 3, 2])

        # pd_op.matmul: (-1x6x197x197xf32) <- (-1x6x197x64xf32, -1x6x64x197xf32)
        matmul_86 = paddle._C_ops.matmul(slice_47, transpose_48, False, False)

        # pd_op.full: (1xf32) <- ()
        full_132 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x6x197x197xf32) <- (-1x6x197x197xf32, 1xf32)
        scale__11 = paddle._C_ops.scale_(matmul_86, full_132, float('0'), True)

        # pd_op.softmax_: (-1x6x197x197xf32) <- (-1x6x197x197xf32)
        softmax__11 = paddle._C_ops.softmax_(scale__11, -1)

        # pd_op.matmul: (-1x6x197x64xf32) <- (-1x6x197x197xf32, -1x6x197x64xf32)
        matmul_87 = paddle._C_ops.matmul(softmax__11, transpose_47, False, False)

        # pd_op.transpose: (-1x197x6x64xf32) <- (-1x6x197x64xf32)
        transpose_49 = paddle._C_ops.transpose(matmul_87, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_133 = paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_134 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_43 = [slice_46, full_133, full_134]

        # pd_op.reshape_: (-1x197x384xf32, 0x-1x197x6x64xf32) <- (-1x197x6x64xf32, [1xi32, 1xi32, 1xi32])
        reshape__74, reshape__75 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_49, [x.reshape([]) for x in combine_43]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x197x384xf32) <- (-1x197x384xf32, 384x384xf32)
        matmul_88 = paddle._C_ops.matmul(reshape__74, parameter_177, False, False)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, 384xf32)
        add__65 = paddle._C_ops.add_(matmul_88, parameter_178)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, -1x197x384xf32)
        add__66 = paddle._C_ops.add_(set_value_with_tensor__5, add__65)

        # pd_op.layer_norm: (-1x197x384xf32, -197xf32, -197xf32) <- (-1x197x384xf32, 384xf32, 384xf32)
        layer_norm_111, layer_norm_112, layer_norm_113 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__66, parameter_179, parameter_180, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x1536xf32) <- (-1x197x384xf32, 384x1536xf32)
        matmul_89 = paddle._C_ops.matmul(layer_norm_111, parameter_181, False, False)

        # pd_op.add_: (-1x197x1536xf32) <- (-1x197x1536xf32, 1536xf32)
        add__67 = paddle._C_ops.add_(matmul_89, parameter_182)

        # pd_op.gelu: (-1x197x1536xf32) <- (-1x197x1536xf32)
        gelu_11 = paddle._C_ops.gelu(add__67, False)

        # pd_op.matmul: (-1x197x384xf32) <- (-1x197x1536xf32, 1536x384xf32)
        matmul_90 = paddle._C_ops.matmul(gelu_11, parameter_183, False, False)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, 384xf32)
        add__68 = paddle._C_ops.add_(matmul_90, parameter_184)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, -1x197x384xf32)
        add__69 = paddle._C_ops.add_(add__66, add__68)

        # pd_op.layer_norm: (-1x16x24xf32, -16xf32, -16xf32) <- (-1x16x24xf32, 24xf32, 24xf32)
        layer_norm_114, layer_norm_115, layer_norm_116 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__63, parameter_185, parameter_186, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x16x24xf32)
        shape_19 = paddle._C_ops.shape(layer_norm_114)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_120 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_121 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_49 = paddle._C_ops.slice(shape_19, [0], full_int_array_120, full_int_array_121, [1], [0])

        # pd_op.matmul: (-1x16x48xf32) <- (-1x16x24xf32, 24x48xf32)
        matmul_91 = paddle._C_ops.matmul(layer_norm_114, parameter_187, False, False)

        # pd_op.full: (1xi32) <- ()
        full_135 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_136 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_137 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_138 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_44 = [slice_49, full_135, full_136, full_137, full_138]

        # pd_op.reshape_: (-1x16x2x4x6xf32, 0x-1x16x48xf32) <- (-1x16x48xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__76, reshape__77 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_91, [x.reshape([]) for x in combine_44]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x4x16x6xf32) <- (-1x16x2x4x6xf32)
        transpose_50 = paddle._C_ops.transpose(reshape__76, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_122 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_123 = [1]

        # pd_op.slice: (-1x4x16x6xf32) <- (2x-1x4x16x6xf32, 1xi64, 1xi64)
        slice_50 = paddle._C_ops.slice(transpose_50, [0], full_int_array_122, full_int_array_123, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_124 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_125 = [2]

        # pd_op.slice: (-1x4x16x6xf32) <- (2x-1x4x16x6xf32, 1xi64, 1xi64)
        slice_51 = paddle._C_ops.slice(transpose_50, [0], full_int_array_124, full_int_array_125, [1], [0])

        # pd_op.matmul: (-1x16x24xf32) <- (-1x16x24xf32, 24x24xf32)
        matmul_92 = paddle._C_ops.matmul(layer_norm_114, parameter_188, False, False)

        # pd_op.full: (1xi32) <- ()
        full_139 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_140 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_141 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_45 = [slice_49, full_139, full_140, full_141]

        # pd_op.reshape_: (-1x16x4x6xf32, 0x-1x16x24xf32) <- (-1x16x24xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__78, reshape__79 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_92, [x.reshape([]) for x in combine_45]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x16x6xf32) <- (-1x16x4x6xf32)
        transpose_51 = paddle._C_ops.transpose(reshape__78, [0, 2, 1, 3])

        # pd_op.transpose: (-1x4x6x16xf32) <- (-1x4x16x6xf32)
        transpose_52 = paddle._C_ops.transpose(slice_51, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x16x16xf32) <- (-1x4x16x6xf32, -1x4x6x16xf32)
        matmul_93 = paddle._C_ops.matmul(slice_50, transpose_52, False, False)

        # pd_op.full: (1xf32) <- ()
        full_142 = paddle._C_ops.full([1], float('0.408248'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x4x16x16xf32) <- (-1x4x16x16xf32, 1xf32)
        scale__12 = paddle._C_ops.scale_(matmul_93, full_142, float('0'), True)

        # pd_op.softmax_: (-1x4x16x16xf32) <- (-1x4x16x16xf32)
        softmax__12 = paddle._C_ops.softmax_(scale__12, -1)

        # pd_op.matmul: (-1x4x16x6xf32) <- (-1x4x16x16xf32, -1x4x16x6xf32)
        matmul_94 = paddle._C_ops.matmul(softmax__12, transpose_51, False, False)

        # pd_op.transpose: (-1x16x4x6xf32) <- (-1x4x16x6xf32)
        transpose_53 = paddle._C_ops.transpose(matmul_94, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_143 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_144 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_46 = [slice_49, full_143, full_144]

        # pd_op.reshape_: (-1x16x24xf32, 0x-1x16x4x6xf32) <- (-1x16x4x6xf32, [1xi32, 1xi32, 1xi32])
        reshape__80, reshape__81 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_53, [x.reshape([]) for x in combine_46]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16x24xf32) <- (-1x16x24xf32, 24x24xf32)
        matmul_95 = paddle._C_ops.matmul(reshape__80, parameter_189, False, False)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, 24xf32)
        add__70 = paddle._C_ops.add_(matmul_95, parameter_190)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, -1x16x24xf32)
        add__71 = paddle._C_ops.add_(add__63, add__70)

        # pd_op.layer_norm: (-1x16x24xf32, -16xf32, -16xf32) <- (-1x16x24xf32, 24xf32, 24xf32)
        layer_norm_117, layer_norm_118, layer_norm_119 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__71, parameter_191, parameter_192, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x16x96xf32) <- (-1x16x24xf32, 24x96xf32)
        matmul_96 = paddle._C_ops.matmul(layer_norm_117, parameter_193, False, False)

        # pd_op.add_: (-1x16x96xf32) <- (-1x16x96xf32, 96xf32)
        add__72 = paddle._C_ops.add_(matmul_96, parameter_194)

        # pd_op.gelu: (-1x16x96xf32) <- (-1x16x96xf32)
        gelu_12 = paddle._C_ops.gelu(add__72, False)

        # pd_op.matmul: (-1x16x24xf32) <- (-1x16x96xf32, 96x24xf32)
        matmul_97 = paddle._C_ops.matmul(gelu_12, parameter_195, False, False)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, 24xf32)
        add__73 = paddle._C_ops.add_(matmul_97, parameter_196)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, -1x16x24xf32)
        add__74 = paddle._C_ops.add_(add__71, add__73)

        # pd_op.shape: (3xi32) <- (-1x197x384xf32)
        shape_20 = paddle._C_ops.shape(add__69)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_126 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_127 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_52 = paddle._C_ops.slice(shape_20, [0], full_int_array_126, full_int_array_127, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_145 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_146 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_47 = [slice_52, full_145, full_146]

        # pd_op.reshape: (-1x196x384xf32, 0x-1x16x24xf32) <- (-1x16x24xf32, [1xi32, 1xi32, 1xi32])
        reshape_16, reshape_17 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__74, [x.reshape([]) for x in combine_47]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.layer_norm: (-1x196x384xf32, -196xf32, -196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_120, layer_norm_121, layer_norm_122 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(reshape_16, parameter_197, parameter_198, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_128 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_129 = [197]

        # pd_op.slice: (-1x196x384xf32) <- (-1x197x384xf32, 1xi64, 1xi64)
        slice_53 = paddle._C_ops.slice(add__69, [1], full_int_array_128, full_int_array_129, [1], [])

        # pd_op.matmul: (-1x196x384xf32) <- (-1x196x384xf32, 384x384xf32)
        matmul_98 = paddle._C_ops.matmul(layer_norm_120, parameter_199, False, False)

        # pd_op.layer_norm: (-1x196x384xf32, -196xf32, -196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_123, layer_norm_124, layer_norm_125 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(matmul_98, parameter_200, parameter_201, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.add_: (-1x196x384xf32) <- (-1x196x384xf32, -1x196x384xf32)
        add__75 = paddle._C_ops.add_(slice_53, layer_norm_123)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_130 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_131 = [2147483647]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_132 = [1]

        # pd_op.set_value_with_tensor_: (-1x197x384xf32) <- (-1x197x384xf32, -1x196x384xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__6 = paddle._C_ops.set_value_with_tensor_(add__69, add__75, full_int_array_130, full_int_array_131, full_int_array_132, [1], [], [])

        # pd_op.layer_norm: (-1x197x384xf32, -197xf32, -197xf32) <- (-1x197x384xf32, 384xf32, 384xf32)
        layer_norm_126, layer_norm_127, layer_norm_128 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(set_value_with_tensor__6, parameter_202, parameter_203, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x197x384xf32)
        shape_21 = paddle._C_ops.shape(layer_norm_126)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_133 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_134 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_54 = paddle._C_ops.slice(shape_21, [0], full_int_array_133, full_int_array_134, [1], [0])

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x384xf32, 384x768xf32)
        matmul_99 = paddle._C_ops.matmul(layer_norm_126, parameter_204, False, False)

        # pd_op.full: (1xi32) <- ()
        full_147 = paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_148 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_149 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_150 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_48 = [slice_54, full_147, full_148, full_149, full_150]

        # pd_op.reshape_: (-1x197x2x6x64xf32, 0x-1x197x768xf32) <- (-1x197x768xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__82, reshape__83 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_99, [x.reshape([]) for x in combine_48]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x6x197x64xf32) <- (-1x197x2x6x64xf32)
        transpose_54 = paddle._C_ops.transpose(reshape__82, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_135 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_136 = [1]

        # pd_op.slice: (-1x6x197x64xf32) <- (2x-1x6x197x64xf32, 1xi64, 1xi64)
        slice_55 = paddle._C_ops.slice(transpose_54, [0], full_int_array_135, full_int_array_136, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_137 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_138 = [2]

        # pd_op.slice: (-1x6x197x64xf32) <- (2x-1x6x197x64xf32, 1xi64, 1xi64)
        slice_56 = paddle._C_ops.slice(transpose_54, [0], full_int_array_137, full_int_array_138, [1], [0])

        # pd_op.matmul: (-1x197x384xf32) <- (-1x197x384xf32, 384x384xf32)
        matmul_100 = paddle._C_ops.matmul(layer_norm_126, parameter_205, False, False)

        # pd_op.full: (1xi32) <- ()
        full_151 = paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_152 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_153 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_49 = [slice_54, full_151, full_152, full_153]

        # pd_op.reshape_: (-1x197x6x64xf32, 0x-1x197x384xf32) <- (-1x197x384xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__84, reshape__85 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_100, [x.reshape([]) for x in combine_49]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x6x197x64xf32) <- (-1x197x6x64xf32)
        transpose_55 = paddle._C_ops.transpose(reshape__84, [0, 2, 1, 3])

        # pd_op.transpose: (-1x6x64x197xf32) <- (-1x6x197x64xf32)
        transpose_56 = paddle._C_ops.transpose(slice_56, [0, 1, 3, 2])

        # pd_op.matmul: (-1x6x197x197xf32) <- (-1x6x197x64xf32, -1x6x64x197xf32)
        matmul_101 = paddle._C_ops.matmul(slice_55, transpose_56, False, False)

        # pd_op.full: (1xf32) <- ()
        full_154 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x6x197x197xf32) <- (-1x6x197x197xf32, 1xf32)
        scale__13 = paddle._C_ops.scale_(matmul_101, full_154, float('0'), True)

        # pd_op.softmax_: (-1x6x197x197xf32) <- (-1x6x197x197xf32)
        softmax__13 = paddle._C_ops.softmax_(scale__13, -1)

        # pd_op.matmul: (-1x6x197x64xf32) <- (-1x6x197x197xf32, -1x6x197x64xf32)
        matmul_102 = paddle._C_ops.matmul(softmax__13, transpose_55, False, False)

        # pd_op.transpose: (-1x197x6x64xf32) <- (-1x6x197x64xf32)
        transpose_57 = paddle._C_ops.transpose(matmul_102, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_155 = paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_156 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_50 = [slice_54, full_155, full_156]

        # pd_op.reshape_: (-1x197x384xf32, 0x-1x197x6x64xf32) <- (-1x197x6x64xf32, [1xi32, 1xi32, 1xi32])
        reshape__86, reshape__87 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_57, [x.reshape([]) for x in combine_50]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x197x384xf32) <- (-1x197x384xf32, 384x384xf32)
        matmul_103 = paddle._C_ops.matmul(reshape__86, parameter_206, False, False)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, 384xf32)
        add__76 = paddle._C_ops.add_(matmul_103, parameter_207)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, -1x197x384xf32)
        add__77 = paddle._C_ops.add_(set_value_with_tensor__6, add__76)

        # pd_op.layer_norm: (-1x197x384xf32, -197xf32, -197xf32) <- (-1x197x384xf32, 384xf32, 384xf32)
        layer_norm_129, layer_norm_130, layer_norm_131 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__77, parameter_208, parameter_209, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x1536xf32) <- (-1x197x384xf32, 384x1536xf32)
        matmul_104 = paddle._C_ops.matmul(layer_norm_129, parameter_210, False, False)

        # pd_op.add_: (-1x197x1536xf32) <- (-1x197x1536xf32, 1536xf32)
        add__78 = paddle._C_ops.add_(matmul_104, parameter_211)

        # pd_op.gelu: (-1x197x1536xf32) <- (-1x197x1536xf32)
        gelu_13 = paddle._C_ops.gelu(add__78, False)

        # pd_op.matmul: (-1x197x384xf32) <- (-1x197x1536xf32, 1536x384xf32)
        matmul_105 = paddle._C_ops.matmul(gelu_13, parameter_212, False, False)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, 384xf32)
        add__79 = paddle._C_ops.add_(matmul_105, parameter_213)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, -1x197x384xf32)
        add__80 = paddle._C_ops.add_(add__77, add__79)

        # pd_op.layer_norm: (-1x16x24xf32, -16xf32, -16xf32) <- (-1x16x24xf32, 24xf32, 24xf32)
        layer_norm_132, layer_norm_133, layer_norm_134 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__74, parameter_214, parameter_215, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x16x24xf32)
        shape_22 = paddle._C_ops.shape(layer_norm_132)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_139 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_140 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_57 = paddle._C_ops.slice(shape_22, [0], full_int_array_139, full_int_array_140, [1], [0])

        # pd_op.matmul: (-1x16x48xf32) <- (-1x16x24xf32, 24x48xf32)
        matmul_106 = paddle._C_ops.matmul(layer_norm_132, parameter_216, False, False)

        # pd_op.full: (1xi32) <- ()
        full_157 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_158 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_159 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_160 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_51 = [slice_57, full_157, full_158, full_159, full_160]

        # pd_op.reshape_: (-1x16x2x4x6xf32, 0x-1x16x48xf32) <- (-1x16x48xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__88, reshape__89 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_106, [x.reshape([]) for x in combine_51]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x4x16x6xf32) <- (-1x16x2x4x6xf32)
        transpose_58 = paddle._C_ops.transpose(reshape__88, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_141 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_142 = [1]

        # pd_op.slice: (-1x4x16x6xf32) <- (2x-1x4x16x6xf32, 1xi64, 1xi64)
        slice_58 = paddle._C_ops.slice(transpose_58, [0], full_int_array_141, full_int_array_142, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_143 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_144 = [2]

        # pd_op.slice: (-1x4x16x6xf32) <- (2x-1x4x16x6xf32, 1xi64, 1xi64)
        slice_59 = paddle._C_ops.slice(transpose_58, [0], full_int_array_143, full_int_array_144, [1], [0])

        # pd_op.matmul: (-1x16x24xf32) <- (-1x16x24xf32, 24x24xf32)
        matmul_107 = paddle._C_ops.matmul(layer_norm_132, parameter_217, False, False)

        # pd_op.full: (1xi32) <- ()
        full_161 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_162 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_163 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_52 = [slice_57, full_161, full_162, full_163]

        # pd_op.reshape_: (-1x16x4x6xf32, 0x-1x16x24xf32) <- (-1x16x24xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__90, reshape__91 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_107, [x.reshape([]) for x in combine_52]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x16x6xf32) <- (-1x16x4x6xf32)
        transpose_59 = paddle._C_ops.transpose(reshape__90, [0, 2, 1, 3])

        # pd_op.transpose: (-1x4x6x16xf32) <- (-1x4x16x6xf32)
        transpose_60 = paddle._C_ops.transpose(slice_59, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x16x16xf32) <- (-1x4x16x6xf32, -1x4x6x16xf32)
        matmul_108 = paddle._C_ops.matmul(slice_58, transpose_60, False, False)

        # pd_op.full: (1xf32) <- ()
        full_164 = paddle._C_ops.full([1], float('0.408248'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x4x16x16xf32) <- (-1x4x16x16xf32, 1xf32)
        scale__14 = paddle._C_ops.scale_(matmul_108, full_164, float('0'), True)

        # pd_op.softmax_: (-1x4x16x16xf32) <- (-1x4x16x16xf32)
        softmax__14 = paddle._C_ops.softmax_(scale__14, -1)

        # pd_op.matmul: (-1x4x16x6xf32) <- (-1x4x16x16xf32, -1x4x16x6xf32)
        matmul_109 = paddle._C_ops.matmul(softmax__14, transpose_59, False, False)

        # pd_op.transpose: (-1x16x4x6xf32) <- (-1x4x16x6xf32)
        transpose_61 = paddle._C_ops.transpose(matmul_109, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_165 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_166 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_53 = [slice_57, full_165, full_166]

        # pd_op.reshape_: (-1x16x24xf32, 0x-1x16x4x6xf32) <- (-1x16x4x6xf32, [1xi32, 1xi32, 1xi32])
        reshape__92, reshape__93 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_61, [x.reshape([]) for x in combine_53]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16x24xf32) <- (-1x16x24xf32, 24x24xf32)
        matmul_110 = paddle._C_ops.matmul(reshape__92, parameter_218, False, False)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, 24xf32)
        add__81 = paddle._C_ops.add_(matmul_110, parameter_219)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, -1x16x24xf32)
        add__82 = paddle._C_ops.add_(add__74, add__81)

        # pd_op.layer_norm: (-1x16x24xf32, -16xf32, -16xf32) <- (-1x16x24xf32, 24xf32, 24xf32)
        layer_norm_135, layer_norm_136, layer_norm_137 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__82, parameter_220, parameter_221, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x16x96xf32) <- (-1x16x24xf32, 24x96xf32)
        matmul_111 = paddle._C_ops.matmul(layer_norm_135, parameter_222, False, False)

        # pd_op.add_: (-1x16x96xf32) <- (-1x16x96xf32, 96xf32)
        add__83 = paddle._C_ops.add_(matmul_111, parameter_223)

        # pd_op.gelu: (-1x16x96xf32) <- (-1x16x96xf32)
        gelu_14 = paddle._C_ops.gelu(add__83, False)

        # pd_op.matmul: (-1x16x24xf32) <- (-1x16x96xf32, 96x24xf32)
        matmul_112 = paddle._C_ops.matmul(gelu_14, parameter_224, False, False)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, 24xf32)
        add__84 = paddle._C_ops.add_(matmul_112, parameter_225)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, -1x16x24xf32)
        add__85 = paddle._C_ops.add_(add__82, add__84)

        # pd_op.shape: (3xi32) <- (-1x197x384xf32)
        shape_23 = paddle._C_ops.shape(add__80)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_145 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_146 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_60 = paddle._C_ops.slice(shape_23, [0], full_int_array_145, full_int_array_146, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_167 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_168 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_54 = [slice_60, full_167, full_168]

        # pd_op.reshape: (-1x196x384xf32, 0x-1x16x24xf32) <- (-1x16x24xf32, [1xi32, 1xi32, 1xi32])
        reshape_18, reshape_19 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__85, [x.reshape([]) for x in combine_54]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.layer_norm: (-1x196x384xf32, -196xf32, -196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_138, layer_norm_139, layer_norm_140 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(reshape_18, parameter_226, parameter_227, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_147 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_148 = [197]

        # pd_op.slice: (-1x196x384xf32) <- (-1x197x384xf32, 1xi64, 1xi64)
        slice_61 = paddle._C_ops.slice(add__80, [1], full_int_array_147, full_int_array_148, [1], [])

        # pd_op.matmul: (-1x196x384xf32) <- (-1x196x384xf32, 384x384xf32)
        matmul_113 = paddle._C_ops.matmul(layer_norm_138, parameter_228, False, False)

        # pd_op.layer_norm: (-1x196x384xf32, -196xf32, -196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_141, layer_norm_142, layer_norm_143 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(matmul_113, parameter_229, parameter_230, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.add_: (-1x196x384xf32) <- (-1x196x384xf32, -1x196x384xf32)
        add__86 = paddle._C_ops.add_(slice_61, layer_norm_141)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_149 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_150 = [2147483647]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_151 = [1]

        # pd_op.set_value_with_tensor_: (-1x197x384xf32) <- (-1x197x384xf32, -1x196x384xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__7 = paddle._C_ops.set_value_with_tensor_(add__80, add__86, full_int_array_149, full_int_array_150, full_int_array_151, [1], [], [])

        # pd_op.layer_norm: (-1x197x384xf32, -197xf32, -197xf32) <- (-1x197x384xf32, 384xf32, 384xf32)
        layer_norm_144, layer_norm_145, layer_norm_146 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(set_value_with_tensor__7, parameter_231, parameter_232, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x197x384xf32)
        shape_24 = paddle._C_ops.shape(layer_norm_144)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_152 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_153 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_62 = paddle._C_ops.slice(shape_24, [0], full_int_array_152, full_int_array_153, [1], [0])

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x384xf32, 384x768xf32)
        matmul_114 = paddle._C_ops.matmul(layer_norm_144, parameter_233, False, False)

        # pd_op.full: (1xi32) <- ()
        full_169 = paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_170 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_171 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_172 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_55 = [slice_62, full_169, full_170, full_171, full_172]

        # pd_op.reshape_: (-1x197x2x6x64xf32, 0x-1x197x768xf32) <- (-1x197x768xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__94, reshape__95 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_114, [x.reshape([]) for x in combine_55]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x6x197x64xf32) <- (-1x197x2x6x64xf32)
        transpose_62 = paddle._C_ops.transpose(reshape__94, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_154 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_155 = [1]

        # pd_op.slice: (-1x6x197x64xf32) <- (2x-1x6x197x64xf32, 1xi64, 1xi64)
        slice_63 = paddle._C_ops.slice(transpose_62, [0], full_int_array_154, full_int_array_155, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_156 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_157 = [2]

        # pd_op.slice: (-1x6x197x64xf32) <- (2x-1x6x197x64xf32, 1xi64, 1xi64)
        slice_64 = paddle._C_ops.slice(transpose_62, [0], full_int_array_156, full_int_array_157, [1], [0])

        # pd_op.matmul: (-1x197x384xf32) <- (-1x197x384xf32, 384x384xf32)
        matmul_115 = paddle._C_ops.matmul(layer_norm_144, parameter_234, False, False)

        # pd_op.full: (1xi32) <- ()
        full_173 = paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_174 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_175 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_56 = [slice_62, full_173, full_174, full_175]

        # pd_op.reshape_: (-1x197x6x64xf32, 0x-1x197x384xf32) <- (-1x197x384xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__96, reshape__97 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_115, [x.reshape([]) for x in combine_56]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x6x197x64xf32) <- (-1x197x6x64xf32)
        transpose_63 = paddle._C_ops.transpose(reshape__96, [0, 2, 1, 3])

        # pd_op.transpose: (-1x6x64x197xf32) <- (-1x6x197x64xf32)
        transpose_64 = paddle._C_ops.transpose(slice_64, [0, 1, 3, 2])

        # pd_op.matmul: (-1x6x197x197xf32) <- (-1x6x197x64xf32, -1x6x64x197xf32)
        matmul_116 = paddle._C_ops.matmul(slice_63, transpose_64, False, False)

        # pd_op.full: (1xf32) <- ()
        full_176 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x6x197x197xf32) <- (-1x6x197x197xf32, 1xf32)
        scale__15 = paddle._C_ops.scale_(matmul_116, full_176, float('0'), True)

        # pd_op.softmax_: (-1x6x197x197xf32) <- (-1x6x197x197xf32)
        softmax__15 = paddle._C_ops.softmax_(scale__15, -1)

        # pd_op.matmul: (-1x6x197x64xf32) <- (-1x6x197x197xf32, -1x6x197x64xf32)
        matmul_117 = paddle._C_ops.matmul(softmax__15, transpose_63, False, False)

        # pd_op.transpose: (-1x197x6x64xf32) <- (-1x6x197x64xf32)
        transpose_65 = paddle._C_ops.transpose(matmul_117, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_177 = paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_178 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_57 = [slice_62, full_177, full_178]

        # pd_op.reshape_: (-1x197x384xf32, 0x-1x197x6x64xf32) <- (-1x197x6x64xf32, [1xi32, 1xi32, 1xi32])
        reshape__98, reshape__99 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_65, [x.reshape([]) for x in combine_57]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x197x384xf32) <- (-1x197x384xf32, 384x384xf32)
        matmul_118 = paddle._C_ops.matmul(reshape__98, parameter_235, False, False)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, 384xf32)
        add__87 = paddle._C_ops.add_(matmul_118, parameter_236)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, -1x197x384xf32)
        add__88 = paddle._C_ops.add_(set_value_with_tensor__7, add__87)

        # pd_op.layer_norm: (-1x197x384xf32, -197xf32, -197xf32) <- (-1x197x384xf32, 384xf32, 384xf32)
        layer_norm_147, layer_norm_148, layer_norm_149 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__88, parameter_237, parameter_238, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x1536xf32) <- (-1x197x384xf32, 384x1536xf32)
        matmul_119 = paddle._C_ops.matmul(layer_norm_147, parameter_239, False, False)

        # pd_op.add_: (-1x197x1536xf32) <- (-1x197x1536xf32, 1536xf32)
        add__89 = paddle._C_ops.add_(matmul_119, parameter_240)

        # pd_op.gelu: (-1x197x1536xf32) <- (-1x197x1536xf32)
        gelu_15 = paddle._C_ops.gelu(add__89, False)

        # pd_op.matmul: (-1x197x384xf32) <- (-1x197x1536xf32, 1536x384xf32)
        matmul_120 = paddle._C_ops.matmul(gelu_15, parameter_241, False, False)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, 384xf32)
        add__90 = paddle._C_ops.add_(matmul_120, parameter_242)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, -1x197x384xf32)
        add__91 = paddle._C_ops.add_(add__88, add__90)

        # pd_op.layer_norm: (-1x16x24xf32, -16xf32, -16xf32) <- (-1x16x24xf32, 24xf32, 24xf32)
        layer_norm_150, layer_norm_151, layer_norm_152 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__85, parameter_243, parameter_244, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x16x24xf32)
        shape_25 = paddle._C_ops.shape(layer_norm_150)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_158 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_159 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_65 = paddle._C_ops.slice(shape_25, [0], full_int_array_158, full_int_array_159, [1], [0])

        # pd_op.matmul: (-1x16x48xf32) <- (-1x16x24xf32, 24x48xf32)
        matmul_121 = paddle._C_ops.matmul(layer_norm_150, parameter_245, False, False)

        # pd_op.full: (1xi32) <- ()
        full_179 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_180 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_181 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_182 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_58 = [slice_65, full_179, full_180, full_181, full_182]

        # pd_op.reshape_: (-1x16x2x4x6xf32, 0x-1x16x48xf32) <- (-1x16x48xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__100, reshape__101 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_121, [x.reshape([]) for x in combine_58]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x4x16x6xf32) <- (-1x16x2x4x6xf32)
        transpose_66 = paddle._C_ops.transpose(reshape__100, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_160 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_161 = [1]

        # pd_op.slice: (-1x4x16x6xf32) <- (2x-1x4x16x6xf32, 1xi64, 1xi64)
        slice_66 = paddle._C_ops.slice(transpose_66, [0], full_int_array_160, full_int_array_161, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_162 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_163 = [2]

        # pd_op.slice: (-1x4x16x6xf32) <- (2x-1x4x16x6xf32, 1xi64, 1xi64)
        slice_67 = paddle._C_ops.slice(transpose_66, [0], full_int_array_162, full_int_array_163, [1], [0])

        # pd_op.matmul: (-1x16x24xf32) <- (-1x16x24xf32, 24x24xf32)
        matmul_122 = paddle._C_ops.matmul(layer_norm_150, parameter_246, False, False)

        # pd_op.full: (1xi32) <- ()
        full_183 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_184 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_185 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_59 = [slice_65, full_183, full_184, full_185]

        # pd_op.reshape_: (-1x16x4x6xf32, 0x-1x16x24xf32) <- (-1x16x24xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__102, reshape__103 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_122, [x.reshape([]) for x in combine_59]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x16x6xf32) <- (-1x16x4x6xf32)
        transpose_67 = paddle._C_ops.transpose(reshape__102, [0, 2, 1, 3])

        # pd_op.transpose: (-1x4x6x16xf32) <- (-1x4x16x6xf32)
        transpose_68 = paddle._C_ops.transpose(slice_67, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x16x16xf32) <- (-1x4x16x6xf32, -1x4x6x16xf32)
        matmul_123 = paddle._C_ops.matmul(slice_66, transpose_68, False, False)

        # pd_op.full: (1xf32) <- ()
        full_186 = paddle._C_ops.full([1], float('0.408248'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x4x16x16xf32) <- (-1x4x16x16xf32, 1xf32)
        scale__16 = paddle._C_ops.scale_(matmul_123, full_186, float('0'), True)

        # pd_op.softmax_: (-1x4x16x16xf32) <- (-1x4x16x16xf32)
        softmax__16 = paddle._C_ops.softmax_(scale__16, -1)

        # pd_op.matmul: (-1x4x16x6xf32) <- (-1x4x16x16xf32, -1x4x16x6xf32)
        matmul_124 = paddle._C_ops.matmul(softmax__16, transpose_67, False, False)

        # pd_op.transpose: (-1x16x4x6xf32) <- (-1x4x16x6xf32)
        transpose_69 = paddle._C_ops.transpose(matmul_124, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_187 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_188 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_60 = [slice_65, full_187, full_188]

        # pd_op.reshape_: (-1x16x24xf32, 0x-1x16x4x6xf32) <- (-1x16x4x6xf32, [1xi32, 1xi32, 1xi32])
        reshape__104, reshape__105 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_69, [x.reshape([]) for x in combine_60]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16x24xf32) <- (-1x16x24xf32, 24x24xf32)
        matmul_125 = paddle._C_ops.matmul(reshape__104, parameter_247, False, False)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, 24xf32)
        add__92 = paddle._C_ops.add_(matmul_125, parameter_248)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, -1x16x24xf32)
        add__93 = paddle._C_ops.add_(add__85, add__92)

        # pd_op.layer_norm: (-1x16x24xf32, -16xf32, -16xf32) <- (-1x16x24xf32, 24xf32, 24xf32)
        layer_norm_153, layer_norm_154, layer_norm_155 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__93, parameter_249, parameter_250, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x16x96xf32) <- (-1x16x24xf32, 24x96xf32)
        matmul_126 = paddle._C_ops.matmul(layer_norm_153, parameter_251, False, False)

        # pd_op.add_: (-1x16x96xf32) <- (-1x16x96xf32, 96xf32)
        add__94 = paddle._C_ops.add_(matmul_126, parameter_252)

        # pd_op.gelu: (-1x16x96xf32) <- (-1x16x96xf32)
        gelu_16 = paddle._C_ops.gelu(add__94, False)

        # pd_op.matmul: (-1x16x24xf32) <- (-1x16x96xf32, 96x24xf32)
        matmul_127 = paddle._C_ops.matmul(gelu_16, parameter_253, False, False)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, 24xf32)
        add__95 = paddle._C_ops.add_(matmul_127, parameter_254)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, -1x16x24xf32)
        add__96 = paddle._C_ops.add_(add__93, add__95)

        # pd_op.shape: (3xi32) <- (-1x197x384xf32)
        shape_26 = paddle._C_ops.shape(add__91)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_164 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_165 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_68 = paddle._C_ops.slice(shape_26, [0], full_int_array_164, full_int_array_165, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_189 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_190 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_61 = [slice_68, full_189, full_190]

        # pd_op.reshape: (-1x196x384xf32, 0x-1x16x24xf32) <- (-1x16x24xf32, [1xi32, 1xi32, 1xi32])
        reshape_20, reshape_21 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__96, [x.reshape([]) for x in combine_61]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.layer_norm: (-1x196x384xf32, -196xf32, -196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_156, layer_norm_157, layer_norm_158 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(reshape_20, parameter_255, parameter_256, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_166 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_167 = [197]

        # pd_op.slice: (-1x196x384xf32) <- (-1x197x384xf32, 1xi64, 1xi64)
        slice_69 = paddle._C_ops.slice(add__91, [1], full_int_array_166, full_int_array_167, [1], [])

        # pd_op.matmul: (-1x196x384xf32) <- (-1x196x384xf32, 384x384xf32)
        matmul_128 = paddle._C_ops.matmul(layer_norm_156, parameter_257, False, False)

        # pd_op.layer_norm: (-1x196x384xf32, -196xf32, -196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_159, layer_norm_160, layer_norm_161 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(matmul_128, parameter_258, parameter_259, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.add_: (-1x196x384xf32) <- (-1x196x384xf32, -1x196x384xf32)
        add__97 = paddle._C_ops.add_(slice_69, layer_norm_159)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_168 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_169 = [2147483647]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_170 = [1]

        # pd_op.set_value_with_tensor_: (-1x197x384xf32) <- (-1x197x384xf32, -1x196x384xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__8 = paddle._C_ops.set_value_with_tensor_(add__91, add__97, full_int_array_168, full_int_array_169, full_int_array_170, [1], [], [])

        # pd_op.layer_norm: (-1x197x384xf32, -197xf32, -197xf32) <- (-1x197x384xf32, 384xf32, 384xf32)
        layer_norm_162, layer_norm_163, layer_norm_164 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(set_value_with_tensor__8, parameter_260, parameter_261, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x197x384xf32)
        shape_27 = paddle._C_ops.shape(layer_norm_162)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_171 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_172 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_70 = paddle._C_ops.slice(shape_27, [0], full_int_array_171, full_int_array_172, [1], [0])

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x384xf32, 384x768xf32)
        matmul_129 = paddle._C_ops.matmul(layer_norm_162, parameter_262, False, False)

        # pd_op.full: (1xi32) <- ()
        full_191 = paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_192 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_193 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_194 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_62 = [slice_70, full_191, full_192, full_193, full_194]

        # pd_op.reshape_: (-1x197x2x6x64xf32, 0x-1x197x768xf32) <- (-1x197x768xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__106, reshape__107 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_129, [x.reshape([]) for x in combine_62]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x6x197x64xf32) <- (-1x197x2x6x64xf32)
        transpose_70 = paddle._C_ops.transpose(reshape__106, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_173 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_174 = [1]

        # pd_op.slice: (-1x6x197x64xf32) <- (2x-1x6x197x64xf32, 1xi64, 1xi64)
        slice_71 = paddle._C_ops.slice(transpose_70, [0], full_int_array_173, full_int_array_174, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_175 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_176 = [2]

        # pd_op.slice: (-1x6x197x64xf32) <- (2x-1x6x197x64xf32, 1xi64, 1xi64)
        slice_72 = paddle._C_ops.slice(transpose_70, [0], full_int_array_175, full_int_array_176, [1], [0])

        # pd_op.matmul: (-1x197x384xf32) <- (-1x197x384xf32, 384x384xf32)
        matmul_130 = paddle._C_ops.matmul(layer_norm_162, parameter_263, False, False)

        # pd_op.full: (1xi32) <- ()
        full_195 = paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_196 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_197 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_63 = [slice_70, full_195, full_196, full_197]

        # pd_op.reshape_: (-1x197x6x64xf32, 0x-1x197x384xf32) <- (-1x197x384xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__108, reshape__109 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_130, [x.reshape([]) for x in combine_63]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x6x197x64xf32) <- (-1x197x6x64xf32)
        transpose_71 = paddle._C_ops.transpose(reshape__108, [0, 2, 1, 3])

        # pd_op.transpose: (-1x6x64x197xf32) <- (-1x6x197x64xf32)
        transpose_72 = paddle._C_ops.transpose(slice_72, [0, 1, 3, 2])

        # pd_op.matmul: (-1x6x197x197xf32) <- (-1x6x197x64xf32, -1x6x64x197xf32)
        matmul_131 = paddle._C_ops.matmul(slice_71, transpose_72, False, False)

        # pd_op.full: (1xf32) <- ()
        full_198 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x6x197x197xf32) <- (-1x6x197x197xf32, 1xf32)
        scale__17 = paddle._C_ops.scale_(matmul_131, full_198, float('0'), True)

        # pd_op.softmax_: (-1x6x197x197xf32) <- (-1x6x197x197xf32)
        softmax__17 = paddle._C_ops.softmax_(scale__17, -1)

        # pd_op.matmul: (-1x6x197x64xf32) <- (-1x6x197x197xf32, -1x6x197x64xf32)
        matmul_132 = paddle._C_ops.matmul(softmax__17, transpose_71, False, False)

        # pd_op.transpose: (-1x197x6x64xf32) <- (-1x6x197x64xf32)
        transpose_73 = paddle._C_ops.transpose(matmul_132, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_199 = paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_200 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_64 = [slice_70, full_199, full_200]

        # pd_op.reshape_: (-1x197x384xf32, 0x-1x197x6x64xf32) <- (-1x197x6x64xf32, [1xi32, 1xi32, 1xi32])
        reshape__110, reshape__111 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_73, [x.reshape([]) for x in combine_64]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x197x384xf32) <- (-1x197x384xf32, 384x384xf32)
        matmul_133 = paddle._C_ops.matmul(reshape__110, parameter_264, False, False)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, 384xf32)
        add__98 = paddle._C_ops.add_(matmul_133, parameter_265)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, -1x197x384xf32)
        add__99 = paddle._C_ops.add_(set_value_with_tensor__8, add__98)

        # pd_op.layer_norm: (-1x197x384xf32, -197xf32, -197xf32) <- (-1x197x384xf32, 384xf32, 384xf32)
        layer_norm_165, layer_norm_166, layer_norm_167 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__99, parameter_266, parameter_267, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x1536xf32) <- (-1x197x384xf32, 384x1536xf32)
        matmul_134 = paddle._C_ops.matmul(layer_norm_165, parameter_268, False, False)

        # pd_op.add_: (-1x197x1536xf32) <- (-1x197x1536xf32, 1536xf32)
        add__100 = paddle._C_ops.add_(matmul_134, parameter_269)

        # pd_op.gelu: (-1x197x1536xf32) <- (-1x197x1536xf32)
        gelu_17 = paddle._C_ops.gelu(add__100, False)

        # pd_op.matmul: (-1x197x384xf32) <- (-1x197x1536xf32, 1536x384xf32)
        matmul_135 = paddle._C_ops.matmul(gelu_17, parameter_270, False, False)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, 384xf32)
        add__101 = paddle._C_ops.add_(matmul_135, parameter_271)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, -1x197x384xf32)
        add__102 = paddle._C_ops.add_(add__99, add__101)

        # pd_op.layer_norm: (-1x16x24xf32, -16xf32, -16xf32) <- (-1x16x24xf32, 24xf32, 24xf32)
        layer_norm_168, layer_norm_169, layer_norm_170 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__96, parameter_272, parameter_273, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x16x24xf32)
        shape_28 = paddle._C_ops.shape(layer_norm_168)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_177 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_178 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_73 = paddle._C_ops.slice(shape_28, [0], full_int_array_177, full_int_array_178, [1], [0])

        # pd_op.matmul: (-1x16x48xf32) <- (-1x16x24xf32, 24x48xf32)
        matmul_136 = paddle._C_ops.matmul(layer_norm_168, parameter_274, False, False)

        # pd_op.full: (1xi32) <- ()
        full_201 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_202 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_203 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_204 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_65 = [slice_73, full_201, full_202, full_203, full_204]

        # pd_op.reshape_: (-1x16x2x4x6xf32, 0x-1x16x48xf32) <- (-1x16x48xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__112, reshape__113 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_136, [x.reshape([]) for x in combine_65]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x4x16x6xf32) <- (-1x16x2x4x6xf32)
        transpose_74 = paddle._C_ops.transpose(reshape__112, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_179 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_180 = [1]

        # pd_op.slice: (-1x4x16x6xf32) <- (2x-1x4x16x6xf32, 1xi64, 1xi64)
        slice_74 = paddle._C_ops.slice(transpose_74, [0], full_int_array_179, full_int_array_180, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_181 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_182 = [2]

        # pd_op.slice: (-1x4x16x6xf32) <- (2x-1x4x16x6xf32, 1xi64, 1xi64)
        slice_75 = paddle._C_ops.slice(transpose_74, [0], full_int_array_181, full_int_array_182, [1], [0])

        # pd_op.matmul: (-1x16x24xf32) <- (-1x16x24xf32, 24x24xf32)
        matmul_137 = paddle._C_ops.matmul(layer_norm_168, parameter_275, False, False)

        # pd_op.full: (1xi32) <- ()
        full_205 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_206 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_207 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_66 = [slice_73, full_205, full_206, full_207]

        # pd_op.reshape_: (-1x16x4x6xf32, 0x-1x16x24xf32) <- (-1x16x24xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__114, reshape__115 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_137, [x.reshape([]) for x in combine_66]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x16x6xf32) <- (-1x16x4x6xf32)
        transpose_75 = paddle._C_ops.transpose(reshape__114, [0, 2, 1, 3])

        # pd_op.transpose: (-1x4x6x16xf32) <- (-1x4x16x6xf32)
        transpose_76 = paddle._C_ops.transpose(slice_75, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x16x16xf32) <- (-1x4x16x6xf32, -1x4x6x16xf32)
        matmul_138 = paddle._C_ops.matmul(slice_74, transpose_76, False, False)

        # pd_op.full: (1xf32) <- ()
        full_208 = paddle._C_ops.full([1], float('0.408248'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x4x16x16xf32) <- (-1x4x16x16xf32, 1xf32)
        scale__18 = paddle._C_ops.scale_(matmul_138, full_208, float('0'), True)

        # pd_op.softmax_: (-1x4x16x16xf32) <- (-1x4x16x16xf32)
        softmax__18 = paddle._C_ops.softmax_(scale__18, -1)

        # pd_op.matmul: (-1x4x16x6xf32) <- (-1x4x16x16xf32, -1x4x16x6xf32)
        matmul_139 = paddle._C_ops.matmul(softmax__18, transpose_75, False, False)

        # pd_op.transpose: (-1x16x4x6xf32) <- (-1x4x16x6xf32)
        transpose_77 = paddle._C_ops.transpose(matmul_139, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_209 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_210 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_67 = [slice_73, full_209, full_210]

        # pd_op.reshape_: (-1x16x24xf32, 0x-1x16x4x6xf32) <- (-1x16x4x6xf32, [1xi32, 1xi32, 1xi32])
        reshape__116, reshape__117 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_77, [x.reshape([]) for x in combine_67]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16x24xf32) <- (-1x16x24xf32, 24x24xf32)
        matmul_140 = paddle._C_ops.matmul(reshape__116, parameter_276, False, False)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, 24xf32)
        add__103 = paddle._C_ops.add_(matmul_140, parameter_277)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, -1x16x24xf32)
        add__104 = paddle._C_ops.add_(add__96, add__103)

        # pd_op.layer_norm: (-1x16x24xf32, -16xf32, -16xf32) <- (-1x16x24xf32, 24xf32, 24xf32)
        layer_norm_171, layer_norm_172, layer_norm_173 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__104, parameter_278, parameter_279, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x16x96xf32) <- (-1x16x24xf32, 24x96xf32)
        matmul_141 = paddle._C_ops.matmul(layer_norm_171, parameter_280, False, False)

        # pd_op.add_: (-1x16x96xf32) <- (-1x16x96xf32, 96xf32)
        add__105 = paddle._C_ops.add_(matmul_141, parameter_281)

        # pd_op.gelu: (-1x16x96xf32) <- (-1x16x96xf32)
        gelu_18 = paddle._C_ops.gelu(add__105, False)

        # pd_op.matmul: (-1x16x24xf32) <- (-1x16x96xf32, 96x24xf32)
        matmul_142 = paddle._C_ops.matmul(gelu_18, parameter_282, False, False)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, 24xf32)
        add__106 = paddle._C_ops.add_(matmul_142, parameter_283)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, -1x16x24xf32)
        add__107 = paddle._C_ops.add_(add__104, add__106)

        # pd_op.shape: (3xi32) <- (-1x197x384xf32)
        shape_29 = paddle._C_ops.shape(add__102)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_183 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_184 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_76 = paddle._C_ops.slice(shape_29, [0], full_int_array_183, full_int_array_184, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_211 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_212 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_68 = [slice_76, full_211, full_212]

        # pd_op.reshape: (-1x196x384xf32, 0x-1x16x24xf32) <- (-1x16x24xf32, [1xi32, 1xi32, 1xi32])
        reshape_22, reshape_23 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__107, [x.reshape([]) for x in combine_68]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.layer_norm: (-1x196x384xf32, -196xf32, -196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_174, layer_norm_175, layer_norm_176 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(reshape_22, parameter_284, parameter_285, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_185 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_186 = [197]

        # pd_op.slice: (-1x196x384xf32) <- (-1x197x384xf32, 1xi64, 1xi64)
        slice_77 = paddle._C_ops.slice(add__102, [1], full_int_array_185, full_int_array_186, [1], [])

        # pd_op.matmul: (-1x196x384xf32) <- (-1x196x384xf32, 384x384xf32)
        matmul_143 = paddle._C_ops.matmul(layer_norm_174, parameter_286, False, False)

        # pd_op.layer_norm: (-1x196x384xf32, -196xf32, -196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_177, layer_norm_178, layer_norm_179 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(matmul_143, parameter_287, parameter_288, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.add_: (-1x196x384xf32) <- (-1x196x384xf32, -1x196x384xf32)
        add__108 = paddle._C_ops.add_(slice_77, layer_norm_177)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_187 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_188 = [2147483647]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_189 = [1]

        # pd_op.set_value_with_tensor_: (-1x197x384xf32) <- (-1x197x384xf32, -1x196x384xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__9 = paddle._C_ops.set_value_with_tensor_(add__102, add__108, full_int_array_187, full_int_array_188, full_int_array_189, [1], [], [])

        # pd_op.layer_norm: (-1x197x384xf32, -197xf32, -197xf32) <- (-1x197x384xf32, 384xf32, 384xf32)
        layer_norm_180, layer_norm_181, layer_norm_182 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(set_value_with_tensor__9, parameter_289, parameter_290, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x197x384xf32)
        shape_30 = paddle._C_ops.shape(layer_norm_180)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_190 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_191 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_78 = paddle._C_ops.slice(shape_30, [0], full_int_array_190, full_int_array_191, [1], [0])

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x384xf32, 384x768xf32)
        matmul_144 = paddle._C_ops.matmul(layer_norm_180, parameter_291, False, False)

        # pd_op.full: (1xi32) <- ()
        full_213 = paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_214 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_215 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_216 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_69 = [slice_78, full_213, full_214, full_215, full_216]

        # pd_op.reshape_: (-1x197x2x6x64xf32, 0x-1x197x768xf32) <- (-1x197x768xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__118, reshape__119 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_144, [x.reshape([]) for x in combine_69]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x6x197x64xf32) <- (-1x197x2x6x64xf32)
        transpose_78 = paddle._C_ops.transpose(reshape__118, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_192 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_193 = [1]

        # pd_op.slice: (-1x6x197x64xf32) <- (2x-1x6x197x64xf32, 1xi64, 1xi64)
        slice_79 = paddle._C_ops.slice(transpose_78, [0], full_int_array_192, full_int_array_193, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_194 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_195 = [2]

        # pd_op.slice: (-1x6x197x64xf32) <- (2x-1x6x197x64xf32, 1xi64, 1xi64)
        slice_80 = paddle._C_ops.slice(transpose_78, [0], full_int_array_194, full_int_array_195, [1], [0])

        # pd_op.matmul: (-1x197x384xf32) <- (-1x197x384xf32, 384x384xf32)
        matmul_145 = paddle._C_ops.matmul(layer_norm_180, parameter_292, False, False)

        # pd_op.full: (1xi32) <- ()
        full_217 = paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_218 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_219 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_70 = [slice_78, full_217, full_218, full_219]

        # pd_op.reshape_: (-1x197x6x64xf32, 0x-1x197x384xf32) <- (-1x197x384xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__120, reshape__121 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_145, [x.reshape([]) for x in combine_70]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x6x197x64xf32) <- (-1x197x6x64xf32)
        transpose_79 = paddle._C_ops.transpose(reshape__120, [0, 2, 1, 3])

        # pd_op.transpose: (-1x6x64x197xf32) <- (-1x6x197x64xf32)
        transpose_80 = paddle._C_ops.transpose(slice_80, [0, 1, 3, 2])

        # pd_op.matmul: (-1x6x197x197xf32) <- (-1x6x197x64xf32, -1x6x64x197xf32)
        matmul_146 = paddle._C_ops.matmul(slice_79, transpose_80, False, False)

        # pd_op.full: (1xf32) <- ()
        full_220 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x6x197x197xf32) <- (-1x6x197x197xf32, 1xf32)
        scale__19 = paddle._C_ops.scale_(matmul_146, full_220, float('0'), True)

        # pd_op.softmax_: (-1x6x197x197xf32) <- (-1x6x197x197xf32)
        softmax__19 = paddle._C_ops.softmax_(scale__19, -1)

        # pd_op.matmul: (-1x6x197x64xf32) <- (-1x6x197x197xf32, -1x6x197x64xf32)
        matmul_147 = paddle._C_ops.matmul(softmax__19, transpose_79, False, False)

        # pd_op.transpose: (-1x197x6x64xf32) <- (-1x6x197x64xf32)
        transpose_81 = paddle._C_ops.transpose(matmul_147, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_221 = paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_222 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_71 = [slice_78, full_221, full_222]

        # pd_op.reshape_: (-1x197x384xf32, 0x-1x197x6x64xf32) <- (-1x197x6x64xf32, [1xi32, 1xi32, 1xi32])
        reshape__122, reshape__123 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_81, [x.reshape([]) for x in combine_71]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x197x384xf32) <- (-1x197x384xf32, 384x384xf32)
        matmul_148 = paddle._C_ops.matmul(reshape__122, parameter_293, False, False)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, 384xf32)
        add__109 = paddle._C_ops.add_(matmul_148, parameter_294)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, -1x197x384xf32)
        add__110 = paddle._C_ops.add_(set_value_with_tensor__9, add__109)

        # pd_op.layer_norm: (-1x197x384xf32, -197xf32, -197xf32) <- (-1x197x384xf32, 384xf32, 384xf32)
        layer_norm_183, layer_norm_184, layer_norm_185 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__110, parameter_295, parameter_296, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x1536xf32) <- (-1x197x384xf32, 384x1536xf32)
        matmul_149 = paddle._C_ops.matmul(layer_norm_183, parameter_297, False, False)

        # pd_op.add_: (-1x197x1536xf32) <- (-1x197x1536xf32, 1536xf32)
        add__111 = paddle._C_ops.add_(matmul_149, parameter_298)

        # pd_op.gelu: (-1x197x1536xf32) <- (-1x197x1536xf32)
        gelu_19 = paddle._C_ops.gelu(add__111, False)

        # pd_op.matmul: (-1x197x384xf32) <- (-1x197x1536xf32, 1536x384xf32)
        matmul_150 = paddle._C_ops.matmul(gelu_19, parameter_299, False, False)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, 384xf32)
        add__112 = paddle._C_ops.add_(matmul_150, parameter_300)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, -1x197x384xf32)
        add__113 = paddle._C_ops.add_(add__110, add__112)

        # pd_op.layer_norm: (-1x16x24xf32, -16xf32, -16xf32) <- (-1x16x24xf32, 24xf32, 24xf32)
        layer_norm_186, layer_norm_187, layer_norm_188 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__107, parameter_301, parameter_302, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x16x24xf32)
        shape_31 = paddle._C_ops.shape(layer_norm_186)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_196 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_197 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_81 = paddle._C_ops.slice(shape_31, [0], full_int_array_196, full_int_array_197, [1], [0])

        # pd_op.matmul: (-1x16x48xf32) <- (-1x16x24xf32, 24x48xf32)
        matmul_151 = paddle._C_ops.matmul(layer_norm_186, parameter_303, False, False)

        # pd_op.full: (1xi32) <- ()
        full_223 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_224 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_225 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_226 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_72 = [slice_81, full_223, full_224, full_225, full_226]

        # pd_op.reshape_: (-1x16x2x4x6xf32, 0x-1x16x48xf32) <- (-1x16x48xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__124, reshape__125 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_151, [x.reshape([]) for x in combine_72]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x4x16x6xf32) <- (-1x16x2x4x6xf32)
        transpose_82 = paddle._C_ops.transpose(reshape__124, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_198 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_199 = [1]

        # pd_op.slice: (-1x4x16x6xf32) <- (2x-1x4x16x6xf32, 1xi64, 1xi64)
        slice_82 = paddle._C_ops.slice(transpose_82, [0], full_int_array_198, full_int_array_199, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_200 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_201 = [2]

        # pd_op.slice: (-1x4x16x6xf32) <- (2x-1x4x16x6xf32, 1xi64, 1xi64)
        slice_83 = paddle._C_ops.slice(transpose_82, [0], full_int_array_200, full_int_array_201, [1], [0])

        # pd_op.matmul: (-1x16x24xf32) <- (-1x16x24xf32, 24x24xf32)
        matmul_152 = paddle._C_ops.matmul(layer_norm_186, parameter_304, False, False)

        # pd_op.full: (1xi32) <- ()
        full_227 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_228 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_229 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_73 = [slice_81, full_227, full_228, full_229]

        # pd_op.reshape_: (-1x16x4x6xf32, 0x-1x16x24xf32) <- (-1x16x24xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__126, reshape__127 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_152, [x.reshape([]) for x in combine_73]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x16x6xf32) <- (-1x16x4x6xf32)
        transpose_83 = paddle._C_ops.transpose(reshape__126, [0, 2, 1, 3])

        # pd_op.transpose: (-1x4x6x16xf32) <- (-1x4x16x6xf32)
        transpose_84 = paddle._C_ops.transpose(slice_83, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x16x16xf32) <- (-1x4x16x6xf32, -1x4x6x16xf32)
        matmul_153 = paddle._C_ops.matmul(slice_82, transpose_84, False, False)

        # pd_op.full: (1xf32) <- ()
        full_230 = paddle._C_ops.full([1], float('0.408248'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x4x16x16xf32) <- (-1x4x16x16xf32, 1xf32)
        scale__20 = paddle._C_ops.scale_(matmul_153, full_230, float('0'), True)

        # pd_op.softmax_: (-1x4x16x16xf32) <- (-1x4x16x16xf32)
        softmax__20 = paddle._C_ops.softmax_(scale__20, -1)

        # pd_op.matmul: (-1x4x16x6xf32) <- (-1x4x16x16xf32, -1x4x16x6xf32)
        matmul_154 = paddle._C_ops.matmul(softmax__20, transpose_83, False, False)

        # pd_op.transpose: (-1x16x4x6xf32) <- (-1x4x16x6xf32)
        transpose_85 = paddle._C_ops.transpose(matmul_154, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_231 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_232 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_74 = [slice_81, full_231, full_232]

        # pd_op.reshape_: (-1x16x24xf32, 0x-1x16x4x6xf32) <- (-1x16x4x6xf32, [1xi32, 1xi32, 1xi32])
        reshape__128, reshape__129 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_85, [x.reshape([]) for x in combine_74]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16x24xf32) <- (-1x16x24xf32, 24x24xf32)
        matmul_155 = paddle._C_ops.matmul(reshape__128, parameter_305, False, False)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, 24xf32)
        add__114 = paddle._C_ops.add_(matmul_155, parameter_306)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, -1x16x24xf32)
        add__115 = paddle._C_ops.add_(add__107, add__114)

        # pd_op.layer_norm: (-1x16x24xf32, -16xf32, -16xf32) <- (-1x16x24xf32, 24xf32, 24xf32)
        layer_norm_189, layer_norm_190, layer_norm_191 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__115, parameter_307, parameter_308, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x16x96xf32) <- (-1x16x24xf32, 24x96xf32)
        matmul_156 = paddle._C_ops.matmul(layer_norm_189, parameter_309, False, False)

        # pd_op.add_: (-1x16x96xf32) <- (-1x16x96xf32, 96xf32)
        add__116 = paddle._C_ops.add_(matmul_156, parameter_310)

        # pd_op.gelu: (-1x16x96xf32) <- (-1x16x96xf32)
        gelu_20 = paddle._C_ops.gelu(add__116, False)

        # pd_op.matmul: (-1x16x24xf32) <- (-1x16x96xf32, 96x24xf32)
        matmul_157 = paddle._C_ops.matmul(gelu_20, parameter_311, False, False)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, 24xf32)
        add__117 = paddle._C_ops.add_(matmul_157, parameter_312)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, -1x16x24xf32)
        add__118 = paddle._C_ops.add_(add__115, add__117)

        # pd_op.shape: (3xi32) <- (-1x197x384xf32)
        shape_32 = paddle._C_ops.shape(add__113)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_202 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_203 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_84 = paddle._C_ops.slice(shape_32, [0], full_int_array_202, full_int_array_203, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_233 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_234 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_75 = [slice_84, full_233, full_234]

        # pd_op.reshape: (-1x196x384xf32, 0x-1x16x24xf32) <- (-1x16x24xf32, [1xi32, 1xi32, 1xi32])
        reshape_24, reshape_25 = (lambda x, f: f(x))(paddle._C_ops.reshape(add__118, [x.reshape([]) for x in combine_75]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.layer_norm: (-1x196x384xf32, -196xf32, -196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_192, layer_norm_193, layer_norm_194 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(reshape_24, parameter_313, parameter_314, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_204 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_205 = [197]

        # pd_op.slice: (-1x196x384xf32) <- (-1x197x384xf32, 1xi64, 1xi64)
        slice_85 = paddle._C_ops.slice(add__113, [1], full_int_array_204, full_int_array_205, [1], [])

        # pd_op.matmul: (-1x196x384xf32) <- (-1x196x384xf32, 384x384xf32)
        matmul_158 = paddle._C_ops.matmul(layer_norm_192, parameter_315, False, False)

        # pd_op.layer_norm: (-1x196x384xf32, -196xf32, -196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_195, layer_norm_196, layer_norm_197 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(matmul_158, parameter_316, parameter_317, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.add_: (-1x196x384xf32) <- (-1x196x384xf32, -1x196x384xf32)
        add__119 = paddle._C_ops.add_(slice_85, layer_norm_195)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_206 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_207 = [2147483647]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_208 = [1]

        # pd_op.set_value_with_tensor_: (-1x197x384xf32) <- (-1x197x384xf32, -1x196x384xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__10 = paddle._C_ops.set_value_with_tensor_(add__113, add__119, full_int_array_206, full_int_array_207, full_int_array_208, [1], [], [])

        # pd_op.layer_norm: (-1x197x384xf32, -197xf32, -197xf32) <- (-1x197x384xf32, 384xf32, 384xf32)
        layer_norm_198, layer_norm_199, layer_norm_200 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(set_value_with_tensor__10, parameter_318, parameter_319, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x197x384xf32)
        shape_33 = paddle._C_ops.shape(layer_norm_198)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_209 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_210 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_86 = paddle._C_ops.slice(shape_33, [0], full_int_array_209, full_int_array_210, [1], [0])

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x384xf32, 384x768xf32)
        matmul_159 = paddle._C_ops.matmul(layer_norm_198, parameter_320, False, False)

        # pd_op.full: (1xi32) <- ()
        full_235 = paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_236 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_237 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_238 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_76 = [slice_86, full_235, full_236, full_237, full_238]

        # pd_op.reshape_: (-1x197x2x6x64xf32, 0x-1x197x768xf32) <- (-1x197x768xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__130, reshape__131 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_159, [x.reshape([]) for x in combine_76]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x6x197x64xf32) <- (-1x197x2x6x64xf32)
        transpose_86 = paddle._C_ops.transpose(reshape__130, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_211 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_212 = [1]

        # pd_op.slice: (-1x6x197x64xf32) <- (2x-1x6x197x64xf32, 1xi64, 1xi64)
        slice_87 = paddle._C_ops.slice(transpose_86, [0], full_int_array_211, full_int_array_212, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_213 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_214 = [2]

        # pd_op.slice: (-1x6x197x64xf32) <- (2x-1x6x197x64xf32, 1xi64, 1xi64)
        slice_88 = paddle._C_ops.slice(transpose_86, [0], full_int_array_213, full_int_array_214, [1], [0])

        # pd_op.matmul: (-1x197x384xf32) <- (-1x197x384xf32, 384x384xf32)
        matmul_160 = paddle._C_ops.matmul(layer_norm_198, parameter_321, False, False)

        # pd_op.full: (1xi32) <- ()
        full_239 = paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_240 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_241 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_77 = [slice_86, full_239, full_240, full_241]

        # pd_op.reshape_: (-1x197x6x64xf32, 0x-1x197x384xf32) <- (-1x197x384xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__132, reshape__133 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_160, [x.reshape([]) for x in combine_77]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x6x197x64xf32) <- (-1x197x6x64xf32)
        transpose_87 = paddle._C_ops.transpose(reshape__132, [0, 2, 1, 3])

        # pd_op.transpose: (-1x6x64x197xf32) <- (-1x6x197x64xf32)
        transpose_88 = paddle._C_ops.transpose(slice_88, [0, 1, 3, 2])

        # pd_op.matmul: (-1x6x197x197xf32) <- (-1x6x197x64xf32, -1x6x64x197xf32)
        matmul_161 = paddle._C_ops.matmul(slice_87, transpose_88, False, False)

        # pd_op.full: (1xf32) <- ()
        full_242 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x6x197x197xf32) <- (-1x6x197x197xf32, 1xf32)
        scale__21 = paddle._C_ops.scale_(matmul_161, full_242, float('0'), True)

        # pd_op.softmax_: (-1x6x197x197xf32) <- (-1x6x197x197xf32)
        softmax__21 = paddle._C_ops.softmax_(scale__21, -1)

        # pd_op.matmul: (-1x6x197x64xf32) <- (-1x6x197x197xf32, -1x6x197x64xf32)
        matmul_162 = paddle._C_ops.matmul(softmax__21, transpose_87, False, False)

        # pd_op.transpose: (-1x197x6x64xf32) <- (-1x6x197x64xf32)
        transpose_89 = paddle._C_ops.transpose(matmul_162, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_243 = paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_244 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_78 = [slice_86, full_243, full_244]

        # pd_op.reshape_: (-1x197x384xf32, 0x-1x197x6x64xf32) <- (-1x197x6x64xf32, [1xi32, 1xi32, 1xi32])
        reshape__134, reshape__135 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_89, [x.reshape([]) for x in combine_78]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x197x384xf32) <- (-1x197x384xf32, 384x384xf32)
        matmul_163 = paddle._C_ops.matmul(reshape__134, parameter_322, False, False)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, 384xf32)
        add__120 = paddle._C_ops.add_(matmul_163, parameter_323)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, -1x197x384xf32)
        add__121 = paddle._C_ops.add_(set_value_with_tensor__10, add__120)

        # pd_op.layer_norm: (-1x197x384xf32, -197xf32, -197xf32) <- (-1x197x384xf32, 384xf32, 384xf32)
        layer_norm_201, layer_norm_202, layer_norm_203 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__121, parameter_324, parameter_325, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x1536xf32) <- (-1x197x384xf32, 384x1536xf32)
        matmul_164 = paddle._C_ops.matmul(layer_norm_201, parameter_326, False, False)

        # pd_op.add_: (-1x197x1536xf32) <- (-1x197x1536xf32, 1536xf32)
        add__122 = paddle._C_ops.add_(matmul_164, parameter_327)

        # pd_op.gelu: (-1x197x1536xf32) <- (-1x197x1536xf32)
        gelu_21 = paddle._C_ops.gelu(add__122, False)

        # pd_op.matmul: (-1x197x384xf32) <- (-1x197x1536xf32, 1536x384xf32)
        matmul_165 = paddle._C_ops.matmul(gelu_21, parameter_328, False, False)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, 384xf32)
        add__123 = paddle._C_ops.add_(matmul_165, parameter_329)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, -1x197x384xf32)
        add__124 = paddle._C_ops.add_(add__121, add__123)

        # pd_op.layer_norm: (-1x16x24xf32, -16xf32, -16xf32) <- (-1x16x24xf32, 24xf32, 24xf32)
        layer_norm_204, layer_norm_205, layer_norm_206 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__118, parameter_330, parameter_331, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x16x24xf32)
        shape_34 = paddle._C_ops.shape(layer_norm_204)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_215 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_216 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_89 = paddle._C_ops.slice(shape_34, [0], full_int_array_215, full_int_array_216, [1], [0])

        # pd_op.matmul: (-1x16x48xf32) <- (-1x16x24xf32, 24x48xf32)
        matmul_166 = paddle._C_ops.matmul(layer_norm_204, parameter_332, False, False)

        # pd_op.full: (1xi32) <- ()
        full_245 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_246 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_247 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_248 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_79 = [slice_89, full_245, full_246, full_247, full_248]

        # pd_op.reshape_: (-1x16x2x4x6xf32, 0x-1x16x48xf32) <- (-1x16x48xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__136, reshape__137 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_166, [x.reshape([]) for x in combine_79]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x4x16x6xf32) <- (-1x16x2x4x6xf32)
        transpose_90 = paddle._C_ops.transpose(reshape__136, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_217 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_218 = [1]

        # pd_op.slice: (-1x4x16x6xf32) <- (2x-1x4x16x6xf32, 1xi64, 1xi64)
        slice_90 = paddle._C_ops.slice(transpose_90, [0], full_int_array_217, full_int_array_218, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_219 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_220 = [2]

        # pd_op.slice: (-1x4x16x6xf32) <- (2x-1x4x16x6xf32, 1xi64, 1xi64)
        slice_91 = paddle._C_ops.slice(transpose_90, [0], full_int_array_219, full_int_array_220, [1], [0])

        # pd_op.matmul: (-1x16x24xf32) <- (-1x16x24xf32, 24x24xf32)
        matmul_167 = paddle._C_ops.matmul(layer_norm_204, parameter_333, False, False)

        # pd_op.full: (1xi32) <- ()
        full_249 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_250 = paddle._C_ops.full([1], float('4'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_251 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_80 = [slice_89, full_249, full_250, full_251]

        # pd_op.reshape_: (-1x16x4x6xf32, 0x-1x16x24xf32) <- (-1x16x24xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__138, reshape__139 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_167, [x.reshape([]) for x in combine_80]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x4x16x6xf32) <- (-1x16x4x6xf32)
        transpose_91 = paddle._C_ops.transpose(reshape__138, [0, 2, 1, 3])

        # pd_op.transpose: (-1x4x6x16xf32) <- (-1x4x16x6xf32)
        transpose_92 = paddle._C_ops.transpose(slice_91, [0, 1, 3, 2])

        # pd_op.matmul: (-1x4x16x16xf32) <- (-1x4x16x6xf32, -1x4x6x16xf32)
        matmul_168 = paddle._C_ops.matmul(slice_90, transpose_92, False, False)

        # pd_op.full: (1xf32) <- ()
        full_252 = paddle._C_ops.full([1], float('0.408248'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x4x16x16xf32) <- (-1x4x16x16xf32, 1xf32)
        scale__22 = paddle._C_ops.scale_(matmul_168, full_252, float('0'), True)

        # pd_op.softmax_: (-1x4x16x16xf32) <- (-1x4x16x16xf32)
        softmax__22 = paddle._C_ops.softmax_(scale__22, -1)

        # pd_op.matmul: (-1x4x16x6xf32) <- (-1x4x16x16xf32, -1x4x16x6xf32)
        matmul_169 = paddle._C_ops.matmul(softmax__22, transpose_91, False, False)

        # pd_op.transpose: (-1x16x4x6xf32) <- (-1x4x16x6xf32)
        transpose_93 = paddle._C_ops.transpose(matmul_169, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_253 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_254 = paddle._C_ops.full([1], float('24'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_81 = [slice_89, full_253, full_254]

        # pd_op.reshape_: (-1x16x24xf32, 0x-1x16x4x6xf32) <- (-1x16x4x6xf32, [1xi32, 1xi32, 1xi32])
        reshape__140, reshape__141 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_93, [x.reshape([]) for x in combine_81]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x16x24xf32) <- (-1x16x24xf32, 24x24xf32)
        matmul_170 = paddle._C_ops.matmul(reshape__140, parameter_334, False, False)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, 24xf32)
        add__125 = paddle._C_ops.add_(matmul_170, parameter_335)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, -1x16x24xf32)
        add__126 = paddle._C_ops.add_(add__118, add__125)

        # pd_op.layer_norm: (-1x16x24xf32, -16xf32, -16xf32) <- (-1x16x24xf32, 24xf32, 24xf32)
        layer_norm_207, layer_norm_208, layer_norm_209 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__126, parameter_336, parameter_337, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x16x96xf32) <- (-1x16x24xf32, 24x96xf32)
        matmul_171 = paddle._C_ops.matmul(layer_norm_207, parameter_338, False, False)

        # pd_op.add_: (-1x16x96xf32) <- (-1x16x96xf32, 96xf32)
        add__127 = paddle._C_ops.add_(matmul_171, parameter_339)

        # pd_op.gelu: (-1x16x96xf32) <- (-1x16x96xf32)
        gelu_22 = paddle._C_ops.gelu(add__127, False)

        # pd_op.matmul: (-1x16x24xf32) <- (-1x16x96xf32, 96x24xf32)
        matmul_172 = paddle._C_ops.matmul(gelu_22, parameter_340, False, False)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, 24xf32)
        add__128 = paddle._C_ops.add_(matmul_172, parameter_341)

        # pd_op.add_: (-1x16x24xf32) <- (-1x16x24xf32, -1x16x24xf32)
        add__129 = paddle._C_ops.add_(add__126, add__128)

        # pd_op.shape: (3xi32) <- (-1x197x384xf32)
        shape_35 = paddle._C_ops.shape(add__124)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_221 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_222 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_92 = paddle._C_ops.slice(shape_35, [0], full_int_array_221, full_int_array_222, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_255 = paddle._C_ops.full([1], float('196'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_256 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_82 = [slice_92, full_255, full_256]

        # pd_op.reshape_: (-1x196x384xf32, 0x-1x16x24xf32) <- (-1x16x24xf32, [1xi32, 1xi32, 1xi32])
        reshape__142, reshape__143 = (lambda x, f: f(x))(paddle._C_ops.reshape_(add__129, [x.reshape([]) for x in combine_82]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.layer_norm: (-1x196x384xf32, -196xf32, -196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_210, layer_norm_211, layer_norm_212 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(reshape__142, parameter_342, parameter_343, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_223 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_224 = [197]

        # pd_op.slice: (-1x196x384xf32) <- (-1x197x384xf32, 1xi64, 1xi64)
        slice_93 = paddle._C_ops.slice(add__124, [1], full_int_array_223, full_int_array_224, [1], [])

        # pd_op.matmul: (-1x196x384xf32) <- (-1x196x384xf32, 384x384xf32)
        matmul_173 = paddle._C_ops.matmul(layer_norm_210, parameter_344, False, False)

        # pd_op.layer_norm: (-1x196x384xf32, -196xf32, -196xf32) <- (-1x196x384xf32, 384xf32, 384xf32)
        layer_norm_213, layer_norm_214, layer_norm_215 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(matmul_173, parameter_345, parameter_346, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.add_: (-1x196x384xf32) <- (-1x196x384xf32, -1x196x384xf32)
        add__130 = paddle._C_ops.add_(slice_93, layer_norm_213)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_225 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_226 = [2147483647]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_227 = [1]

        # pd_op.set_value_with_tensor_: (-1x197x384xf32) <- (-1x197x384xf32, -1x196x384xf32, 1xi64, 1xi64, 1xi64)
        set_value_with_tensor__11 = paddle._C_ops.set_value_with_tensor_(add__124, add__130, full_int_array_225, full_int_array_226, full_int_array_227, [1], [], [])

        # pd_op.layer_norm: (-1x197x384xf32, -197xf32, -197xf32) <- (-1x197x384xf32, 384xf32, 384xf32)
        layer_norm_216, layer_norm_217, layer_norm_218 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(set_value_with_tensor__11, parameter_347, parameter_348, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.shape: (3xi32) <- (-1x197x384xf32)
        shape_36 = paddle._C_ops.shape(layer_norm_216)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_228 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_229 = [1]

        # pd_op.slice: (1xi32) <- (3xi32, 1xi64, 1xi64)
        slice_94 = paddle._C_ops.slice(shape_36, [0], full_int_array_228, full_int_array_229, [1], [0])

        # pd_op.matmul: (-1x197x768xf32) <- (-1x197x384xf32, 384x768xf32)
        matmul_174 = paddle._C_ops.matmul(layer_norm_216, parameter_349, False, False)

        # pd_op.full: (1xi32) <- ()
        full_257 = paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_258 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_259 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_260 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_83 = [slice_94, full_257, full_258, full_259, full_260]

        # pd_op.reshape_: (-1x197x2x6x64xf32, 0x-1x197x768xf32) <- (-1x197x768xf32, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__144, reshape__145 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_174, [x.reshape([]) for x in combine_83]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (2x-1x6x197x64xf32) <- (-1x197x2x6x64xf32)
        transpose_94 = paddle._C_ops.transpose(reshape__144, [2, 0, 3, 1, 4])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_230 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_231 = [1]

        # pd_op.slice: (-1x6x197x64xf32) <- (2x-1x6x197x64xf32, 1xi64, 1xi64)
        slice_95 = paddle._C_ops.slice(transpose_94, [0], full_int_array_230, full_int_array_231, [1], [0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_232 = [1]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_233 = [2]

        # pd_op.slice: (-1x6x197x64xf32) <- (2x-1x6x197x64xf32, 1xi64, 1xi64)
        slice_96 = paddle._C_ops.slice(transpose_94, [0], full_int_array_232, full_int_array_233, [1], [0])

        # pd_op.matmul: (-1x197x384xf32) <- (-1x197x384xf32, 384x384xf32)
        matmul_175 = paddle._C_ops.matmul(layer_norm_216, parameter_350, False, False)

        # pd_op.full: (1xi32) <- ()
        full_261 = paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_262 = paddle._C_ops.full([1], float('6'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_263 = paddle._C_ops.full([1], float('64'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_84 = [slice_94, full_261, full_262, full_263]

        # pd_op.reshape_: (-1x197x6x64xf32, 0x-1x197x384xf32) <- (-1x197x384xf32, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__146, reshape__147 = (lambda x, f: f(x))(paddle._C_ops.reshape_(matmul_175, [x.reshape([]) for x in combine_84]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x6x197x64xf32) <- (-1x197x6x64xf32)
        transpose_95 = paddle._C_ops.transpose(reshape__146, [0, 2, 1, 3])

        # pd_op.transpose: (-1x6x64x197xf32) <- (-1x6x197x64xf32)
        transpose_96 = paddle._C_ops.transpose(slice_96, [0, 1, 3, 2])

        # pd_op.matmul: (-1x6x197x197xf32) <- (-1x6x197x64xf32, -1x6x64x197xf32)
        matmul_176 = paddle._C_ops.matmul(slice_95, transpose_96, False, False)

        # pd_op.full: (1xf32) <- ()
        full_264 = paddle._C_ops.full([1], float('0.125'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale_: (-1x6x197x197xf32) <- (-1x6x197x197xf32, 1xf32)
        scale__23 = paddle._C_ops.scale_(matmul_176, full_264, float('0'), True)

        # pd_op.softmax_: (-1x6x197x197xf32) <- (-1x6x197x197xf32)
        softmax__23 = paddle._C_ops.softmax_(scale__23, -1)

        # pd_op.matmul: (-1x6x197x64xf32) <- (-1x6x197x197xf32, -1x6x197x64xf32)
        matmul_177 = paddle._C_ops.matmul(softmax__23, transpose_95, False, False)

        # pd_op.transpose: (-1x197x6x64xf32) <- (-1x6x197x64xf32)
        transpose_97 = paddle._C_ops.transpose(matmul_177, [0, 2, 1, 3])

        # pd_op.full: (1xi32) <- ()
        full_265 = paddle._C_ops.full([1], float('197'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_266 = paddle._C_ops.full([1], float('384'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32)
        combine_85 = [slice_94, full_265, full_266]

        # pd_op.reshape_: (-1x197x384xf32, 0x-1x197x6x64xf32) <- (-1x197x6x64xf32, [1xi32, 1xi32, 1xi32])
        reshape__148, reshape__149 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_97, [x.reshape([]) for x in combine_85]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x197x384xf32) <- (-1x197x384xf32, 384x384xf32)
        matmul_178 = paddle._C_ops.matmul(reshape__148, parameter_351, False, False)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, 384xf32)
        add__131 = paddle._C_ops.add_(matmul_178, parameter_352)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, -1x197x384xf32)
        add__132 = paddle._C_ops.add_(set_value_with_tensor__11, add__131)

        # pd_op.layer_norm: (-1x197x384xf32, -197xf32, -197xf32) <- (-1x197x384xf32, 384xf32, 384xf32)
        layer_norm_219, layer_norm_220, layer_norm_221 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__132, parameter_353, parameter_354, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.matmul: (-1x197x1536xf32) <- (-1x197x384xf32, 384x1536xf32)
        matmul_179 = paddle._C_ops.matmul(layer_norm_219, parameter_355, False, False)

        # pd_op.add_: (-1x197x1536xf32) <- (-1x197x1536xf32, 1536xf32)
        add__133 = paddle._C_ops.add_(matmul_179, parameter_356)

        # pd_op.gelu: (-1x197x1536xf32) <- (-1x197x1536xf32)
        gelu_23 = paddle._C_ops.gelu(add__133, False)

        # pd_op.matmul: (-1x197x384xf32) <- (-1x197x1536xf32, 1536x384xf32)
        matmul_180 = paddle._C_ops.matmul(gelu_23, parameter_357, False, False)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, 384xf32)
        add__134 = paddle._C_ops.add_(matmul_180, parameter_358)

        # pd_op.add_: (-1x197x384xf32) <- (-1x197x384xf32, -1x197x384xf32)
        add__135 = paddle._C_ops.add_(add__132, add__134)

        # pd_op.layer_norm: (-1x197x384xf32, -197xf32, -197xf32) <- (-1x197x384xf32, 384xf32, 384xf32)
        layer_norm_222, layer_norm_223, layer_norm_224 = (lambda x, f: f(x))(paddle._C_ops.layer_norm(add__135, parameter_359, parameter_360, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_234 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_235 = [1]

        # pd_op.slice: (-1x384xf32) <- (-1x197x384xf32, 1xi64, 1xi64)
        slice_97 = paddle._C_ops.slice(layer_norm_222, [1], full_int_array_234, full_int_array_235, [1], [1])

        # pd_op.matmul: (-1x1000xf32) <- (-1x384xf32, 384x1000xf32)
        matmul_181 = paddle._C_ops.matmul(slice_97, parameter_361, False, False)

        # pd_op.add_: (-1x1000xf32) <- (-1x1000xf32, 1000xf32)
        add__136 = paddle._C_ops.add_(matmul_181, parameter_362)

        # pd_op.softmax_: (-1x1000xf32) <- (-1x1000xf32)
        softmax__24 = paddle._C_ops.softmax_(add__136, -1)
        return softmax__24



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

    def forward(self, parameter_0, parameter_1, parameter_2, parameter_4, parameter_3, parameter_5, parameter_6, parameter_8, parameter_7, parameter_9, parameter_10, parameter_12, parameter_11, parameter_13, parameter_14, parameter_15, parameter_16, parameter_18, parameter_17, parameter_19, parameter_20, parameter_21, parameter_22, parameter_24, parameter_23, parameter_25, parameter_27, parameter_26, parameter_29, parameter_28, parameter_30, parameter_31, parameter_32, parameter_33, parameter_35, parameter_34, parameter_36, parameter_37, parameter_38, parameter_39, parameter_41, parameter_40, parameter_42, parameter_43, parameter_44, parameter_45, parameter_47, parameter_46, parameter_48, parameter_49, parameter_50, parameter_51, parameter_53, parameter_52, parameter_54, parameter_56, parameter_55, parameter_58, parameter_57, parameter_59, parameter_60, parameter_61, parameter_62, parameter_64, parameter_63, parameter_65, parameter_66, parameter_67, parameter_68, parameter_70, parameter_69, parameter_71, parameter_72, parameter_73, parameter_74, parameter_76, parameter_75, parameter_77, parameter_78, parameter_79, parameter_80, parameter_82, parameter_81, parameter_83, parameter_85, parameter_84, parameter_87, parameter_86, parameter_88, parameter_89, parameter_90, parameter_91, parameter_93, parameter_92, parameter_94, parameter_95, parameter_96, parameter_97, parameter_99, parameter_98, parameter_100, parameter_101, parameter_102, parameter_103, parameter_105, parameter_104, parameter_106, parameter_107, parameter_108, parameter_109, parameter_111, parameter_110, parameter_112, parameter_114, parameter_113, parameter_116, parameter_115, parameter_117, parameter_118, parameter_119, parameter_120, parameter_122, parameter_121, parameter_123, parameter_124, parameter_125, parameter_126, parameter_128, parameter_127, parameter_129, parameter_130, parameter_131, parameter_132, parameter_134, parameter_133, parameter_135, parameter_136, parameter_137, parameter_138, parameter_140, parameter_139, parameter_141, parameter_143, parameter_142, parameter_145, parameter_144, parameter_146, parameter_147, parameter_148, parameter_149, parameter_151, parameter_150, parameter_152, parameter_153, parameter_154, parameter_155, parameter_157, parameter_156, parameter_158, parameter_159, parameter_160, parameter_161, parameter_163, parameter_162, parameter_164, parameter_165, parameter_166, parameter_167, parameter_169, parameter_168, parameter_170, parameter_172, parameter_171, parameter_174, parameter_173, parameter_175, parameter_176, parameter_177, parameter_178, parameter_180, parameter_179, parameter_181, parameter_182, parameter_183, parameter_184, parameter_186, parameter_185, parameter_187, parameter_188, parameter_189, parameter_190, parameter_192, parameter_191, parameter_193, parameter_194, parameter_195, parameter_196, parameter_198, parameter_197, parameter_199, parameter_201, parameter_200, parameter_203, parameter_202, parameter_204, parameter_205, parameter_206, parameter_207, parameter_209, parameter_208, parameter_210, parameter_211, parameter_212, parameter_213, parameter_215, parameter_214, parameter_216, parameter_217, parameter_218, parameter_219, parameter_221, parameter_220, parameter_222, parameter_223, parameter_224, parameter_225, parameter_227, parameter_226, parameter_228, parameter_230, parameter_229, parameter_232, parameter_231, parameter_233, parameter_234, parameter_235, parameter_236, parameter_238, parameter_237, parameter_239, parameter_240, parameter_241, parameter_242, parameter_244, parameter_243, parameter_245, parameter_246, parameter_247, parameter_248, parameter_250, parameter_249, parameter_251, parameter_252, parameter_253, parameter_254, parameter_256, parameter_255, parameter_257, parameter_259, parameter_258, parameter_261, parameter_260, parameter_262, parameter_263, parameter_264, parameter_265, parameter_267, parameter_266, parameter_268, parameter_269, parameter_270, parameter_271, parameter_273, parameter_272, parameter_274, parameter_275, parameter_276, parameter_277, parameter_279, parameter_278, parameter_280, parameter_281, parameter_282, parameter_283, parameter_285, parameter_284, parameter_286, parameter_288, parameter_287, parameter_290, parameter_289, parameter_291, parameter_292, parameter_293, parameter_294, parameter_296, parameter_295, parameter_297, parameter_298, parameter_299, parameter_300, parameter_302, parameter_301, parameter_303, parameter_304, parameter_305, parameter_306, parameter_308, parameter_307, parameter_309, parameter_310, parameter_311, parameter_312, parameter_314, parameter_313, parameter_315, parameter_317, parameter_316, parameter_319, parameter_318, parameter_320, parameter_321, parameter_322, parameter_323, parameter_325, parameter_324, parameter_326, parameter_327, parameter_328, parameter_329, parameter_331, parameter_330, parameter_332, parameter_333, parameter_334, parameter_335, parameter_337, parameter_336, parameter_338, parameter_339, parameter_340, parameter_341, parameter_343, parameter_342, parameter_344, parameter_346, parameter_345, parameter_348, parameter_347, parameter_349, parameter_350, parameter_351, parameter_352, parameter_354, parameter_353, parameter_355, parameter_356, parameter_357, parameter_358, parameter_360, parameter_359, parameter_361, parameter_362, feed_0):
        return self.builtin_module_1764_0_0(parameter_0, parameter_1, parameter_2, parameter_4, parameter_3, parameter_5, parameter_6, parameter_8, parameter_7, parameter_9, parameter_10, parameter_12, parameter_11, parameter_13, parameter_14, parameter_15, parameter_16, parameter_18, parameter_17, parameter_19, parameter_20, parameter_21, parameter_22, parameter_24, parameter_23, parameter_25, parameter_27, parameter_26, parameter_29, parameter_28, parameter_30, parameter_31, parameter_32, parameter_33, parameter_35, parameter_34, parameter_36, parameter_37, parameter_38, parameter_39, parameter_41, parameter_40, parameter_42, parameter_43, parameter_44, parameter_45, parameter_47, parameter_46, parameter_48, parameter_49, parameter_50, parameter_51, parameter_53, parameter_52, parameter_54, parameter_56, parameter_55, parameter_58, parameter_57, parameter_59, parameter_60, parameter_61, parameter_62, parameter_64, parameter_63, parameter_65, parameter_66, parameter_67, parameter_68, parameter_70, parameter_69, parameter_71, parameter_72, parameter_73, parameter_74, parameter_76, parameter_75, parameter_77, parameter_78, parameter_79, parameter_80, parameter_82, parameter_81, parameter_83, parameter_85, parameter_84, parameter_87, parameter_86, parameter_88, parameter_89, parameter_90, parameter_91, parameter_93, parameter_92, parameter_94, parameter_95, parameter_96, parameter_97, parameter_99, parameter_98, parameter_100, parameter_101, parameter_102, parameter_103, parameter_105, parameter_104, parameter_106, parameter_107, parameter_108, parameter_109, parameter_111, parameter_110, parameter_112, parameter_114, parameter_113, parameter_116, parameter_115, parameter_117, parameter_118, parameter_119, parameter_120, parameter_122, parameter_121, parameter_123, parameter_124, parameter_125, parameter_126, parameter_128, parameter_127, parameter_129, parameter_130, parameter_131, parameter_132, parameter_134, parameter_133, parameter_135, parameter_136, parameter_137, parameter_138, parameter_140, parameter_139, parameter_141, parameter_143, parameter_142, parameter_145, parameter_144, parameter_146, parameter_147, parameter_148, parameter_149, parameter_151, parameter_150, parameter_152, parameter_153, parameter_154, parameter_155, parameter_157, parameter_156, parameter_158, parameter_159, parameter_160, parameter_161, parameter_163, parameter_162, parameter_164, parameter_165, parameter_166, parameter_167, parameter_169, parameter_168, parameter_170, parameter_172, parameter_171, parameter_174, parameter_173, parameter_175, parameter_176, parameter_177, parameter_178, parameter_180, parameter_179, parameter_181, parameter_182, parameter_183, parameter_184, parameter_186, parameter_185, parameter_187, parameter_188, parameter_189, parameter_190, parameter_192, parameter_191, parameter_193, parameter_194, parameter_195, parameter_196, parameter_198, parameter_197, parameter_199, parameter_201, parameter_200, parameter_203, parameter_202, parameter_204, parameter_205, parameter_206, parameter_207, parameter_209, parameter_208, parameter_210, parameter_211, parameter_212, parameter_213, parameter_215, parameter_214, parameter_216, parameter_217, parameter_218, parameter_219, parameter_221, parameter_220, parameter_222, parameter_223, parameter_224, parameter_225, parameter_227, parameter_226, parameter_228, parameter_230, parameter_229, parameter_232, parameter_231, parameter_233, parameter_234, parameter_235, parameter_236, parameter_238, parameter_237, parameter_239, parameter_240, parameter_241, parameter_242, parameter_244, parameter_243, parameter_245, parameter_246, parameter_247, parameter_248, parameter_250, parameter_249, parameter_251, parameter_252, parameter_253, parameter_254, parameter_256, parameter_255, parameter_257, parameter_259, parameter_258, parameter_261, parameter_260, parameter_262, parameter_263, parameter_264, parameter_265, parameter_267, parameter_266, parameter_268, parameter_269, parameter_270, parameter_271, parameter_273, parameter_272, parameter_274, parameter_275, parameter_276, parameter_277, parameter_279, parameter_278, parameter_280, parameter_281, parameter_282, parameter_283, parameter_285, parameter_284, parameter_286, parameter_288, parameter_287, parameter_290, parameter_289, parameter_291, parameter_292, parameter_293, parameter_294, parameter_296, parameter_295, parameter_297, parameter_298, parameter_299, parameter_300, parameter_302, parameter_301, parameter_303, parameter_304, parameter_305, parameter_306, parameter_308, parameter_307, parameter_309, parameter_310, parameter_311, parameter_312, parameter_314, parameter_313, parameter_315, parameter_317, parameter_316, parameter_319, parameter_318, parameter_320, parameter_321, parameter_322, parameter_323, parameter_325, parameter_324, parameter_326, parameter_327, parameter_328, parameter_329, parameter_331, parameter_330, parameter_332, parameter_333, parameter_334, parameter_335, parameter_337, parameter_336, parameter_338, parameter_339, parameter_340, parameter_341, parameter_343, parameter_342, parameter_344, parameter_346, parameter_345, parameter_348, parameter_347, parameter_349, parameter_350, parameter_351, parameter_352, parameter_354, parameter_353, parameter_355, parameter_356, parameter_357, parameter_358, parameter_360, parameter_359, parameter_361, parameter_362, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_1764_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([24, 3, 7, 7], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([1, 16, 24], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([1, 1, 384], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([1, 197, 384], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([24, 48], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([24, 96], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([96, 24], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([384, 1536], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
            # parameter_38
            paddle.uniform([1536, 384], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([24, 48], dtype='float32', min=0, max=0.5),
            # parameter_43
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([24, 96], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([96, 24], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([384, 1536], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([1536, 384], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([24, 48], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([24, 96], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([96, 24], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([384, 1536], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([1536, 384], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([24, 48], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([24, 96], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([96, 24], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([384, 1536], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([1536, 384], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([24, 48], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_133
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([24, 96], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([96, 24], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_139
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_149
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([384, 1536], dtype='float32', min=0, max=0.5),
            # parameter_153
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
            # parameter_154
            paddle.uniform([1536, 384], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_158
            paddle.uniform([24, 48], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([24, 96], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([96, 24], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([384, 1536], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([1536, 384], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([24, 48], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            # parameter_189
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([24, 96], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([96, 24], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_209
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([384, 1536], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([1536, 384], dtype='float32', min=0, max=0.5),
            # parameter_213
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_215
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_214
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([24, 48], dtype='float32', min=0, max=0.5),
            # parameter_217
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            # parameter_218
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            # parameter_219
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_221
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([24, 96], dtype='float32', min=0, max=0.5),
            # parameter_223
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_224
            paddle.uniform([96, 24], dtype='float32', min=0, max=0.5),
            # parameter_225
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_227
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_228
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_229
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_233
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            # parameter_234
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_238
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_239
            paddle.uniform([384, 1536], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
            # parameter_241
            paddle.uniform([1536, 384], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_244
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_243
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_245
            paddle.uniform([24, 48], dtype='float32', min=0, max=0.5),
            # parameter_246
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            # parameter_248
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_250
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_249
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_251
            paddle.uniform([24, 96], dtype='float32', min=0, max=0.5),
            # parameter_252
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_253
            paddle.uniform([96, 24], dtype='float32', min=0, max=0.5),
            # parameter_254
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_256
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_255
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_257
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_259
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_258
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_261
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_262
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            # parameter_263
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_264
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_265
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_267
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_266
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_268
            paddle.uniform([384, 1536], dtype='float32', min=0, max=0.5),
            # parameter_269
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
            # parameter_270
            paddle.uniform([1536, 384], dtype='float32', min=0, max=0.5),
            # parameter_271
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_273
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_272
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_274
            paddle.uniform([24, 48], dtype='float32', min=0, max=0.5),
            # parameter_275
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            # parameter_276
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            # parameter_277
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_279
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_278
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_280
            paddle.uniform([24, 96], dtype='float32', min=0, max=0.5),
            # parameter_281
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_282
            paddle.uniform([96, 24], dtype='float32', min=0, max=0.5),
            # parameter_283
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_285
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_284
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_286
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_288
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_287
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_290
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_289
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_291
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            # parameter_292
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_293
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_294
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_296
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_295
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_297
            paddle.uniform([384, 1536], dtype='float32', min=0, max=0.5),
            # parameter_298
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
            # parameter_299
            paddle.uniform([1536, 384], dtype='float32', min=0, max=0.5),
            # parameter_300
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_302
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_301
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_303
            paddle.uniform([24, 48], dtype='float32', min=0, max=0.5),
            # parameter_304
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            # parameter_305
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            # parameter_306
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_308
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_307
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_309
            paddle.uniform([24, 96], dtype='float32', min=0, max=0.5),
            # parameter_310
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_311
            paddle.uniform([96, 24], dtype='float32', min=0, max=0.5),
            # parameter_312
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_314
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_313
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_315
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_317
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_316
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_319
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_318
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_320
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            # parameter_321
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_322
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_323
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_325
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_324
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_326
            paddle.uniform([384, 1536], dtype='float32', min=0, max=0.5),
            # parameter_327
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
            # parameter_328
            paddle.uniform([1536, 384], dtype='float32', min=0, max=0.5),
            # parameter_329
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_331
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_330
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_332
            paddle.uniform([24, 48], dtype='float32', min=0, max=0.5),
            # parameter_333
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            # parameter_334
            paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            # parameter_335
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_337
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_336
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_338
            paddle.uniform([24, 96], dtype='float32', min=0, max=0.5),
            # parameter_339
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            # parameter_340
            paddle.uniform([96, 24], dtype='float32', min=0, max=0.5),
            # parameter_341
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_343
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_342
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_344
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_346
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_345
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_348
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_347
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_349
            paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            # parameter_350
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_351
            paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            # parameter_352
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_354
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_353
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_355
            paddle.uniform([384, 1536], dtype='float32', min=0, max=0.5),
            # parameter_356
            paddle.uniform([1536], dtype='float32', min=0, max=0.5),
            # parameter_357
            paddle.uniform([1536, 384], dtype='float32', min=0, max=0.5),
            # parameter_358
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_360
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_359
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_361
            paddle.uniform([384, 1000], dtype='float32', min=0, max=0.5),
            # parameter_362
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
            paddle.static.InputSpec(shape=[24, 3, 7, 7], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[1, 16, 24], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[1, 1, 384], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[1, 197, 384], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[24, 48], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[24, 96], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[96, 24], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[384, 1536], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[1536], dtype='float32'),
            # parameter_38
            paddle.static.InputSpec(shape=[1536, 384], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[24, 48], dtype='float32'),
            # parameter_43
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[24, 96], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[96, 24], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[384, 1536], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[1536], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[1536, 384], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[24, 48], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[24, 96], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[96, 24], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[384, 1536], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[1536], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[1536, 384], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[24, 48], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[24, 96], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[96, 24], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[384, 1536], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[1536], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[1536, 384], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[24, 48], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_133
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[24, 96], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[96, 24], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_139
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_149
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[384, 1536], dtype='float32'),
            # parameter_153
            paddle.static.InputSpec(shape=[1536], dtype='float32'),
            # parameter_154
            paddle.static.InputSpec(shape=[1536, 384], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_158
            paddle.static.InputSpec(shape=[24, 48], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[24, 96], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[96, 24], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[384, 1536], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[1536], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[1536, 384], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[24, 48], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
            # parameter_189
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[24, 96], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[96, 24], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_209
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[384, 1536], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[1536], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[1536, 384], dtype='float32'),
            # parameter_213
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_215
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_214
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[24, 48], dtype='float32'),
            # parameter_217
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
            # parameter_218
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
            # parameter_219
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_221
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[24, 96], dtype='float32'),
            # parameter_223
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_224
            paddle.static.InputSpec(shape=[96, 24], dtype='float32'),
            # parameter_225
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_227
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_228
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_229
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_233
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
            # parameter_234
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_238
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_239
            paddle.static.InputSpec(shape=[384, 1536], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[1536], dtype='float32'),
            # parameter_241
            paddle.static.InputSpec(shape=[1536, 384], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_244
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_243
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_245
            paddle.static.InputSpec(shape=[24, 48], dtype='float32'),
            # parameter_246
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
            # parameter_248
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_250
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_249
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_251
            paddle.static.InputSpec(shape=[24, 96], dtype='float32'),
            # parameter_252
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_253
            paddle.static.InputSpec(shape=[96, 24], dtype='float32'),
            # parameter_254
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_256
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_255
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_257
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_259
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_258
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_261
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_262
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
            # parameter_263
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_264
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_265
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_267
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_266
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_268
            paddle.static.InputSpec(shape=[384, 1536], dtype='float32'),
            # parameter_269
            paddle.static.InputSpec(shape=[1536], dtype='float32'),
            # parameter_270
            paddle.static.InputSpec(shape=[1536, 384], dtype='float32'),
            # parameter_271
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_273
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_272
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_274
            paddle.static.InputSpec(shape=[24, 48], dtype='float32'),
            # parameter_275
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
            # parameter_276
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
            # parameter_277
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_279
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_278
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_280
            paddle.static.InputSpec(shape=[24, 96], dtype='float32'),
            # parameter_281
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_282
            paddle.static.InputSpec(shape=[96, 24], dtype='float32'),
            # parameter_283
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_285
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_284
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_286
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_288
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_287
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_290
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_289
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_291
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
            # parameter_292
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_293
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_294
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_296
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_295
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_297
            paddle.static.InputSpec(shape=[384, 1536], dtype='float32'),
            # parameter_298
            paddle.static.InputSpec(shape=[1536], dtype='float32'),
            # parameter_299
            paddle.static.InputSpec(shape=[1536, 384], dtype='float32'),
            # parameter_300
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_302
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_301
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_303
            paddle.static.InputSpec(shape=[24, 48], dtype='float32'),
            # parameter_304
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
            # parameter_305
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
            # parameter_306
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_308
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_307
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_309
            paddle.static.InputSpec(shape=[24, 96], dtype='float32'),
            # parameter_310
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_311
            paddle.static.InputSpec(shape=[96, 24], dtype='float32'),
            # parameter_312
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_314
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_313
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_315
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_317
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_316
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_319
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_318
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_320
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
            # parameter_321
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_322
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_323
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_325
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_324
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_326
            paddle.static.InputSpec(shape=[384, 1536], dtype='float32'),
            # parameter_327
            paddle.static.InputSpec(shape=[1536], dtype='float32'),
            # parameter_328
            paddle.static.InputSpec(shape=[1536, 384], dtype='float32'),
            # parameter_329
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_331
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_330
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_332
            paddle.static.InputSpec(shape=[24, 48], dtype='float32'),
            # parameter_333
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
            # parameter_334
            paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
            # parameter_335
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_337
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_336
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_338
            paddle.static.InputSpec(shape=[24, 96], dtype='float32'),
            # parameter_339
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            # parameter_340
            paddle.static.InputSpec(shape=[96, 24], dtype='float32'),
            # parameter_341
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_343
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_342
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_344
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_346
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_345
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_348
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_347
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_349
            paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
            # parameter_350
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_351
            paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            # parameter_352
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_354
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_353
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_355
            paddle.static.InputSpec(shape=[384, 1536], dtype='float32'),
            # parameter_356
            paddle.static.InputSpec(shape=[1536], dtype='float32'),
            # parameter_357
            paddle.static.InputSpec(shape=[1536, 384], dtype='float32'),
            # parameter_358
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_360
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_359
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_361
            paddle.static.InputSpec(shape=[384, 1000], dtype='float32'),
            # parameter_362
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