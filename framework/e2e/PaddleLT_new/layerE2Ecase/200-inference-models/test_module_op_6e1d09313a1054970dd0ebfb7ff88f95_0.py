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
    return [509][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_877_0_0(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_8, parameter_5, parameter_7, parameter_6, parameter_9, parameter_13, parameter_10, parameter_12, parameter_11, parameter_14, parameter_18, parameter_15, parameter_17, parameter_16, parameter_19, parameter_23, parameter_20, parameter_22, parameter_21, parameter_24, parameter_28, parameter_25, parameter_27, parameter_26, parameter_29, parameter_33, parameter_30, parameter_32, parameter_31, parameter_34, parameter_38, parameter_35, parameter_37, parameter_36, parameter_39, parameter_43, parameter_40, parameter_42, parameter_41, parameter_44, parameter_48, parameter_45, parameter_47, parameter_46, parameter_49, parameter_53, parameter_50, parameter_52, parameter_51, parameter_54, parameter_58, parameter_55, parameter_57, parameter_56, parameter_59, parameter_63, parameter_60, parameter_62, parameter_61, parameter_64, parameter_68, parameter_65, parameter_67, parameter_66, parameter_69, parameter_73, parameter_70, parameter_72, parameter_71, parameter_74, parameter_78, parameter_75, parameter_77, parameter_76, parameter_79, parameter_83, parameter_80, parameter_82, parameter_81, parameter_84, parameter_88, parameter_85, parameter_87, parameter_86, parameter_89, parameter_93, parameter_90, parameter_92, parameter_91, parameter_94, parameter_98, parameter_95, parameter_97, parameter_96, parameter_99, parameter_103, parameter_100, parameter_102, parameter_101, parameter_104, parameter_108, parameter_105, parameter_107, parameter_106, parameter_109, parameter_113, parameter_110, parameter_112, parameter_111, parameter_114, parameter_118, parameter_115, parameter_117, parameter_116, parameter_119, parameter_123, parameter_120, parameter_122, parameter_121, parameter_124, parameter_128, parameter_125, parameter_127, parameter_126, parameter_129, parameter_133, parameter_130, parameter_132, parameter_131, parameter_134, parameter_138, parameter_135, parameter_137, parameter_136, parameter_139, parameter_143, parameter_140, parameter_142, parameter_141, parameter_144, parameter_148, parameter_145, parameter_147, parameter_146, parameter_149, parameter_153, parameter_150, parameter_152, parameter_151, parameter_154, parameter_158, parameter_155, parameter_157, parameter_156, parameter_159, parameter_163, parameter_160, parameter_162, parameter_161, parameter_164, parameter_168, parameter_165, parameter_167, parameter_166, parameter_169, parameter_173, parameter_170, parameter_172, parameter_171, parameter_174, parameter_178, parameter_175, parameter_177, parameter_176, parameter_179, parameter_183, parameter_180, parameter_182, parameter_181, parameter_184, parameter_188, parameter_185, parameter_187, parameter_186, parameter_189, parameter_193, parameter_190, parameter_192, parameter_191, parameter_194, parameter_198, parameter_195, parameter_197, parameter_196, parameter_199, parameter_203, parameter_200, parameter_202, parameter_201, parameter_204, parameter_208, parameter_205, parameter_207, parameter_206, parameter_209, parameter_213, parameter_210, parameter_212, parameter_211, parameter_214, parameter_218, parameter_215, parameter_217, parameter_216, parameter_219, parameter_223, parameter_220, parameter_222, parameter_221, parameter_224, parameter_228, parameter_225, parameter_227, parameter_226, parameter_229, parameter_233, parameter_230, parameter_232, parameter_231, parameter_234, parameter_238, parameter_235, parameter_237, parameter_236, parameter_239, parameter_243, parameter_240, parameter_242, parameter_241, parameter_244, parameter_248, parameter_245, parameter_247, parameter_246, parameter_249, parameter_253, parameter_250, parameter_252, parameter_251, parameter_254, parameter_258, parameter_255, parameter_257, parameter_256, parameter_259, parameter_263, parameter_260, parameter_262, parameter_261, parameter_264, parameter_268, parameter_265, parameter_267, parameter_266, parameter_269, parameter_273, parameter_270, parameter_272, parameter_271, parameter_274, parameter_278, parameter_275, parameter_277, parameter_276, parameter_279, parameter_283, parameter_280, parameter_282, parameter_281, parameter_284, parameter_288, parameter_285, parameter_287, parameter_286, parameter_289, parameter_293, parameter_290, parameter_292, parameter_291, parameter_294, parameter_298, parameter_295, parameter_297, parameter_296, parameter_299, parameter_303, parameter_300, parameter_302, parameter_301, parameter_304, parameter_308, parameter_305, parameter_307, parameter_306, parameter_309, parameter_313, parameter_310, parameter_312, parameter_311, parameter_314, parameter_318, parameter_315, parameter_317, parameter_316, parameter_319, parameter_323, parameter_320, parameter_322, parameter_321, parameter_324, parameter_328, parameter_325, parameter_327, parameter_326, parameter_329, parameter_333, parameter_330, parameter_332, parameter_331, parameter_334, parameter_338, parameter_335, parameter_337, parameter_336, parameter_339, parameter_343, parameter_340, parameter_342, parameter_341, parameter_344, parameter_348, parameter_345, parameter_347, parameter_346, parameter_349, parameter_353, parameter_350, parameter_352, parameter_351, parameter_354, parameter_358, parameter_355, parameter_357, parameter_356, parameter_359, parameter_360, feed_0):

        # pd_op.cast: (-1x3x224x224xf16) <- (-1x3x224x224xf32)
        cast_0 = paddle._C_ops.cast(feed_0, paddle.float16)

        # pd_op.conv2d: (-1x10x112x112xf16) <- (-1x3x224x224xf16, 10x3x3x3xf16)
        conv2d_0 = paddle._C_ops.conv2d(cast_0, parameter_0, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x10x112x112xf16, 10xf32, 10xf32, 10xf32, 10xf32, None) <- (-1x10x112x112xf16, 10xf32, 10xf32, 10xf32, 10xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_0, parameter_1, parameter_2, parameter_3, parameter_4, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x10x112x112xf16) <- (-1x10x112x112xf16)
        relu__0 = paddle._C_ops.relu_(batch_norm__0)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [3, 3]

        # pd_op.pool2d: (-1x10x56x56xf16) <- (-1x10x112x112xf16, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(relu__0, full_int_array_0, [2, 2], [1, 1], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.batch_norm_: (-1x10x56x56xf16, 10xf32, 10xf32, 10xf32, 10xf32, None) <- (-1x10x56x56xf16, 10xf32, 10xf32, 10xf32, 10xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(pool2d_0, parameter_5, parameter_6, parameter_7, parameter_8, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x10x56x56xf16) <- (-1x10x56x56xf16)
        relu__1 = paddle._C_ops.relu_(batch_norm__6)

        # pd_op.conv2d: (-1x96x56x56xf16) <- (-1x10x56x56xf16, 96x10x1x1xf16)
        conv2d_1 = paddle._C_ops.conv2d(relu__1, parameter_9, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [64, 32]

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x64x56x56xf16, -1x32x56x56xf16]) <- (-1x96x56x56xf16, 2xi64, 1xi32)
        split_0 = paddle._C_ops.split(conv2d_1, full_int_array_1, full_0)

        # pd_op.batch_norm_: (-1x10x56x56xf16, 10xf32, 10xf32, 10xf32, 10xf32, None) <- (-1x10x56x56xf16, 10xf32, 10xf32, 10xf32, 10xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(pool2d_0, parameter_10, parameter_11, parameter_12, parameter_13, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x10x56x56xf16) <- (-1x10x56x56xf16)
        relu__2 = paddle._C_ops.relu_(batch_norm__12)

        # pd_op.conv2d: (-1x128x56x56xf16) <- (-1x10x56x56xf16, 128x10x1x1xf16)
        conv2d_2 = paddle._C_ops.conv2d(relu__2, parameter_14, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x56x56xf16, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x56x56xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_2, parameter_15, parameter_16, parameter_17, parameter_18, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x56x56xf16) <- (-1x128x56x56xf16)
        relu__3 = paddle._C_ops.relu_(batch_norm__18)

        # pd_op.conv2d: (-1x128x56x56xf16) <- (-1x128x56x56xf16, 128x4x3x3xf16)
        conv2d_3 = paddle._C_ops.conv2d(relu__3, parameter_19, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x128x56x56xf16, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x56x56xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_3, parameter_20, parameter_21, parameter_22, parameter_23, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x56x56xf16) <- (-1x128x56x56xf16)
        relu__4 = paddle._C_ops.relu_(batch_norm__24)

        # pd_op.conv2d: (-1x80x56x56xf16) <- (-1x128x56x56xf16, 80x128x1x1xf16)
        conv2d_4 = paddle._C_ops.conv2d(relu__4, parameter_24, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [64, 16]

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x64x56x56xf16, -1x16x56x56xf16]) <- (-1x80x56x56xf16, 2xi64, 1xi32)
        split_1 = paddle._C_ops.split(conv2d_4, full_int_array_2, full_1)

        # builtin.slice: (-1x64x56x56xf16) <- ([-1x64x56x56xf16, -1x32x56x56xf16])
        slice_0 = split_0[0]

        # builtin.slice: (-1x64x56x56xf16) <- ([-1x64x56x56xf16, -1x16x56x56xf16])
        slice_1 = split_1[0]

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, -1x64x56x56xf16)
        add__0 = paddle._C_ops.add_(slice_0, slice_1)

        # builtin.slice: (-1x32x56x56xf16) <- ([-1x64x56x56xf16, -1x32x56x56xf16])
        slice_2 = split_0[1]

        # builtin.slice: (-1x16x56x56xf16) <- ([-1x64x56x56xf16, -1x16x56x56xf16])
        slice_3 = split_1[1]

        # builtin.combine: ([-1x32x56x56xf16, -1x16x56x56xf16]) <- (-1x32x56x56xf16, -1x16x56x56xf16)
        combine_0 = [slice_2, slice_3]

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x48x56x56xf16) <- ([-1x32x56x56xf16, -1x16x56x56xf16], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_2)

        # builtin.combine: ([-1x64x56x56xf16, -1x48x56x56xf16]) <- (-1x64x56x56xf16, -1x48x56x56xf16)
        combine_1 = [add__0, concat_0]

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x112x56x56xf16) <- ([-1x64x56x56xf16, -1x48x56x56xf16], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_3)

        # pd_op.batch_norm_: (-1x112x56x56xf16, 112xf32, 112xf32, 112xf32, 112xf32, None) <- (-1x112x56x56xf16, 112xf32, 112xf32, 112xf32, 112xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_1, parameter_25, parameter_26, parameter_27, parameter_28, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x112x56x56xf16) <- (-1x112x56x56xf16)
        relu__5 = paddle._C_ops.relu_(batch_norm__30)

        # pd_op.conv2d: (-1x128x56x56xf16) <- (-1x112x56x56xf16, 128x112x1x1xf16)
        conv2d_5 = paddle._C_ops.conv2d(relu__5, parameter_29, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x56x56xf16, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x56x56xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_5, parameter_30, parameter_31, parameter_32, parameter_33, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x56x56xf16) <- (-1x128x56x56xf16)
        relu__6 = paddle._C_ops.relu_(batch_norm__36)

        # pd_op.conv2d: (-1x128x56x56xf16) <- (-1x128x56x56xf16, 128x4x3x3xf16)
        conv2d_6 = paddle._C_ops.conv2d(relu__6, parameter_34, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x128x56x56xf16, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x56x56xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_6, parameter_35, parameter_36, parameter_37, parameter_38, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x56x56xf16) <- (-1x128x56x56xf16)
        relu__7 = paddle._C_ops.relu_(batch_norm__42)

        # pd_op.conv2d: (-1x80x56x56xf16) <- (-1x128x56x56xf16, 80x128x1x1xf16)
        conv2d_7 = paddle._C_ops.conv2d(relu__7, parameter_39, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [64, 16]

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x64x56x56xf16, -1x16x56x56xf16]) <- (-1x80x56x56xf16, 2xi64, 1xi32)
        split_2 = paddle._C_ops.split(conv2d_7, full_int_array_3, full_4)

        # builtin.slice: (-1x64x56x56xf16) <- ([-1x64x56x56xf16, -1x16x56x56xf16])
        slice_4 = split_2[0]

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, -1x64x56x56xf16)
        add__1 = paddle._C_ops.add_(add__0, slice_4)

        # builtin.slice: (-1x16x56x56xf16) <- ([-1x64x56x56xf16, -1x16x56x56xf16])
        slice_5 = split_2[1]

        # builtin.combine: ([-1x48x56x56xf16, -1x16x56x56xf16]) <- (-1x48x56x56xf16, -1x16x56x56xf16)
        combine_2 = [concat_0, slice_5]

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x64x56x56xf16) <- ([-1x48x56x56xf16, -1x16x56x56xf16], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_5)

        # builtin.combine: ([-1x64x56x56xf16, -1x64x56x56xf16]) <- (-1x64x56x56xf16, -1x64x56x56xf16)
        combine_3 = [add__1, concat_2]

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x128x56x56xf16) <- ([-1x64x56x56xf16, -1x64x56x56xf16], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_3, full_6)

        # pd_op.batch_norm_: (-1x128x56x56xf16, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x56x56xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_3, parameter_40, parameter_41, parameter_42, parameter_43, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x56x56xf16) <- (-1x128x56x56xf16)
        relu__8 = paddle._C_ops.relu_(batch_norm__48)

        # pd_op.conv2d: (-1x128x56x56xf16) <- (-1x128x56x56xf16, 128x128x1x1xf16)
        conv2d_8 = paddle._C_ops.conv2d(relu__8, parameter_44, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x56x56xf16, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x56x56xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_8, parameter_45, parameter_46, parameter_47, parameter_48, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x56x56xf16) <- (-1x128x56x56xf16)
        relu__9 = paddle._C_ops.relu_(batch_norm__54)

        # pd_op.conv2d: (-1x128x56x56xf16) <- (-1x128x56x56xf16, 128x4x3x3xf16)
        conv2d_9 = paddle._C_ops.conv2d(relu__9, parameter_49, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x128x56x56xf16, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x56x56xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_9, parameter_50, parameter_51, parameter_52, parameter_53, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x128x56x56xf16) <- (-1x128x56x56xf16)
        relu__10 = paddle._C_ops.relu_(batch_norm__60)

        # pd_op.conv2d: (-1x80x56x56xf16) <- (-1x128x56x56xf16, 80x128x1x1xf16)
        conv2d_10 = paddle._C_ops.conv2d(relu__10, parameter_54, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [64, 16]

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x64x56x56xf16, -1x16x56x56xf16]) <- (-1x80x56x56xf16, 2xi64, 1xi32)
        split_3 = paddle._C_ops.split(conv2d_10, full_int_array_4, full_7)

        # builtin.slice: (-1x64x56x56xf16) <- ([-1x64x56x56xf16, -1x16x56x56xf16])
        slice_6 = split_3[0]

        # pd_op.add_: (-1x64x56x56xf16) <- (-1x64x56x56xf16, -1x64x56x56xf16)
        add__2 = paddle._C_ops.add_(add__1, slice_6)

        # builtin.slice: (-1x16x56x56xf16) <- ([-1x64x56x56xf16, -1x16x56x56xf16])
        slice_7 = split_3[1]

        # builtin.combine: ([-1x64x56x56xf16, -1x16x56x56xf16]) <- (-1x64x56x56xf16, -1x16x56x56xf16)
        combine_4 = [concat_2, slice_7]

        # pd_op.full: (1xi32) <- ()
        full_8 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x80x56x56xf16) <- ([-1x64x56x56xf16, -1x16x56x56xf16], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_4, full_8)

        # builtin.combine: ([-1x64x56x56xf16, -1x80x56x56xf16]) <- (-1x64x56x56xf16, -1x80x56x56xf16)
        combine_5 = [add__2, concat_4]

        # pd_op.full: (1xi32) <- ()
        full_9 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x144x56x56xf16) <- ([-1x64x56x56xf16, -1x80x56x56xf16], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_5, full_9)

        # pd_op.batch_norm_: (-1x144x56x56xf16, 144xf32, 144xf32, 144xf32, 144xf32, None) <- (-1x144x56x56xf16, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_5, parameter_55, parameter_56, parameter_57, parameter_58, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x144x56x56xf16) <- (-1x144x56x56xf16)
        relu__11 = paddle._C_ops.relu_(batch_norm__66)

        # pd_op.conv2d: (-1x192x28x28xf16) <- (-1x144x56x56xf16, 192x144x1x1xf16)
        conv2d_11 = paddle._C_ops.conv2d(relu__11, parameter_59, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [128, 64]

        # pd_op.full: (1xi32) <- ()
        full_10 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x128x28x28xf16, -1x64x28x28xf16]) <- (-1x192x28x28xf16, 2xi64, 1xi32)
        split_4 = paddle._C_ops.split(conv2d_11, full_int_array_5, full_10)

        # pd_op.batch_norm_: (-1x144x56x56xf16, 144xf32, 144xf32, 144xf32, 144xf32, None) <- (-1x144x56x56xf16, 144xf32, 144xf32, 144xf32, 144xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_5, parameter_60, parameter_61, parameter_62, parameter_63, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x144x56x56xf16) <- (-1x144x56x56xf16)
        relu__12 = paddle._C_ops.relu_(batch_norm__72)

        # pd_op.conv2d: (-1x256x56x56xf16) <- (-1x144x56x56xf16, 256x144x1x1xf16)
        conv2d_12 = paddle._C_ops.conv2d(relu__12, parameter_64, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x56x56xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x56x56xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_12, parameter_65, parameter_66, parameter_67, parameter_68, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x56x56xf16) <- (-1x256x56x56xf16)
        relu__13 = paddle._C_ops.relu_(batch_norm__78)

        # pd_op.conv2d: (-1x256x28x28xf16) <- (-1x256x56x56xf16, 256x8x3x3xf16)
        conv2d_13 = paddle._C_ops.conv2d(relu__13, parameter_69, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x256x28x28xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x28x28xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_13, parameter_70, parameter_71, parameter_72, parameter_73, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x28x28xf16) <- (-1x256x28x28xf16)
        relu__14 = paddle._C_ops.relu_(batch_norm__84)

        # pd_op.conv2d: (-1x160x28x28xf16) <- (-1x256x28x28xf16, 160x256x1x1xf16)
        conv2d_14 = paddle._C_ops.conv2d(relu__14, parameter_74, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_6 = [128, 32]

        # pd_op.full: (1xi32) <- ()
        full_11 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x128x28x28xf16, -1x32x28x28xf16]) <- (-1x160x28x28xf16, 2xi64, 1xi32)
        split_5 = paddle._C_ops.split(conv2d_14, full_int_array_6, full_11)

        # builtin.slice: (-1x128x28x28xf16) <- ([-1x128x28x28xf16, -1x64x28x28xf16])
        slice_8 = split_4[0]

        # builtin.slice: (-1x128x28x28xf16) <- ([-1x128x28x28xf16, -1x32x28x28xf16])
        slice_9 = split_5[0]

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, -1x128x28x28xf16)
        add__3 = paddle._C_ops.add_(slice_8, slice_9)

        # builtin.slice: (-1x64x28x28xf16) <- ([-1x128x28x28xf16, -1x64x28x28xf16])
        slice_10 = split_4[1]

        # builtin.slice: (-1x32x28x28xf16) <- ([-1x128x28x28xf16, -1x32x28x28xf16])
        slice_11 = split_5[1]

        # builtin.combine: ([-1x64x28x28xf16, -1x32x28x28xf16]) <- (-1x64x28x28xf16, -1x32x28x28xf16)
        combine_6 = [slice_10, slice_11]

        # pd_op.full: (1xi32) <- ()
        full_12 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x96x28x28xf16) <- ([-1x64x28x28xf16, -1x32x28x28xf16], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_6, full_12)

        # builtin.combine: ([-1x128x28x28xf16, -1x96x28x28xf16]) <- (-1x128x28x28xf16, -1x96x28x28xf16)
        combine_7 = [add__3, concat_6]

        # pd_op.full: (1xi32) <- ()
        full_13 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x224x28x28xf16) <- ([-1x128x28x28xf16, -1x96x28x28xf16], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_7, full_13)

        # pd_op.batch_norm_: (-1x224x28x28xf16, 224xf32, 224xf32, 224xf32, 224xf32, None) <- (-1x224x28x28xf16, 224xf32, 224xf32, 224xf32, 224xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_7, parameter_75, parameter_76, parameter_77, parameter_78, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x224x28x28xf16) <- (-1x224x28x28xf16)
        relu__15 = paddle._C_ops.relu_(batch_norm__90)

        # pd_op.conv2d: (-1x256x28x28xf16) <- (-1x224x28x28xf16, 256x224x1x1xf16)
        conv2d_15 = paddle._C_ops.conv2d(relu__15, parameter_79, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x28x28xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x28x28xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_15, parameter_80, parameter_81, parameter_82, parameter_83, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x28x28xf16) <- (-1x256x28x28xf16)
        relu__16 = paddle._C_ops.relu_(batch_norm__96)

        # pd_op.conv2d: (-1x256x28x28xf16) <- (-1x256x28x28xf16, 256x8x3x3xf16)
        conv2d_16 = paddle._C_ops.conv2d(relu__16, parameter_84, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x256x28x28xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x28x28xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_16, parameter_85, parameter_86, parameter_87, parameter_88, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x28x28xf16) <- (-1x256x28x28xf16)
        relu__17 = paddle._C_ops.relu_(batch_norm__102)

        # pd_op.conv2d: (-1x160x28x28xf16) <- (-1x256x28x28xf16, 160x256x1x1xf16)
        conv2d_17 = paddle._C_ops.conv2d(relu__17, parameter_89, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_7 = [128, 32]

        # pd_op.full: (1xi32) <- ()
        full_14 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x128x28x28xf16, -1x32x28x28xf16]) <- (-1x160x28x28xf16, 2xi64, 1xi32)
        split_6 = paddle._C_ops.split(conv2d_17, full_int_array_7, full_14)

        # builtin.slice: (-1x128x28x28xf16) <- ([-1x128x28x28xf16, -1x32x28x28xf16])
        slice_12 = split_6[0]

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, -1x128x28x28xf16)
        add__4 = paddle._C_ops.add_(add__3, slice_12)

        # builtin.slice: (-1x32x28x28xf16) <- ([-1x128x28x28xf16, -1x32x28x28xf16])
        slice_13 = split_6[1]

        # builtin.combine: ([-1x96x28x28xf16, -1x32x28x28xf16]) <- (-1x96x28x28xf16, -1x32x28x28xf16)
        combine_8 = [concat_6, slice_13]

        # pd_op.full: (1xi32) <- ()
        full_15 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x128x28x28xf16) <- ([-1x96x28x28xf16, -1x32x28x28xf16], 1xi32)
        concat_8 = paddle._C_ops.concat(combine_8, full_15)

        # builtin.combine: ([-1x128x28x28xf16, -1x128x28x28xf16]) <- (-1x128x28x28xf16, -1x128x28x28xf16)
        combine_9 = [add__4, concat_8]

        # pd_op.full: (1xi32) <- ()
        full_16 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x256x28x28xf16) <- ([-1x128x28x28xf16, -1x128x28x28xf16], 1xi32)
        concat_9 = paddle._C_ops.concat(combine_9, full_16)

        # pd_op.batch_norm_: (-1x256x28x28xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x28x28xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_9, parameter_90, parameter_91, parameter_92, parameter_93, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x28x28xf16) <- (-1x256x28x28xf16)
        relu__18 = paddle._C_ops.relu_(batch_norm__108)

        # pd_op.conv2d: (-1x256x28x28xf16) <- (-1x256x28x28xf16, 256x256x1x1xf16)
        conv2d_18 = paddle._C_ops.conv2d(relu__18, parameter_94, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x28x28xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x28x28xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_18, parameter_95, parameter_96, parameter_97, parameter_98, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x28x28xf16) <- (-1x256x28x28xf16)
        relu__19 = paddle._C_ops.relu_(batch_norm__114)

        # pd_op.conv2d: (-1x256x28x28xf16) <- (-1x256x28x28xf16, 256x8x3x3xf16)
        conv2d_19 = paddle._C_ops.conv2d(relu__19, parameter_99, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x256x28x28xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x28x28xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_19, parameter_100, parameter_101, parameter_102, parameter_103, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x28x28xf16) <- (-1x256x28x28xf16)
        relu__20 = paddle._C_ops.relu_(batch_norm__120)

        # pd_op.conv2d: (-1x160x28x28xf16) <- (-1x256x28x28xf16, 160x256x1x1xf16)
        conv2d_20 = paddle._C_ops.conv2d(relu__20, parameter_104, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_8 = [128, 32]

        # pd_op.full: (1xi32) <- ()
        full_17 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x128x28x28xf16, -1x32x28x28xf16]) <- (-1x160x28x28xf16, 2xi64, 1xi32)
        split_7 = paddle._C_ops.split(conv2d_20, full_int_array_8, full_17)

        # builtin.slice: (-1x128x28x28xf16) <- ([-1x128x28x28xf16, -1x32x28x28xf16])
        slice_14 = split_7[0]

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, -1x128x28x28xf16)
        add__5 = paddle._C_ops.add_(add__4, slice_14)

        # builtin.slice: (-1x32x28x28xf16) <- ([-1x128x28x28xf16, -1x32x28x28xf16])
        slice_15 = split_7[1]

        # builtin.combine: ([-1x128x28x28xf16, -1x32x28x28xf16]) <- (-1x128x28x28xf16, -1x32x28x28xf16)
        combine_10 = [concat_8, slice_15]

        # pd_op.full: (1xi32) <- ()
        full_18 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x160x28x28xf16) <- ([-1x128x28x28xf16, -1x32x28x28xf16], 1xi32)
        concat_10 = paddle._C_ops.concat(combine_10, full_18)

        # builtin.combine: ([-1x128x28x28xf16, -1x160x28x28xf16]) <- (-1x128x28x28xf16, -1x160x28x28xf16)
        combine_11 = [add__5, concat_10]

        # pd_op.full: (1xi32) <- ()
        full_19 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x288x28x28xf16) <- ([-1x128x28x28xf16, -1x160x28x28xf16], 1xi32)
        concat_11 = paddle._C_ops.concat(combine_11, full_19)

        # pd_op.batch_norm_: (-1x288x28x28xf16, 288xf32, 288xf32, 288xf32, 288xf32, None) <- (-1x288x28x28xf16, 288xf32, 288xf32, 288xf32, 288xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_11, parameter_105, parameter_106, parameter_107, parameter_108, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x288x28x28xf16) <- (-1x288x28x28xf16)
        relu__21 = paddle._C_ops.relu_(batch_norm__126)

        # pd_op.conv2d: (-1x256x28x28xf16) <- (-1x288x28x28xf16, 256x288x1x1xf16)
        conv2d_21 = paddle._C_ops.conv2d(relu__21, parameter_109, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x28x28xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x28x28xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_21, parameter_110, parameter_111, parameter_112, parameter_113, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x28x28xf16) <- (-1x256x28x28xf16)
        relu__22 = paddle._C_ops.relu_(batch_norm__132)

        # pd_op.conv2d: (-1x256x28x28xf16) <- (-1x256x28x28xf16, 256x8x3x3xf16)
        conv2d_22 = paddle._C_ops.conv2d(relu__22, parameter_114, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x256x28x28xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x28x28xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_22, parameter_115, parameter_116, parameter_117, parameter_118, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x256x28x28xf16) <- (-1x256x28x28xf16)
        relu__23 = paddle._C_ops.relu_(batch_norm__138)

        # pd_op.conv2d: (-1x160x28x28xf16) <- (-1x256x28x28xf16, 160x256x1x1xf16)
        conv2d_23 = paddle._C_ops.conv2d(relu__23, parameter_119, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_9 = [128, 32]

        # pd_op.full: (1xi32) <- ()
        full_20 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x128x28x28xf16, -1x32x28x28xf16]) <- (-1x160x28x28xf16, 2xi64, 1xi32)
        split_8 = paddle._C_ops.split(conv2d_23, full_int_array_9, full_20)

        # builtin.slice: (-1x128x28x28xf16) <- ([-1x128x28x28xf16, -1x32x28x28xf16])
        slice_16 = split_8[0]

        # pd_op.add_: (-1x128x28x28xf16) <- (-1x128x28x28xf16, -1x128x28x28xf16)
        add__6 = paddle._C_ops.add_(add__5, slice_16)

        # builtin.slice: (-1x32x28x28xf16) <- ([-1x128x28x28xf16, -1x32x28x28xf16])
        slice_17 = split_8[1]

        # builtin.combine: ([-1x160x28x28xf16, -1x32x28x28xf16]) <- (-1x160x28x28xf16, -1x32x28x28xf16)
        combine_12 = [concat_10, slice_17]

        # pd_op.full: (1xi32) <- ()
        full_21 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x192x28x28xf16) <- ([-1x160x28x28xf16, -1x32x28x28xf16], 1xi32)
        concat_12 = paddle._C_ops.concat(combine_12, full_21)

        # builtin.combine: ([-1x128x28x28xf16, -1x192x28x28xf16]) <- (-1x128x28x28xf16, -1x192x28x28xf16)
        combine_13 = [add__6, concat_12]

        # pd_op.full: (1xi32) <- ()
        full_22 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x320x28x28xf16) <- ([-1x128x28x28xf16, -1x192x28x28xf16], 1xi32)
        concat_13 = paddle._C_ops.concat(combine_13, full_22)

        # pd_op.batch_norm_: (-1x320x28x28xf16, 320xf32, 320xf32, 320xf32, 320xf32, None) <- (-1x320x28x28xf16, 320xf32, 320xf32, 320xf32, 320xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_13, parameter_120, parameter_121, parameter_122, parameter_123, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x320x28x28xf16) <- (-1x320x28x28xf16)
        relu__24 = paddle._C_ops.relu_(batch_norm__144)

        # pd_op.conv2d: (-1x320x14x14xf16) <- (-1x320x28x28xf16, 320x320x1x1xf16)
        conv2d_24 = paddle._C_ops.conv2d(relu__24, parameter_124, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_10 = [256, 64]

        # pd_op.full: (1xi32) <- ()
        full_23 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x256x14x14xf16, -1x64x14x14xf16]) <- (-1x320x14x14xf16, 2xi64, 1xi32)
        split_9 = paddle._C_ops.split(conv2d_24, full_int_array_10, full_23)

        # pd_op.batch_norm_: (-1x320x28x28xf16, 320xf32, 320xf32, 320xf32, 320xf32, None) <- (-1x320x28x28xf16, 320xf32, 320xf32, 320xf32, 320xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_13, parameter_125, parameter_126, parameter_127, parameter_128, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x320x28x28xf16) <- (-1x320x28x28xf16)
        relu__25 = paddle._C_ops.relu_(batch_norm__150)

        # pd_op.conv2d: (-1x512x28x28xf16) <- (-1x320x28x28xf16, 512x320x1x1xf16)
        conv2d_25 = paddle._C_ops.conv2d(relu__25, parameter_129, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x28x28xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x28x28xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_25, parameter_130, parameter_131, parameter_132, parameter_133, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x28x28xf16) <- (-1x512x28x28xf16)
        relu__26 = paddle._C_ops.relu_(batch_norm__156)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x512x28x28xf16, 512x16x3x3xf16)
        conv2d_26 = paddle._C_ops.conv2d(relu__26, parameter_134, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_26, parameter_135, parameter_136, parameter_137, parameter_138, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__27 = paddle._C_ops.relu_(batch_norm__162)

        # pd_op.conv2d: (-1x288x14x14xf16) <- (-1x512x14x14xf16, 288x512x1x1xf16)
        conv2d_27 = paddle._C_ops.conv2d(relu__27, parameter_139, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_11 = [256, 32]

        # pd_op.full: (1xi32) <- ()
        full_24 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x256x14x14xf16, -1x32x14x14xf16]) <- (-1x288x14x14xf16, 2xi64, 1xi32)
        split_10 = paddle._C_ops.split(conv2d_27, full_int_array_11, full_24)

        # builtin.slice: (-1x256x14x14xf16) <- ([-1x256x14x14xf16, -1x64x14x14xf16])
        slice_18 = split_9[0]

        # builtin.slice: (-1x256x14x14xf16) <- ([-1x256x14x14xf16, -1x32x14x14xf16])
        slice_19 = split_10[0]

        # pd_op.add_: (-1x256x14x14xf16) <- (-1x256x14x14xf16, -1x256x14x14xf16)
        add__7 = paddle._C_ops.add_(slice_18, slice_19)

        # builtin.slice: (-1x64x14x14xf16) <- ([-1x256x14x14xf16, -1x64x14x14xf16])
        slice_20 = split_9[1]

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x256x14x14xf16, -1x32x14x14xf16])
        slice_21 = split_10[1]

        # builtin.combine: ([-1x64x14x14xf16, -1x32x14x14xf16]) <- (-1x64x14x14xf16, -1x32x14x14xf16)
        combine_14 = [slice_20, slice_21]

        # pd_op.full: (1xi32) <- ()
        full_25 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x96x14x14xf16) <- ([-1x64x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_14 = paddle._C_ops.concat(combine_14, full_25)

        # builtin.combine: ([-1x256x14x14xf16, -1x96x14x14xf16]) <- (-1x256x14x14xf16, -1x96x14x14xf16)
        combine_15 = [add__7, concat_14]

        # pd_op.full: (1xi32) <- ()
        full_26 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x352x14x14xf16) <- ([-1x256x14x14xf16, -1x96x14x14xf16], 1xi32)
        concat_15 = paddle._C_ops.concat(combine_15, full_26)

        # pd_op.batch_norm_: (-1x352x14x14xf16, 352xf32, 352xf32, 352xf32, 352xf32, None) <- (-1x352x14x14xf16, 352xf32, 352xf32, 352xf32, 352xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_15, parameter_140, parameter_141, parameter_142, parameter_143, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x352x14x14xf16) <- (-1x352x14x14xf16)
        relu__28 = paddle._C_ops.relu_(batch_norm__168)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x352x14x14xf16, 512x352x1x1xf16)
        conv2d_28 = paddle._C_ops.conv2d(relu__28, parameter_144, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_28, parameter_145, parameter_146, parameter_147, parameter_148, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__29 = paddle._C_ops.relu_(batch_norm__174)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x512x14x14xf16, 512x16x3x3xf16)
        conv2d_29 = paddle._C_ops.conv2d(relu__29, parameter_149, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__180, batch_norm__181, batch_norm__182, batch_norm__183, batch_norm__184, batch_norm__185 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_29, parameter_150, parameter_151, parameter_152, parameter_153, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__30 = paddle._C_ops.relu_(batch_norm__180)

        # pd_op.conv2d: (-1x288x14x14xf16) <- (-1x512x14x14xf16, 288x512x1x1xf16)
        conv2d_30 = paddle._C_ops.conv2d(relu__30, parameter_154, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_12 = [256, 32]

        # pd_op.full: (1xi32) <- ()
        full_27 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x256x14x14xf16, -1x32x14x14xf16]) <- (-1x288x14x14xf16, 2xi64, 1xi32)
        split_11 = paddle._C_ops.split(conv2d_30, full_int_array_12, full_27)

        # builtin.slice: (-1x256x14x14xf16) <- ([-1x256x14x14xf16, -1x32x14x14xf16])
        slice_22 = split_11[0]

        # pd_op.add_: (-1x256x14x14xf16) <- (-1x256x14x14xf16, -1x256x14x14xf16)
        add__8 = paddle._C_ops.add_(add__7, slice_22)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x256x14x14xf16, -1x32x14x14xf16])
        slice_23 = split_11[1]

        # builtin.combine: ([-1x96x14x14xf16, -1x32x14x14xf16]) <- (-1x96x14x14xf16, -1x32x14x14xf16)
        combine_16 = [concat_14, slice_23]

        # pd_op.full: (1xi32) <- ()
        full_28 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x128x14x14xf16) <- ([-1x96x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_16 = paddle._C_ops.concat(combine_16, full_28)

        # builtin.combine: ([-1x256x14x14xf16, -1x128x14x14xf16]) <- (-1x256x14x14xf16, -1x128x14x14xf16)
        combine_17 = [add__8, concat_16]

        # pd_op.full: (1xi32) <- ()
        full_29 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x384x14x14xf16) <- ([-1x256x14x14xf16, -1x128x14x14xf16], 1xi32)
        concat_17 = paddle._C_ops.concat(combine_17, full_29)

        # pd_op.batch_norm_: (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32, None) <- (-1x384x14x14xf16, 384xf32, 384xf32, 384xf32, 384xf32)
        batch_norm__186, batch_norm__187, batch_norm__188, batch_norm__189, batch_norm__190, batch_norm__191 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_17, parameter_155, parameter_156, parameter_157, parameter_158, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x384x14x14xf16) <- (-1x384x14x14xf16)
        relu__31 = paddle._C_ops.relu_(batch_norm__186)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x384x14x14xf16, 512x384x1x1xf16)
        conv2d_31 = paddle._C_ops.conv2d(relu__31, parameter_159, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__192, batch_norm__193, batch_norm__194, batch_norm__195, batch_norm__196, batch_norm__197 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_31, parameter_160, parameter_161, parameter_162, parameter_163, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__32 = paddle._C_ops.relu_(batch_norm__192)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x512x14x14xf16, 512x16x3x3xf16)
        conv2d_32 = paddle._C_ops.conv2d(relu__32, parameter_164, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__198, batch_norm__199, batch_norm__200, batch_norm__201, batch_norm__202, batch_norm__203 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_32, parameter_165, parameter_166, parameter_167, parameter_168, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__33 = paddle._C_ops.relu_(batch_norm__198)

        # pd_op.conv2d: (-1x288x14x14xf16) <- (-1x512x14x14xf16, 288x512x1x1xf16)
        conv2d_33 = paddle._C_ops.conv2d(relu__33, parameter_169, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_13 = [256, 32]

        # pd_op.full: (1xi32) <- ()
        full_30 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x256x14x14xf16, -1x32x14x14xf16]) <- (-1x288x14x14xf16, 2xi64, 1xi32)
        split_12 = paddle._C_ops.split(conv2d_33, full_int_array_13, full_30)

        # builtin.slice: (-1x256x14x14xf16) <- ([-1x256x14x14xf16, -1x32x14x14xf16])
        slice_24 = split_12[0]

        # pd_op.add_: (-1x256x14x14xf16) <- (-1x256x14x14xf16, -1x256x14x14xf16)
        add__9 = paddle._C_ops.add_(add__8, slice_24)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x256x14x14xf16, -1x32x14x14xf16])
        slice_25 = split_12[1]

        # builtin.combine: ([-1x128x14x14xf16, -1x32x14x14xf16]) <- (-1x128x14x14xf16, -1x32x14x14xf16)
        combine_18 = [concat_16, slice_25]

        # pd_op.full: (1xi32) <- ()
        full_31 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x160x14x14xf16) <- ([-1x128x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_18 = paddle._C_ops.concat(combine_18, full_31)

        # builtin.combine: ([-1x256x14x14xf16, -1x160x14x14xf16]) <- (-1x256x14x14xf16, -1x160x14x14xf16)
        combine_19 = [add__9, concat_18]

        # pd_op.full: (1xi32) <- ()
        full_32 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x416x14x14xf16) <- ([-1x256x14x14xf16, -1x160x14x14xf16], 1xi32)
        concat_19 = paddle._C_ops.concat(combine_19, full_32)

        # pd_op.batch_norm_: (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32, None) <- (-1x416x14x14xf16, 416xf32, 416xf32, 416xf32, 416xf32)
        batch_norm__204, batch_norm__205, batch_norm__206, batch_norm__207, batch_norm__208, batch_norm__209 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_19, parameter_170, parameter_171, parameter_172, parameter_173, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x416x14x14xf16) <- (-1x416x14x14xf16)
        relu__34 = paddle._C_ops.relu_(batch_norm__204)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x416x14x14xf16, 512x416x1x1xf16)
        conv2d_34 = paddle._C_ops.conv2d(relu__34, parameter_174, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__210, batch_norm__211, batch_norm__212, batch_norm__213, batch_norm__214, batch_norm__215 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_34, parameter_175, parameter_176, parameter_177, parameter_178, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__35 = paddle._C_ops.relu_(batch_norm__210)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x512x14x14xf16, 512x16x3x3xf16)
        conv2d_35 = paddle._C_ops.conv2d(relu__35, parameter_179, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__216, batch_norm__217, batch_norm__218, batch_norm__219, batch_norm__220, batch_norm__221 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_35, parameter_180, parameter_181, parameter_182, parameter_183, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__36 = paddle._C_ops.relu_(batch_norm__216)

        # pd_op.conv2d: (-1x288x14x14xf16) <- (-1x512x14x14xf16, 288x512x1x1xf16)
        conv2d_36 = paddle._C_ops.conv2d(relu__36, parameter_184, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_14 = [256, 32]

        # pd_op.full: (1xi32) <- ()
        full_33 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x256x14x14xf16, -1x32x14x14xf16]) <- (-1x288x14x14xf16, 2xi64, 1xi32)
        split_13 = paddle._C_ops.split(conv2d_36, full_int_array_14, full_33)

        # builtin.slice: (-1x256x14x14xf16) <- ([-1x256x14x14xf16, -1x32x14x14xf16])
        slice_26 = split_13[0]

        # pd_op.add_: (-1x256x14x14xf16) <- (-1x256x14x14xf16, -1x256x14x14xf16)
        add__10 = paddle._C_ops.add_(add__9, slice_26)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x256x14x14xf16, -1x32x14x14xf16])
        slice_27 = split_13[1]

        # builtin.combine: ([-1x160x14x14xf16, -1x32x14x14xf16]) <- (-1x160x14x14xf16, -1x32x14x14xf16)
        combine_20 = [concat_18, slice_27]

        # pd_op.full: (1xi32) <- ()
        full_34 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x192x14x14xf16) <- ([-1x160x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_20 = paddle._C_ops.concat(combine_20, full_34)

        # builtin.combine: ([-1x256x14x14xf16, -1x192x14x14xf16]) <- (-1x256x14x14xf16, -1x192x14x14xf16)
        combine_21 = [add__10, concat_20]

        # pd_op.full: (1xi32) <- ()
        full_35 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x448x14x14xf16) <- ([-1x256x14x14xf16, -1x192x14x14xf16], 1xi32)
        concat_21 = paddle._C_ops.concat(combine_21, full_35)

        # pd_op.batch_norm_: (-1x448x14x14xf16, 448xf32, 448xf32, 448xf32, 448xf32, None) <- (-1x448x14x14xf16, 448xf32, 448xf32, 448xf32, 448xf32)
        batch_norm__222, batch_norm__223, batch_norm__224, batch_norm__225, batch_norm__226, batch_norm__227 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_21, parameter_185, parameter_186, parameter_187, parameter_188, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x448x14x14xf16) <- (-1x448x14x14xf16)
        relu__37 = paddle._C_ops.relu_(batch_norm__222)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x448x14x14xf16, 512x448x1x1xf16)
        conv2d_37 = paddle._C_ops.conv2d(relu__37, parameter_189, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__228, batch_norm__229, batch_norm__230, batch_norm__231, batch_norm__232, batch_norm__233 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_37, parameter_190, parameter_191, parameter_192, parameter_193, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__38 = paddle._C_ops.relu_(batch_norm__228)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x512x14x14xf16, 512x16x3x3xf16)
        conv2d_38 = paddle._C_ops.conv2d(relu__38, parameter_194, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__234, batch_norm__235, batch_norm__236, batch_norm__237, batch_norm__238, batch_norm__239 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_38, parameter_195, parameter_196, parameter_197, parameter_198, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__39 = paddle._C_ops.relu_(batch_norm__234)

        # pd_op.conv2d: (-1x288x14x14xf16) <- (-1x512x14x14xf16, 288x512x1x1xf16)
        conv2d_39 = paddle._C_ops.conv2d(relu__39, parameter_199, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_15 = [256, 32]

        # pd_op.full: (1xi32) <- ()
        full_36 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x256x14x14xf16, -1x32x14x14xf16]) <- (-1x288x14x14xf16, 2xi64, 1xi32)
        split_14 = paddle._C_ops.split(conv2d_39, full_int_array_15, full_36)

        # builtin.slice: (-1x256x14x14xf16) <- ([-1x256x14x14xf16, -1x32x14x14xf16])
        slice_28 = split_14[0]

        # pd_op.add_: (-1x256x14x14xf16) <- (-1x256x14x14xf16, -1x256x14x14xf16)
        add__11 = paddle._C_ops.add_(add__10, slice_28)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x256x14x14xf16, -1x32x14x14xf16])
        slice_29 = split_14[1]

        # builtin.combine: ([-1x192x14x14xf16, -1x32x14x14xf16]) <- (-1x192x14x14xf16, -1x32x14x14xf16)
        combine_22 = [concat_20, slice_29]

        # pd_op.full: (1xi32) <- ()
        full_37 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x224x14x14xf16) <- ([-1x192x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_22 = paddle._C_ops.concat(combine_22, full_37)

        # builtin.combine: ([-1x256x14x14xf16, -1x224x14x14xf16]) <- (-1x256x14x14xf16, -1x224x14x14xf16)
        combine_23 = [add__11, concat_22]

        # pd_op.full: (1xi32) <- ()
        full_38 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x480x14x14xf16) <- ([-1x256x14x14xf16, -1x224x14x14xf16], 1xi32)
        concat_23 = paddle._C_ops.concat(combine_23, full_38)

        # pd_op.batch_norm_: (-1x480x14x14xf16, 480xf32, 480xf32, 480xf32, 480xf32, None) <- (-1x480x14x14xf16, 480xf32, 480xf32, 480xf32, 480xf32)
        batch_norm__240, batch_norm__241, batch_norm__242, batch_norm__243, batch_norm__244, batch_norm__245 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_23, parameter_200, parameter_201, parameter_202, parameter_203, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x480x14x14xf16) <- (-1x480x14x14xf16)
        relu__40 = paddle._C_ops.relu_(batch_norm__240)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x480x14x14xf16, 512x480x1x1xf16)
        conv2d_40 = paddle._C_ops.conv2d(relu__40, parameter_204, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__246, batch_norm__247, batch_norm__248, batch_norm__249, batch_norm__250, batch_norm__251 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_40, parameter_205, parameter_206, parameter_207, parameter_208, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__41 = paddle._C_ops.relu_(batch_norm__246)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x512x14x14xf16, 512x16x3x3xf16)
        conv2d_41 = paddle._C_ops.conv2d(relu__41, parameter_209, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__252, batch_norm__253, batch_norm__254, batch_norm__255, batch_norm__256, batch_norm__257 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_41, parameter_210, parameter_211, parameter_212, parameter_213, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__42 = paddle._C_ops.relu_(batch_norm__252)

        # pd_op.conv2d: (-1x288x14x14xf16) <- (-1x512x14x14xf16, 288x512x1x1xf16)
        conv2d_42 = paddle._C_ops.conv2d(relu__42, parameter_214, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_16 = [256, 32]

        # pd_op.full: (1xi32) <- ()
        full_39 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x256x14x14xf16, -1x32x14x14xf16]) <- (-1x288x14x14xf16, 2xi64, 1xi32)
        split_15 = paddle._C_ops.split(conv2d_42, full_int_array_16, full_39)

        # builtin.slice: (-1x256x14x14xf16) <- ([-1x256x14x14xf16, -1x32x14x14xf16])
        slice_30 = split_15[0]

        # pd_op.add_: (-1x256x14x14xf16) <- (-1x256x14x14xf16, -1x256x14x14xf16)
        add__12 = paddle._C_ops.add_(add__11, slice_30)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x256x14x14xf16, -1x32x14x14xf16])
        slice_31 = split_15[1]

        # builtin.combine: ([-1x224x14x14xf16, -1x32x14x14xf16]) <- (-1x224x14x14xf16, -1x32x14x14xf16)
        combine_24 = [concat_22, slice_31]

        # pd_op.full: (1xi32) <- ()
        full_40 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x256x14x14xf16) <- ([-1x224x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_24 = paddle._C_ops.concat(combine_24, full_40)

        # builtin.combine: ([-1x256x14x14xf16, -1x256x14x14xf16]) <- (-1x256x14x14xf16, -1x256x14x14xf16)
        combine_25 = [add__12, concat_24]

        # pd_op.full: (1xi32) <- ()
        full_41 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x512x14x14xf16) <- ([-1x256x14x14xf16, -1x256x14x14xf16], 1xi32)
        concat_25 = paddle._C_ops.concat(combine_25, full_41)

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__258, batch_norm__259, batch_norm__260, batch_norm__261, batch_norm__262, batch_norm__263 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_25, parameter_215, parameter_216, parameter_217, parameter_218, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__43 = paddle._C_ops.relu_(batch_norm__258)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x512x14x14xf16, 512x512x1x1xf16)
        conv2d_43 = paddle._C_ops.conv2d(relu__43, parameter_219, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__264, batch_norm__265, batch_norm__266, batch_norm__267, batch_norm__268, batch_norm__269 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_43, parameter_220, parameter_221, parameter_222, parameter_223, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__44 = paddle._C_ops.relu_(batch_norm__264)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x512x14x14xf16, 512x16x3x3xf16)
        conv2d_44 = paddle._C_ops.conv2d(relu__44, parameter_224, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__270, batch_norm__271, batch_norm__272, batch_norm__273, batch_norm__274, batch_norm__275 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_44, parameter_225, parameter_226, parameter_227, parameter_228, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__45 = paddle._C_ops.relu_(batch_norm__270)

        # pd_op.conv2d: (-1x288x14x14xf16) <- (-1x512x14x14xf16, 288x512x1x1xf16)
        conv2d_45 = paddle._C_ops.conv2d(relu__45, parameter_229, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_17 = [256, 32]

        # pd_op.full: (1xi32) <- ()
        full_42 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x256x14x14xf16, -1x32x14x14xf16]) <- (-1x288x14x14xf16, 2xi64, 1xi32)
        split_16 = paddle._C_ops.split(conv2d_45, full_int_array_17, full_42)

        # builtin.slice: (-1x256x14x14xf16) <- ([-1x256x14x14xf16, -1x32x14x14xf16])
        slice_32 = split_16[0]

        # pd_op.add_: (-1x256x14x14xf16) <- (-1x256x14x14xf16, -1x256x14x14xf16)
        add__13 = paddle._C_ops.add_(add__12, slice_32)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x256x14x14xf16, -1x32x14x14xf16])
        slice_33 = split_16[1]

        # builtin.combine: ([-1x256x14x14xf16, -1x32x14x14xf16]) <- (-1x256x14x14xf16, -1x32x14x14xf16)
        combine_26 = [concat_24, slice_33]

        # pd_op.full: (1xi32) <- ()
        full_43 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x288x14x14xf16) <- ([-1x256x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_26 = paddle._C_ops.concat(combine_26, full_43)

        # builtin.combine: ([-1x256x14x14xf16, -1x288x14x14xf16]) <- (-1x256x14x14xf16, -1x288x14x14xf16)
        combine_27 = [add__13, concat_26]

        # pd_op.full: (1xi32) <- ()
        full_44 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x544x14x14xf16) <- ([-1x256x14x14xf16, -1x288x14x14xf16], 1xi32)
        concat_27 = paddle._C_ops.concat(combine_27, full_44)

        # pd_op.batch_norm_: (-1x544x14x14xf16, 544xf32, 544xf32, 544xf32, 544xf32, None) <- (-1x544x14x14xf16, 544xf32, 544xf32, 544xf32, 544xf32)
        batch_norm__276, batch_norm__277, batch_norm__278, batch_norm__279, batch_norm__280, batch_norm__281 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_27, parameter_230, parameter_231, parameter_232, parameter_233, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x544x14x14xf16) <- (-1x544x14x14xf16)
        relu__46 = paddle._C_ops.relu_(batch_norm__276)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x544x14x14xf16, 512x544x1x1xf16)
        conv2d_46 = paddle._C_ops.conv2d(relu__46, parameter_234, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__282, batch_norm__283, batch_norm__284, batch_norm__285, batch_norm__286, batch_norm__287 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_46, parameter_235, parameter_236, parameter_237, parameter_238, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__47 = paddle._C_ops.relu_(batch_norm__282)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x512x14x14xf16, 512x16x3x3xf16)
        conv2d_47 = paddle._C_ops.conv2d(relu__47, parameter_239, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__288, batch_norm__289, batch_norm__290, batch_norm__291, batch_norm__292, batch_norm__293 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_47, parameter_240, parameter_241, parameter_242, parameter_243, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__48 = paddle._C_ops.relu_(batch_norm__288)

        # pd_op.conv2d: (-1x288x14x14xf16) <- (-1x512x14x14xf16, 288x512x1x1xf16)
        conv2d_48 = paddle._C_ops.conv2d(relu__48, parameter_244, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_18 = [256, 32]

        # pd_op.full: (1xi32) <- ()
        full_45 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x256x14x14xf16, -1x32x14x14xf16]) <- (-1x288x14x14xf16, 2xi64, 1xi32)
        split_17 = paddle._C_ops.split(conv2d_48, full_int_array_18, full_45)

        # builtin.slice: (-1x256x14x14xf16) <- ([-1x256x14x14xf16, -1x32x14x14xf16])
        slice_34 = split_17[0]

        # pd_op.add_: (-1x256x14x14xf16) <- (-1x256x14x14xf16, -1x256x14x14xf16)
        add__14 = paddle._C_ops.add_(add__13, slice_34)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x256x14x14xf16, -1x32x14x14xf16])
        slice_35 = split_17[1]

        # builtin.combine: ([-1x288x14x14xf16, -1x32x14x14xf16]) <- (-1x288x14x14xf16, -1x32x14x14xf16)
        combine_28 = [concat_26, slice_35]

        # pd_op.full: (1xi32) <- ()
        full_46 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x320x14x14xf16) <- ([-1x288x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_28 = paddle._C_ops.concat(combine_28, full_46)

        # builtin.combine: ([-1x256x14x14xf16, -1x320x14x14xf16]) <- (-1x256x14x14xf16, -1x320x14x14xf16)
        combine_29 = [add__14, concat_28]

        # pd_op.full: (1xi32) <- ()
        full_47 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x576x14x14xf16) <- ([-1x256x14x14xf16, -1x320x14x14xf16], 1xi32)
        concat_29 = paddle._C_ops.concat(combine_29, full_47)

        # pd_op.batch_norm_: (-1x576x14x14xf16, 576xf32, 576xf32, 576xf32, 576xf32, None) <- (-1x576x14x14xf16, 576xf32, 576xf32, 576xf32, 576xf32)
        batch_norm__294, batch_norm__295, batch_norm__296, batch_norm__297, batch_norm__298, batch_norm__299 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_29, parameter_245, parameter_246, parameter_247, parameter_248, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x576x14x14xf16) <- (-1x576x14x14xf16)
        relu__49 = paddle._C_ops.relu_(batch_norm__294)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x576x14x14xf16, 512x576x1x1xf16)
        conv2d_49 = paddle._C_ops.conv2d(relu__49, parameter_249, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__300, batch_norm__301, batch_norm__302, batch_norm__303, batch_norm__304, batch_norm__305 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_49, parameter_250, parameter_251, parameter_252, parameter_253, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__50 = paddle._C_ops.relu_(batch_norm__300)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x512x14x14xf16, 512x16x3x3xf16)
        conv2d_50 = paddle._C_ops.conv2d(relu__50, parameter_254, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__306, batch_norm__307, batch_norm__308, batch_norm__309, batch_norm__310, batch_norm__311 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_50, parameter_255, parameter_256, parameter_257, parameter_258, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__51 = paddle._C_ops.relu_(batch_norm__306)

        # pd_op.conv2d: (-1x288x14x14xf16) <- (-1x512x14x14xf16, 288x512x1x1xf16)
        conv2d_51 = paddle._C_ops.conv2d(relu__51, parameter_259, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_19 = [256, 32]

        # pd_op.full: (1xi32) <- ()
        full_48 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x256x14x14xf16, -1x32x14x14xf16]) <- (-1x288x14x14xf16, 2xi64, 1xi32)
        split_18 = paddle._C_ops.split(conv2d_51, full_int_array_19, full_48)

        # builtin.slice: (-1x256x14x14xf16) <- ([-1x256x14x14xf16, -1x32x14x14xf16])
        slice_36 = split_18[0]

        # pd_op.add_: (-1x256x14x14xf16) <- (-1x256x14x14xf16, -1x256x14x14xf16)
        add__15 = paddle._C_ops.add_(add__14, slice_36)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x256x14x14xf16, -1x32x14x14xf16])
        slice_37 = split_18[1]

        # builtin.combine: ([-1x320x14x14xf16, -1x32x14x14xf16]) <- (-1x320x14x14xf16, -1x32x14x14xf16)
        combine_30 = [concat_28, slice_37]

        # pd_op.full: (1xi32) <- ()
        full_49 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x352x14x14xf16) <- ([-1x320x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_30 = paddle._C_ops.concat(combine_30, full_49)

        # builtin.combine: ([-1x256x14x14xf16, -1x352x14x14xf16]) <- (-1x256x14x14xf16, -1x352x14x14xf16)
        combine_31 = [add__15, concat_30]

        # pd_op.full: (1xi32) <- ()
        full_50 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x608x14x14xf16) <- ([-1x256x14x14xf16, -1x352x14x14xf16], 1xi32)
        concat_31 = paddle._C_ops.concat(combine_31, full_50)

        # pd_op.batch_norm_: (-1x608x14x14xf16, 608xf32, 608xf32, 608xf32, 608xf32, None) <- (-1x608x14x14xf16, 608xf32, 608xf32, 608xf32, 608xf32)
        batch_norm__312, batch_norm__313, batch_norm__314, batch_norm__315, batch_norm__316, batch_norm__317 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_31, parameter_260, parameter_261, parameter_262, parameter_263, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x608x14x14xf16) <- (-1x608x14x14xf16)
        relu__52 = paddle._C_ops.relu_(batch_norm__312)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x608x14x14xf16, 512x608x1x1xf16)
        conv2d_52 = paddle._C_ops.conv2d(relu__52, parameter_264, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__318, batch_norm__319, batch_norm__320, batch_norm__321, batch_norm__322, batch_norm__323 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_52, parameter_265, parameter_266, parameter_267, parameter_268, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__53 = paddle._C_ops.relu_(batch_norm__318)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x512x14x14xf16, 512x16x3x3xf16)
        conv2d_53 = paddle._C_ops.conv2d(relu__53, parameter_269, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__324, batch_norm__325, batch_norm__326, batch_norm__327, batch_norm__328, batch_norm__329 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_53, parameter_270, parameter_271, parameter_272, parameter_273, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__54 = paddle._C_ops.relu_(batch_norm__324)

        # pd_op.conv2d: (-1x288x14x14xf16) <- (-1x512x14x14xf16, 288x512x1x1xf16)
        conv2d_54 = paddle._C_ops.conv2d(relu__54, parameter_274, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_20 = [256, 32]

        # pd_op.full: (1xi32) <- ()
        full_51 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x256x14x14xf16, -1x32x14x14xf16]) <- (-1x288x14x14xf16, 2xi64, 1xi32)
        split_19 = paddle._C_ops.split(conv2d_54, full_int_array_20, full_51)

        # builtin.slice: (-1x256x14x14xf16) <- ([-1x256x14x14xf16, -1x32x14x14xf16])
        slice_38 = split_19[0]

        # pd_op.add_: (-1x256x14x14xf16) <- (-1x256x14x14xf16, -1x256x14x14xf16)
        add__16 = paddle._C_ops.add_(add__15, slice_38)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x256x14x14xf16, -1x32x14x14xf16])
        slice_39 = split_19[1]

        # builtin.combine: ([-1x352x14x14xf16, -1x32x14x14xf16]) <- (-1x352x14x14xf16, -1x32x14x14xf16)
        combine_32 = [concat_30, slice_39]

        # pd_op.full: (1xi32) <- ()
        full_52 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x384x14x14xf16) <- ([-1x352x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_32 = paddle._C_ops.concat(combine_32, full_52)

        # builtin.combine: ([-1x256x14x14xf16, -1x384x14x14xf16]) <- (-1x256x14x14xf16, -1x384x14x14xf16)
        combine_33 = [add__16, concat_32]

        # pd_op.full: (1xi32) <- ()
        full_53 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x640x14x14xf16) <- ([-1x256x14x14xf16, -1x384x14x14xf16], 1xi32)
        concat_33 = paddle._C_ops.concat(combine_33, full_53)

        # pd_op.batch_norm_: (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32, None) <- (-1x640x14x14xf16, 640xf32, 640xf32, 640xf32, 640xf32)
        batch_norm__330, batch_norm__331, batch_norm__332, batch_norm__333, batch_norm__334, batch_norm__335 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_33, parameter_275, parameter_276, parameter_277, parameter_278, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x640x14x14xf16) <- (-1x640x14x14xf16)
        relu__55 = paddle._C_ops.relu_(batch_norm__330)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x640x14x14xf16, 512x640x1x1xf16)
        conv2d_55 = paddle._C_ops.conv2d(relu__55, parameter_279, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__336, batch_norm__337, batch_norm__338, batch_norm__339, batch_norm__340, batch_norm__341 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_55, parameter_280, parameter_281, parameter_282, parameter_283, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__56 = paddle._C_ops.relu_(batch_norm__336)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x512x14x14xf16, 512x16x3x3xf16)
        conv2d_56 = paddle._C_ops.conv2d(relu__56, parameter_284, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__342, batch_norm__343, batch_norm__344, batch_norm__345, batch_norm__346, batch_norm__347 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_56, parameter_285, parameter_286, parameter_287, parameter_288, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__57 = paddle._C_ops.relu_(batch_norm__342)

        # pd_op.conv2d: (-1x288x14x14xf16) <- (-1x512x14x14xf16, 288x512x1x1xf16)
        conv2d_57 = paddle._C_ops.conv2d(relu__57, parameter_289, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_21 = [256, 32]

        # pd_op.full: (1xi32) <- ()
        full_54 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x256x14x14xf16, -1x32x14x14xf16]) <- (-1x288x14x14xf16, 2xi64, 1xi32)
        split_20 = paddle._C_ops.split(conv2d_57, full_int_array_21, full_54)

        # builtin.slice: (-1x256x14x14xf16) <- ([-1x256x14x14xf16, -1x32x14x14xf16])
        slice_40 = split_20[0]

        # pd_op.add_: (-1x256x14x14xf16) <- (-1x256x14x14xf16, -1x256x14x14xf16)
        add__17 = paddle._C_ops.add_(add__16, slice_40)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x256x14x14xf16, -1x32x14x14xf16])
        slice_41 = split_20[1]

        # builtin.combine: ([-1x384x14x14xf16, -1x32x14x14xf16]) <- (-1x384x14x14xf16, -1x32x14x14xf16)
        combine_34 = [concat_32, slice_41]

        # pd_op.full: (1xi32) <- ()
        full_55 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x416x14x14xf16) <- ([-1x384x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_34 = paddle._C_ops.concat(combine_34, full_55)

        # builtin.combine: ([-1x256x14x14xf16, -1x416x14x14xf16]) <- (-1x256x14x14xf16, -1x416x14x14xf16)
        combine_35 = [add__17, concat_34]

        # pd_op.full: (1xi32) <- ()
        full_56 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x672x14x14xf16) <- ([-1x256x14x14xf16, -1x416x14x14xf16], 1xi32)
        concat_35 = paddle._C_ops.concat(combine_35, full_56)

        # pd_op.batch_norm_: (-1x672x14x14xf16, 672xf32, 672xf32, 672xf32, 672xf32, None) <- (-1x672x14x14xf16, 672xf32, 672xf32, 672xf32, 672xf32)
        batch_norm__348, batch_norm__349, batch_norm__350, batch_norm__351, batch_norm__352, batch_norm__353 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_35, parameter_290, parameter_291, parameter_292, parameter_293, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x672x14x14xf16) <- (-1x672x14x14xf16)
        relu__58 = paddle._C_ops.relu_(batch_norm__348)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x672x14x14xf16, 512x672x1x1xf16)
        conv2d_58 = paddle._C_ops.conv2d(relu__58, parameter_294, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__354, batch_norm__355, batch_norm__356, batch_norm__357, batch_norm__358, batch_norm__359 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_58, parameter_295, parameter_296, parameter_297, parameter_298, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__59 = paddle._C_ops.relu_(batch_norm__354)

        # pd_op.conv2d: (-1x512x14x14xf16) <- (-1x512x14x14xf16, 512x16x3x3xf16)
        conv2d_59 = paddle._C_ops.conv2d(relu__59, parameter_299, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x14x14xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__360, batch_norm__361, batch_norm__362, batch_norm__363, batch_norm__364, batch_norm__365 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_59, parameter_300, parameter_301, parameter_302, parameter_303, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x512x14x14xf16) <- (-1x512x14x14xf16)
        relu__60 = paddle._C_ops.relu_(batch_norm__360)

        # pd_op.conv2d: (-1x288x14x14xf16) <- (-1x512x14x14xf16, 288x512x1x1xf16)
        conv2d_60 = paddle._C_ops.conv2d(relu__60, parameter_304, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_22 = [256, 32]

        # pd_op.full: (1xi32) <- ()
        full_57 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x256x14x14xf16, -1x32x14x14xf16]) <- (-1x288x14x14xf16, 2xi64, 1xi32)
        split_21 = paddle._C_ops.split(conv2d_60, full_int_array_22, full_57)

        # builtin.slice: (-1x256x14x14xf16) <- ([-1x256x14x14xf16, -1x32x14x14xf16])
        slice_42 = split_21[0]

        # pd_op.add_: (-1x256x14x14xf16) <- (-1x256x14x14xf16, -1x256x14x14xf16)
        add__18 = paddle._C_ops.add_(add__17, slice_42)

        # builtin.slice: (-1x32x14x14xf16) <- ([-1x256x14x14xf16, -1x32x14x14xf16])
        slice_43 = split_21[1]

        # builtin.combine: ([-1x416x14x14xf16, -1x32x14x14xf16]) <- (-1x416x14x14xf16, -1x32x14x14xf16)
        combine_36 = [concat_34, slice_43]

        # pd_op.full: (1xi32) <- ()
        full_58 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x448x14x14xf16) <- ([-1x416x14x14xf16, -1x32x14x14xf16], 1xi32)
        concat_36 = paddle._C_ops.concat(combine_36, full_58)

        # builtin.combine: ([-1x256x14x14xf16, -1x448x14x14xf16]) <- (-1x256x14x14xf16, -1x448x14x14xf16)
        combine_37 = [add__18, concat_36]

        # pd_op.full: (1xi32) <- ()
        full_59 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x704x14x14xf16) <- ([-1x256x14x14xf16, -1x448x14x14xf16], 1xi32)
        concat_37 = paddle._C_ops.concat(combine_37, full_59)

        # pd_op.batch_norm_: (-1x704x14x14xf16, 704xf32, 704xf32, 704xf32, 704xf32, None) <- (-1x704x14x14xf16, 704xf32, 704xf32, 704xf32, 704xf32)
        batch_norm__366, batch_norm__367, batch_norm__368, batch_norm__369, batch_norm__370, batch_norm__371 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_37, parameter_305, parameter_306, parameter_307, parameter_308, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x704x14x14xf16) <- (-1x704x14x14xf16)
        relu__61 = paddle._C_ops.relu_(batch_norm__366)

        # pd_op.conv2d: (-1x640x7x7xf16) <- (-1x704x14x14xf16, 640x704x1x1xf16)
        conv2d_61 = paddle._C_ops.conv2d(relu__61, parameter_309, [2, 2], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_23 = [512, 128]

        # pd_op.full: (1xi32) <- ()
        full_60 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x512x7x7xf16, -1x128x7x7xf16]) <- (-1x640x7x7xf16, 2xi64, 1xi32)
        split_22 = paddle._C_ops.split(conv2d_61, full_int_array_23, full_60)

        # pd_op.batch_norm_: (-1x704x14x14xf16, 704xf32, 704xf32, 704xf32, 704xf32, None) <- (-1x704x14x14xf16, 704xf32, 704xf32, 704xf32, 704xf32)
        batch_norm__372, batch_norm__373, batch_norm__374, batch_norm__375, batch_norm__376, batch_norm__377 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_37, parameter_310, parameter_311, parameter_312, parameter_313, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x704x14x14xf16) <- (-1x704x14x14xf16)
        relu__62 = paddle._C_ops.relu_(batch_norm__372)

        # pd_op.conv2d: (-1x1024x14x14xf16) <- (-1x704x14x14xf16, 1024x704x1x1xf16)
        conv2d_62 = paddle._C_ops.conv2d(relu__62, parameter_314, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x14x14xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__378, batch_norm__379, batch_norm__380, batch_norm__381, batch_norm__382, batch_norm__383 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_62, parameter_315, parameter_316, parameter_317, parameter_318, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1024x14x14xf16) <- (-1x1024x14x14xf16)
        relu__63 = paddle._C_ops.relu_(batch_norm__378)

        # pd_op.conv2d: (-1x1024x7x7xf16) <- (-1x1024x14x14xf16, 1024x32x3x3xf16)
        conv2d_63 = paddle._C_ops.conv2d(relu__63, parameter_319, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x7x7xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x7x7xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__384, batch_norm__385, batch_norm__386, batch_norm__387, batch_norm__388, batch_norm__389 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_63, parameter_320, parameter_321, parameter_322, parameter_323, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1024x7x7xf16) <- (-1x1024x7x7xf16)
        relu__64 = paddle._C_ops.relu_(batch_norm__384)

        # pd_op.conv2d: (-1x576x7x7xf16) <- (-1x1024x7x7xf16, 576x1024x1x1xf16)
        conv2d_64 = paddle._C_ops.conv2d(relu__64, parameter_324, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_24 = [512, 64]

        # pd_op.full: (1xi32) <- ()
        full_61 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x512x7x7xf16, -1x64x7x7xf16]) <- (-1x576x7x7xf16, 2xi64, 1xi32)
        split_23 = paddle._C_ops.split(conv2d_64, full_int_array_24, full_61)

        # builtin.slice: (-1x512x7x7xf16) <- ([-1x512x7x7xf16, -1x128x7x7xf16])
        slice_44 = split_22[0]

        # builtin.slice: (-1x512x7x7xf16) <- ([-1x512x7x7xf16, -1x64x7x7xf16])
        slice_45 = split_23[0]

        # pd_op.add_: (-1x512x7x7xf16) <- (-1x512x7x7xf16, -1x512x7x7xf16)
        add__19 = paddle._C_ops.add_(slice_44, slice_45)

        # builtin.slice: (-1x128x7x7xf16) <- ([-1x512x7x7xf16, -1x128x7x7xf16])
        slice_46 = split_22[1]

        # builtin.slice: (-1x64x7x7xf16) <- ([-1x512x7x7xf16, -1x64x7x7xf16])
        slice_47 = split_23[1]

        # builtin.combine: ([-1x128x7x7xf16, -1x64x7x7xf16]) <- (-1x128x7x7xf16, -1x64x7x7xf16)
        combine_38 = [slice_46, slice_47]

        # pd_op.full: (1xi32) <- ()
        full_62 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x192x7x7xf16) <- ([-1x128x7x7xf16, -1x64x7x7xf16], 1xi32)
        concat_38 = paddle._C_ops.concat(combine_38, full_62)

        # builtin.combine: ([-1x512x7x7xf16, -1x192x7x7xf16]) <- (-1x512x7x7xf16, -1x192x7x7xf16)
        combine_39 = [add__19, concat_38]

        # pd_op.full: (1xi32) <- ()
        full_63 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x704x7x7xf16) <- ([-1x512x7x7xf16, -1x192x7x7xf16], 1xi32)
        concat_39 = paddle._C_ops.concat(combine_39, full_63)

        # pd_op.batch_norm_: (-1x704x7x7xf16, 704xf32, 704xf32, 704xf32, 704xf32, None) <- (-1x704x7x7xf16, 704xf32, 704xf32, 704xf32, 704xf32)
        batch_norm__390, batch_norm__391, batch_norm__392, batch_norm__393, batch_norm__394, batch_norm__395 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_39, parameter_325, parameter_326, parameter_327, parameter_328, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x704x7x7xf16) <- (-1x704x7x7xf16)
        relu__65 = paddle._C_ops.relu_(batch_norm__390)

        # pd_op.conv2d: (-1x1024x7x7xf16) <- (-1x704x7x7xf16, 1024x704x1x1xf16)
        conv2d_65 = paddle._C_ops.conv2d(relu__65, parameter_329, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x7x7xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x7x7xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__396, batch_norm__397, batch_norm__398, batch_norm__399, batch_norm__400, batch_norm__401 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_65, parameter_330, parameter_331, parameter_332, parameter_333, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1024x7x7xf16) <- (-1x1024x7x7xf16)
        relu__66 = paddle._C_ops.relu_(batch_norm__396)

        # pd_op.conv2d: (-1x1024x7x7xf16) <- (-1x1024x7x7xf16, 1024x32x3x3xf16)
        conv2d_66 = paddle._C_ops.conv2d(relu__66, parameter_334, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x7x7xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x7x7xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__402, batch_norm__403, batch_norm__404, batch_norm__405, batch_norm__406, batch_norm__407 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_66, parameter_335, parameter_336, parameter_337, parameter_338, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1024x7x7xf16) <- (-1x1024x7x7xf16)
        relu__67 = paddle._C_ops.relu_(batch_norm__402)

        # pd_op.conv2d: (-1x576x7x7xf16) <- (-1x1024x7x7xf16, 576x1024x1x1xf16)
        conv2d_67 = paddle._C_ops.conv2d(relu__67, parameter_339, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_25 = [512, 64]

        # pd_op.full: (1xi32) <- ()
        full_64 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x512x7x7xf16, -1x64x7x7xf16]) <- (-1x576x7x7xf16, 2xi64, 1xi32)
        split_24 = paddle._C_ops.split(conv2d_67, full_int_array_25, full_64)

        # builtin.slice: (-1x512x7x7xf16) <- ([-1x512x7x7xf16, -1x64x7x7xf16])
        slice_48 = split_24[0]

        # pd_op.add_: (-1x512x7x7xf16) <- (-1x512x7x7xf16, -1x512x7x7xf16)
        add__20 = paddle._C_ops.add_(add__19, slice_48)

        # builtin.slice: (-1x64x7x7xf16) <- ([-1x512x7x7xf16, -1x64x7x7xf16])
        slice_49 = split_24[1]

        # builtin.combine: ([-1x192x7x7xf16, -1x64x7x7xf16]) <- (-1x192x7x7xf16, -1x64x7x7xf16)
        combine_40 = [concat_38, slice_49]

        # pd_op.full: (1xi32) <- ()
        full_65 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x256x7x7xf16) <- ([-1x192x7x7xf16, -1x64x7x7xf16], 1xi32)
        concat_40 = paddle._C_ops.concat(combine_40, full_65)

        # builtin.combine: ([-1x512x7x7xf16, -1x256x7x7xf16]) <- (-1x512x7x7xf16, -1x256x7x7xf16)
        combine_41 = [add__20, concat_40]

        # pd_op.full: (1xi32) <- ()
        full_66 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x768x7x7xf16) <- ([-1x512x7x7xf16, -1x256x7x7xf16], 1xi32)
        concat_41 = paddle._C_ops.concat(combine_41, full_66)

        # pd_op.batch_norm_: (-1x768x7x7xf16, 768xf32, 768xf32, 768xf32, 768xf32, None) <- (-1x768x7x7xf16, 768xf32, 768xf32, 768xf32, 768xf32)
        batch_norm__408, batch_norm__409, batch_norm__410, batch_norm__411, batch_norm__412, batch_norm__413 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_41, parameter_340, parameter_341, parameter_342, parameter_343, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x768x7x7xf16) <- (-1x768x7x7xf16)
        relu__68 = paddle._C_ops.relu_(batch_norm__408)

        # pd_op.conv2d: (-1x1024x7x7xf16) <- (-1x768x7x7xf16, 1024x768x1x1xf16)
        conv2d_68 = paddle._C_ops.conv2d(relu__68, parameter_344, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x7x7xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x7x7xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__414, batch_norm__415, batch_norm__416, batch_norm__417, batch_norm__418, batch_norm__419 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_68, parameter_345, parameter_346, parameter_347, parameter_348, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1024x7x7xf16) <- (-1x1024x7x7xf16)
        relu__69 = paddle._C_ops.relu_(batch_norm__414)

        # pd_op.conv2d: (-1x1024x7x7xf16) <- (-1x1024x7x7xf16, 1024x32x3x3xf16)
        conv2d_69 = paddle._C_ops.conv2d(relu__69, parameter_349, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 32, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x7x7xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x7x7xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__420, batch_norm__421, batch_norm__422, batch_norm__423, batch_norm__424, batch_norm__425 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_69, parameter_350, parameter_351, parameter_352, parameter_353, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x1024x7x7xf16) <- (-1x1024x7x7xf16)
        relu__70 = paddle._C_ops.relu_(batch_norm__420)

        # pd_op.conv2d: (-1x576x7x7xf16) <- (-1x1024x7x7xf16, 576x1024x1x1xf16)
        conv2d_70 = paddle._C_ops.conv2d(relu__70, parameter_354, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_26 = [512, 64]

        # pd_op.full: (1xi32) <- ()
        full_67 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x512x7x7xf16, -1x64x7x7xf16]) <- (-1x576x7x7xf16, 2xi64, 1xi32)
        split_25 = paddle._C_ops.split(conv2d_70, full_int_array_26, full_67)

        # builtin.slice: (-1x512x7x7xf16) <- ([-1x512x7x7xf16, -1x64x7x7xf16])
        slice_50 = split_25[0]

        # pd_op.add_: (-1x512x7x7xf16) <- (-1x512x7x7xf16, -1x512x7x7xf16)
        add__21 = paddle._C_ops.add_(add__20, slice_50)

        # builtin.slice: (-1x64x7x7xf16) <- ([-1x512x7x7xf16, -1x64x7x7xf16])
        slice_51 = split_25[1]

        # builtin.combine: ([-1x256x7x7xf16, -1x64x7x7xf16]) <- (-1x256x7x7xf16, -1x64x7x7xf16)
        combine_42 = [concat_40, slice_51]

        # pd_op.full: (1xi32) <- ()
        full_68 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x320x7x7xf16) <- ([-1x256x7x7xf16, -1x64x7x7xf16], 1xi32)
        concat_42 = paddle._C_ops.concat(combine_42, full_68)

        # builtin.combine: ([-1x512x7x7xf16, -1x320x7x7xf16]) <- (-1x512x7x7xf16, -1x320x7x7xf16)
        combine_43 = [add__21, concat_42]

        # pd_op.full: (1xi32) <- ()
        full_69 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x832x7x7xf16) <- ([-1x512x7x7xf16, -1x320x7x7xf16], 1xi32)
        concat_43 = paddle._C_ops.concat(combine_43, full_69)

        # pd_op.batch_norm_: (-1x832x7x7xf16, 832xf32, 832xf32, 832xf32, 832xf32, None) <- (-1x832x7x7xf16, 832xf32, 832xf32, 832xf32, 832xf32)
        batch_norm__426, batch_norm__427, batch_norm__428, batch_norm__429, batch_norm__430, batch_norm__431 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(concat_43, parameter_355, parameter_356, parameter_357, parameter_358, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu_: (-1x832x7x7xf16) <- (-1x832x7x7xf16)
        relu__71 = paddle._C_ops.relu_(batch_norm__426)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_27 = [1, 1]

        # pd_op.pool2d: (-1x832x1x1xf16) <- (-1x832x7x7xf16, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(relu__71, full_int_array_27, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.flatten_: (-1x832xf16, None) <- (-1x832x1x1xf16)
        flatten__0, flatten__1 = (lambda x, f: f(x))(paddle._C_ops.flatten_(pool2d_1, 1, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x1000xf16) <- (-1x832xf16, 832x1000xf16)
        matmul_0 = paddle._C_ops.matmul(flatten__0, parameter_359, False, False)

        # pd_op.add_: (-1x1000xf16) <- (-1x1000xf16, 1000xf16)
        add__22 = paddle._C_ops.add_(matmul_0, parameter_360)

        # pd_op.softmax_: (-1x1000xf16) <- (-1x1000xf16)
        softmax__0 = paddle._C_ops.softmax_(add__22, -1)

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

    def forward(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_8, parameter_5, parameter_7, parameter_6, parameter_9, parameter_13, parameter_10, parameter_12, parameter_11, parameter_14, parameter_18, parameter_15, parameter_17, parameter_16, parameter_19, parameter_23, parameter_20, parameter_22, parameter_21, parameter_24, parameter_28, parameter_25, parameter_27, parameter_26, parameter_29, parameter_33, parameter_30, parameter_32, parameter_31, parameter_34, parameter_38, parameter_35, parameter_37, parameter_36, parameter_39, parameter_43, parameter_40, parameter_42, parameter_41, parameter_44, parameter_48, parameter_45, parameter_47, parameter_46, parameter_49, parameter_53, parameter_50, parameter_52, parameter_51, parameter_54, parameter_58, parameter_55, parameter_57, parameter_56, parameter_59, parameter_63, parameter_60, parameter_62, parameter_61, parameter_64, parameter_68, parameter_65, parameter_67, parameter_66, parameter_69, parameter_73, parameter_70, parameter_72, parameter_71, parameter_74, parameter_78, parameter_75, parameter_77, parameter_76, parameter_79, parameter_83, parameter_80, parameter_82, parameter_81, parameter_84, parameter_88, parameter_85, parameter_87, parameter_86, parameter_89, parameter_93, parameter_90, parameter_92, parameter_91, parameter_94, parameter_98, parameter_95, parameter_97, parameter_96, parameter_99, parameter_103, parameter_100, parameter_102, parameter_101, parameter_104, parameter_108, parameter_105, parameter_107, parameter_106, parameter_109, parameter_113, parameter_110, parameter_112, parameter_111, parameter_114, parameter_118, parameter_115, parameter_117, parameter_116, parameter_119, parameter_123, parameter_120, parameter_122, parameter_121, parameter_124, parameter_128, parameter_125, parameter_127, parameter_126, parameter_129, parameter_133, parameter_130, parameter_132, parameter_131, parameter_134, parameter_138, parameter_135, parameter_137, parameter_136, parameter_139, parameter_143, parameter_140, parameter_142, parameter_141, parameter_144, parameter_148, parameter_145, parameter_147, parameter_146, parameter_149, parameter_153, parameter_150, parameter_152, parameter_151, parameter_154, parameter_158, parameter_155, parameter_157, parameter_156, parameter_159, parameter_163, parameter_160, parameter_162, parameter_161, parameter_164, parameter_168, parameter_165, parameter_167, parameter_166, parameter_169, parameter_173, parameter_170, parameter_172, parameter_171, parameter_174, parameter_178, parameter_175, parameter_177, parameter_176, parameter_179, parameter_183, parameter_180, parameter_182, parameter_181, parameter_184, parameter_188, parameter_185, parameter_187, parameter_186, parameter_189, parameter_193, parameter_190, parameter_192, parameter_191, parameter_194, parameter_198, parameter_195, parameter_197, parameter_196, parameter_199, parameter_203, parameter_200, parameter_202, parameter_201, parameter_204, parameter_208, parameter_205, parameter_207, parameter_206, parameter_209, parameter_213, parameter_210, parameter_212, parameter_211, parameter_214, parameter_218, parameter_215, parameter_217, parameter_216, parameter_219, parameter_223, parameter_220, parameter_222, parameter_221, parameter_224, parameter_228, parameter_225, parameter_227, parameter_226, parameter_229, parameter_233, parameter_230, parameter_232, parameter_231, parameter_234, parameter_238, parameter_235, parameter_237, parameter_236, parameter_239, parameter_243, parameter_240, parameter_242, parameter_241, parameter_244, parameter_248, parameter_245, parameter_247, parameter_246, parameter_249, parameter_253, parameter_250, parameter_252, parameter_251, parameter_254, parameter_258, parameter_255, parameter_257, parameter_256, parameter_259, parameter_263, parameter_260, parameter_262, parameter_261, parameter_264, parameter_268, parameter_265, parameter_267, parameter_266, parameter_269, parameter_273, parameter_270, parameter_272, parameter_271, parameter_274, parameter_278, parameter_275, parameter_277, parameter_276, parameter_279, parameter_283, parameter_280, parameter_282, parameter_281, parameter_284, parameter_288, parameter_285, parameter_287, parameter_286, parameter_289, parameter_293, parameter_290, parameter_292, parameter_291, parameter_294, parameter_298, parameter_295, parameter_297, parameter_296, parameter_299, parameter_303, parameter_300, parameter_302, parameter_301, parameter_304, parameter_308, parameter_305, parameter_307, parameter_306, parameter_309, parameter_313, parameter_310, parameter_312, parameter_311, parameter_314, parameter_318, parameter_315, parameter_317, parameter_316, parameter_319, parameter_323, parameter_320, parameter_322, parameter_321, parameter_324, parameter_328, parameter_325, parameter_327, parameter_326, parameter_329, parameter_333, parameter_330, parameter_332, parameter_331, parameter_334, parameter_338, parameter_335, parameter_337, parameter_336, parameter_339, parameter_343, parameter_340, parameter_342, parameter_341, parameter_344, parameter_348, parameter_345, parameter_347, parameter_346, parameter_349, parameter_353, parameter_350, parameter_352, parameter_351, parameter_354, parameter_358, parameter_355, parameter_357, parameter_356, parameter_359, parameter_360, feed_0):
        return self.builtin_module_877_0_0(parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_8, parameter_5, parameter_7, parameter_6, parameter_9, parameter_13, parameter_10, parameter_12, parameter_11, parameter_14, parameter_18, parameter_15, parameter_17, parameter_16, parameter_19, parameter_23, parameter_20, parameter_22, parameter_21, parameter_24, parameter_28, parameter_25, parameter_27, parameter_26, parameter_29, parameter_33, parameter_30, parameter_32, parameter_31, parameter_34, parameter_38, parameter_35, parameter_37, parameter_36, parameter_39, parameter_43, parameter_40, parameter_42, parameter_41, parameter_44, parameter_48, parameter_45, parameter_47, parameter_46, parameter_49, parameter_53, parameter_50, parameter_52, parameter_51, parameter_54, parameter_58, parameter_55, parameter_57, parameter_56, parameter_59, parameter_63, parameter_60, parameter_62, parameter_61, parameter_64, parameter_68, parameter_65, parameter_67, parameter_66, parameter_69, parameter_73, parameter_70, parameter_72, parameter_71, parameter_74, parameter_78, parameter_75, parameter_77, parameter_76, parameter_79, parameter_83, parameter_80, parameter_82, parameter_81, parameter_84, parameter_88, parameter_85, parameter_87, parameter_86, parameter_89, parameter_93, parameter_90, parameter_92, parameter_91, parameter_94, parameter_98, parameter_95, parameter_97, parameter_96, parameter_99, parameter_103, parameter_100, parameter_102, parameter_101, parameter_104, parameter_108, parameter_105, parameter_107, parameter_106, parameter_109, parameter_113, parameter_110, parameter_112, parameter_111, parameter_114, parameter_118, parameter_115, parameter_117, parameter_116, parameter_119, parameter_123, parameter_120, parameter_122, parameter_121, parameter_124, parameter_128, parameter_125, parameter_127, parameter_126, parameter_129, parameter_133, parameter_130, parameter_132, parameter_131, parameter_134, parameter_138, parameter_135, parameter_137, parameter_136, parameter_139, parameter_143, parameter_140, parameter_142, parameter_141, parameter_144, parameter_148, parameter_145, parameter_147, parameter_146, parameter_149, parameter_153, parameter_150, parameter_152, parameter_151, parameter_154, parameter_158, parameter_155, parameter_157, parameter_156, parameter_159, parameter_163, parameter_160, parameter_162, parameter_161, parameter_164, parameter_168, parameter_165, parameter_167, parameter_166, parameter_169, parameter_173, parameter_170, parameter_172, parameter_171, parameter_174, parameter_178, parameter_175, parameter_177, parameter_176, parameter_179, parameter_183, parameter_180, parameter_182, parameter_181, parameter_184, parameter_188, parameter_185, parameter_187, parameter_186, parameter_189, parameter_193, parameter_190, parameter_192, parameter_191, parameter_194, parameter_198, parameter_195, parameter_197, parameter_196, parameter_199, parameter_203, parameter_200, parameter_202, parameter_201, parameter_204, parameter_208, parameter_205, parameter_207, parameter_206, parameter_209, parameter_213, parameter_210, parameter_212, parameter_211, parameter_214, parameter_218, parameter_215, parameter_217, parameter_216, parameter_219, parameter_223, parameter_220, parameter_222, parameter_221, parameter_224, parameter_228, parameter_225, parameter_227, parameter_226, parameter_229, parameter_233, parameter_230, parameter_232, parameter_231, parameter_234, parameter_238, parameter_235, parameter_237, parameter_236, parameter_239, parameter_243, parameter_240, parameter_242, parameter_241, parameter_244, parameter_248, parameter_245, parameter_247, parameter_246, parameter_249, parameter_253, parameter_250, parameter_252, parameter_251, parameter_254, parameter_258, parameter_255, parameter_257, parameter_256, parameter_259, parameter_263, parameter_260, parameter_262, parameter_261, parameter_264, parameter_268, parameter_265, parameter_267, parameter_266, parameter_269, parameter_273, parameter_270, parameter_272, parameter_271, parameter_274, parameter_278, parameter_275, parameter_277, parameter_276, parameter_279, parameter_283, parameter_280, parameter_282, parameter_281, parameter_284, parameter_288, parameter_285, parameter_287, parameter_286, parameter_289, parameter_293, parameter_290, parameter_292, parameter_291, parameter_294, parameter_298, parameter_295, parameter_297, parameter_296, parameter_299, parameter_303, parameter_300, parameter_302, parameter_301, parameter_304, parameter_308, parameter_305, parameter_307, parameter_306, parameter_309, parameter_313, parameter_310, parameter_312, parameter_311, parameter_314, parameter_318, parameter_315, parameter_317, parameter_316, parameter_319, parameter_323, parameter_320, parameter_322, parameter_321, parameter_324, parameter_328, parameter_325, parameter_327, parameter_326, parameter_329, parameter_333, parameter_330, parameter_332, parameter_331, parameter_334, parameter_338, parameter_335, parameter_337, parameter_336, parameter_339, parameter_343, parameter_340, parameter_342, parameter_341, parameter_344, parameter_348, parameter_345, parameter_347, parameter_346, parameter_349, parameter_353, parameter_350, parameter_352, parameter_351, parameter_354, parameter_358, parameter_355, parameter_357, parameter_356, parameter_359, parameter_360, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_877_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([10, 3, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_4
            paddle.uniform([10], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([10], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([10], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([10], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([10], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([10], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([10], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([10], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([96, 10, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_13
            paddle.uniform([10], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([10], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([10], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([10], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([128, 10, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_18
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_19
            paddle.uniform([128, 4, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_23
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_24
            paddle.uniform([80, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_28
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([112], dtype='float32', min=0, max=0.5),
            # parameter_29
            paddle.uniform([128, 112, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_33
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([128, 4, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_38
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([80, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_43
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([128, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_48
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([128, 4, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_53
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([80, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_58
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([192, 144, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_63
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([256, 144, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_68
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_67
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([256, 8, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_73
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_72
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_74
            paddle.uniform([160, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_78
            paddle.uniform([224], dtype='float32', min=0, max=0.5),
            # parameter_75
            paddle.uniform([224], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([224], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([224], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([256, 224, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_83
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([256, 8, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_88
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_86
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([160, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_93
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_91
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([256, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_98
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([256, 8, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_103
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_104
            paddle.uniform([160, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_108
            paddle.uniform([288], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([288], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([288], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([288], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([256, 288, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_113
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([256, 8, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_118
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([160, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_123
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([320, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_128
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([512, 320, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_133
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_132
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_134
            paddle.uniform([512, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_138
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_135
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_139
            paddle.uniform([288, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_143
            paddle.uniform([352], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([352], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([352], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([352], dtype='float32', min=0, max=0.5),
            # parameter_144
            paddle.uniform([512, 352, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_148
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_149
            paddle.uniform([512, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_153
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_152
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_151
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_154
            paddle.uniform([288, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_158
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.uniform([512, 384, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_163
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([512, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_168
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([288, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_173
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([416], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([512, 416, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_178
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([512, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_183
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.uniform([288, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_188
            paddle.uniform([448], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([448], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([448], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([448], dtype='float32', min=0, max=0.5),
            # parameter_189
            paddle.uniform([512, 448, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_193
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.uniform([512, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_198
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([288, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_203
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([480], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([512, 480, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_208
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_209
            paddle.uniform([512, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_213
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_214
            paddle.uniform([288, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_218
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_215
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_217
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_219
            paddle.uniform([512, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_223
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_220
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_221
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_224
            paddle.uniform([512, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_228
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_225
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_227
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_229
            paddle.uniform([288, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_233
            paddle.uniform([544], dtype='float32', min=0, max=0.5),
            # parameter_230
            paddle.uniform([544], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([544], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([544], dtype='float32', min=0, max=0.5),
            # parameter_234
            paddle.uniform([512, 544, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_238
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_239
            paddle.uniform([512, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_243
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_241
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_244
            paddle.uniform([288, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_248
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
            # parameter_245
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
            # parameter_246
            paddle.uniform([576], dtype='float32', min=0, max=0.5),
            # parameter_249
            paddle.uniform([512, 576, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_253
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_250
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_252
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_251
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_254
            paddle.uniform([512, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_258
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_255
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_257
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_256
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_259
            paddle.uniform([288, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_263
            paddle.uniform([608], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([608], dtype='float32', min=0, max=0.5),
            # parameter_262
            paddle.uniform([608], dtype='float32', min=0, max=0.5),
            # parameter_261
            paddle.uniform([608], dtype='float32', min=0, max=0.5),
            # parameter_264
            paddle.uniform([512, 608, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_268
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_265
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_267
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_266
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_269
            paddle.uniform([512, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_273
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_270
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_272
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_271
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_274
            paddle.uniform([288, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_278
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_275
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_277
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_276
            paddle.uniform([640], dtype='float32', min=0, max=0.5),
            # parameter_279
            paddle.uniform([512, 640, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_283
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_280
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_282
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_281
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_284
            paddle.uniform([512, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_288
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_285
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_287
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_286
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_289
            paddle.uniform([288, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_293
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_290
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_292
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_291
            paddle.uniform([672], dtype='float32', min=0, max=0.5),
            # parameter_294
            paddle.uniform([512, 672, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_298
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_295
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_297
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_296
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_299
            paddle.uniform([512, 16, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_303
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_300
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_302
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_301
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_304
            paddle.uniform([288, 512, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_308
            paddle.uniform([704], dtype='float32', min=0, max=0.5),
            # parameter_305
            paddle.uniform([704], dtype='float32', min=0, max=0.5),
            # parameter_307
            paddle.uniform([704], dtype='float32', min=0, max=0.5),
            # parameter_306
            paddle.uniform([704], dtype='float32', min=0, max=0.5),
            # parameter_309
            paddle.uniform([640, 704, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_313
            paddle.uniform([704], dtype='float32', min=0, max=0.5),
            # parameter_310
            paddle.uniform([704], dtype='float32', min=0, max=0.5),
            # parameter_312
            paddle.uniform([704], dtype='float32', min=0, max=0.5),
            # parameter_311
            paddle.uniform([704], dtype='float32', min=0, max=0.5),
            # parameter_314
            paddle.uniform([1024, 704, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_318
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_315
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_317
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_316
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_319
            paddle.uniform([1024, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_323
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_320
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_322
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_321
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_324
            paddle.uniform([576, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_328
            paddle.uniform([704], dtype='float32', min=0, max=0.5),
            # parameter_325
            paddle.uniform([704], dtype='float32', min=0, max=0.5),
            # parameter_327
            paddle.uniform([704], dtype='float32', min=0, max=0.5),
            # parameter_326
            paddle.uniform([704], dtype='float32', min=0, max=0.5),
            # parameter_329
            paddle.uniform([1024, 704, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_333
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_330
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_332
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_331
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_334
            paddle.uniform([1024, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_338
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_335
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_337
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_336
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_339
            paddle.uniform([576, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_343
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_340
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_342
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_341
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            # parameter_344
            paddle.uniform([1024, 768, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_348
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_345
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_347
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_346
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_349
            paddle.uniform([1024, 32, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_353
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_350
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_352
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_351
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_354
            paddle.uniform([576, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_358
            paddle.uniform([832], dtype='float32', min=0, max=0.5),
            # parameter_355
            paddle.uniform([832], dtype='float32', min=0, max=0.5),
            # parameter_357
            paddle.uniform([832], dtype='float32', min=0, max=0.5),
            # parameter_356
            paddle.uniform([832], dtype='float32', min=0, max=0.5),
            # parameter_359
            paddle.uniform([832, 1000], dtype='float16', min=0, max=0.5),
            # parameter_360
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
            paddle.static.InputSpec(shape=[10, 3, 3, 3], dtype='float16'),
            # parameter_4
            paddle.static.InputSpec(shape=[10], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[10], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[10], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[10], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[10], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[10], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[10], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[10], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[96, 10, 1, 1], dtype='float16'),
            # parameter_13
            paddle.static.InputSpec(shape=[10], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[10], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[10], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[10], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[128, 10, 1, 1], dtype='float16'),
            # parameter_18
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_19
            paddle.static.InputSpec(shape=[128, 4, 3, 3], dtype='float16'),
            # parameter_23
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_24
            paddle.static.InputSpec(shape=[80, 128, 1, 1], dtype='float16'),
            # parameter_28
            paddle.static.InputSpec(shape=[112], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[112], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[112], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[112], dtype='float32'),
            # parameter_29
            paddle.static.InputSpec(shape=[128, 112, 1, 1], dtype='float16'),
            # parameter_33
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[128, 4, 3, 3], dtype='float16'),
            # parameter_38
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[80, 128, 1, 1], dtype='float16'),
            # parameter_43
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float16'),
            # parameter_48
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[128, 4, 3, 3], dtype='float16'),
            # parameter_53
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[80, 128, 1, 1], dtype='float16'),
            # parameter_58
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[192, 144, 1, 1], dtype='float16'),
            # parameter_63
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[256, 144, 1, 1], dtype='float16'),
            # parameter_68
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_67
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[256, 8, 3, 3], dtype='float16'),
            # parameter_73
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_72
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_74
            paddle.static.InputSpec(shape=[160, 256, 1, 1], dtype='float16'),
            # parameter_78
            paddle.static.InputSpec(shape=[224], dtype='float32'),
            # parameter_75
            paddle.static.InputSpec(shape=[224], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[224], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[224], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[256, 224, 1, 1], dtype='float16'),
            # parameter_83
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[256, 8, 3, 3], dtype='float16'),
            # parameter_88
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_86
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[160, 256, 1, 1], dtype='float16'),
            # parameter_93
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_91
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float16'),
            # parameter_98
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[256, 8, 3, 3], dtype='float16'),
            # parameter_103
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_104
            paddle.static.InputSpec(shape=[160, 256, 1, 1], dtype='float16'),
            # parameter_108
            paddle.static.InputSpec(shape=[288], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[288], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[288], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[288], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[256, 288, 1, 1], dtype='float16'),
            # parameter_113
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[256, 8, 3, 3], dtype='float16'),
            # parameter_118
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[160, 256, 1, 1], dtype='float16'),
            # parameter_123
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[320, 320, 1, 1], dtype='float16'),
            # parameter_128
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[512, 320, 1, 1], dtype='float16'),
            # parameter_133
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_132
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_134
            paddle.static.InputSpec(shape=[512, 16, 3, 3], dtype='float16'),
            # parameter_138
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_135
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_139
            paddle.static.InputSpec(shape=[288, 512, 1, 1], dtype='float16'),
            # parameter_143
            paddle.static.InputSpec(shape=[352], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[352], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[352], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[352], dtype='float32'),
            # parameter_144
            paddle.static.InputSpec(shape=[512, 352, 1, 1], dtype='float16'),
            # parameter_148
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_149
            paddle.static.InputSpec(shape=[512, 16, 3, 3], dtype='float16'),
            # parameter_153
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_152
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_151
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_154
            paddle.static.InputSpec(shape=[288, 512, 1, 1], dtype='float16'),
            # parameter_158
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[512, 384, 1, 1], dtype='float16'),
            # parameter_163
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[512, 16, 3, 3], dtype='float16'),
            # parameter_168
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[288, 512, 1, 1], dtype='float16'),
            # parameter_173
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[416], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[512, 416, 1, 1], dtype='float16'),
            # parameter_178
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[512, 16, 3, 3], dtype='float16'),
            # parameter_183
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[288, 512, 1, 1], dtype='float16'),
            # parameter_188
            paddle.static.InputSpec(shape=[448], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[448], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[448], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[448], dtype='float32'),
            # parameter_189
            paddle.static.InputSpec(shape=[512, 448, 1, 1], dtype='float16'),
            # parameter_193
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[512, 16, 3, 3], dtype='float16'),
            # parameter_198
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[288, 512, 1, 1], dtype='float16'),
            # parameter_203
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[480], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[512, 480, 1, 1], dtype='float16'),
            # parameter_208
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_209
            paddle.static.InputSpec(shape=[512, 16, 3, 3], dtype='float16'),
            # parameter_213
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_214
            paddle.static.InputSpec(shape=[288, 512, 1, 1], dtype='float16'),
            # parameter_218
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_215
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_217
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_219
            paddle.static.InputSpec(shape=[512, 512, 1, 1], dtype='float16'),
            # parameter_223
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_220
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_221
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_224
            paddle.static.InputSpec(shape=[512, 16, 3, 3], dtype='float16'),
            # parameter_228
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_225
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_227
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_229
            paddle.static.InputSpec(shape=[288, 512, 1, 1], dtype='float16'),
            # parameter_233
            paddle.static.InputSpec(shape=[544], dtype='float32'),
            # parameter_230
            paddle.static.InputSpec(shape=[544], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[544], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[544], dtype='float32'),
            # parameter_234
            paddle.static.InputSpec(shape=[512, 544, 1, 1], dtype='float16'),
            # parameter_238
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_239
            paddle.static.InputSpec(shape=[512, 16, 3, 3], dtype='float16'),
            # parameter_243
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_241
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_244
            paddle.static.InputSpec(shape=[288, 512, 1, 1], dtype='float16'),
            # parameter_248
            paddle.static.InputSpec(shape=[576], dtype='float32'),
            # parameter_245
            paddle.static.InputSpec(shape=[576], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[576], dtype='float32'),
            # parameter_246
            paddle.static.InputSpec(shape=[576], dtype='float32'),
            # parameter_249
            paddle.static.InputSpec(shape=[512, 576, 1, 1], dtype='float16'),
            # parameter_253
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_250
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_252
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_251
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_254
            paddle.static.InputSpec(shape=[512, 16, 3, 3], dtype='float16'),
            # parameter_258
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_255
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_257
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_256
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_259
            paddle.static.InputSpec(shape=[288, 512, 1, 1], dtype='float16'),
            # parameter_263
            paddle.static.InputSpec(shape=[608], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[608], dtype='float32'),
            # parameter_262
            paddle.static.InputSpec(shape=[608], dtype='float32'),
            # parameter_261
            paddle.static.InputSpec(shape=[608], dtype='float32'),
            # parameter_264
            paddle.static.InputSpec(shape=[512, 608, 1, 1], dtype='float16'),
            # parameter_268
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_265
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_267
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_266
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_269
            paddle.static.InputSpec(shape=[512, 16, 3, 3], dtype='float16'),
            # parameter_273
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_270
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_272
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_271
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_274
            paddle.static.InputSpec(shape=[288, 512, 1, 1], dtype='float16'),
            # parameter_278
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_275
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_277
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_276
            paddle.static.InputSpec(shape=[640], dtype='float32'),
            # parameter_279
            paddle.static.InputSpec(shape=[512, 640, 1, 1], dtype='float16'),
            # parameter_283
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_280
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_282
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_281
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_284
            paddle.static.InputSpec(shape=[512, 16, 3, 3], dtype='float16'),
            # parameter_288
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_285
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_287
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_286
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_289
            paddle.static.InputSpec(shape=[288, 512, 1, 1], dtype='float16'),
            # parameter_293
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_290
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_292
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_291
            paddle.static.InputSpec(shape=[672], dtype='float32'),
            # parameter_294
            paddle.static.InputSpec(shape=[512, 672, 1, 1], dtype='float16'),
            # parameter_298
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_295
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_297
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_296
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_299
            paddle.static.InputSpec(shape=[512, 16, 3, 3], dtype='float16'),
            # parameter_303
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_300
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_302
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_301
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_304
            paddle.static.InputSpec(shape=[288, 512, 1, 1], dtype='float16'),
            # parameter_308
            paddle.static.InputSpec(shape=[704], dtype='float32'),
            # parameter_305
            paddle.static.InputSpec(shape=[704], dtype='float32'),
            # parameter_307
            paddle.static.InputSpec(shape=[704], dtype='float32'),
            # parameter_306
            paddle.static.InputSpec(shape=[704], dtype='float32'),
            # parameter_309
            paddle.static.InputSpec(shape=[640, 704, 1, 1], dtype='float16'),
            # parameter_313
            paddle.static.InputSpec(shape=[704], dtype='float32'),
            # parameter_310
            paddle.static.InputSpec(shape=[704], dtype='float32'),
            # parameter_312
            paddle.static.InputSpec(shape=[704], dtype='float32'),
            # parameter_311
            paddle.static.InputSpec(shape=[704], dtype='float32'),
            # parameter_314
            paddle.static.InputSpec(shape=[1024, 704, 1, 1], dtype='float16'),
            # parameter_318
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_315
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_317
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_316
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_319
            paddle.static.InputSpec(shape=[1024, 32, 3, 3], dtype='float16'),
            # parameter_323
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_320
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_322
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_321
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_324
            paddle.static.InputSpec(shape=[576, 1024, 1, 1], dtype='float16'),
            # parameter_328
            paddle.static.InputSpec(shape=[704], dtype='float32'),
            # parameter_325
            paddle.static.InputSpec(shape=[704], dtype='float32'),
            # parameter_327
            paddle.static.InputSpec(shape=[704], dtype='float32'),
            # parameter_326
            paddle.static.InputSpec(shape=[704], dtype='float32'),
            # parameter_329
            paddle.static.InputSpec(shape=[1024, 704, 1, 1], dtype='float16'),
            # parameter_333
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_330
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_332
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_331
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_334
            paddle.static.InputSpec(shape=[1024, 32, 3, 3], dtype='float16'),
            # parameter_338
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_335
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_337
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_336
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_339
            paddle.static.InputSpec(shape=[576, 1024, 1, 1], dtype='float16'),
            # parameter_343
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_340
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_342
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_341
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            # parameter_344
            paddle.static.InputSpec(shape=[1024, 768, 1, 1], dtype='float16'),
            # parameter_348
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_345
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_347
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_346
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_349
            paddle.static.InputSpec(shape=[1024, 32, 3, 3], dtype='float16'),
            # parameter_353
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_350
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_352
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_351
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_354
            paddle.static.InputSpec(shape=[576, 1024, 1, 1], dtype='float16'),
            # parameter_358
            paddle.static.InputSpec(shape=[832], dtype='float32'),
            # parameter_355
            paddle.static.InputSpec(shape=[832], dtype='float32'),
            # parameter_357
            paddle.static.InputSpec(shape=[832], dtype='float32'),
            # parameter_356
            paddle.static.InputSpec(shape=[832], dtype='float32'),
            # parameter_359
            paddle.static.InputSpec(shape=[832, 1000], dtype='float16'),
            # parameter_360
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