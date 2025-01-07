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
    return [605][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_932_0_0(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_26, parameter_27, parameter_28, parameter_29, parameter_33, parameter_30, parameter_32, parameter_31, parameter_34, parameter_38, parameter_35, parameter_37, parameter_36, parameter_39, parameter_43, parameter_40, parameter_42, parameter_41, parameter_44, parameter_48, parameter_45, parameter_47, parameter_46, parameter_49, parameter_53, parameter_50, parameter_52, parameter_51, parameter_54, parameter_55, parameter_56, parameter_57, parameter_58, parameter_62, parameter_59, parameter_61, parameter_60, parameter_63, parameter_67, parameter_64, parameter_66, parameter_65, parameter_68, parameter_72, parameter_69, parameter_71, parameter_70, parameter_73, parameter_74, parameter_75, parameter_76, parameter_77, parameter_81, parameter_78, parameter_80, parameter_79, parameter_82, parameter_86, parameter_83, parameter_85, parameter_84, parameter_87, parameter_91, parameter_88, parameter_90, parameter_89, parameter_92, parameter_96, parameter_93, parameter_95, parameter_94, parameter_97, parameter_101, parameter_98, parameter_100, parameter_99, parameter_102, parameter_103, parameter_104, parameter_105, parameter_106, parameter_110, parameter_107, parameter_109, parameter_108, parameter_111, parameter_115, parameter_112, parameter_114, parameter_113, parameter_116, parameter_120, parameter_117, parameter_119, parameter_118, parameter_121, parameter_125, parameter_122, parameter_124, parameter_123, parameter_126, parameter_130, parameter_127, parameter_129, parameter_128, parameter_131, parameter_132, parameter_133, parameter_134, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_151, parameter_152, parameter_153, parameter_154, parameter_158, parameter_155, parameter_157, parameter_156, parameter_159, parameter_163, parameter_160, parameter_162, parameter_161, parameter_164, parameter_168, parameter_165, parameter_167, parameter_166, parameter_169, parameter_170, parameter_171, parameter_172, parameter_173, parameter_177, parameter_174, parameter_176, parameter_175, parameter_178, parameter_182, parameter_179, parameter_181, parameter_180, parameter_183, parameter_187, parameter_184, parameter_186, parameter_185, parameter_188, parameter_189, parameter_190, parameter_191, parameter_192, parameter_196, parameter_193, parameter_195, parameter_194, parameter_197, parameter_201, parameter_198, parameter_200, parameter_199, parameter_202, parameter_206, parameter_203, parameter_205, parameter_204, parameter_207, parameter_208, parameter_209, parameter_210, parameter_211, parameter_215, parameter_212, parameter_214, parameter_213, parameter_216, parameter_220, parameter_217, parameter_219, parameter_218, parameter_221, parameter_225, parameter_222, parameter_224, parameter_223, parameter_226, parameter_227, parameter_228, parameter_229, parameter_230, parameter_234, parameter_231, parameter_233, parameter_232, parameter_235, parameter_239, parameter_236, parameter_238, parameter_237, parameter_240, parameter_244, parameter_241, parameter_243, parameter_242, parameter_245, parameter_249, parameter_246, parameter_248, parameter_247, parameter_250, parameter_254, parameter_251, parameter_253, parameter_252, parameter_255, parameter_256, parameter_257, parameter_258, parameter_259, parameter_263, parameter_260, parameter_262, parameter_261, parameter_264, parameter_268, parameter_265, parameter_267, parameter_266, parameter_269, parameter_273, parameter_270, parameter_272, parameter_271, parameter_274, parameter_278, parameter_275, parameter_277, parameter_276, parameter_279, parameter_283, parameter_280, parameter_282, parameter_281, parameter_284, parameter_285, parameter_286, parameter_287, parameter_288, parameter_292, parameter_289, parameter_291, parameter_290, parameter_293, parameter_297, parameter_294, parameter_296, parameter_295, parameter_298, parameter_302, parameter_299, parameter_301, parameter_300, parameter_303, parameter_304, parameter_305, parameter_306, parameter_307, parameter_311, parameter_308, parameter_310, parameter_309, parameter_312, parameter_316, parameter_313, parameter_315, parameter_314, parameter_317, parameter_318, parameter_319, feed_0):

        # pd_op.cast: (-1x3x224x224xf16) <- (-1x3x224x224xf32)
        cast_0 = paddle._C_ops.cast(feed_0, paddle.float16)

        # pd_op.conv2d: (-1x24x112x112xf16) <- (-1x3x224x224xf16, 24x3x3x3xf16)
        conv2d_0 = paddle._C_ops.conv2d(cast_0, parameter_0, [2, 2], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x24x112x112xf16, 24xf32, 24xf32, 24xf32, 24xf32, None) <- (-1x24x112x112xf16, 24xf32, 24xf32, 24xf32, 24xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_0, parameter_1, parameter_2, parameter_3, parameter_4, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x24x112x112xf16) <- (-1x24x112x112xf16)
        hardswish_0 = paddle._C_ops.hardswish(batch_norm__0)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [3, 3]

        # pd_op.pool2d: (-1x24x56x56xf16) <- (-1x24x112x112xf16, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(hardswish_0, full_int_array_0, [2, 2], [1, 1], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.depthwise_conv2d: (-1x24x28x28xf16) <- (-1x24x56x56xf16, 24x1x3x3xf16)
        depthwise_conv2d_0 = paddle._C_ops.depthwise_conv2d(pool2d_0, parameter_5, [2, 2], [1, 1], 'EXPLICIT', 24, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x24x28x28xf16, 24xf32, 24xf32, 24xf32, 24xf32, None) <- (-1x24x28x28xf16, 24xf32, 24xf32, 24xf32, 24xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_0, parameter_6, parameter_7, parameter_8, parameter_9, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x16x28x28xf16) <- (-1x24x28x28xf16, 16x24x1x1xf16)
        conv2d_1 = paddle._C_ops.conv2d(batch_norm__6, parameter_10, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x16x28x28xf16, 16xf32, 16xf32, 16xf32, 16xf32, None) <- (-1x16x28x28xf16, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_1, parameter_11, parameter_12, parameter_13, parameter_14, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x16x28x28xf16) <- (-1x16x28x28xf16)
        hardswish_1 = paddle._C_ops.hardswish(batch_norm__12)

        # pd_op.conv2d: (-1x16x56x56xf16) <- (-1x24x56x56xf16, 16x24x1x1xf16)
        conv2d_2 = paddle._C_ops.conv2d(pool2d_0, parameter_15, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x16x56x56xf16, 16xf32, 16xf32, 16xf32, 16xf32, None) <- (-1x16x56x56xf16, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_2, parameter_16, parameter_17, parameter_18, parameter_19, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x16x56x56xf16) <- (-1x16x56x56xf16)
        hardswish_2 = paddle._C_ops.hardswish(batch_norm__18)

        # pd_op.depthwise_conv2d: (-1x16x28x28xf16) <- (-1x16x56x56xf16, 16x1x3x3xf16)
        depthwise_conv2d_1 = paddle._C_ops.depthwise_conv2d(hardswish_2, parameter_20, [2, 2], [1, 1], 'EXPLICIT', 16, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x16x28x28xf16, 16xf32, 16xf32, 16xf32, 16xf32, None) <- (-1x16x28x28xf16, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_1, parameter_21, parameter_22, parameter_23, parameter_24, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [1, 1]

        # pd_op.pool2d: (-1x16x1x1xf16) <- (-1x16x28x28xf16, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(batch_norm__24, full_int_array_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x4x1x1xf16) <- (-1x16x1x1xf16, 4x16x1x1xf16)
        conv2d_3 = paddle._C_ops.conv2d(pool2d_1, parameter_25, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_2 = [1, 4, 1, 1]

        # pd_op.reshape: (1x4x1x1xf16, 0x4xf16) <- (4xf16, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_26, full_int_array_2), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x4x1x1xf16) <- (-1x4x1x1xf16, 1x4x1x1xf16)
        add__0 = paddle._C_ops.add_(conv2d_3, reshape_0)

        # pd_op.relu_: (-1x4x1x1xf16) <- (-1x4x1x1xf16)
        relu__0 = paddle._C_ops.relu_(add__0)

        # pd_op.conv2d: (-1x16x1x1xf16) <- (-1x4x1x1xf16, 16x4x1x1xf16)
        conv2d_4 = paddle._C_ops.conv2d(relu__0, parameter_27, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_3 = [1, 16, 1, 1]

        # pd_op.reshape: (1x16x1x1xf16, 0x16xf16) <- (16xf16, 4xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_28, full_int_array_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x16x1x1xf16) <- (-1x16x1x1xf16, 1x16x1x1xf16)
        add__1 = paddle._C_ops.add_(conv2d_4, reshape_2)

        # pd_op.hardsigmoid: (-1x16x1x1xf16) <- (-1x16x1x1xf16)
        hardsigmoid_0 = paddle._C_ops.hardsigmoid(add__1, float('0.166667'), float('0.5'))

        # pd_op.multiply_: (-1x16x28x28xf16) <- (-1x16x28x28xf16, -1x16x1x1xf16)
        multiply__0 = paddle._C_ops.multiply_(batch_norm__24, hardsigmoid_0)

        # pd_op.conv2d: (-1x16x28x28xf16) <- (-1x16x28x28xf16, 16x16x1x1xf16)
        conv2d_5 = paddle._C_ops.conv2d(multiply__0, parameter_29, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x16x28x28xf16, 16xf32, 16xf32, 16xf32, 16xf32, None) <- (-1x16x28x28xf16, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_5, parameter_30, parameter_31, parameter_32, parameter_33, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x16x28x28xf16) <- (-1x16x28x28xf16)
        hardswish_3 = paddle._C_ops.hardswish(batch_norm__30)

        # builtin.combine: ([-1x16x28x28xf16, -1x16x28x28xf16]) <- (-1x16x28x28xf16, -1x16x28x28xf16)
        combine_0 = [hardswish_1, hardswish_3]

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x32x28x28xf16) <- ([-1x16x28x28xf16, -1x16x28x28xf16], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        # pd_op.depthwise_conv2d: (-1x32x28x28xf16) <- (-1x32x28x28xf16, 32x1x3x3xf16)
        depthwise_conv2d_2 = paddle._C_ops.depthwise_conv2d(concat_0, parameter_34, [1, 1], [1, 1], 'EXPLICIT', 32, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x32x28x28xf16, 32xf32, 32xf32, 32xf32, 32xf32, None) <- (-1x32x28x28xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_2, parameter_35, parameter_36, parameter_37, parameter_38, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x32x28x28xf16) <- (-1x32x28x28xf16)
        hardswish_4 = paddle._C_ops.hardswish(batch_norm__36)

        # pd_op.conv2d: (-1x32x28x28xf16) <- (-1x32x28x28xf16, 32x32x1x1xf16)
        conv2d_6 = paddle._C_ops.conv2d(hardswish_4, parameter_39, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x28x28xf16, 32xf32, 32xf32, 32xf32, 32xf32, None) <- (-1x32x28x28xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_6, parameter_40, parameter_41, parameter_42, parameter_43, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x32x28x28xf16) <- (-1x32x28x28xf16)
        hardswish_5 = paddle._C_ops.hardswish(batch_norm__42)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [16, 16]

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x16x28x28xf16, -1x16x28x28xf16]) <- (-1x32x28x28xf16, 2xi64, 1xi32)
        split_0 = paddle._C_ops.split(hardswish_5, full_int_array_4, full_1)

        # builtin.slice: (-1x16x28x28xf16) <- ([-1x16x28x28xf16, -1x16x28x28xf16])
        slice_0 = split_0[1]

        # pd_op.conv2d: (-1x16x28x28xf16) <- (-1x16x28x28xf16, 16x16x1x1xf16)
        conv2d_7 = paddle._C_ops.conv2d(slice_0, parameter_44, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x16x28x28xf16, 16xf32, 16xf32, 16xf32, 16xf32, None) <- (-1x16x28x28xf16, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_7, parameter_45, parameter_46, parameter_47, parameter_48, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x16x28x28xf16) <- (-1x16x28x28xf16)
        hardswish_6 = paddle._C_ops.hardswish(batch_norm__48)

        # pd_op.depthwise_conv2d: (-1x16x28x28xf16) <- (-1x16x28x28xf16, 16x1x3x3xf16)
        depthwise_conv2d_3 = paddle._C_ops.depthwise_conv2d(hardswish_6, parameter_49, [1, 1], [1, 1], 'EXPLICIT', 16, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x16x28x28xf16, 16xf32, 16xf32, 16xf32, 16xf32, None) <- (-1x16x28x28xf16, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_3, parameter_50, parameter_51, parameter_52, parameter_53, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # builtin.combine: ([-1x16x28x28xf16, -1x16x28x28xf16]) <- (-1x16x28x28xf16, -1x16x28x28xf16)
        combine_1 = [hardswish_6, batch_norm__54]

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x32x28x28xf16) <- ([-1x16x28x28xf16, -1x16x28x28xf16], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, full_2)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_5 = [1, 1]

        # pd_op.pool2d: (-1x32x1x1xf16) <- (-1x32x28x28xf16, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(concat_1, full_int_array_5, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x8x1x1xf16) <- (-1x32x1x1xf16, 8x32x1x1xf16)
        conv2d_8 = paddle._C_ops.conv2d(pool2d_2, parameter_54, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_6 = [1, 8, 1, 1]

        # pd_op.reshape: (1x8x1x1xf16, 0x8xf16) <- (8xf16, 4xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_55, full_int_array_6), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x8x1x1xf16) <- (-1x8x1x1xf16, 1x8x1x1xf16)
        add__2 = paddle._C_ops.add_(conv2d_8, reshape_4)

        # pd_op.relu_: (-1x8x1x1xf16) <- (-1x8x1x1xf16)
        relu__1 = paddle._C_ops.relu_(add__2)

        # pd_op.conv2d: (-1x32x1x1xf16) <- (-1x8x1x1xf16, 32x8x1x1xf16)
        conv2d_9 = paddle._C_ops.conv2d(relu__1, parameter_56, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_7 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf16, 0x32xf16) <- (32xf16, 4xi64)
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_57, full_int_array_7), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x32x1x1xf16) <- (-1x32x1x1xf16, 1x32x1x1xf16)
        add__3 = paddle._C_ops.add_(conv2d_9, reshape_6)

        # pd_op.hardsigmoid: (-1x32x1x1xf16) <- (-1x32x1x1xf16)
        hardsigmoid_1 = paddle._C_ops.hardsigmoid(add__3, float('0.166667'), float('0.5'))

        # pd_op.multiply_: (-1x32x28x28xf16) <- (-1x32x28x28xf16, -1x32x1x1xf16)
        multiply__1 = paddle._C_ops.multiply_(concat_1, hardsigmoid_1)

        # pd_op.conv2d: (-1x16x28x28xf16) <- (-1x32x28x28xf16, 16x32x1x1xf16)
        conv2d_10 = paddle._C_ops.conv2d(multiply__1, parameter_58, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x16x28x28xf16, 16xf32, 16xf32, 16xf32, 16xf32, None) <- (-1x16x28x28xf16, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_10, parameter_59, parameter_60, parameter_61, parameter_62, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x16x28x28xf16) <- (-1x16x28x28xf16)
        hardswish_7 = paddle._C_ops.hardswish(batch_norm__60)

        # builtin.slice: (-1x16x28x28xf16) <- ([-1x16x28x28xf16, -1x16x28x28xf16])
        slice_1 = split_0[0]

        # builtin.combine: ([-1x16x28x28xf16, -1x16x28x28xf16]) <- (-1x16x28x28xf16, -1x16x28x28xf16)
        combine_2 = [slice_1, hardswish_7]

        # pd_op.full: (1xi32) <- ()
        full_3 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x32x28x28xf16) <- ([-1x16x28x28xf16, -1x16x28x28xf16], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, full_3)

        # pd_op.shape: (4xi32) <- (-1x32x28x28xf16)
        shape_0 = paddle._C_ops.shape(paddle.cast(concat_2, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(shape_0, [0], full_int_array_8, full_int_array_9, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_4 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_5 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_7 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_3 = [slice_2, full_4, full_5, full_6, full_7]

        # pd_op.reshape_: (-1x2x16x28x28xf16, 0x-1x32x28x28xf16) <- (-1x32x28x28xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__0, reshape__1 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_2, [x.reshape([]) for x in combine_3]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x16x2x28x28xf16) <- (-1x2x16x28x28xf16)
        transpose_0 = paddle._C_ops.transpose(reshape__0, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_8 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_9 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_10 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_4 = [slice_2, full_8, full_9, full_10]

        # pd_op.reshape_: (-1x32x28x28xf16, 0x-1x16x2x28x28xf16) <- (-1x16x2x28x28xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__2, reshape__3 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_0, [x.reshape([]) for x in combine_4]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_10 = [16, 16]

        # pd_op.full: (1xi32) <- ()
        full_11 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x16x28x28xf16, -1x16x28x28xf16]) <- (-1x32x28x28xf16, 2xi64, 1xi32)
        split_1 = paddle._C_ops.split(reshape__2, full_int_array_10, full_11)

        # builtin.slice: (-1x16x28x28xf16) <- ([-1x16x28x28xf16, -1x16x28x28xf16])
        slice_3 = split_1[1]

        # pd_op.conv2d: (-1x16x28x28xf16) <- (-1x16x28x28xf16, 16x16x1x1xf16)
        conv2d_11 = paddle._C_ops.conv2d(slice_3, parameter_63, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x16x28x28xf16, 16xf32, 16xf32, 16xf32, 16xf32, None) <- (-1x16x28x28xf16, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_11, parameter_64, parameter_65, parameter_66, parameter_67, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x16x28x28xf16) <- (-1x16x28x28xf16)
        hardswish_8 = paddle._C_ops.hardswish(batch_norm__66)

        # pd_op.depthwise_conv2d: (-1x16x28x28xf16) <- (-1x16x28x28xf16, 16x1x3x3xf16)
        depthwise_conv2d_4 = paddle._C_ops.depthwise_conv2d(hardswish_8, parameter_68, [1, 1], [1, 1], 'EXPLICIT', 16, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x16x28x28xf16, 16xf32, 16xf32, 16xf32, 16xf32, None) <- (-1x16x28x28xf16, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_4, parameter_69, parameter_70, parameter_71, parameter_72, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # builtin.combine: ([-1x16x28x28xf16, -1x16x28x28xf16]) <- (-1x16x28x28xf16, -1x16x28x28xf16)
        combine_5 = [hardswish_8, batch_norm__72]

        # pd_op.full: (1xi32) <- ()
        full_12 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x32x28x28xf16) <- ([-1x16x28x28xf16, -1x16x28x28xf16], 1xi32)
        concat_3 = paddle._C_ops.concat(combine_5, full_12)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_11 = [1, 1]

        # pd_op.pool2d: (-1x32x1x1xf16) <- (-1x32x28x28xf16, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(concat_3, full_int_array_11, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x8x1x1xf16) <- (-1x32x1x1xf16, 8x32x1x1xf16)
        conv2d_12 = paddle._C_ops.conv2d(pool2d_3, parameter_73, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_12 = [1, 8, 1, 1]

        # pd_op.reshape: (1x8x1x1xf16, 0x8xf16) <- (8xf16, 4xi64)
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_74, full_int_array_12), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x8x1x1xf16) <- (-1x8x1x1xf16, 1x8x1x1xf16)
        add__4 = paddle._C_ops.add_(conv2d_12, reshape_8)

        # pd_op.relu_: (-1x8x1x1xf16) <- (-1x8x1x1xf16)
        relu__2 = paddle._C_ops.relu_(add__4)

        # pd_op.conv2d: (-1x32x1x1xf16) <- (-1x8x1x1xf16, 32x8x1x1xf16)
        conv2d_13 = paddle._C_ops.conv2d(relu__2, parameter_75, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_13 = [1, 32, 1, 1]

        # pd_op.reshape: (1x32x1x1xf16, 0x32xf16) <- (32xf16, 4xi64)
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_76, full_int_array_13), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x32x1x1xf16) <- (-1x32x1x1xf16, 1x32x1x1xf16)
        add__5 = paddle._C_ops.add_(conv2d_13, reshape_10)

        # pd_op.hardsigmoid: (-1x32x1x1xf16) <- (-1x32x1x1xf16)
        hardsigmoid_2 = paddle._C_ops.hardsigmoid(add__5, float('0.166667'), float('0.5'))

        # pd_op.multiply_: (-1x32x28x28xf16) <- (-1x32x28x28xf16, -1x32x1x1xf16)
        multiply__2 = paddle._C_ops.multiply_(concat_3, hardsigmoid_2)

        # pd_op.conv2d: (-1x16x28x28xf16) <- (-1x32x28x28xf16, 16x32x1x1xf16)
        conv2d_14 = paddle._C_ops.conv2d(multiply__2, parameter_77, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x16x28x28xf16, 16xf32, 16xf32, 16xf32, 16xf32, None) <- (-1x16x28x28xf16, 16xf32, 16xf32, 16xf32, 16xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_14, parameter_78, parameter_79, parameter_80, parameter_81, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x16x28x28xf16) <- (-1x16x28x28xf16)
        hardswish_9 = paddle._C_ops.hardswish(batch_norm__78)

        # builtin.slice: (-1x16x28x28xf16) <- ([-1x16x28x28xf16, -1x16x28x28xf16])
        slice_4 = split_1[0]

        # builtin.combine: ([-1x16x28x28xf16, -1x16x28x28xf16]) <- (-1x16x28x28xf16, -1x16x28x28xf16)
        combine_6 = [slice_4, hardswish_9]

        # pd_op.full: (1xi32) <- ()
        full_13 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x32x28x28xf16) <- ([-1x16x28x28xf16, -1x16x28x28xf16], 1xi32)
        concat_4 = paddle._C_ops.concat(combine_6, full_13)

        # pd_op.shape: (4xi32) <- (-1x32x28x28xf16)
        shape_1 = paddle._C_ops.shape(paddle.cast(concat_4, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_14 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_15 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(shape_1, [0], full_int_array_14, full_int_array_15, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_14 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_15 = paddle._C_ops.full([1], float('16'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_16 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_17 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_7 = [slice_5, full_14, full_15, full_16, full_17]

        # pd_op.reshape_: (-1x2x16x28x28xf16, 0x-1x32x28x28xf16) <- (-1x32x28x28xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__4, reshape__5 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_4, [x.reshape([]) for x in combine_7]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x16x2x28x28xf16) <- (-1x2x16x28x28xf16)
        transpose_1 = paddle._C_ops.transpose(reshape__4, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_18 = paddle._C_ops.full([1], float('32'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_19 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_20 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_8 = [slice_5, full_18, full_19, full_20]

        # pd_op.reshape_: (-1x32x28x28xf16, 0x-1x16x2x28x28xf16) <- (-1x16x2x28x28xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__6, reshape__7 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_1, [x.reshape([]) for x in combine_8]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x32x14x14xf16) <- (-1x32x28x28xf16, 32x1x3x3xf16)
        depthwise_conv2d_5 = paddle._C_ops.depthwise_conv2d(reshape__6, parameter_82, [2, 2], [1, 1], 'EXPLICIT', 32, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x32x14x14xf16, 32xf32, 32xf32, 32xf32, 32xf32, None) <- (-1x32x14x14xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_5, parameter_83, parameter_84, parameter_85, parameter_86, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x28x14x14xf16) <- (-1x32x14x14xf16, 28x32x1x1xf16)
        conv2d_15 = paddle._C_ops.conv2d(batch_norm__84, parameter_87, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32, None) <- (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_15, parameter_88, parameter_89, parameter_90, parameter_91, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x28x14x14xf16) <- (-1x28x14x14xf16)
        hardswish_10 = paddle._C_ops.hardswish(batch_norm__90)

        # pd_op.conv2d: (-1x28x28x28xf16) <- (-1x32x28x28xf16, 28x32x1x1xf16)
        conv2d_16 = paddle._C_ops.conv2d(reshape__6, parameter_92, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x28x28x28xf16, 28xf32, 28xf32, 28xf32, 28xf32, None) <- (-1x28x28x28xf16, 28xf32, 28xf32, 28xf32, 28xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_16, parameter_93, parameter_94, parameter_95, parameter_96, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x28x28x28xf16) <- (-1x28x28x28xf16)
        hardswish_11 = paddle._C_ops.hardswish(batch_norm__96)

        # pd_op.depthwise_conv2d: (-1x28x14x14xf16) <- (-1x28x28x28xf16, 28x1x3x3xf16)
        depthwise_conv2d_6 = paddle._C_ops.depthwise_conv2d(hardswish_11, parameter_97, [2, 2], [1, 1], 'EXPLICIT', 28, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32, None) <- (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_6, parameter_98, parameter_99, parameter_100, parameter_101, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_16 = [1, 1]

        # pd_op.pool2d: (-1x28x1x1xf16) <- (-1x28x14x14xf16, 2xi64)
        pool2d_4 = paddle._C_ops.pool2d(batch_norm__102, full_int_array_16, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x7x1x1xf16) <- (-1x28x1x1xf16, 7x28x1x1xf16)
        conv2d_17 = paddle._C_ops.conv2d(pool2d_4, parameter_102, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_17 = [1, 7, 1, 1]

        # pd_op.reshape: (1x7x1x1xf16, 0x7xf16) <- (7xf16, 4xi64)
        reshape_12, reshape_13 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_103, full_int_array_17), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x7x1x1xf16) <- (-1x7x1x1xf16, 1x7x1x1xf16)
        add__6 = paddle._C_ops.add_(conv2d_17, reshape_12)

        # pd_op.relu_: (-1x7x1x1xf16) <- (-1x7x1x1xf16)
        relu__3 = paddle._C_ops.relu_(add__6)

        # pd_op.conv2d: (-1x28x1x1xf16) <- (-1x7x1x1xf16, 28x7x1x1xf16)
        conv2d_18 = paddle._C_ops.conv2d(relu__3, parameter_104, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_18 = [1, 28, 1, 1]

        # pd_op.reshape: (1x28x1x1xf16, 0x28xf16) <- (28xf16, 4xi64)
        reshape_14, reshape_15 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_105, full_int_array_18), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x28x1x1xf16) <- (-1x28x1x1xf16, 1x28x1x1xf16)
        add__7 = paddle._C_ops.add_(conv2d_18, reshape_14)

        # pd_op.hardsigmoid: (-1x28x1x1xf16) <- (-1x28x1x1xf16)
        hardsigmoid_3 = paddle._C_ops.hardsigmoid(add__7, float('0.166667'), float('0.5'))

        # pd_op.multiply_: (-1x28x14x14xf16) <- (-1x28x14x14xf16, -1x28x1x1xf16)
        multiply__3 = paddle._C_ops.multiply_(batch_norm__102, hardsigmoid_3)

        # pd_op.conv2d: (-1x28x14x14xf16) <- (-1x28x14x14xf16, 28x28x1x1xf16)
        conv2d_19 = paddle._C_ops.conv2d(multiply__3, parameter_106, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32, None) <- (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_19, parameter_107, parameter_108, parameter_109, parameter_110, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x28x14x14xf16) <- (-1x28x14x14xf16)
        hardswish_12 = paddle._C_ops.hardswish(batch_norm__108)

        # builtin.combine: ([-1x28x14x14xf16, -1x28x14x14xf16]) <- (-1x28x14x14xf16, -1x28x14x14xf16)
        combine_9 = [hardswish_10, hardswish_12]

        # pd_op.full: (1xi32) <- ()
        full_21 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x56x14x14xf16) <- ([-1x28x14x14xf16, -1x28x14x14xf16], 1xi32)
        concat_5 = paddle._C_ops.concat(combine_9, full_21)

        # pd_op.depthwise_conv2d: (-1x56x14x14xf16) <- (-1x56x14x14xf16, 56x1x3x3xf16)
        depthwise_conv2d_7 = paddle._C_ops.depthwise_conv2d(concat_5, parameter_111, [1, 1], [1, 1], 'EXPLICIT', 56, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x56x14x14xf16, 56xf32, 56xf32, 56xf32, 56xf32, None) <- (-1x56x14x14xf16, 56xf32, 56xf32, 56xf32, 56xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_7, parameter_112, parameter_113, parameter_114, parameter_115, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x56x14x14xf16) <- (-1x56x14x14xf16)
        hardswish_13 = paddle._C_ops.hardswish(batch_norm__114)

        # pd_op.conv2d: (-1x56x14x14xf16) <- (-1x56x14x14xf16, 56x56x1x1xf16)
        conv2d_20 = paddle._C_ops.conv2d(hardswish_13, parameter_116, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x56x14x14xf16, 56xf32, 56xf32, 56xf32, 56xf32, None) <- (-1x56x14x14xf16, 56xf32, 56xf32, 56xf32, 56xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_20, parameter_117, parameter_118, parameter_119, parameter_120, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x56x14x14xf16) <- (-1x56x14x14xf16)
        hardswish_14 = paddle._C_ops.hardswish(batch_norm__120)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_19 = [28, 28]

        # pd_op.full: (1xi32) <- ()
        full_22 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x28x14x14xf16, -1x28x14x14xf16]) <- (-1x56x14x14xf16, 2xi64, 1xi32)
        split_2 = paddle._C_ops.split(hardswish_14, full_int_array_19, full_22)

        # builtin.slice: (-1x28x14x14xf16) <- ([-1x28x14x14xf16, -1x28x14x14xf16])
        slice_6 = split_2[1]

        # pd_op.conv2d: (-1x28x14x14xf16) <- (-1x28x14x14xf16, 28x28x1x1xf16)
        conv2d_21 = paddle._C_ops.conv2d(slice_6, parameter_121, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32, None) <- (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_21, parameter_122, parameter_123, parameter_124, parameter_125, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x28x14x14xf16) <- (-1x28x14x14xf16)
        hardswish_15 = paddle._C_ops.hardswish(batch_norm__126)

        # pd_op.depthwise_conv2d: (-1x28x14x14xf16) <- (-1x28x14x14xf16, 28x1x3x3xf16)
        depthwise_conv2d_8 = paddle._C_ops.depthwise_conv2d(hardswish_15, parameter_126, [1, 1], [1, 1], 'EXPLICIT', 28, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32, None) <- (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_8, parameter_127, parameter_128, parameter_129, parameter_130, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # builtin.combine: ([-1x28x14x14xf16, -1x28x14x14xf16]) <- (-1x28x14x14xf16, -1x28x14x14xf16)
        combine_10 = [hardswish_15, batch_norm__132]

        # pd_op.full: (1xi32) <- ()
        full_23 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x56x14x14xf16) <- ([-1x28x14x14xf16, -1x28x14x14xf16], 1xi32)
        concat_6 = paddle._C_ops.concat(combine_10, full_23)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_20 = [1, 1]

        # pd_op.pool2d: (-1x56x1x1xf16) <- (-1x56x14x14xf16, 2xi64)
        pool2d_5 = paddle._C_ops.pool2d(concat_6, full_int_array_20, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x14x1x1xf16) <- (-1x56x1x1xf16, 14x56x1x1xf16)
        conv2d_22 = paddle._C_ops.conv2d(pool2d_5, parameter_131, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_21 = [1, 14, 1, 1]

        # pd_op.reshape: (1x14x1x1xf16, 0x14xf16) <- (14xf16, 4xi64)
        reshape_16, reshape_17 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_132, full_int_array_21), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x14x1x1xf16) <- (-1x14x1x1xf16, 1x14x1x1xf16)
        add__8 = paddle._C_ops.add_(conv2d_22, reshape_16)

        # pd_op.relu_: (-1x14x1x1xf16) <- (-1x14x1x1xf16)
        relu__4 = paddle._C_ops.relu_(add__8)

        # pd_op.conv2d: (-1x56x1x1xf16) <- (-1x14x1x1xf16, 56x14x1x1xf16)
        conv2d_23 = paddle._C_ops.conv2d(relu__4, parameter_133, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_22 = [1, 56, 1, 1]

        # pd_op.reshape: (1x56x1x1xf16, 0x56xf16) <- (56xf16, 4xi64)
        reshape_18, reshape_19 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_134, full_int_array_22), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x56x1x1xf16) <- (-1x56x1x1xf16, 1x56x1x1xf16)
        add__9 = paddle._C_ops.add_(conv2d_23, reshape_18)

        # pd_op.hardsigmoid: (-1x56x1x1xf16) <- (-1x56x1x1xf16)
        hardsigmoid_4 = paddle._C_ops.hardsigmoid(add__9, float('0.166667'), float('0.5'))

        # pd_op.multiply_: (-1x56x14x14xf16) <- (-1x56x14x14xf16, -1x56x1x1xf16)
        multiply__4 = paddle._C_ops.multiply_(concat_6, hardsigmoid_4)

        # pd_op.conv2d: (-1x28x14x14xf16) <- (-1x56x14x14xf16, 28x56x1x1xf16)
        conv2d_24 = paddle._C_ops.conv2d(multiply__4, parameter_135, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32, None) <- (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_24, parameter_136, parameter_137, parameter_138, parameter_139, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x28x14x14xf16) <- (-1x28x14x14xf16)
        hardswish_16 = paddle._C_ops.hardswish(batch_norm__138)

        # builtin.slice: (-1x28x14x14xf16) <- ([-1x28x14x14xf16, -1x28x14x14xf16])
        slice_7 = split_2[0]

        # builtin.combine: ([-1x28x14x14xf16, -1x28x14x14xf16]) <- (-1x28x14x14xf16, -1x28x14x14xf16)
        combine_11 = [slice_7, hardswish_16]

        # pd_op.full: (1xi32) <- ()
        full_24 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x56x14x14xf16) <- ([-1x28x14x14xf16, -1x28x14x14xf16], 1xi32)
        concat_7 = paddle._C_ops.concat(combine_11, full_24)

        # pd_op.shape: (4xi32) <- (-1x56x14x14xf16)
        shape_2 = paddle._C_ops.shape(paddle.cast(concat_7, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_23 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_24 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(shape_2, [0], full_int_array_23, full_int_array_24, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_25 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_26 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_27 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_28 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_12 = [slice_8, full_25, full_26, full_27, full_28]

        # pd_op.reshape_: (-1x2x28x14x14xf16, 0x-1x56x14x14xf16) <- (-1x56x14x14xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__8, reshape__9 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_7, [x.reshape([]) for x in combine_12]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x28x2x14x14xf16) <- (-1x2x28x14x14xf16)
        transpose_2 = paddle._C_ops.transpose(reshape__8, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_29 = paddle._C_ops.full([1], float('56'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_30 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_31 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_13 = [slice_8, full_29, full_30, full_31]

        # pd_op.reshape_: (-1x56x14x14xf16, 0x-1x28x2x14x14xf16) <- (-1x28x2x14x14xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__10, reshape__11 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_2, [x.reshape([]) for x in combine_13]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_25 = [28, 28]

        # pd_op.full: (1xi32) <- ()
        full_32 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x28x14x14xf16, -1x28x14x14xf16]) <- (-1x56x14x14xf16, 2xi64, 1xi32)
        split_3 = paddle._C_ops.split(reshape__10, full_int_array_25, full_32)

        # builtin.slice: (-1x28x14x14xf16) <- ([-1x28x14x14xf16, -1x28x14x14xf16])
        slice_9 = split_3[1]

        # pd_op.conv2d: (-1x28x14x14xf16) <- (-1x28x14x14xf16, 28x28x1x1xf16)
        conv2d_25 = paddle._C_ops.conv2d(slice_9, parameter_140, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32, None) <- (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_25, parameter_141, parameter_142, parameter_143, parameter_144, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x28x14x14xf16) <- (-1x28x14x14xf16)
        hardswish_17 = paddle._C_ops.hardswish(batch_norm__144)

        # pd_op.depthwise_conv2d: (-1x28x14x14xf16) <- (-1x28x14x14xf16, 28x1x3x3xf16)
        depthwise_conv2d_9 = paddle._C_ops.depthwise_conv2d(hardswish_17, parameter_145, [1, 1], [1, 1], 'EXPLICIT', 28, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32, None) <- (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_9, parameter_146, parameter_147, parameter_148, parameter_149, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # builtin.combine: ([-1x28x14x14xf16, -1x28x14x14xf16]) <- (-1x28x14x14xf16, -1x28x14x14xf16)
        combine_14 = [hardswish_17, batch_norm__150]

        # pd_op.full: (1xi32) <- ()
        full_33 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x56x14x14xf16) <- ([-1x28x14x14xf16, -1x28x14x14xf16], 1xi32)
        concat_8 = paddle._C_ops.concat(combine_14, full_33)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_26 = [1, 1]

        # pd_op.pool2d: (-1x56x1x1xf16) <- (-1x56x14x14xf16, 2xi64)
        pool2d_6 = paddle._C_ops.pool2d(concat_8, full_int_array_26, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x14x1x1xf16) <- (-1x56x1x1xf16, 14x56x1x1xf16)
        conv2d_26 = paddle._C_ops.conv2d(pool2d_6, parameter_150, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_27 = [1, 14, 1, 1]

        # pd_op.reshape: (1x14x1x1xf16, 0x14xf16) <- (14xf16, 4xi64)
        reshape_20, reshape_21 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_151, full_int_array_27), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x14x1x1xf16) <- (-1x14x1x1xf16, 1x14x1x1xf16)
        add__10 = paddle._C_ops.add_(conv2d_26, reshape_20)

        # pd_op.relu_: (-1x14x1x1xf16) <- (-1x14x1x1xf16)
        relu__5 = paddle._C_ops.relu_(add__10)

        # pd_op.conv2d: (-1x56x1x1xf16) <- (-1x14x1x1xf16, 56x14x1x1xf16)
        conv2d_27 = paddle._C_ops.conv2d(relu__5, parameter_152, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_28 = [1, 56, 1, 1]

        # pd_op.reshape: (1x56x1x1xf16, 0x56xf16) <- (56xf16, 4xi64)
        reshape_22, reshape_23 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_153, full_int_array_28), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x56x1x1xf16) <- (-1x56x1x1xf16, 1x56x1x1xf16)
        add__11 = paddle._C_ops.add_(conv2d_27, reshape_22)

        # pd_op.hardsigmoid: (-1x56x1x1xf16) <- (-1x56x1x1xf16)
        hardsigmoid_5 = paddle._C_ops.hardsigmoid(add__11, float('0.166667'), float('0.5'))

        # pd_op.multiply_: (-1x56x14x14xf16) <- (-1x56x14x14xf16, -1x56x1x1xf16)
        multiply__5 = paddle._C_ops.multiply_(concat_8, hardsigmoid_5)

        # pd_op.conv2d: (-1x28x14x14xf16) <- (-1x56x14x14xf16, 28x56x1x1xf16)
        conv2d_28 = paddle._C_ops.conv2d(multiply__5, parameter_154, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32, None) <- (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_28, parameter_155, parameter_156, parameter_157, parameter_158, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x28x14x14xf16) <- (-1x28x14x14xf16)
        hardswish_18 = paddle._C_ops.hardswish(batch_norm__156)

        # builtin.slice: (-1x28x14x14xf16) <- ([-1x28x14x14xf16, -1x28x14x14xf16])
        slice_10 = split_3[0]

        # builtin.combine: ([-1x28x14x14xf16, -1x28x14x14xf16]) <- (-1x28x14x14xf16, -1x28x14x14xf16)
        combine_15 = [slice_10, hardswish_18]

        # pd_op.full: (1xi32) <- ()
        full_34 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x56x14x14xf16) <- ([-1x28x14x14xf16, -1x28x14x14xf16], 1xi32)
        concat_9 = paddle._C_ops.concat(combine_15, full_34)

        # pd_op.shape: (4xi32) <- (-1x56x14x14xf16)
        shape_3 = paddle._C_ops.shape(paddle.cast(concat_9, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_29 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_30 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(shape_3, [0], full_int_array_29, full_int_array_30, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_35 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_36 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_37 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_38 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_16 = [slice_11, full_35, full_36, full_37, full_38]

        # pd_op.reshape_: (-1x2x28x14x14xf16, 0x-1x56x14x14xf16) <- (-1x56x14x14xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__12, reshape__13 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_9, [x.reshape([]) for x in combine_16]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x28x2x14x14xf16) <- (-1x2x28x14x14xf16)
        transpose_3 = paddle._C_ops.transpose(reshape__12, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_39 = paddle._C_ops.full([1], float('56'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_40 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_41 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_17 = [slice_11, full_39, full_40, full_41]

        # pd_op.reshape_: (-1x56x14x14xf16, 0x-1x28x2x14x14xf16) <- (-1x28x2x14x14xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__14, reshape__15 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_3, [x.reshape([]) for x in combine_17]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_31 = [28, 28]

        # pd_op.full: (1xi32) <- ()
        full_42 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x28x14x14xf16, -1x28x14x14xf16]) <- (-1x56x14x14xf16, 2xi64, 1xi32)
        split_4 = paddle._C_ops.split(reshape__14, full_int_array_31, full_42)

        # builtin.slice: (-1x28x14x14xf16) <- ([-1x28x14x14xf16, -1x28x14x14xf16])
        slice_12 = split_4[1]

        # pd_op.conv2d: (-1x28x14x14xf16) <- (-1x28x14x14xf16, 28x28x1x1xf16)
        conv2d_29 = paddle._C_ops.conv2d(slice_12, parameter_159, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32, None) <- (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_29, parameter_160, parameter_161, parameter_162, parameter_163, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x28x14x14xf16) <- (-1x28x14x14xf16)
        hardswish_19 = paddle._C_ops.hardswish(batch_norm__162)

        # pd_op.depthwise_conv2d: (-1x28x14x14xf16) <- (-1x28x14x14xf16, 28x1x3x3xf16)
        depthwise_conv2d_10 = paddle._C_ops.depthwise_conv2d(hardswish_19, parameter_164, [1, 1], [1, 1], 'EXPLICIT', 28, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32, None) <- (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_10, parameter_165, parameter_166, parameter_167, parameter_168, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # builtin.combine: ([-1x28x14x14xf16, -1x28x14x14xf16]) <- (-1x28x14x14xf16, -1x28x14x14xf16)
        combine_18 = [hardswish_19, batch_norm__168]

        # pd_op.full: (1xi32) <- ()
        full_43 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x56x14x14xf16) <- ([-1x28x14x14xf16, -1x28x14x14xf16], 1xi32)
        concat_10 = paddle._C_ops.concat(combine_18, full_43)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_32 = [1, 1]

        # pd_op.pool2d: (-1x56x1x1xf16) <- (-1x56x14x14xf16, 2xi64)
        pool2d_7 = paddle._C_ops.pool2d(concat_10, full_int_array_32, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x14x1x1xf16) <- (-1x56x1x1xf16, 14x56x1x1xf16)
        conv2d_30 = paddle._C_ops.conv2d(pool2d_7, parameter_169, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_33 = [1, 14, 1, 1]

        # pd_op.reshape: (1x14x1x1xf16, 0x14xf16) <- (14xf16, 4xi64)
        reshape_24, reshape_25 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_170, full_int_array_33), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x14x1x1xf16) <- (-1x14x1x1xf16, 1x14x1x1xf16)
        add__12 = paddle._C_ops.add_(conv2d_30, reshape_24)

        # pd_op.relu_: (-1x14x1x1xf16) <- (-1x14x1x1xf16)
        relu__6 = paddle._C_ops.relu_(add__12)

        # pd_op.conv2d: (-1x56x1x1xf16) <- (-1x14x1x1xf16, 56x14x1x1xf16)
        conv2d_31 = paddle._C_ops.conv2d(relu__6, parameter_171, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_34 = [1, 56, 1, 1]

        # pd_op.reshape: (1x56x1x1xf16, 0x56xf16) <- (56xf16, 4xi64)
        reshape_26, reshape_27 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_172, full_int_array_34), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x56x1x1xf16) <- (-1x56x1x1xf16, 1x56x1x1xf16)
        add__13 = paddle._C_ops.add_(conv2d_31, reshape_26)

        # pd_op.hardsigmoid: (-1x56x1x1xf16) <- (-1x56x1x1xf16)
        hardsigmoid_6 = paddle._C_ops.hardsigmoid(add__13, float('0.166667'), float('0.5'))

        # pd_op.multiply_: (-1x56x14x14xf16) <- (-1x56x14x14xf16, -1x56x1x1xf16)
        multiply__6 = paddle._C_ops.multiply_(concat_10, hardsigmoid_6)

        # pd_op.conv2d: (-1x28x14x14xf16) <- (-1x56x14x14xf16, 28x56x1x1xf16)
        conv2d_32 = paddle._C_ops.conv2d(multiply__6, parameter_173, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32, None) <- (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_32, parameter_174, parameter_175, parameter_176, parameter_177, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x28x14x14xf16) <- (-1x28x14x14xf16)
        hardswish_20 = paddle._C_ops.hardswish(batch_norm__174)

        # builtin.slice: (-1x28x14x14xf16) <- ([-1x28x14x14xf16, -1x28x14x14xf16])
        slice_13 = split_4[0]

        # builtin.combine: ([-1x28x14x14xf16, -1x28x14x14xf16]) <- (-1x28x14x14xf16, -1x28x14x14xf16)
        combine_19 = [slice_13, hardswish_20]

        # pd_op.full: (1xi32) <- ()
        full_44 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x56x14x14xf16) <- ([-1x28x14x14xf16, -1x28x14x14xf16], 1xi32)
        concat_11 = paddle._C_ops.concat(combine_19, full_44)

        # pd_op.shape: (4xi32) <- (-1x56x14x14xf16)
        shape_4 = paddle._C_ops.shape(paddle.cast(concat_11, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_35 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_36 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(shape_4, [0], full_int_array_35, full_int_array_36, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_45 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_46 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_47 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_48 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_20 = [slice_14, full_45, full_46, full_47, full_48]

        # pd_op.reshape_: (-1x2x28x14x14xf16, 0x-1x56x14x14xf16) <- (-1x56x14x14xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__16, reshape__17 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_11, [x.reshape([]) for x in combine_20]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x28x2x14x14xf16) <- (-1x2x28x14x14xf16)
        transpose_4 = paddle._C_ops.transpose(reshape__16, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_49 = paddle._C_ops.full([1], float('56'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_50 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_51 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_21 = [slice_14, full_49, full_50, full_51]

        # pd_op.reshape_: (-1x56x14x14xf16, 0x-1x28x2x14x14xf16) <- (-1x28x2x14x14xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__18, reshape__19 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_4, [x.reshape([]) for x in combine_21]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_37 = [28, 28]

        # pd_op.full: (1xi32) <- ()
        full_52 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x28x14x14xf16, -1x28x14x14xf16]) <- (-1x56x14x14xf16, 2xi64, 1xi32)
        split_5 = paddle._C_ops.split(reshape__18, full_int_array_37, full_52)

        # builtin.slice: (-1x28x14x14xf16) <- ([-1x28x14x14xf16, -1x28x14x14xf16])
        slice_15 = split_5[1]

        # pd_op.conv2d: (-1x28x14x14xf16) <- (-1x28x14x14xf16, 28x28x1x1xf16)
        conv2d_33 = paddle._C_ops.conv2d(slice_15, parameter_178, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32, None) <- (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32)
        batch_norm__180, batch_norm__181, batch_norm__182, batch_norm__183, batch_norm__184, batch_norm__185 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_33, parameter_179, parameter_180, parameter_181, parameter_182, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x28x14x14xf16) <- (-1x28x14x14xf16)
        hardswish_21 = paddle._C_ops.hardswish(batch_norm__180)

        # pd_op.depthwise_conv2d: (-1x28x14x14xf16) <- (-1x28x14x14xf16, 28x1x3x3xf16)
        depthwise_conv2d_11 = paddle._C_ops.depthwise_conv2d(hardswish_21, parameter_183, [1, 1], [1, 1], 'EXPLICIT', 28, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32, None) <- (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32)
        batch_norm__186, batch_norm__187, batch_norm__188, batch_norm__189, batch_norm__190, batch_norm__191 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_11, parameter_184, parameter_185, parameter_186, parameter_187, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # builtin.combine: ([-1x28x14x14xf16, -1x28x14x14xf16]) <- (-1x28x14x14xf16, -1x28x14x14xf16)
        combine_22 = [hardswish_21, batch_norm__186]

        # pd_op.full: (1xi32) <- ()
        full_53 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x56x14x14xf16) <- ([-1x28x14x14xf16, -1x28x14x14xf16], 1xi32)
        concat_12 = paddle._C_ops.concat(combine_22, full_53)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_38 = [1, 1]

        # pd_op.pool2d: (-1x56x1x1xf16) <- (-1x56x14x14xf16, 2xi64)
        pool2d_8 = paddle._C_ops.pool2d(concat_12, full_int_array_38, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x14x1x1xf16) <- (-1x56x1x1xf16, 14x56x1x1xf16)
        conv2d_34 = paddle._C_ops.conv2d(pool2d_8, parameter_188, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_39 = [1, 14, 1, 1]

        # pd_op.reshape: (1x14x1x1xf16, 0x14xf16) <- (14xf16, 4xi64)
        reshape_28, reshape_29 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_189, full_int_array_39), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x14x1x1xf16) <- (-1x14x1x1xf16, 1x14x1x1xf16)
        add__14 = paddle._C_ops.add_(conv2d_34, reshape_28)

        # pd_op.relu_: (-1x14x1x1xf16) <- (-1x14x1x1xf16)
        relu__7 = paddle._C_ops.relu_(add__14)

        # pd_op.conv2d: (-1x56x1x1xf16) <- (-1x14x1x1xf16, 56x14x1x1xf16)
        conv2d_35 = paddle._C_ops.conv2d(relu__7, parameter_190, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_40 = [1, 56, 1, 1]

        # pd_op.reshape: (1x56x1x1xf16, 0x56xf16) <- (56xf16, 4xi64)
        reshape_30, reshape_31 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_191, full_int_array_40), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x56x1x1xf16) <- (-1x56x1x1xf16, 1x56x1x1xf16)
        add__15 = paddle._C_ops.add_(conv2d_35, reshape_30)

        # pd_op.hardsigmoid: (-1x56x1x1xf16) <- (-1x56x1x1xf16)
        hardsigmoid_7 = paddle._C_ops.hardsigmoid(add__15, float('0.166667'), float('0.5'))

        # pd_op.multiply_: (-1x56x14x14xf16) <- (-1x56x14x14xf16, -1x56x1x1xf16)
        multiply__7 = paddle._C_ops.multiply_(concat_12, hardsigmoid_7)

        # pd_op.conv2d: (-1x28x14x14xf16) <- (-1x56x14x14xf16, 28x56x1x1xf16)
        conv2d_36 = paddle._C_ops.conv2d(multiply__7, parameter_192, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32, None) <- (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32)
        batch_norm__192, batch_norm__193, batch_norm__194, batch_norm__195, batch_norm__196, batch_norm__197 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_36, parameter_193, parameter_194, parameter_195, parameter_196, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x28x14x14xf16) <- (-1x28x14x14xf16)
        hardswish_22 = paddle._C_ops.hardswish(batch_norm__192)

        # builtin.slice: (-1x28x14x14xf16) <- ([-1x28x14x14xf16, -1x28x14x14xf16])
        slice_16 = split_5[0]

        # builtin.combine: ([-1x28x14x14xf16, -1x28x14x14xf16]) <- (-1x28x14x14xf16, -1x28x14x14xf16)
        combine_23 = [slice_16, hardswish_22]

        # pd_op.full: (1xi32) <- ()
        full_54 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x56x14x14xf16) <- ([-1x28x14x14xf16, -1x28x14x14xf16], 1xi32)
        concat_13 = paddle._C_ops.concat(combine_23, full_54)

        # pd_op.shape: (4xi32) <- (-1x56x14x14xf16)
        shape_5 = paddle._C_ops.shape(paddle.cast(concat_13, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_41 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_42 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(shape_5, [0], full_int_array_41, full_int_array_42, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_55 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_56 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_57 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_58 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_24 = [slice_17, full_55, full_56, full_57, full_58]

        # pd_op.reshape_: (-1x2x28x14x14xf16, 0x-1x56x14x14xf16) <- (-1x56x14x14xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__20, reshape__21 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_13, [x.reshape([]) for x in combine_24]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x28x2x14x14xf16) <- (-1x2x28x14x14xf16)
        transpose_5 = paddle._C_ops.transpose(reshape__20, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_59 = paddle._C_ops.full([1], float('56'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_60 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_61 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_25 = [slice_17, full_59, full_60, full_61]

        # pd_op.reshape_: (-1x56x14x14xf16, 0x-1x28x2x14x14xf16) <- (-1x28x2x14x14xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__22, reshape__23 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_5, [x.reshape([]) for x in combine_25]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_43 = [28, 28]

        # pd_op.full: (1xi32) <- ()
        full_62 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x28x14x14xf16, -1x28x14x14xf16]) <- (-1x56x14x14xf16, 2xi64, 1xi32)
        split_6 = paddle._C_ops.split(reshape__22, full_int_array_43, full_62)

        # builtin.slice: (-1x28x14x14xf16) <- ([-1x28x14x14xf16, -1x28x14x14xf16])
        slice_18 = split_6[1]

        # pd_op.conv2d: (-1x28x14x14xf16) <- (-1x28x14x14xf16, 28x28x1x1xf16)
        conv2d_37 = paddle._C_ops.conv2d(slice_18, parameter_197, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32, None) <- (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32)
        batch_norm__198, batch_norm__199, batch_norm__200, batch_norm__201, batch_norm__202, batch_norm__203 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_37, parameter_198, parameter_199, parameter_200, parameter_201, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x28x14x14xf16) <- (-1x28x14x14xf16)
        hardswish_23 = paddle._C_ops.hardswish(batch_norm__198)

        # pd_op.depthwise_conv2d: (-1x28x14x14xf16) <- (-1x28x14x14xf16, 28x1x3x3xf16)
        depthwise_conv2d_12 = paddle._C_ops.depthwise_conv2d(hardswish_23, parameter_202, [1, 1], [1, 1], 'EXPLICIT', 28, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32, None) <- (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32)
        batch_norm__204, batch_norm__205, batch_norm__206, batch_norm__207, batch_norm__208, batch_norm__209 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_12, parameter_203, parameter_204, parameter_205, parameter_206, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # builtin.combine: ([-1x28x14x14xf16, -1x28x14x14xf16]) <- (-1x28x14x14xf16, -1x28x14x14xf16)
        combine_26 = [hardswish_23, batch_norm__204]

        # pd_op.full: (1xi32) <- ()
        full_63 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x56x14x14xf16) <- ([-1x28x14x14xf16, -1x28x14x14xf16], 1xi32)
        concat_14 = paddle._C_ops.concat(combine_26, full_63)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_44 = [1, 1]

        # pd_op.pool2d: (-1x56x1x1xf16) <- (-1x56x14x14xf16, 2xi64)
        pool2d_9 = paddle._C_ops.pool2d(concat_14, full_int_array_44, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x14x1x1xf16) <- (-1x56x1x1xf16, 14x56x1x1xf16)
        conv2d_38 = paddle._C_ops.conv2d(pool2d_9, parameter_207, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_45 = [1, 14, 1, 1]

        # pd_op.reshape: (1x14x1x1xf16, 0x14xf16) <- (14xf16, 4xi64)
        reshape_32, reshape_33 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_208, full_int_array_45), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x14x1x1xf16) <- (-1x14x1x1xf16, 1x14x1x1xf16)
        add__16 = paddle._C_ops.add_(conv2d_38, reshape_32)

        # pd_op.relu_: (-1x14x1x1xf16) <- (-1x14x1x1xf16)
        relu__8 = paddle._C_ops.relu_(add__16)

        # pd_op.conv2d: (-1x56x1x1xf16) <- (-1x14x1x1xf16, 56x14x1x1xf16)
        conv2d_39 = paddle._C_ops.conv2d(relu__8, parameter_209, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_46 = [1, 56, 1, 1]

        # pd_op.reshape: (1x56x1x1xf16, 0x56xf16) <- (56xf16, 4xi64)
        reshape_34, reshape_35 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_210, full_int_array_46), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x56x1x1xf16) <- (-1x56x1x1xf16, 1x56x1x1xf16)
        add__17 = paddle._C_ops.add_(conv2d_39, reshape_34)

        # pd_op.hardsigmoid: (-1x56x1x1xf16) <- (-1x56x1x1xf16)
        hardsigmoid_8 = paddle._C_ops.hardsigmoid(add__17, float('0.166667'), float('0.5'))

        # pd_op.multiply_: (-1x56x14x14xf16) <- (-1x56x14x14xf16, -1x56x1x1xf16)
        multiply__8 = paddle._C_ops.multiply_(concat_14, hardsigmoid_8)

        # pd_op.conv2d: (-1x28x14x14xf16) <- (-1x56x14x14xf16, 28x56x1x1xf16)
        conv2d_40 = paddle._C_ops.conv2d(multiply__8, parameter_211, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32, None) <- (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32)
        batch_norm__210, batch_norm__211, batch_norm__212, batch_norm__213, batch_norm__214, batch_norm__215 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_40, parameter_212, parameter_213, parameter_214, parameter_215, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x28x14x14xf16) <- (-1x28x14x14xf16)
        hardswish_24 = paddle._C_ops.hardswish(batch_norm__210)

        # builtin.slice: (-1x28x14x14xf16) <- ([-1x28x14x14xf16, -1x28x14x14xf16])
        slice_19 = split_6[0]

        # builtin.combine: ([-1x28x14x14xf16, -1x28x14x14xf16]) <- (-1x28x14x14xf16, -1x28x14x14xf16)
        combine_27 = [slice_19, hardswish_24]

        # pd_op.full: (1xi32) <- ()
        full_64 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x56x14x14xf16) <- ([-1x28x14x14xf16, -1x28x14x14xf16], 1xi32)
        concat_15 = paddle._C_ops.concat(combine_27, full_64)

        # pd_op.shape: (4xi32) <- (-1x56x14x14xf16)
        shape_6 = paddle._C_ops.shape(paddle.cast(concat_15, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_47 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_48 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(shape_6, [0], full_int_array_47, full_int_array_48, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_65 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_66 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_67 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_68 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_28 = [slice_20, full_65, full_66, full_67, full_68]

        # pd_op.reshape_: (-1x2x28x14x14xf16, 0x-1x56x14x14xf16) <- (-1x56x14x14xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__24, reshape__25 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_15, [x.reshape([]) for x in combine_28]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x28x2x14x14xf16) <- (-1x2x28x14x14xf16)
        transpose_6 = paddle._C_ops.transpose(reshape__24, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_69 = paddle._C_ops.full([1], float('56'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_70 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_71 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_29 = [slice_20, full_69, full_70, full_71]

        # pd_op.reshape_: (-1x56x14x14xf16, 0x-1x28x2x14x14xf16) <- (-1x28x2x14x14xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__26, reshape__27 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_6, [x.reshape([]) for x in combine_29]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_49 = [28, 28]

        # pd_op.full: (1xi32) <- ()
        full_72 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x28x14x14xf16, -1x28x14x14xf16]) <- (-1x56x14x14xf16, 2xi64, 1xi32)
        split_7 = paddle._C_ops.split(reshape__26, full_int_array_49, full_72)

        # builtin.slice: (-1x28x14x14xf16) <- ([-1x28x14x14xf16, -1x28x14x14xf16])
        slice_21 = split_7[1]

        # pd_op.conv2d: (-1x28x14x14xf16) <- (-1x28x14x14xf16, 28x28x1x1xf16)
        conv2d_41 = paddle._C_ops.conv2d(slice_21, parameter_216, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32, None) <- (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32)
        batch_norm__216, batch_norm__217, batch_norm__218, batch_norm__219, batch_norm__220, batch_norm__221 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_41, parameter_217, parameter_218, parameter_219, parameter_220, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x28x14x14xf16) <- (-1x28x14x14xf16)
        hardswish_25 = paddle._C_ops.hardswish(batch_norm__216)

        # pd_op.depthwise_conv2d: (-1x28x14x14xf16) <- (-1x28x14x14xf16, 28x1x3x3xf16)
        depthwise_conv2d_13 = paddle._C_ops.depthwise_conv2d(hardswish_25, parameter_221, [1, 1], [1, 1], 'EXPLICIT', 28, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32, None) <- (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32)
        batch_norm__222, batch_norm__223, batch_norm__224, batch_norm__225, batch_norm__226, batch_norm__227 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_13, parameter_222, parameter_223, parameter_224, parameter_225, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # builtin.combine: ([-1x28x14x14xf16, -1x28x14x14xf16]) <- (-1x28x14x14xf16, -1x28x14x14xf16)
        combine_30 = [hardswish_25, batch_norm__222]

        # pd_op.full: (1xi32) <- ()
        full_73 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x56x14x14xf16) <- ([-1x28x14x14xf16, -1x28x14x14xf16], 1xi32)
        concat_16 = paddle._C_ops.concat(combine_30, full_73)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_50 = [1, 1]

        # pd_op.pool2d: (-1x56x1x1xf16) <- (-1x56x14x14xf16, 2xi64)
        pool2d_10 = paddle._C_ops.pool2d(concat_16, full_int_array_50, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x14x1x1xf16) <- (-1x56x1x1xf16, 14x56x1x1xf16)
        conv2d_42 = paddle._C_ops.conv2d(pool2d_10, parameter_226, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_51 = [1, 14, 1, 1]

        # pd_op.reshape: (1x14x1x1xf16, 0x14xf16) <- (14xf16, 4xi64)
        reshape_36, reshape_37 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_227, full_int_array_51), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x14x1x1xf16) <- (-1x14x1x1xf16, 1x14x1x1xf16)
        add__18 = paddle._C_ops.add_(conv2d_42, reshape_36)

        # pd_op.relu_: (-1x14x1x1xf16) <- (-1x14x1x1xf16)
        relu__9 = paddle._C_ops.relu_(add__18)

        # pd_op.conv2d: (-1x56x1x1xf16) <- (-1x14x1x1xf16, 56x14x1x1xf16)
        conv2d_43 = paddle._C_ops.conv2d(relu__9, parameter_228, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_52 = [1, 56, 1, 1]

        # pd_op.reshape: (1x56x1x1xf16, 0x56xf16) <- (56xf16, 4xi64)
        reshape_38, reshape_39 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_229, full_int_array_52), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x56x1x1xf16) <- (-1x56x1x1xf16, 1x56x1x1xf16)
        add__19 = paddle._C_ops.add_(conv2d_43, reshape_38)

        # pd_op.hardsigmoid: (-1x56x1x1xf16) <- (-1x56x1x1xf16)
        hardsigmoid_9 = paddle._C_ops.hardsigmoid(add__19, float('0.166667'), float('0.5'))

        # pd_op.multiply_: (-1x56x14x14xf16) <- (-1x56x14x14xf16, -1x56x1x1xf16)
        multiply__9 = paddle._C_ops.multiply_(concat_16, hardsigmoid_9)

        # pd_op.conv2d: (-1x28x14x14xf16) <- (-1x56x14x14xf16, 28x56x1x1xf16)
        conv2d_44 = paddle._C_ops.conv2d(multiply__9, parameter_230, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32, None) <- (-1x28x14x14xf16, 28xf32, 28xf32, 28xf32, 28xf32)
        batch_norm__228, batch_norm__229, batch_norm__230, batch_norm__231, batch_norm__232, batch_norm__233 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_44, parameter_231, parameter_232, parameter_233, parameter_234, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x28x14x14xf16) <- (-1x28x14x14xf16)
        hardswish_26 = paddle._C_ops.hardswish(batch_norm__228)

        # builtin.slice: (-1x28x14x14xf16) <- ([-1x28x14x14xf16, -1x28x14x14xf16])
        slice_22 = split_7[0]

        # builtin.combine: ([-1x28x14x14xf16, -1x28x14x14xf16]) <- (-1x28x14x14xf16, -1x28x14x14xf16)
        combine_31 = [slice_22, hardswish_26]

        # pd_op.full: (1xi32) <- ()
        full_74 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x56x14x14xf16) <- ([-1x28x14x14xf16, -1x28x14x14xf16], 1xi32)
        concat_17 = paddle._C_ops.concat(combine_31, full_74)

        # pd_op.shape: (4xi32) <- (-1x56x14x14xf16)
        shape_7 = paddle._C_ops.shape(paddle.cast(concat_17, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_53 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_54 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(shape_7, [0], full_int_array_53, full_int_array_54, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_75 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_76 = paddle._C_ops.full([1], float('28'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_77 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_78 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_32 = [slice_23, full_75, full_76, full_77, full_78]

        # pd_op.reshape_: (-1x2x28x14x14xf16, 0x-1x56x14x14xf16) <- (-1x56x14x14xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__28, reshape__29 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_17, [x.reshape([]) for x in combine_32]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x28x2x14x14xf16) <- (-1x2x28x14x14xf16)
        transpose_7 = paddle._C_ops.transpose(reshape__28, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_79 = paddle._C_ops.full([1], float('56'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_80 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_81 = paddle._C_ops.full([1], float('14'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_33 = [slice_23, full_79, full_80, full_81]

        # pd_op.reshape_: (-1x56x14x14xf16, 0x-1x28x2x14x14xf16) <- (-1x28x2x14x14xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__30, reshape__31 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_7, [x.reshape([]) for x in combine_33]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.depthwise_conv2d: (-1x56x7x7xf16) <- (-1x56x14x14xf16, 56x1x3x3xf16)
        depthwise_conv2d_14 = paddle._C_ops.depthwise_conv2d(reshape__30, parameter_235, [2, 2], [1, 1], 'EXPLICIT', 56, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x56x7x7xf16, 56xf32, 56xf32, 56xf32, 56xf32, None) <- (-1x56x7x7xf16, 56xf32, 56xf32, 56xf32, 56xf32)
        batch_norm__234, batch_norm__235, batch_norm__236, batch_norm__237, batch_norm__238, batch_norm__239 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_14, parameter_236, parameter_237, parameter_238, parameter_239, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x60x7x7xf16) <- (-1x56x7x7xf16, 60x56x1x1xf16)
        conv2d_45 = paddle._C_ops.conv2d(batch_norm__234, parameter_240, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x60x7x7xf16, 60xf32, 60xf32, 60xf32, 60xf32, None) <- (-1x60x7x7xf16, 60xf32, 60xf32, 60xf32, 60xf32)
        batch_norm__240, batch_norm__241, batch_norm__242, batch_norm__243, batch_norm__244, batch_norm__245 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_45, parameter_241, parameter_242, parameter_243, parameter_244, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x60x7x7xf16) <- (-1x60x7x7xf16)
        hardswish_27 = paddle._C_ops.hardswish(batch_norm__240)

        # pd_op.conv2d: (-1x60x14x14xf16) <- (-1x56x14x14xf16, 60x56x1x1xf16)
        conv2d_46 = paddle._C_ops.conv2d(reshape__30, parameter_245, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x60x14x14xf16, 60xf32, 60xf32, 60xf32, 60xf32, None) <- (-1x60x14x14xf16, 60xf32, 60xf32, 60xf32, 60xf32)
        batch_norm__246, batch_norm__247, batch_norm__248, batch_norm__249, batch_norm__250, batch_norm__251 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_46, parameter_246, parameter_247, parameter_248, parameter_249, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x60x14x14xf16) <- (-1x60x14x14xf16)
        hardswish_28 = paddle._C_ops.hardswish(batch_norm__246)

        # pd_op.depthwise_conv2d: (-1x60x7x7xf16) <- (-1x60x14x14xf16, 60x1x3x3xf16)
        depthwise_conv2d_15 = paddle._C_ops.depthwise_conv2d(hardswish_28, parameter_250, [2, 2], [1, 1], 'EXPLICIT', 60, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x60x7x7xf16, 60xf32, 60xf32, 60xf32, 60xf32, None) <- (-1x60x7x7xf16, 60xf32, 60xf32, 60xf32, 60xf32)
        batch_norm__252, batch_norm__253, batch_norm__254, batch_norm__255, batch_norm__256, batch_norm__257 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_15, parameter_251, parameter_252, parameter_253, parameter_254, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_55 = [1, 1]

        # pd_op.pool2d: (-1x60x1x1xf16) <- (-1x60x7x7xf16, 2xi64)
        pool2d_11 = paddle._C_ops.pool2d(batch_norm__252, full_int_array_55, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x15x1x1xf16) <- (-1x60x1x1xf16, 15x60x1x1xf16)
        conv2d_47 = paddle._C_ops.conv2d(pool2d_11, parameter_255, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_56 = [1, 15, 1, 1]

        # pd_op.reshape: (1x15x1x1xf16, 0x15xf16) <- (15xf16, 4xi64)
        reshape_40, reshape_41 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_256, full_int_array_56), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x15x1x1xf16) <- (-1x15x1x1xf16, 1x15x1x1xf16)
        add__20 = paddle._C_ops.add_(conv2d_47, reshape_40)

        # pd_op.relu_: (-1x15x1x1xf16) <- (-1x15x1x1xf16)
        relu__10 = paddle._C_ops.relu_(add__20)

        # pd_op.conv2d: (-1x60x1x1xf16) <- (-1x15x1x1xf16, 60x15x1x1xf16)
        conv2d_48 = paddle._C_ops.conv2d(relu__10, parameter_257, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_57 = [1, 60, 1, 1]

        # pd_op.reshape: (1x60x1x1xf16, 0x60xf16) <- (60xf16, 4xi64)
        reshape_42, reshape_43 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_258, full_int_array_57), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x60x1x1xf16) <- (-1x60x1x1xf16, 1x60x1x1xf16)
        add__21 = paddle._C_ops.add_(conv2d_48, reshape_42)

        # pd_op.hardsigmoid: (-1x60x1x1xf16) <- (-1x60x1x1xf16)
        hardsigmoid_10 = paddle._C_ops.hardsigmoid(add__21, float('0.166667'), float('0.5'))

        # pd_op.multiply_: (-1x60x7x7xf16) <- (-1x60x7x7xf16, -1x60x1x1xf16)
        multiply__10 = paddle._C_ops.multiply_(batch_norm__252, hardsigmoid_10)

        # pd_op.conv2d: (-1x60x7x7xf16) <- (-1x60x7x7xf16, 60x60x1x1xf16)
        conv2d_49 = paddle._C_ops.conv2d(multiply__10, parameter_259, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x60x7x7xf16, 60xf32, 60xf32, 60xf32, 60xf32, None) <- (-1x60x7x7xf16, 60xf32, 60xf32, 60xf32, 60xf32)
        batch_norm__258, batch_norm__259, batch_norm__260, batch_norm__261, batch_norm__262, batch_norm__263 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_49, parameter_260, parameter_261, parameter_262, parameter_263, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x60x7x7xf16) <- (-1x60x7x7xf16)
        hardswish_29 = paddle._C_ops.hardswish(batch_norm__258)

        # builtin.combine: ([-1x60x7x7xf16, -1x60x7x7xf16]) <- (-1x60x7x7xf16, -1x60x7x7xf16)
        combine_34 = [hardswish_27, hardswish_29]

        # pd_op.full: (1xi32) <- ()
        full_82 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x120x7x7xf16) <- ([-1x60x7x7xf16, -1x60x7x7xf16], 1xi32)
        concat_18 = paddle._C_ops.concat(combine_34, full_82)

        # pd_op.depthwise_conv2d: (-1x120x7x7xf16) <- (-1x120x7x7xf16, 120x1x3x3xf16)
        depthwise_conv2d_16 = paddle._C_ops.depthwise_conv2d(concat_18, parameter_264, [1, 1], [1, 1], 'EXPLICIT', 120, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x120x7x7xf16, 120xf32, 120xf32, 120xf32, 120xf32, None) <- (-1x120x7x7xf16, 120xf32, 120xf32, 120xf32, 120xf32)
        batch_norm__264, batch_norm__265, batch_norm__266, batch_norm__267, batch_norm__268, batch_norm__269 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_16, parameter_265, parameter_266, parameter_267, parameter_268, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x120x7x7xf16) <- (-1x120x7x7xf16)
        hardswish_30 = paddle._C_ops.hardswish(batch_norm__264)

        # pd_op.conv2d: (-1x120x7x7xf16) <- (-1x120x7x7xf16, 120x120x1x1xf16)
        conv2d_50 = paddle._C_ops.conv2d(hardswish_30, parameter_269, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x120x7x7xf16, 120xf32, 120xf32, 120xf32, 120xf32, None) <- (-1x120x7x7xf16, 120xf32, 120xf32, 120xf32, 120xf32)
        batch_norm__270, batch_norm__271, batch_norm__272, batch_norm__273, batch_norm__274, batch_norm__275 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_50, parameter_270, parameter_271, parameter_272, parameter_273, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x120x7x7xf16) <- (-1x120x7x7xf16)
        hardswish_31 = paddle._C_ops.hardswish(batch_norm__270)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_58 = [60, 60]

        # pd_op.full: (1xi32) <- ()
        full_83 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x60x7x7xf16, -1x60x7x7xf16]) <- (-1x120x7x7xf16, 2xi64, 1xi32)
        split_8 = paddle._C_ops.split(hardswish_31, full_int_array_58, full_83)

        # builtin.slice: (-1x60x7x7xf16) <- ([-1x60x7x7xf16, -1x60x7x7xf16])
        slice_24 = split_8[1]

        # pd_op.conv2d: (-1x60x7x7xf16) <- (-1x60x7x7xf16, 60x60x1x1xf16)
        conv2d_51 = paddle._C_ops.conv2d(slice_24, parameter_274, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x60x7x7xf16, 60xf32, 60xf32, 60xf32, 60xf32, None) <- (-1x60x7x7xf16, 60xf32, 60xf32, 60xf32, 60xf32)
        batch_norm__276, batch_norm__277, batch_norm__278, batch_norm__279, batch_norm__280, batch_norm__281 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_51, parameter_275, parameter_276, parameter_277, parameter_278, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x60x7x7xf16) <- (-1x60x7x7xf16)
        hardswish_32 = paddle._C_ops.hardswish(batch_norm__276)

        # pd_op.depthwise_conv2d: (-1x60x7x7xf16) <- (-1x60x7x7xf16, 60x1x3x3xf16)
        depthwise_conv2d_17 = paddle._C_ops.depthwise_conv2d(hardswish_32, parameter_279, [1, 1], [1, 1], 'EXPLICIT', 60, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x60x7x7xf16, 60xf32, 60xf32, 60xf32, 60xf32, None) <- (-1x60x7x7xf16, 60xf32, 60xf32, 60xf32, 60xf32)
        batch_norm__282, batch_norm__283, batch_norm__284, batch_norm__285, batch_norm__286, batch_norm__287 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_17, parameter_280, parameter_281, parameter_282, parameter_283, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # builtin.combine: ([-1x60x7x7xf16, -1x60x7x7xf16]) <- (-1x60x7x7xf16, -1x60x7x7xf16)
        combine_35 = [hardswish_32, batch_norm__282]

        # pd_op.full: (1xi32) <- ()
        full_84 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x120x7x7xf16) <- ([-1x60x7x7xf16, -1x60x7x7xf16], 1xi32)
        concat_19 = paddle._C_ops.concat(combine_35, full_84)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_59 = [1, 1]

        # pd_op.pool2d: (-1x120x1x1xf16) <- (-1x120x7x7xf16, 2xi64)
        pool2d_12 = paddle._C_ops.pool2d(concat_19, full_int_array_59, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x30x1x1xf16) <- (-1x120x1x1xf16, 30x120x1x1xf16)
        conv2d_52 = paddle._C_ops.conv2d(pool2d_12, parameter_284, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_60 = [1, 30, 1, 1]

        # pd_op.reshape: (1x30x1x1xf16, 0x30xf16) <- (30xf16, 4xi64)
        reshape_44, reshape_45 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_285, full_int_array_60), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x30x1x1xf16) <- (-1x30x1x1xf16, 1x30x1x1xf16)
        add__22 = paddle._C_ops.add_(conv2d_52, reshape_44)

        # pd_op.relu_: (-1x30x1x1xf16) <- (-1x30x1x1xf16)
        relu__11 = paddle._C_ops.relu_(add__22)

        # pd_op.conv2d: (-1x120x1x1xf16) <- (-1x30x1x1xf16, 120x30x1x1xf16)
        conv2d_53 = paddle._C_ops.conv2d(relu__11, parameter_286, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_61 = [1, 120, 1, 1]

        # pd_op.reshape: (1x120x1x1xf16, 0x120xf16) <- (120xf16, 4xi64)
        reshape_46, reshape_47 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_287, full_int_array_61), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x120x1x1xf16) <- (-1x120x1x1xf16, 1x120x1x1xf16)
        add__23 = paddle._C_ops.add_(conv2d_53, reshape_46)

        # pd_op.hardsigmoid: (-1x120x1x1xf16) <- (-1x120x1x1xf16)
        hardsigmoid_11 = paddle._C_ops.hardsigmoid(add__23, float('0.166667'), float('0.5'))

        # pd_op.multiply_: (-1x120x7x7xf16) <- (-1x120x7x7xf16, -1x120x1x1xf16)
        multiply__11 = paddle._C_ops.multiply_(concat_19, hardsigmoid_11)

        # pd_op.conv2d: (-1x60x7x7xf16) <- (-1x120x7x7xf16, 60x120x1x1xf16)
        conv2d_54 = paddle._C_ops.conv2d(multiply__11, parameter_288, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x60x7x7xf16, 60xf32, 60xf32, 60xf32, 60xf32, None) <- (-1x60x7x7xf16, 60xf32, 60xf32, 60xf32, 60xf32)
        batch_norm__288, batch_norm__289, batch_norm__290, batch_norm__291, batch_norm__292, batch_norm__293 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_54, parameter_289, parameter_290, parameter_291, parameter_292, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x60x7x7xf16) <- (-1x60x7x7xf16)
        hardswish_33 = paddle._C_ops.hardswish(batch_norm__288)

        # builtin.slice: (-1x60x7x7xf16) <- ([-1x60x7x7xf16, -1x60x7x7xf16])
        slice_25 = split_8[0]

        # builtin.combine: ([-1x60x7x7xf16, -1x60x7x7xf16]) <- (-1x60x7x7xf16, -1x60x7x7xf16)
        combine_36 = [slice_25, hardswish_33]

        # pd_op.full: (1xi32) <- ()
        full_85 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x120x7x7xf16) <- ([-1x60x7x7xf16, -1x60x7x7xf16], 1xi32)
        concat_20 = paddle._C_ops.concat(combine_36, full_85)

        # pd_op.shape: (4xi32) <- (-1x120x7x7xf16)
        shape_8 = paddle._C_ops.shape(paddle.cast(concat_20, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_62 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_63 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(shape_8, [0], full_int_array_62, full_int_array_63, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_86 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_87 = paddle._C_ops.full([1], float('60'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_88 = paddle._C_ops.full([1], float('7'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_89 = paddle._C_ops.full([1], float('7'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_37 = [slice_26, full_86, full_87, full_88, full_89]

        # pd_op.reshape_: (-1x2x60x7x7xf16, 0x-1x120x7x7xf16) <- (-1x120x7x7xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__32, reshape__33 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_20, [x.reshape([]) for x in combine_37]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x60x2x7x7xf16) <- (-1x2x60x7x7xf16)
        transpose_8 = paddle._C_ops.transpose(reshape__32, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_90 = paddle._C_ops.full([1], float('120'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_91 = paddle._C_ops.full([1], float('7'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_92 = paddle._C_ops.full([1], float('7'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_38 = [slice_26, full_90, full_91, full_92]

        # pd_op.reshape_: (-1x120x7x7xf16, 0x-1x60x2x7x7xf16) <- (-1x60x2x7x7xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__34, reshape__35 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_8, [x.reshape([]) for x in combine_38]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_64 = [60, 60]

        # pd_op.full: (1xi32) <- ()
        full_93 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([-1x60x7x7xf16, -1x60x7x7xf16]) <- (-1x120x7x7xf16, 2xi64, 1xi32)
        split_9 = paddle._C_ops.split(reshape__34, full_int_array_64, full_93)

        # builtin.slice: (-1x60x7x7xf16) <- ([-1x60x7x7xf16, -1x60x7x7xf16])
        slice_27 = split_9[1]

        # pd_op.conv2d: (-1x60x7x7xf16) <- (-1x60x7x7xf16, 60x60x1x1xf16)
        conv2d_55 = paddle._C_ops.conv2d(slice_27, parameter_293, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x60x7x7xf16, 60xf32, 60xf32, 60xf32, 60xf32, None) <- (-1x60x7x7xf16, 60xf32, 60xf32, 60xf32, 60xf32)
        batch_norm__294, batch_norm__295, batch_norm__296, batch_norm__297, batch_norm__298, batch_norm__299 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_55, parameter_294, parameter_295, parameter_296, parameter_297, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x60x7x7xf16) <- (-1x60x7x7xf16)
        hardswish_34 = paddle._C_ops.hardswish(batch_norm__294)

        # pd_op.depthwise_conv2d: (-1x60x7x7xf16) <- (-1x60x7x7xf16, 60x1x3x3xf16)
        depthwise_conv2d_18 = paddle._C_ops.depthwise_conv2d(hardswish_34, parameter_298, [1, 1], [1, 1], 'EXPLICIT', 60, [1, 1], 'NCHW')

        # pd_op.batch_norm_: (-1x60x7x7xf16, 60xf32, 60xf32, 60xf32, 60xf32, None) <- (-1x60x7x7xf16, 60xf32, 60xf32, 60xf32, 60xf32)
        batch_norm__300, batch_norm__301, batch_norm__302, batch_norm__303, batch_norm__304, batch_norm__305 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(depthwise_conv2d_18, parameter_299, parameter_300, parameter_301, parameter_302, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # builtin.combine: ([-1x60x7x7xf16, -1x60x7x7xf16]) <- (-1x60x7x7xf16, -1x60x7x7xf16)
        combine_39 = [hardswish_34, batch_norm__300]

        # pd_op.full: (1xi32) <- ()
        full_94 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x120x7x7xf16) <- ([-1x60x7x7xf16, -1x60x7x7xf16], 1xi32)
        concat_21 = paddle._C_ops.concat(combine_39, full_94)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_65 = [1, 1]

        # pd_op.pool2d: (-1x120x1x1xf16) <- (-1x120x7x7xf16, 2xi64)
        pool2d_13 = paddle._C_ops.pool2d(concat_21, full_int_array_65, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x30x1x1xf16) <- (-1x120x1x1xf16, 30x120x1x1xf16)
        conv2d_56 = paddle._C_ops.conv2d(pool2d_13, parameter_303, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_66 = [1, 30, 1, 1]

        # pd_op.reshape: (1x30x1x1xf16, 0x30xf16) <- (30xf16, 4xi64)
        reshape_48, reshape_49 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_304, full_int_array_66), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x30x1x1xf16) <- (-1x30x1x1xf16, 1x30x1x1xf16)
        add__24 = paddle._C_ops.add_(conv2d_56, reshape_48)

        # pd_op.relu_: (-1x30x1x1xf16) <- (-1x30x1x1xf16)
        relu__12 = paddle._C_ops.relu_(add__24)

        # pd_op.conv2d: (-1x120x1x1xf16) <- (-1x30x1x1xf16, 120x30x1x1xf16)
        conv2d_57 = paddle._C_ops.conv2d(relu__12, parameter_305, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_67 = [1, 120, 1, 1]

        # pd_op.reshape: (1x120x1x1xf16, 0x120xf16) <- (120xf16, 4xi64)
        reshape_50, reshape_51 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_306, full_int_array_67), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add_: (-1x120x1x1xf16) <- (-1x120x1x1xf16, 1x120x1x1xf16)
        add__25 = paddle._C_ops.add_(conv2d_57, reshape_50)

        # pd_op.hardsigmoid: (-1x120x1x1xf16) <- (-1x120x1x1xf16)
        hardsigmoid_12 = paddle._C_ops.hardsigmoid(add__25, float('0.166667'), float('0.5'))

        # pd_op.multiply_: (-1x120x7x7xf16) <- (-1x120x7x7xf16, -1x120x1x1xf16)
        multiply__12 = paddle._C_ops.multiply_(concat_21, hardsigmoid_12)

        # pd_op.conv2d: (-1x60x7x7xf16) <- (-1x120x7x7xf16, 60x120x1x1xf16)
        conv2d_58 = paddle._C_ops.conv2d(multiply__12, parameter_307, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x60x7x7xf16, 60xf32, 60xf32, 60xf32, 60xf32, None) <- (-1x60x7x7xf16, 60xf32, 60xf32, 60xf32, 60xf32)
        batch_norm__306, batch_norm__307, batch_norm__308, batch_norm__309, batch_norm__310, batch_norm__311 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_58, parameter_308, parameter_309, parameter_310, parameter_311, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x60x7x7xf16) <- (-1x60x7x7xf16)
        hardswish_35 = paddle._C_ops.hardswish(batch_norm__306)

        # builtin.slice: (-1x60x7x7xf16) <- ([-1x60x7x7xf16, -1x60x7x7xf16])
        slice_28 = split_9[0]

        # builtin.combine: ([-1x60x7x7xf16, -1x60x7x7xf16]) <- (-1x60x7x7xf16, -1x60x7x7xf16)
        combine_40 = [slice_28, hardswish_35]

        # pd_op.full: (1xi32) <- ()
        full_95 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.concat: (-1x120x7x7xf16) <- ([-1x60x7x7xf16, -1x60x7x7xf16], 1xi32)
        concat_22 = paddle._C_ops.concat(combine_40, full_95)

        # pd_op.shape: (4xi32) <- (-1x120x7x7xf16)
        shape_9 = paddle._C_ops.shape(paddle.cast(concat_22, 'float32'))

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_68 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_69 = [1]

        # pd_op.slice: (1xi32) <- (4xi32, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(shape_9, [0], full_int_array_68, full_int_array_69, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_96 = paddle._C_ops.full([1], float('2'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_97 = paddle._C_ops.full([1], float('60'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_98 = paddle._C_ops.full([1], float('7'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_99 = paddle._C_ops.full([1], float('7'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32, 1xi32)
        combine_41 = [slice_29, full_96, full_97, full_98, full_99]

        # pd_op.reshape_: (-1x2x60x7x7xf16, 0x-1x120x7x7xf16) <- (-1x120x7x7xf16, [1xi32, 1xi32, 1xi32, 1xi32, 1xi32])
        reshape__36, reshape__37 = (lambda x, f: f(x))(paddle._C_ops.reshape_(concat_22, [x.reshape([]) for x in combine_41]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x60x2x7x7xf16) <- (-1x2x60x7x7xf16)
        transpose_9 = paddle._C_ops.transpose(reshape__36, [0, 2, 1, 3, 4])

        # pd_op.full: (1xi32) <- ()
        full_100 = paddle._C_ops.full([1], float('120'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_101 = paddle._C_ops.full([1], float('7'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.full: (1xi32) <- ()
        full_102 = paddle._C_ops.full([1], float('7'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([1xi32, 1xi32, 1xi32, 1xi32]) <- (1xi32, 1xi32, 1xi32, 1xi32)
        combine_42 = [slice_29, full_100, full_101, full_102]

        # pd_op.reshape_: (-1x120x7x7xf16, 0x-1x60x2x7x7xf16) <- (-1x60x2x7x7xf16, [1xi32, 1xi32, 1xi32, 1xi32])
        reshape__38, reshape__39 = (lambda x, f: f(x))(paddle._C_ops.reshape_(transpose_9, [x.reshape([]) for x in combine_42]), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.conv2d: (-1x1024x7x7xf16) <- (-1x120x7x7xf16, 1024x120x1x1xf16)
        conv2d_59 = paddle._C_ops.conv2d(reshape__38, parameter_312, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x1024x7x7xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32, None) <- (-1x1024x7x7xf16, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        batch_norm__312, batch_norm__313, batch_norm__314, batch_norm__315, batch_norm__316, batch_norm__317 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_59, parameter_313, parameter_314, parameter_315, parameter_316, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.hardswish: (-1x1024x7x7xf16) <- (-1x1024x7x7xf16)
        hardswish_36 = paddle._C_ops.hardswish(batch_norm__312)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_70 = [1, 1]

        # pd_op.pool2d: (-1x1024x1x1xf16) <- (-1x1024x7x7xf16, 2xi64)
        pool2d_14 = paddle._C_ops.pool2d(hardswish_36, full_int_array_70, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        # pd_op.conv2d: (-1x1280x1x1xf16) <- (-1x1024x1x1xf16, 1280x1024x1x1xf16)
        conv2d_60 = paddle._C_ops.conv2d(pool2d_14, parameter_317, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.hardswish: (-1x1280x1x1xf16) <- (-1x1280x1x1xf16)
        hardswish_37 = paddle._C_ops.hardswish(conv2d_60)

        # pd_op.full: (1xf32) <- ()
        full_103 = paddle._C_ops.full([1], float('0.2'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.dropout: (-1x1280x1x1xf16, None) <- (-1x1280x1x1xf16, None, 1xf32)
        dropout_0, dropout_1 = (lambda x, f: f(x))(paddle._C_ops.dropout(hardswish_37, None, full_103, True, 'downgrade_in_infer', 0, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.flatten_: (-1x1280xf16, None) <- (-1x1280x1x1xf16)
        flatten__0, flatten__1 = (lambda x, f: f(x))(paddle._C_ops.flatten_(dropout_0, 1, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.matmul: (-1x1000xf16) <- (-1x1280xf16, 1280x1000xf16)
        matmul_0 = paddle._C_ops.matmul(flatten__0, parameter_318, False, False)

        # pd_op.add_: (-1x1000xf16) <- (-1x1000xf16, 1000xf16)
        add__26 = paddle._C_ops.add_(matmul_0, parameter_319)

        # pd_op.softmax_: (-1x1000xf16) <- (-1x1000xf16)
        softmax__0 = paddle._C_ops.softmax_(add__26, -1)

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

    def forward(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_26, parameter_27, parameter_28, parameter_29, parameter_33, parameter_30, parameter_32, parameter_31, parameter_34, parameter_38, parameter_35, parameter_37, parameter_36, parameter_39, parameter_43, parameter_40, parameter_42, parameter_41, parameter_44, parameter_48, parameter_45, parameter_47, parameter_46, parameter_49, parameter_53, parameter_50, parameter_52, parameter_51, parameter_54, parameter_55, parameter_56, parameter_57, parameter_58, parameter_62, parameter_59, parameter_61, parameter_60, parameter_63, parameter_67, parameter_64, parameter_66, parameter_65, parameter_68, parameter_72, parameter_69, parameter_71, parameter_70, parameter_73, parameter_74, parameter_75, parameter_76, parameter_77, parameter_81, parameter_78, parameter_80, parameter_79, parameter_82, parameter_86, parameter_83, parameter_85, parameter_84, parameter_87, parameter_91, parameter_88, parameter_90, parameter_89, parameter_92, parameter_96, parameter_93, parameter_95, parameter_94, parameter_97, parameter_101, parameter_98, parameter_100, parameter_99, parameter_102, parameter_103, parameter_104, parameter_105, parameter_106, parameter_110, parameter_107, parameter_109, parameter_108, parameter_111, parameter_115, parameter_112, parameter_114, parameter_113, parameter_116, parameter_120, parameter_117, parameter_119, parameter_118, parameter_121, parameter_125, parameter_122, parameter_124, parameter_123, parameter_126, parameter_130, parameter_127, parameter_129, parameter_128, parameter_131, parameter_132, parameter_133, parameter_134, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_151, parameter_152, parameter_153, parameter_154, parameter_158, parameter_155, parameter_157, parameter_156, parameter_159, parameter_163, parameter_160, parameter_162, parameter_161, parameter_164, parameter_168, parameter_165, parameter_167, parameter_166, parameter_169, parameter_170, parameter_171, parameter_172, parameter_173, parameter_177, parameter_174, parameter_176, parameter_175, parameter_178, parameter_182, parameter_179, parameter_181, parameter_180, parameter_183, parameter_187, parameter_184, parameter_186, parameter_185, parameter_188, parameter_189, parameter_190, parameter_191, parameter_192, parameter_196, parameter_193, parameter_195, parameter_194, parameter_197, parameter_201, parameter_198, parameter_200, parameter_199, parameter_202, parameter_206, parameter_203, parameter_205, parameter_204, parameter_207, parameter_208, parameter_209, parameter_210, parameter_211, parameter_215, parameter_212, parameter_214, parameter_213, parameter_216, parameter_220, parameter_217, parameter_219, parameter_218, parameter_221, parameter_225, parameter_222, parameter_224, parameter_223, parameter_226, parameter_227, parameter_228, parameter_229, parameter_230, parameter_234, parameter_231, parameter_233, parameter_232, parameter_235, parameter_239, parameter_236, parameter_238, parameter_237, parameter_240, parameter_244, parameter_241, parameter_243, parameter_242, parameter_245, parameter_249, parameter_246, parameter_248, parameter_247, parameter_250, parameter_254, parameter_251, parameter_253, parameter_252, parameter_255, parameter_256, parameter_257, parameter_258, parameter_259, parameter_263, parameter_260, parameter_262, parameter_261, parameter_264, parameter_268, parameter_265, parameter_267, parameter_266, parameter_269, parameter_273, parameter_270, parameter_272, parameter_271, parameter_274, parameter_278, parameter_275, parameter_277, parameter_276, parameter_279, parameter_283, parameter_280, parameter_282, parameter_281, parameter_284, parameter_285, parameter_286, parameter_287, parameter_288, parameter_292, parameter_289, parameter_291, parameter_290, parameter_293, parameter_297, parameter_294, parameter_296, parameter_295, parameter_298, parameter_302, parameter_299, parameter_301, parameter_300, parameter_303, parameter_304, parameter_305, parameter_306, parameter_307, parameter_311, parameter_308, parameter_310, parameter_309, parameter_312, parameter_316, parameter_313, parameter_315, parameter_314, parameter_317, parameter_318, parameter_319, feed_0):
        return self.builtin_module_932_0_0(parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_26, parameter_27, parameter_28, parameter_29, parameter_33, parameter_30, parameter_32, parameter_31, parameter_34, parameter_38, parameter_35, parameter_37, parameter_36, parameter_39, parameter_43, parameter_40, parameter_42, parameter_41, parameter_44, parameter_48, parameter_45, parameter_47, parameter_46, parameter_49, parameter_53, parameter_50, parameter_52, parameter_51, parameter_54, parameter_55, parameter_56, parameter_57, parameter_58, parameter_62, parameter_59, parameter_61, parameter_60, parameter_63, parameter_67, parameter_64, parameter_66, parameter_65, parameter_68, parameter_72, parameter_69, parameter_71, parameter_70, parameter_73, parameter_74, parameter_75, parameter_76, parameter_77, parameter_81, parameter_78, parameter_80, parameter_79, parameter_82, parameter_86, parameter_83, parameter_85, parameter_84, parameter_87, parameter_91, parameter_88, parameter_90, parameter_89, parameter_92, parameter_96, parameter_93, parameter_95, parameter_94, parameter_97, parameter_101, parameter_98, parameter_100, parameter_99, parameter_102, parameter_103, parameter_104, parameter_105, parameter_106, parameter_110, parameter_107, parameter_109, parameter_108, parameter_111, parameter_115, parameter_112, parameter_114, parameter_113, parameter_116, parameter_120, parameter_117, parameter_119, parameter_118, parameter_121, parameter_125, parameter_122, parameter_124, parameter_123, parameter_126, parameter_130, parameter_127, parameter_129, parameter_128, parameter_131, parameter_132, parameter_133, parameter_134, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_151, parameter_152, parameter_153, parameter_154, parameter_158, parameter_155, parameter_157, parameter_156, parameter_159, parameter_163, parameter_160, parameter_162, parameter_161, parameter_164, parameter_168, parameter_165, parameter_167, parameter_166, parameter_169, parameter_170, parameter_171, parameter_172, parameter_173, parameter_177, parameter_174, parameter_176, parameter_175, parameter_178, parameter_182, parameter_179, parameter_181, parameter_180, parameter_183, parameter_187, parameter_184, parameter_186, parameter_185, parameter_188, parameter_189, parameter_190, parameter_191, parameter_192, parameter_196, parameter_193, parameter_195, parameter_194, parameter_197, parameter_201, parameter_198, parameter_200, parameter_199, parameter_202, parameter_206, parameter_203, parameter_205, parameter_204, parameter_207, parameter_208, parameter_209, parameter_210, parameter_211, parameter_215, parameter_212, parameter_214, parameter_213, parameter_216, parameter_220, parameter_217, parameter_219, parameter_218, parameter_221, parameter_225, parameter_222, parameter_224, parameter_223, parameter_226, parameter_227, parameter_228, parameter_229, parameter_230, parameter_234, parameter_231, parameter_233, parameter_232, parameter_235, parameter_239, parameter_236, parameter_238, parameter_237, parameter_240, parameter_244, parameter_241, parameter_243, parameter_242, parameter_245, parameter_249, parameter_246, parameter_248, parameter_247, parameter_250, parameter_254, parameter_251, parameter_253, parameter_252, parameter_255, parameter_256, parameter_257, parameter_258, parameter_259, parameter_263, parameter_260, parameter_262, parameter_261, parameter_264, parameter_268, parameter_265, parameter_267, parameter_266, parameter_269, parameter_273, parameter_270, parameter_272, parameter_271, parameter_274, parameter_278, parameter_275, parameter_277, parameter_276, parameter_279, parameter_283, parameter_280, parameter_282, parameter_281, parameter_284, parameter_285, parameter_286, parameter_287, parameter_288, parameter_292, parameter_289, parameter_291, parameter_290, parameter_293, parameter_297, parameter_294, parameter_296, parameter_295, parameter_298, parameter_302, parameter_299, parameter_301, parameter_300, parameter_303, parameter_304, parameter_305, parameter_306, parameter_307, parameter_311, parameter_308, parameter_310, parameter_309, parameter_312, parameter_316, parameter_313, parameter_315, parameter_314, parameter_317, parameter_318, parameter_319, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_932_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([24, 3, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_4
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([24, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_9
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([24], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([16, 24, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_14
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([16, 24, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_19
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_16
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_18
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_17
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_20
            paddle.uniform([16, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_24
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_21
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_23
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_22
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_25
            paddle.uniform([4, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_26
            paddle.uniform([4], dtype='float16', min=0, max=0.5),
            # parameter_27
            paddle.uniform([16, 4, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_28
            paddle.uniform([16], dtype='float16', min=0, max=0.5),
            # parameter_29
            paddle.uniform([16, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_33
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_34
            paddle.uniform([32, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_38
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_37
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_36
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_39
            paddle.uniform([32, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_43
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_40
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_42
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_41
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_44
            paddle.uniform([16, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_48
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_45
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_49
            paddle.uniform([16, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_53
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_54
            paddle.uniform([8, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_55
            paddle.uniform([8], dtype='float16', min=0, max=0.5),
            # parameter_56
            paddle.uniform([32, 8, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_57
            paddle.uniform([32], dtype='float16', min=0, max=0.5),
            # parameter_58
            paddle.uniform([16, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_62
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_59
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([16, 16, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_67
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_64
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_66
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_68
            paddle.uniform([16, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_72
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_69
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_71
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_70
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_73
            paddle.uniform([8, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_74
            paddle.uniform([8], dtype='float16', min=0, max=0.5),
            # parameter_75
            paddle.uniform([32, 8, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_76
            paddle.uniform([32], dtype='float16', min=0, max=0.5),
            # parameter_77
            paddle.uniform([16, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_81
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_79
            paddle.uniform([16], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([32, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_86
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_84
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            # parameter_87
            paddle.uniform([28, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_91
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_88
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_90
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_89
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_92
            paddle.uniform([28, 32, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_96
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_93
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_95
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_94
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([28, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_101
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_99
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([7, 28, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_103
            paddle.uniform([7], dtype='float16', min=0, max=0.5),
            # parameter_104
            paddle.uniform([28, 7, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_105
            paddle.uniform([28], dtype='float16', min=0, max=0.5),
            # parameter_106
            paddle.uniform([28, 28, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_110
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_109
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([56, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_115
            paddle.uniform([56], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([56], dtype='float32', min=0, max=0.5),
            # parameter_114
            paddle.uniform([56], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([56], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([56, 56, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_120
            paddle.uniform([56], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([56], dtype='float32', min=0, max=0.5),
            # parameter_119
            paddle.uniform([56], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([56], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([28, 28, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_125
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_124
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([28, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_130
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_129
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_131
            paddle.uniform([14, 56, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_132
            paddle.uniform([14], dtype='float16', min=0, max=0.5),
            # parameter_133
            paddle.uniform([56, 14, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_134
            paddle.uniform([56], dtype='float16', min=0, max=0.5),
            # parameter_135
            paddle.uniform([28, 56, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_139
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_136
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_138
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_137
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_140
            paddle.uniform([28, 28, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_144
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([28, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_149
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([14, 56, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_151
            paddle.uniform([14], dtype='float16', min=0, max=0.5),
            # parameter_152
            paddle.uniform([56, 14, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_153
            paddle.uniform([56], dtype='float16', min=0, max=0.5),
            # parameter_154
            paddle.uniform([28, 56, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_158
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_155
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_157
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_156
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_159
            paddle.uniform([28, 28, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_163
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_160
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_164
            paddle.uniform([28, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_168
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_169
            paddle.uniform([14, 56, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_170
            paddle.uniform([14], dtype='float16', min=0, max=0.5),
            # parameter_171
            paddle.uniform([56, 14, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_172
            paddle.uniform([56], dtype='float16', min=0, max=0.5),
            # parameter_173
            paddle.uniform([28, 56, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_177
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_174
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([28, 28, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_182
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_179
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([28, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_187
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_184
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([14, 56, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_189
            paddle.uniform([14], dtype='float16', min=0, max=0.5),
            # parameter_190
            paddle.uniform([56, 14, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_191
            paddle.uniform([56], dtype='float16', min=0, max=0.5),
            # parameter_192
            paddle.uniform([28, 56, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_196
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_194
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([28, 28, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_201
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([28, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_206
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([14, 56, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_208
            paddle.uniform([14], dtype='float16', min=0, max=0.5),
            # parameter_209
            paddle.uniform([56, 14, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_210
            paddle.uniform([56], dtype='float16', min=0, max=0.5),
            # parameter_211
            paddle.uniform([28, 56, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_215
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_212
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_214
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_213
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_216
            paddle.uniform([28, 28, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_220
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_217
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_219
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_218
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_221
            paddle.uniform([28, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_225
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_222
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_224
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_223
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_226
            paddle.uniform([14, 56, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_227
            paddle.uniform([14], dtype='float16', min=0, max=0.5),
            # parameter_228
            paddle.uniform([56, 14, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_229
            paddle.uniform([56], dtype='float16', min=0, max=0.5),
            # parameter_230
            paddle.uniform([28, 56, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_234
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_231
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_233
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_232
            paddle.uniform([28], dtype='float32', min=0, max=0.5),
            # parameter_235
            paddle.uniform([56, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_239
            paddle.uniform([56], dtype='float32', min=0, max=0.5),
            # parameter_236
            paddle.uniform([56], dtype='float32', min=0, max=0.5),
            # parameter_238
            paddle.uniform([56], dtype='float32', min=0, max=0.5),
            # parameter_237
            paddle.uniform([56], dtype='float32', min=0, max=0.5),
            # parameter_240
            paddle.uniform([60, 56, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_244
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_241
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_243
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_242
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_245
            paddle.uniform([60, 56, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_249
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_246
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_248
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_247
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_250
            paddle.uniform([60, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_254
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_251
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_253
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_252
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_255
            paddle.uniform([15, 60, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_256
            paddle.uniform([15], dtype='float16', min=0, max=0.5),
            # parameter_257
            paddle.uniform([60, 15, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_258
            paddle.uniform([60], dtype='float16', min=0, max=0.5),
            # parameter_259
            paddle.uniform([60, 60, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_263
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_260
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_262
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_261
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_264
            paddle.uniform([120, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_268
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            # parameter_265
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            # parameter_267
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            # parameter_266
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            # parameter_269
            paddle.uniform([120, 120, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_273
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            # parameter_270
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            # parameter_272
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            # parameter_271
            paddle.uniform([120], dtype='float32', min=0, max=0.5),
            # parameter_274
            paddle.uniform([60, 60, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_278
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_275
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_277
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_276
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_279
            paddle.uniform([60, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_283
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_280
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_282
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_281
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_284
            paddle.uniform([30, 120, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_285
            paddle.uniform([30], dtype='float16', min=0, max=0.5),
            # parameter_286
            paddle.uniform([120, 30, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_287
            paddle.uniform([120], dtype='float16', min=0, max=0.5),
            # parameter_288
            paddle.uniform([60, 120, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_292
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_289
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_291
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_290
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_293
            paddle.uniform([60, 60, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_297
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_294
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_296
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_295
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_298
            paddle.uniform([60, 1, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_302
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_299
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_301
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_300
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_303
            paddle.uniform([30, 120, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_304
            paddle.uniform([30], dtype='float16', min=0, max=0.5),
            # parameter_305
            paddle.uniform([120, 30, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_306
            paddle.uniform([120], dtype='float16', min=0, max=0.5),
            # parameter_307
            paddle.uniform([60, 120, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_311
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_308
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_310
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_309
            paddle.uniform([60], dtype='float32', min=0, max=0.5),
            # parameter_312
            paddle.uniform([1024, 120, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_316
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_313
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_315
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_314
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_317
            paddle.uniform([1280, 1024, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_318
            paddle.uniform([1280, 1000], dtype='float16', min=0, max=0.5),
            # parameter_319
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
            paddle.static.InputSpec(shape=[24, 3, 3, 3], dtype='float16'),
            # parameter_4
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[24, 1, 3, 3], dtype='float16'),
            # parameter_9
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[16, 24, 1, 1], dtype='float16'),
            # parameter_14
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[16, 24, 1, 1], dtype='float16'),
            # parameter_19
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_16
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_18
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_17
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_20
            paddle.static.InputSpec(shape=[16, 1, 3, 3], dtype='float16'),
            # parameter_24
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_21
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_23
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_22
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_25
            paddle.static.InputSpec(shape=[4, 16, 1, 1], dtype='float16'),
            # parameter_26
            paddle.static.InputSpec(shape=[4], dtype='float16'),
            # parameter_27
            paddle.static.InputSpec(shape=[16, 4, 1, 1], dtype='float16'),
            # parameter_28
            paddle.static.InputSpec(shape=[16], dtype='float16'),
            # parameter_29
            paddle.static.InputSpec(shape=[16, 16, 1, 1], dtype='float16'),
            # parameter_33
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_34
            paddle.static.InputSpec(shape=[32, 1, 3, 3], dtype='float16'),
            # parameter_38
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_37
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_36
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_39
            paddle.static.InputSpec(shape=[32, 32, 1, 1], dtype='float16'),
            # parameter_43
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_40
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_42
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_41
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_44
            paddle.static.InputSpec(shape=[16, 16, 1, 1], dtype='float16'),
            # parameter_48
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_45
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_49
            paddle.static.InputSpec(shape=[16, 1, 3, 3], dtype='float16'),
            # parameter_53
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_54
            paddle.static.InputSpec(shape=[8, 32, 1, 1], dtype='float16'),
            # parameter_55
            paddle.static.InputSpec(shape=[8], dtype='float16'),
            # parameter_56
            paddle.static.InputSpec(shape=[32, 8, 1, 1], dtype='float16'),
            # parameter_57
            paddle.static.InputSpec(shape=[32], dtype='float16'),
            # parameter_58
            paddle.static.InputSpec(shape=[16, 32, 1, 1], dtype='float16'),
            # parameter_62
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_59
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[16, 16, 1, 1], dtype='float16'),
            # parameter_67
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_64
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_66
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_68
            paddle.static.InputSpec(shape=[16, 1, 3, 3], dtype='float16'),
            # parameter_72
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_69
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_71
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_70
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_73
            paddle.static.InputSpec(shape=[8, 32, 1, 1], dtype='float16'),
            # parameter_74
            paddle.static.InputSpec(shape=[8], dtype='float16'),
            # parameter_75
            paddle.static.InputSpec(shape=[32, 8, 1, 1], dtype='float16'),
            # parameter_76
            paddle.static.InputSpec(shape=[32], dtype='float16'),
            # parameter_77
            paddle.static.InputSpec(shape=[16, 32, 1, 1], dtype='float16'),
            # parameter_81
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_79
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[32, 1, 3, 3], dtype='float16'),
            # parameter_86
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_84
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            # parameter_87
            paddle.static.InputSpec(shape=[28, 32, 1, 1], dtype='float16'),
            # parameter_91
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_88
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_90
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_89
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_92
            paddle.static.InputSpec(shape=[28, 32, 1, 1], dtype='float16'),
            # parameter_96
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_93
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_95
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_94
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[28, 1, 3, 3], dtype='float16'),
            # parameter_101
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_99
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[7, 28, 1, 1], dtype='float16'),
            # parameter_103
            paddle.static.InputSpec(shape=[7], dtype='float16'),
            # parameter_104
            paddle.static.InputSpec(shape=[28, 7, 1, 1], dtype='float16'),
            # parameter_105
            paddle.static.InputSpec(shape=[28], dtype='float16'),
            # parameter_106
            paddle.static.InputSpec(shape=[28, 28, 1, 1], dtype='float16'),
            # parameter_110
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_109
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[56, 1, 3, 3], dtype='float16'),
            # parameter_115
            paddle.static.InputSpec(shape=[56], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[56], dtype='float32'),
            # parameter_114
            paddle.static.InputSpec(shape=[56], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[56], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[56, 56, 1, 1], dtype='float16'),
            # parameter_120
            paddle.static.InputSpec(shape=[56], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[56], dtype='float32'),
            # parameter_119
            paddle.static.InputSpec(shape=[56], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[56], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[28, 28, 1, 1], dtype='float16'),
            # parameter_125
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_124
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[28, 1, 3, 3], dtype='float16'),
            # parameter_130
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_129
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_131
            paddle.static.InputSpec(shape=[14, 56, 1, 1], dtype='float16'),
            # parameter_132
            paddle.static.InputSpec(shape=[14], dtype='float16'),
            # parameter_133
            paddle.static.InputSpec(shape=[56, 14, 1, 1], dtype='float16'),
            # parameter_134
            paddle.static.InputSpec(shape=[56], dtype='float16'),
            # parameter_135
            paddle.static.InputSpec(shape=[28, 56, 1, 1], dtype='float16'),
            # parameter_139
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_136
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_138
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_137
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_140
            paddle.static.InputSpec(shape=[28, 28, 1, 1], dtype='float16'),
            # parameter_144
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[28, 1, 3, 3], dtype='float16'),
            # parameter_149
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[14, 56, 1, 1], dtype='float16'),
            # parameter_151
            paddle.static.InputSpec(shape=[14], dtype='float16'),
            # parameter_152
            paddle.static.InputSpec(shape=[56, 14, 1, 1], dtype='float16'),
            # parameter_153
            paddle.static.InputSpec(shape=[56], dtype='float16'),
            # parameter_154
            paddle.static.InputSpec(shape=[28, 56, 1, 1], dtype='float16'),
            # parameter_158
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_155
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_157
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_156
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_159
            paddle.static.InputSpec(shape=[28, 28, 1, 1], dtype='float16'),
            # parameter_163
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_160
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_164
            paddle.static.InputSpec(shape=[28, 1, 3, 3], dtype='float16'),
            # parameter_168
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_169
            paddle.static.InputSpec(shape=[14, 56, 1, 1], dtype='float16'),
            # parameter_170
            paddle.static.InputSpec(shape=[14], dtype='float16'),
            # parameter_171
            paddle.static.InputSpec(shape=[56, 14, 1, 1], dtype='float16'),
            # parameter_172
            paddle.static.InputSpec(shape=[56], dtype='float16'),
            # parameter_173
            paddle.static.InputSpec(shape=[28, 56, 1, 1], dtype='float16'),
            # parameter_177
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_174
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[28, 28, 1, 1], dtype='float16'),
            # parameter_182
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_179
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[28, 1, 3, 3], dtype='float16'),
            # parameter_187
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_184
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[14, 56, 1, 1], dtype='float16'),
            # parameter_189
            paddle.static.InputSpec(shape=[14], dtype='float16'),
            # parameter_190
            paddle.static.InputSpec(shape=[56, 14, 1, 1], dtype='float16'),
            # parameter_191
            paddle.static.InputSpec(shape=[56], dtype='float16'),
            # parameter_192
            paddle.static.InputSpec(shape=[28, 56, 1, 1], dtype='float16'),
            # parameter_196
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_194
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[28, 28, 1, 1], dtype='float16'),
            # parameter_201
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[28, 1, 3, 3], dtype='float16'),
            # parameter_206
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[14, 56, 1, 1], dtype='float16'),
            # parameter_208
            paddle.static.InputSpec(shape=[14], dtype='float16'),
            # parameter_209
            paddle.static.InputSpec(shape=[56, 14, 1, 1], dtype='float16'),
            # parameter_210
            paddle.static.InputSpec(shape=[56], dtype='float16'),
            # parameter_211
            paddle.static.InputSpec(shape=[28, 56, 1, 1], dtype='float16'),
            # parameter_215
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_212
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_214
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_213
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_216
            paddle.static.InputSpec(shape=[28, 28, 1, 1], dtype='float16'),
            # parameter_220
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_217
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_219
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_218
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_221
            paddle.static.InputSpec(shape=[28, 1, 3, 3], dtype='float16'),
            # parameter_225
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_222
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_224
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_223
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_226
            paddle.static.InputSpec(shape=[14, 56, 1, 1], dtype='float16'),
            # parameter_227
            paddle.static.InputSpec(shape=[14], dtype='float16'),
            # parameter_228
            paddle.static.InputSpec(shape=[56, 14, 1, 1], dtype='float16'),
            # parameter_229
            paddle.static.InputSpec(shape=[56], dtype='float16'),
            # parameter_230
            paddle.static.InputSpec(shape=[28, 56, 1, 1], dtype='float16'),
            # parameter_234
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_231
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_233
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_232
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            # parameter_235
            paddle.static.InputSpec(shape=[56, 1, 3, 3], dtype='float16'),
            # parameter_239
            paddle.static.InputSpec(shape=[56], dtype='float32'),
            # parameter_236
            paddle.static.InputSpec(shape=[56], dtype='float32'),
            # parameter_238
            paddle.static.InputSpec(shape=[56], dtype='float32'),
            # parameter_237
            paddle.static.InputSpec(shape=[56], dtype='float32'),
            # parameter_240
            paddle.static.InputSpec(shape=[60, 56, 1, 1], dtype='float16'),
            # parameter_244
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_241
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_243
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_242
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_245
            paddle.static.InputSpec(shape=[60, 56, 1, 1], dtype='float16'),
            # parameter_249
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_246
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_248
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_247
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_250
            paddle.static.InputSpec(shape=[60, 1, 3, 3], dtype='float16'),
            # parameter_254
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_251
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_253
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_252
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_255
            paddle.static.InputSpec(shape=[15, 60, 1, 1], dtype='float16'),
            # parameter_256
            paddle.static.InputSpec(shape=[15], dtype='float16'),
            # parameter_257
            paddle.static.InputSpec(shape=[60, 15, 1, 1], dtype='float16'),
            # parameter_258
            paddle.static.InputSpec(shape=[60], dtype='float16'),
            # parameter_259
            paddle.static.InputSpec(shape=[60, 60, 1, 1], dtype='float16'),
            # parameter_263
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_260
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_262
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_261
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_264
            paddle.static.InputSpec(shape=[120, 1, 3, 3], dtype='float16'),
            # parameter_268
            paddle.static.InputSpec(shape=[120], dtype='float32'),
            # parameter_265
            paddle.static.InputSpec(shape=[120], dtype='float32'),
            # parameter_267
            paddle.static.InputSpec(shape=[120], dtype='float32'),
            # parameter_266
            paddle.static.InputSpec(shape=[120], dtype='float32'),
            # parameter_269
            paddle.static.InputSpec(shape=[120, 120, 1, 1], dtype='float16'),
            # parameter_273
            paddle.static.InputSpec(shape=[120], dtype='float32'),
            # parameter_270
            paddle.static.InputSpec(shape=[120], dtype='float32'),
            # parameter_272
            paddle.static.InputSpec(shape=[120], dtype='float32'),
            # parameter_271
            paddle.static.InputSpec(shape=[120], dtype='float32'),
            # parameter_274
            paddle.static.InputSpec(shape=[60, 60, 1, 1], dtype='float16'),
            # parameter_278
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_275
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_277
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_276
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_279
            paddle.static.InputSpec(shape=[60, 1, 3, 3], dtype='float16'),
            # parameter_283
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_280
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_282
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_281
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_284
            paddle.static.InputSpec(shape=[30, 120, 1, 1], dtype='float16'),
            # parameter_285
            paddle.static.InputSpec(shape=[30], dtype='float16'),
            # parameter_286
            paddle.static.InputSpec(shape=[120, 30, 1, 1], dtype='float16'),
            # parameter_287
            paddle.static.InputSpec(shape=[120], dtype='float16'),
            # parameter_288
            paddle.static.InputSpec(shape=[60, 120, 1, 1], dtype='float16'),
            # parameter_292
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_289
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_291
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_290
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_293
            paddle.static.InputSpec(shape=[60, 60, 1, 1], dtype='float16'),
            # parameter_297
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_294
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_296
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_295
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_298
            paddle.static.InputSpec(shape=[60, 1, 3, 3], dtype='float16'),
            # parameter_302
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_299
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_301
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_300
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_303
            paddle.static.InputSpec(shape=[30, 120, 1, 1], dtype='float16'),
            # parameter_304
            paddle.static.InputSpec(shape=[30], dtype='float16'),
            # parameter_305
            paddle.static.InputSpec(shape=[120, 30, 1, 1], dtype='float16'),
            # parameter_306
            paddle.static.InputSpec(shape=[120], dtype='float16'),
            # parameter_307
            paddle.static.InputSpec(shape=[60, 120, 1, 1], dtype='float16'),
            # parameter_311
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_308
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_310
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_309
            paddle.static.InputSpec(shape=[60], dtype='float32'),
            # parameter_312
            paddle.static.InputSpec(shape=[1024, 120, 1, 1], dtype='float16'),
            # parameter_316
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_313
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_315
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_314
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_317
            paddle.static.InputSpec(shape=[1280, 1024, 1, 1], dtype='float16'),
            # parameter_318
            paddle.static.InputSpec(shape=[1280, 1000], dtype='float16'),
            # parameter_319
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