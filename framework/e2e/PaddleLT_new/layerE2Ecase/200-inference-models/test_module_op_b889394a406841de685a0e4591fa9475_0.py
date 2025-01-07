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
    return [159][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_379_0_0(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_114, parameter_111, parameter_113, parameter_112, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_154, parameter_151, parameter_153, parameter_152, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_169, parameter_166, parameter_168, parameter_167, parameter_170, parameter_174, parameter_171, parameter_173, parameter_172, parameter_175, parameter_179, parameter_176, parameter_178, parameter_177, parameter_180, parameter_184, parameter_181, parameter_183, parameter_182, parameter_185, parameter_189, parameter_186, parameter_188, parameter_187, parameter_190, parameter_194, parameter_191, parameter_193, parameter_192, parameter_195, parameter_196, parameter_197, parameter_198, parameter_199, parameter_200, parameter_201, parameter_202, parameter_203, parameter_204, parameter_205, parameter_206, parameter_207, parameter_208, parameter_209, parameter_210, parameter_211, parameter_212, feed_0):

        # pd_op.cast: (-1x3x32x-1xf16) <- (-1x3x32x-1xf32)
        cast_0 = paddle._C_ops.cast(feed_0, paddle.float16)

        # pd_op.conv2d: (-1x32x32x-1xf16) <- (-1x3x32x-1xf16, 32x3x3x3xf16)
        conv2d_0 = paddle._C_ops.conv2d(cast_0, parameter_0, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x32x-1xf16, 32xf32, 32xf32, 32xf32, 32xf32, None) <- (-1x32x32x-1xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__0, batch_norm__1, batch_norm__2, batch_norm__3, batch_norm__4, batch_norm__5 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_0, parameter_1, parameter_2, parameter_3, parameter_4, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x32x32x-1xf16) <- (-1x32x32x-1xf16)
        relu_0 = paddle._C_ops.relu(batch_norm__0)

        # pd_op.conv2d: (-1x32x32x-1xf16) <- (-1x32x32x-1xf16, 32x32x3x3xf16)
        conv2d_1 = paddle._C_ops.conv2d(relu_0, parameter_5, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x32x32x-1xf16, 32xf32, 32xf32, 32xf32, 32xf32, None) <- (-1x32x32x-1xf16, 32xf32, 32xf32, 32xf32, 32xf32)
        batch_norm__6, batch_norm__7, batch_norm__8, batch_norm__9, batch_norm__10, batch_norm__11 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_1, parameter_6, parameter_7, parameter_8, parameter_9, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x32x32x-1xf16) <- (-1x32x32x-1xf16)
        relu_1 = paddle._C_ops.relu(batch_norm__6)

        # pd_op.conv2d: (-1x64x32x-1xf16) <- (-1x32x32x-1xf16, 64x32x3x3xf16)
        conv2d_2 = paddle._C_ops.conv2d(relu_1, parameter_10, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x32x-1xf16, 64xf32, 64xf32, 64xf32, 64xf32, None) <- (-1x64x32x-1xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__12, batch_norm__13, batch_norm__14, batch_norm__15, batch_norm__16, batch_norm__17 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_2, parameter_11, parameter_12, parameter_13, parameter_14, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x64x32x-1xf16) <- (-1x64x32x-1xf16)
        relu_2 = paddle._C_ops.relu(batch_norm__12)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [3, 3]

        # pd_op.pool2d: (-1x64x16x-1xf16) <- (-1x64x32x-1xf16, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(relu_2, full_int_array_0, [2, 2], [1, 1], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x64x16x-1xf16) <- (-1x64x16x-1xf16, 64x64x3x3xf16)
        conv2d_3 = paddle._C_ops.conv2d(pool2d_0, parameter_15, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x16x-1xf16, 64xf32, 64xf32, 64xf32, 64xf32, None) <- (-1x64x16x-1xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__18, batch_norm__19, batch_norm__20, batch_norm__21, batch_norm__22, batch_norm__23 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_3, parameter_16, parameter_17, parameter_18, parameter_19, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x64x16x-1xf16) <- (-1x64x16x-1xf16)
        relu_3 = paddle._C_ops.relu(batch_norm__18)

        # pd_op.conv2d: (-1x64x16x-1xf16) <- (-1x64x16x-1xf16, 64x64x3x3xf16)
        conv2d_4 = paddle._C_ops.conv2d(relu_3, parameter_20, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x16x-1xf16, 64xf32, 64xf32, 64xf32, 64xf32, None) <- (-1x64x16x-1xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__24, batch_norm__25, batch_norm__26, batch_norm__27, batch_norm__28, batch_norm__29 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_4, parameter_21, parameter_22, parameter_23, parameter_24, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.conv2d: (-1x64x16x-1xf16) <- (-1x64x16x-1xf16, 64x64x1x1xf16)
        conv2d_5 = paddle._C_ops.conv2d(pool2d_0, parameter_25, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x16x-1xf16, 64xf32, 64xf32, 64xf32, 64xf32, None) <- (-1x64x16x-1xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__30, batch_norm__31, batch_norm__32, batch_norm__33, batch_norm__34, batch_norm__35 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_5, parameter_26, parameter_27, parameter_28, parameter_29, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x64x16x-1xf16) <- (-1x64x16x-1xf16, -1x64x16x-1xf16)
        add_0 = paddle._C_ops.add(batch_norm__30, batch_norm__24)

        # pd_op.relu: (-1x64x16x-1xf16) <- (-1x64x16x-1xf16)
        relu_4 = paddle._C_ops.relu(add_0)

        # pd_op.conv2d: (-1x64x16x-1xf16) <- (-1x64x16x-1xf16, 64x64x3x3xf16)
        conv2d_6 = paddle._C_ops.conv2d(relu_4, parameter_30, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x16x-1xf16, 64xf32, 64xf32, 64xf32, 64xf32, None) <- (-1x64x16x-1xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__36, batch_norm__37, batch_norm__38, batch_norm__39, batch_norm__40, batch_norm__41 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_6, parameter_31, parameter_32, parameter_33, parameter_34, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x64x16x-1xf16) <- (-1x64x16x-1xf16)
        relu_5 = paddle._C_ops.relu(batch_norm__36)

        # pd_op.conv2d: (-1x64x16x-1xf16) <- (-1x64x16x-1xf16, 64x64x3x3xf16)
        conv2d_7 = paddle._C_ops.conv2d(relu_5, parameter_35, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x16x-1xf16, 64xf32, 64xf32, 64xf32, 64xf32, None) <- (-1x64x16x-1xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__42, batch_norm__43, batch_norm__44, batch_norm__45, batch_norm__46, batch_norm__47 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_7, parameter_36, parameter_37, parameter_38, parameter_39, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x64x16x-1xf16) <- (-1x64x16x-1xf16, -1x64x16x-1xf16)
        add_1 = paddle._C_ops.add(relu_4, batch_norm__42)

        # pd_op.relu: (-1x64x16x-1xf16) <- (-1x64x16x-1xf16)
        relu_6 = paddle._C_ops.relu(add_1)

        # pd_op.conv2d: (-1x64x16x-1xf16) <- (-1x64x16x-1xf16, 64x64x3x3xf16)
        conv2d_8 = paddle._C_ops.conv2d(relu_6, parameter_40, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x16x-1xf16, 64xf32, 64xf32, 64xf32, 64xf32, None) <- (-1x64x16x-1xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__48, batch_norm__49, batch_norm__50, batch_norm__51, batch_norm__52, batch_norm__53 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_8, parameter_41, parameter_42, parameter_43, parameter_44, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x64x16x-1xf16) <- (-1x64x16x-1xf16)
        relu_7 = paddle._C_ops.relu(batch_norm__48)

        # pd_op.conv2d: (-1x64x16x-1xf16) <- (-1x64x16x-1xf16, 64x64x3x3xf16)
        conv2d_9 = paddle._C_ops.conv2d(relu_7, parameter_45, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x64x16x-1xf16, 64xf32, 64xf32, 64xf32, 64xf32, None) <- (-1x64x16x-1xf16, 64xf32, 64xf32, 64xf32, 64xf32)
        batch_norm__54, batch_norm__55, batch_norm__56, batch_norm__57, batch_norm__58, batch_norm__59 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_9, parameter_46, parameter_47, parameter_48, parameter_49, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x64x16x-1xf16) <- (-1x64x16x-1xf16, -1x64x16x-1xf16)
        add_2 = paddle._C_ops.add(relu_6, batch_norm__54)

        # pd_op.relu: (-1x64x16x-1xf16) <- (-1x64x16x-1xf16)
        relu_8 = paddle._C_ops.relu(add_2)

        # pd_op.conv2d: (-1x128x8x-1xf16) <- (-1x64x16x-1xf16, 128x64x3x3xf16)
        conv2d_10 = paddle._C_ops.conv2d(relu_8, parameter_50, [2, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x8x-1xf16, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x8x-1xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__60, batch_norm__61, batch_norm__62, batch_norm__63, batch_norm__64, batch_norm__65 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_10, parameter_51, parameter_52, parameter_53, parameter_54, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x128x8x-1xf16) <- (-1x128x8x-1xf16)
        relu_9 = paddle._C_ops.relu(batch_norm__60)

        # pd_op.conv2d: (-1x128x8x-1xf16) <- (-1x128x8x-1xf16, 128x128x3x3xf16)
        conv2d_11 = paddle._C_ops.conv2d(relu_9, parameter_55, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x8x-1xf16, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x8x-1xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__66, batch_norm__67, batch_norm__68, batch_norm__69, batch_norm__70, batch_norm__71 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_11, parameter_56, parameter_57, parameter_58, parameter_59, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [2, 1]

        # pd_op.pool2d: (-1x64x8x-1xf16) <- (-1x64x16x-1xf16, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(relu_8, full_int_array_1, [2, 1], [0, 0], True, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x128x8x-1xf16) <- (-1x64x8x-1xf16, 128x64x1x1xf16)
        conv2d_12 = paddle._C_ops.conv2d(pool2d_1, parameter_60, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x8x-1xf16, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x8x-1xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__72, batch_norm__73, batch_norm__74, batch_norm__75, batch_norm__76, batch_norm__77 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_12, parameter_61, parameter_62, parameter_63, parameter_64, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x128x8x-1xf16) <- (-1x128x8x-1xf16, -1x128x8x-1xf16)
        add_3 = paddle._C_ops.add(batch_norm__72, batch_norm__66)

        # pd_op.relu: (-1x128x8x-1xf16) <- (-1x128x8x-1xf16)
        relu_10 = paddle._C_ops.relu(add_3)

        # pd_op.conv2d: (-1x128x8x-1xf16) <- (-1x128x8x-1xf16, 128x128x3x3xf16)
        conv2d_13 = paddle._C_ops.conv2d(relu_10, parameter_65, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x8x-1xf16, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x8x-1xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__78, batch_norm__79, batch_norm__80, batch_norm__81, batch_norm__82, batch_norm__83 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_13, parameter_66, parameter_67, parameter_68, parameter_69, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x128x8x-1xf16) <- (-1x128x8x-1xf16)
        relu_11 = paddle._C_ops.relu(batch_norm__78)

        # pd_op.conv2d: (-1x128x8x-1xf16) <- (-1x128x8x-1xf16, 128x128x3x3xf16)
        conv2d_14 = paddle._C_ops.conv2d(relu_11, parameter_70, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x8x-1xf16, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x8x-1xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__84, batch_norm__85, batch_norm__86, batch_norm__87, batch_norm__88, batch_norm__89 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_14, parameter_71, parameter_72, parameter_73, parameter_74, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x128x8x-1xf16) <- (-1x128x8x-1xf16, -1x128x8x-1xf16)
        add_4 = paddle._C_ops.add(relu_10, batch_norm__84)

        # pd_op.relu: (-1x128x8x-1xf16) <- (-1x128x8x-1xf16)
        relu_12 = paddle._C_ops.relu(add_4)

        # pd_op.conv2d: (-1x128x8x-1xf16) <- (-1x128x8x-1xf16, 128x128x3x3xf16)
        conv2d_15 = paddle._C_ops.conv2d(relu_12, parameter_75, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x8x-1xf16, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x8x-1xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__90, batch_norm__91, batch_norm__92, batch_norm__93, batch_norm__94, batch_norm__95 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_15, parameter_76, parameter_77, parameter_78, parameter_79, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x128x8x-1xf16) <- (-1x128x8x-1xf16)
        relu_13 = paddle._C_ops.relu(batch_norm__90)

        # pd_op.conv2d: (-1x128x8x-1xf16) <- (-1x128x8x-1xf16, 128x128x3x3xf16)
        conv2d_16 = paddle._C_ops.conv2d(relu_13, parameter_80, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x8x-1xf16, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x8x-1xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__96, batch_norm__97, batch_norm__98, batch_norm__99, batch_norm__100, batch_norm__101 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_16, parameter_81, parameter_82, parameter_83, parameter_84, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x128x8x-1xf16) <- (-1x128x8x-1xf16, -1x128x8x-1xf16)
        add_5 = paddle._C_ops.add(relu_12, batch_norm__96)

        # pd_op.relu: (-1x128x8x-1xf16) <- (-1x128x8x-1xf16)
        relu_14 = paddle._C_ops.relu(add_5)

        # pd_op.conv2d: (-1x128x8x-1xf16) <- (-1x128x8x-1xf16, 128x128x3x3xf16)
        conv2d_17 = paddle._C_ops.conv2d(relu_14, parameter_85, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x8x-1xf16, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x8x-1xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__102, batch_norm__103, batch_norm__104, batch_norm__105, batch_norm__106, batch_norm__107 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_17, parameter_86, parameter_87, parameter_88, parameter_89, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x128x8x-1xf16) <- (-1x128x8x-1xf16)
        relu_15 = paddle._C_ops.relu(batch_norm__102)

        # pd_op.conv2d: (-1x128x8x-1xf16) <- (-1x128x8x-1xf16, 128x128x3x3xf16)
        conv2d_18 = paddle._C_ops.conv2d(relu_15, parameter_90, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x128x8x-1xf16, 128xf32, 128xf32, 128xf32, 128xf32, None) <- (-1x128x8x-1xf16, 128xf32, 128xf32, 128xf32, 128xf32)
        batch_norm__108, batch_norm__109, batch_norm__110, batch_norm__111, batch_norm__112, batch_norm__113 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_18, parameter_91, parameter_92, parameter_93, parameter_94, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x128x8x-1xf16) <- (-1x128x8x-1xf16, -1x128x8x-1xf16)
        add_6 = paddle._C_ops.add(relu_14, batch_norm__108)

        # pd_op.relu: (-1x128x8x-1xf16) <- (-1x128x8x-1xf16)
        relu_16 = paddle._C_ops.relu(add_6)

        # pd_op.conv2d: (-1x256x4x-1xf16) <- (-1x128x8x-1xf16, 256x128x3x3xf16)
        conv2d_19 = paddle._C_ops.conv2d(relu_16, parameter_95, [2, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x4x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x4x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__114, batch_norm__115, batch_norm__116, batch_norm__117, batch_norm__118, batch_norm__119 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_19, parameter_96, parameter_97, parameter_98, parameter_99, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x256x4x-1xf16) <- (-1x256x4x-1xf16)
        relu_17 = paddle._C_ops.relu(batch_norm__114)

        # pd_op.conv2d: (-1x256x4x-1xf16) <- (-1x256x4x-1xf16, 256x256x3x3xf16)
        conv2d_20 = paddle._C_ops.conv2d(relu_17, parameter_100, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x4x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x4x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__120, batch_norm__121, batch_norm__122, batch_norm__123, batch_norm__124, batch_norm__125 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_20, parameter_101, parameter_102, parameter_103, parameter_104, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [2, 1]

        # pd_op.pool2d: (-1x128x4x-1xf16) <- (-1x128x8x-1xf16, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(relu_16, full_int_array_2, [2, 1], [0, 0], True, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x256x4x-1xf16) <- (-1x128x4x-1xf16, 256x128x1x1xf16)
        conv2d_21 = paddle._C_ops.conv2d(pool2d_2, parameter_105, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x4x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x4x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__126, batch_norm__127, batch_norm__128, batch_norm__129, batch_norm__130, batch_norm__131 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_21, parameter_106, parameter_107, parameter_108, parameter_109, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x256x4x-1xf16) <- (-1x256x4x-1xf16, -1x256x4x-1xf16)
        add_7 = paddle._C_ops.add(batch_norm__126, batch_norm__120)

        # pd_op.relu: (-1x256x4x-1xf16) <- (-1x256x4x-1xf16)
        relu_18 = paddle._C_ops.relu(add_7)

        # pd_op.conv2d: (-1x256x4x-1xf16) <- (-1x256x4x-1xf16, 256x256x3x3xf16)
        conv2d_22 = paddle._C_ops.conv2d(relu_18, parameter_110, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x4x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x4x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__132, batch_norm__133, batch_norm__134, batch_norm__135, batch_norm__136, batch_norm__137 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_22, parameter_111, parameter_112, parameter_113, parameter_114, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x256x4x-1xf16) <- (-1x256x4x-1xf16)
        relu_19 = paddle._C_ops.relu(batch_norm__132)

        # pd_op.conv2d: (-1x256x4x-1xf16) <- (-1x256x4x-1xf16, 256x256x3x3xf16)
        conv2d_23 = paddle._C_ops.conv2d(relu_19, parameter_115, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x4x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x4x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__138, batch_norm__139, batch_norm__140, batch_norm__141, batch_norm__142, batch_norm__143 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_23, parameter_116, parameter_117, parameter_118, parameter_119, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x256x4x-1xf16) <- (-1x256x4x-1xf16, -1x256x4x-1xf16)
        add_8 = paddle._C_ops.add(relu_18, batch_norm__138)

        # pd_op.relu: (-1x256x4x-1xf16) <- (-1x256x4x-1xf16)
        relu_20 = paddle._C_ops.relu(add_8)

        # pd_op.conv2d: (-1x256x4x-1xf16) <- (-1x256x4x-1xf16, 256x256x3x3xf16)
        conv2d_24 = paddle._C_ops.conv2d(relu_20, parameter_120, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x4x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x4x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__144, batch_norm__145, batch_norm__146, batch_norm__147, batch_norm__148, batch_norm__149 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_24, parameter_121, parameter_122, parameter_123, parameter_124, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x256x4x-1xf16) <- (-1x256x4x-1xf16)
        relu_21 = paddle._C_ops.relu(batch_norm__144)

        # pd_op.conv2d: (-1x256x4x-1xf16) <- (-1x256x4x-1xf16, 256x256x3x3xf16)
        conv2d_25 = paddle._C_ops.conv2d(relu_21, parameter_125, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x4x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x4x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__150, batch_norm__151, batch_norm__152, batch_norm__153, batch_norm__154, batch_norm__155 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_25, parameter_126, parameter_127, parameter_128, parameter_129, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x256x4x-1xf16) <- (-1x256x4x-1xf16, -1x256x4x-1xf16)
        add_9 = paddle._C_ops.add(relu_20, batch_norm__150)

        # pd_op.relu: (-1x256x4x-1xf16) <- (-1x256x4x-1xf16)
        relu_22 = paddle._C_ops.relu(add_9)

        # pd_op.conv2d: (-1x256x4x-1xf16) <- (-1x256x4x-1xf16, 256x256x3x3xf16)
        conv2d_26 = paddle._C_ops.conv2d(relu_22, parameter_130, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x4x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x4x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__156, batch_norm__157, batch_norm__158, batch_norm__159, batch_norm__160, batch_norm__161 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_26, parameter_131, parameter_132, parameter_133, parameter_134, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x256x4x-1xf16) <- (-1x256x4x-1xf16)
        relu_23 = paddle._C_ops.relu(batch_norm__156)

        # pd_op.conv2d: (-1x256x4x-1xf16) <- (-1x256x4x-1xf16, 256x256x3x3xf16)
        conv2d_27 = paddle._C_ops.conv2d(relu_23, parameter_135, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x4x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x4x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__162, batch_norm__163, batch_norm__164, batch_norm__165, batch_norm__166, batch_norm__167 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_27, parameter_136, parameter_137, parameter_138, parameter_139, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x256x4x-1xf16) <- (-1x256x4x-1xf16, -1x256x4x-1xf16)
        add_10 = paddle._C_ops.add(relu_22, batch_norm__162)

        # pd_op.relu: (-1x256x4x-1xf16) <- (-1x256x4x-1xf16)
        relu_24 = paddle._C_ops.relu(add_10)

        # pd_op.conv2d: (-1x256x4x-1xf16) <- (-1x256x4x-1xf16, 256x256x3x3xf16)
        conv2d_28 = paddle._C_ops.conv2d(relu_24, parameter_140, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x4x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x4x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__168, batch_norm__169, batch_norm__170, batch_norm__171, batch_norm__172, batch_norm__173 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_28, parameter_141, parameter_142, parameter_143, parameter_144, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x256x4x-1xf16) <- (-1x256x4x-1xf16)
        relu_25 = paddle._C_ops.relu(batch_norm__168)

        # pd_op.conv2d: (-1x256x4x-1xf16) <- (-1x256x4x-1xf16, 256x256x3x3xf16)
        conv2d_29 = paddle._C_ops.conv2d(relu_25, parameter_145, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x4x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x4x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__174, batch_norm__175, batch_norm__176, batch_norm__177, batch_norm__178, batch_norm__179 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_29, parameter_146, parameter_147, parameter_148, parameter_149, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x256x4x-1xf16) <- (-1x256x4x-1xf16, -1x256x4x-1xf16)
        add_11 = paddle._C_ops.add(relu_24, batch_norm__174)

        # pd_op.relu: (-1x256x4x-1xf16) <- (-1x256x4x-1xf16)
        relu_26 = paddle._C_ops.relu(add_11)

        # pd_op.conv2d: (-1x256x4x-1xf16) <- (-1x256x4x-1xf16, 256x256x3x3xf16)
        conv2d_30 = paddle._C_ops.conv2d(relu_26, parameter_150, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x4x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x4x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__180, batch_norm__181, batch_norm__182, batch_norm__183, batch_norm__184, batch_norm__185 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_30, parameter_151, parameter_152, parameter_153, parameter_154, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x256x4x-1xf16) <- (-1x256x4x-1xf16)
        relu_27 = paddle._C_ops.relu(batch_norm__180)

        # pd_op.conv2d: (-1x256x4x-1xf16) <- (-1x256x4x-1xf16, 256x256x3x3xf16)
        conv2d_31 = paddle._C_ops.conv2d(relu_27, parameter_155, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x256x4x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32, None) <- (-1x256x4x-1xf16, 256xf32, 256xf32, 256xf32, 256xf32)
        batch_norm__186, batch_norm__187, batch_norm__188, batch_norm__189, batch_norm__190, batch_norm__191 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_31, parameter_156, parameter_157, parameter_158, parameter_159, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x256x4x-1xf16) <- (-1x256x4x-1xf16, -1x256x4x-1xf16)
        add_12 = paddle._C_ops.add(relu_26, batch_norm__186)

        # pd_op.relu: (-1x256x4x-1xf16) <- (-1x256x4x-1xf16)
        relu_28 = paddle._C_ops.relu(add_12)

        # pd_op.conv2d: (-1x512x2x-1xf16) <- (-1x256x4x-1xf16, 512x256x3x3xf16)
        conv2d_32 = paddle._C_ops.conv2d(relu_28, parameter_160, [2, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x2x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x2x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__192, batch_norm__193, batch_norm__194, batch_norm__195, batch_norm__196, batch_norm__197 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_32, parameter_161, parameter_162, parameter_163, parameter_164, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x512x2x-1xf16) <- (-1x512x2x-1xf16)
        relu_29 = paddle._C_ops.relu(batch_norm__192)

        # pd_op.conv2d: (-1x512x2x-1xf16) <- (-1x512x2x-1xf16, 512x512x3x3xf16)
        conv2d_33 = paddle._C_ops.conv2d(relu_29, parameter_165, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x2x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x2x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__198, batch_norm__199, batch_norm__200, batch_norm__201, batch_norm__202, batch_norm__203 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_33, parameter_166, parameter_167, parameter_168, parameter_169, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [2, 1]

        # pd_op.pool2d: (-1x256x2x-1xf16) <- (-1x256x4x-1xf16, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(relu_28, full_int_array_3, [2, 1], [0, 0], True, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x512x2x-1xf16) <- (-1x256x2x-1xf16, 512x256x1x1xf16)
        conv2d_34 = paddle._C_ops.conv2d(pool2d_3, parameter_170, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x2x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x2x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__204, batch_norm__205, batch_norm__206, batch_norm__207, batch_norm__208, batch_norm__209 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_34, parameter_171, parameter_172, parameter_173, parameter_174, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x512x2x-1xf16) <- (-1x512x2x-1xf16, -1x512x2x-1xf16)
        add_13 = paddle._C_ops.add(batch_norm__204, batch_norm__198)

        # pd_op.relu: (-1x512x2x-1xf16) <- (-1x512x2x-1xf16)
        relu_30 = paddle._C_ops.relu(add_13)

        # pd_op.conv2d: (-1x512x2x-1xf16) <- (-1x512x2x-1xf16, 512x512x3x3xf16)
        conv2d_35 = paddle._C_ops.conv2d(relu_30, parameter_175, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x2x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x2x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__210, batch_norm__211, batch_norm__212, batch_norm__213, batch_norm__214, batch_norm__215 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_35, parameter_176, parameter_177, parameter_178, parameter_179, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x512x2x-1xf16) <- (-1x512x2x-1xf16)
        relu_31 = paddle._C_ops.relu(batch_norm__210)

        # pd_op.conv2d: (-1x512x2x-1xf16) <- (-1x512x2x-1xf16, 512x512x3x3xf16)
        conv2d_36 = paddle._C_ops.conv2d(relu_31, parameter_180, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x2x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x2x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__216, batch_norm__217, batch_norm__218, batch_norm__219, batch_norm__220, batch_norm__221 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_36, parameter_181, parameter_182, parameter_183, parameter_184, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x512x2x-1xf16) <- (-1x512x2x-1xf16, -1x512x2x-1xf16)
        add_14 = paddle._C_ops.add(relu_30, batch_norm__216)

        # pd_op.relu: (-1x512x2x-1xf16) <- (-1x512x2x-1xf16)
        relu_32 = paddle._C_ops.relu(add_14)

        # pd_op.conv2d: (-1x512x2x-1xf16) <- (-1x512x2x-1xf16, 512x512x3x3xf16)
        conv2d_37 = paddle._C_ops.conv2d(relu_32, parameter_185, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x2x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x2x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__222, batch_norm__223, batch_norm__224, batch_norm__225, batch_norm__226, batch_norm__227 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_37, parameter_186, parameter_187, parameter_188, parameter_189, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.relu: (-1x512x2x-1xf16) <- (-1x512x2x-1xf16)
        relu_33 = paddle._C_ops.relu(batch_norm__222)

        # pd_op.conv2d: (-1x512x2x-1xf16) <- (-1x512x2x-1xf16, 512x512x3x3xf16)
        conv2d_38 = paddle._C_ops.conv2d(relu_33, parameter_190, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.batch_norm_: (-1x512x2x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32, None) <- (-1x512x2x-1xf16, 512xf32, 512xf32, 512xf32, 512xf32)
        batch_norm__228, batch_norm__229, batch_norm__230, batch_norm__231, batch_norm__232, batch_norm__233 = (lambda x, f: f(x))(paddle._C_ops.batch_norm(conv2d_38, parameter_191, parameter_192, parameter_193, parameter_194, True, float('0.9'), float('1e-05'), 'NCHW', False, False), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None,None,None))

        # pd_op.add: (-1x512x2x-1xf16) <- (-1x512x2x-1xf16, -1x512x2x-1xf16)
        add_15 = paddle._C_ops.add(relu_32, batch_norm__228)

        # pd_op.relu: (-1x512x2x-1xf16) <- (-1x512x2x-1xf16)
        relu_34 = paddle._C_ops.relu(add_15)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_4 = [2, 2]

        # pd_op.pool2d: (-1x512x1x-1xf16) <- (-1x512x2x-1xf16, 2xi64)
        pool2d_4 = paddle._C_ops.pool2d(relu_34, full_int_array_4, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [2]

        # pd_op.squeeze_: (-1x512x-1xf16, None) <- (-1x512x1x-1xf16, 1xi64)
        squeeze__0, squeeze__1 = (lambda x, f: f(x))(paddle._C_ops.squeeze_(pool2d_4, full_int_array_5), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x-1x512xf16) <- (-1x512x-1xf16)
        transpose_0 = paddle._C_ops.transpose(squeeze__0, [0, 2, 1])

        # pd_op.full_batch_size_like: (4x-1x256xf16) <- (-1x-1x512xf16)
        full_batch_size_like_0 = paddle._C_ops.full_batch_size_like(transpose_0, [4, -1, 256], paddle.float16, float('0'), 0, 1, paddle.framework._current_expected_place())

        # pd_op.full_batch_size_like: (4x-1x256xf16) <- (-1x-1x512xf16)
        full_batch_size_like_1 = paddle._C_ops.full_batch_size_like(transpose_0, [4, -1, 256], paddle.float16, float('0'), 0, 1, paddle.framework._current_expected_place())

        # pd_op.transpose: (-1x-1x512xf16) <- (-1x-1x512xf16)
        transpose_1 = paddle._C_ops.transpose(transpose_0, [1, 0, 2])

        # pd_op.cast: (-1x-1x512xf32) <- (-1x-1x512xf16)
        cast_1 = paddle._C_ops.cast(transpose_1, paddle.float32)

        # pd_op.cast: (4x-1x256xf32) <- (4x-1x256xf16)
        cast_2 = paddle._C_ops.cast(full_batch_size_like_0, paddle.float32)

        # pd_op.cast: (4x-1x256xf32) <- (4x-1x256xf16)
        cast_3 = paddle._C_ops.cast(full_batch_size_like_1, paddle.float32)

        # builtin.combine: ([4x-1x256xf32, 4x-1x256xf32]) <- (4x-1x256xf32, 4x-1x256xf32)
        combine_0 = [cast_2, cast_3]

        # builtin.combine: ([1024x512xf32, 1024x256xf32, 1024x512xf32, 1024x256xf32, 1024x512xf32, 1024x256xf32, 1024x512xf32, 1024x256xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32]) <- (1024x512xf32, 1024x256xf32, 1024x512xf32, 1024x256xf32, 1024x512xf32, 1024x256xf32, 1024x512xf32, 1024x256xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32)
        combine_1 = [parameter_195, parameter_196, parameter_197, parameter_198, parameter_199, parameter_200, parameter_201, parameter_202, parameter_203, parameter_204, parameter_205, parameter_206, parameter_207, parameter_208, parameter_209, parameter_210]

        # pd_op.full: (xui8) <- ()
        full_0 = paddle._C_ops.full([], float('0'), paddle.uint8, paddle.core.CPUPlace())

        # pd_op.rnn_: (-1x-1x512xf32, xui8, [4x-1x256xf32, 4x-1x256xf32], xui8) <- (-1x-1x512xf32, [4x-1x256xf32, 4x-1x256xf32], [1024x512xf32, 1024x256xf32, 1024x512xf32, 1024x256xf32, 1024x512xf32, 1024x256xf32, 1024x512xf32, 1024x256xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32, 1024xf32], None, xui8)
        rnn__0, rnn__1, rnn__2, rnn__3 = (lambda x, f: f(x))(paddle._C_ops.rnn(cast_1, combine_0, combine_1, None, full_0, float('0'), True, 512, 256, 2, 'LSTM', 0, True), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None,None)) + (None,)

        # pd_op.cast: (-1x-1x512xf16) <- (-1x-1x512xf32)
        cast_4 = paddle._C_ops.cast(rnn__0, paddle.float16)

        # pd_op.transpose: (-1x-1x512xf16) <- (-1x-1x512xf16)
        transpose_2 = paddle._C_ops.transpose(cast_4, [1, 0, 2])

        # pd_op.matmul: (-1x-1x37xf16) <- (-1x-1x512xf16, 512x37xf16)
        matmul_0 = paddle._C_ops.matmul(transpose_2, parameter_211, False, False)

        # pd_op.add: (-1x-1x37xf16) <- (-1x-1x37xf16, 37xf16)
        add_16 = paddle._C_ops.add(matmul_0, parameter_212)

        # pd_op.softmax: (-1x-1x37xf16) <- (-1x-1x37xf16)
        softmax_0 = paddle._C_ops.softmax(add_16, 2)

        # pd_op.cast: (-1x-1x37xf32) <- (-1x-1x37xf16)
        cast_5 = paddle._C_ops.cast(softmax_0, paddle.float32)
        return cast_5



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

    def forward(self, parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_114, parameter_111, parameter_113, parameter_112, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_154, parameter_151, parameter_153, parameter_152, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_169, parameter_166, parameter_168, parameter_167, parameter_170, parameter_174, parameter_171, parameter_173, parameter_172, parameter_175, parameter_179, parameter_176, parameter_178, parameter_177, parameter_180, parameter_184, parameter_181, parameter_183, parameter_182, parameter_185, parameter_189, parameter_186, parameter_188, parameter_187, parameter_190, parameter_194, parameter_191, parameter_193, parameter_192, parameter_195, parameter_196, parameter_197, parameter_198, parameter_199, parameter_200, parameter_201, parameter_202, parameter_203, parameter_204, parameter_205, parameter_206, parameter_207, parameter_208, parameter_209, parameter_210, parameter_211, parameter_212, feed_0):
        return self.builtin_module_379_0_0(parameter_0, parameter_4, parameter_1, parameter_3, parameter_2, parameter_5, parameter_9, parameter_6, parameter_8, parameter_7, parameter_10, parameter_14, parameter_11, parameter_13, parameter_12, parameter_15, parameter_19, parameter_16, parameter_18, parameter_17, parameter_20, parameter_24, parameter_21, parameter_23, parameter_22, parameter_25, parameter_29, parameter_26, parameter_28, parameter_27, parameter_30, parameter_34, parameter_31, parameter_33, parameter_32, parameter_35, parameter_39, parameter_36, parameter_38, parameter_37, parameter_40, parameter_44, parameter_41, parameter_43, parameter_42, parameter_45, parameter_49, parameter_46, parameter_48, parameter_47, parameter_50, parameter_54, parameter_51, parameter_53, parameter_52, parameter_55, parameter_59, parameter_56, parameter_58, parameter_57, parameter_60, parameter_64, parameter_61, parameter_63, parameter_62, parameter_65, parameter_69, parameter_66, parameter_68, parameter_67, parameter_70, parameter_74, parameter_71, parameter_73, parameter_72, parameter_75, parameter_79, parameter_76, parameter_78, parameter_77, parameter_80, parameter_84, parameter_81, parameter_83, parameter_82, parameter_85, parameter_89, parameter_86, parameter_88, parameter_87, parameter_90, parameter_94, parameter_91, parameter_93, parameter_92, parameter_95, parameter_99, parameter_96, parameter_98, parameter_97, parameter_100, parameter_104, parameter_101, parameter_103, parameter_102, parameter_105, parameter_109, parameter_106, parameter_108, parameter_107, parameter_110, parameter_114, parameter_111, parameter_113, parameter_112, parameter_115, parameter_119, parameter_116, parameter_118, parameter_117, parameter_120, parameter_124, parameter_121, parameter_123, parameter_122, parameter_125, parameter_129, parameter_126, parameter_128, parameter_127, parameter_130, parameter_134, parameter_131, parameter_133, parameter_132, parameter_135, parameter_139, parameter_136, parameter_138, parameter_137, parameter_140, parameter_144, parameter_141, parameter_143, parameter_142, parameter_145, parameter_149, parameter_146, parameter_148, parameter_147, parameter_150, parameter_154, parameter_151, parameter_153, parameter_152, parameter_155, parameter_159, parameter_156, parameter_158, parameter_157, parameter_160, parameter_164, parameter_161, parameter_163, parameter_162, parameter_165, parameter_169, parameter_166, parameter_168, parameter_167, parameter_170, parameter_174, parameter_171, parameter_173, parameter_172, parameter_175, parameter_179, parameter_176, parameter_178, parameter_177, parameter_180, parameter_184, parameter_181, parameter_183, parameter_182, parameter_185, parameter_189, parameter_186, parameter_188, parameter_187, parameter_190, parameter_194, parameter_191, parameter_193, parameter_192, parameter_195, parameter_196, parameter_197, parameter_198, parameter_199, parameter_200, parameter_201, parameter_202, parameter_203, parameter_204, parameter_205, parameter_206, parameter_207, parameter_208, parameter_209, parameter_210, parameter_211, parameter_212, feed_0)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_379_0_0(CinnTestBase, unittest.TestCase):
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
            paddle.uniform([64, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_29
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_26
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_28
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_27
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_30
            paddle.uniform([64, 64, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_34
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_31
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_33
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_32
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_35
            paddle.uniform([64, 64, 3, 3], dtype='float16', min=0, max=0.5),
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
            paddle.uniform([64, 64, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_49
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_46
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_48
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_47
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            # parameter_50
            paddle.uniform([128, 64, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_54
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_51
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_53
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_52
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_55
            paddle.uniform([128, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_59
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_56
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_58
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_57
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_60
            paddle.uniform([128, 64, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_64
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_61
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_63
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_62
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_65
            paddle.uniform([128, 128, 3, 3], dtype='float16', min=0, max=0.5),
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
            paddle.uniform([128, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_79
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_76
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_78
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_77
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_80
            paddle.uniform([128, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_84
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_81
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_83
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_82
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            # parameter_85
            paddle.uniform([128, 128, 3, 3], dtype='float16', min=0, max=0.5),
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
            paddle.uniform([256, 128, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_99
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_96
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_98
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_97
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_100
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_104
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_101
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_103
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_102
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_105
            paddle.uniform([256, 128, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_109
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_106
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_108
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_107
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_110
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_114
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_111
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_113
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_112
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_115
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_119
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_116
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_118
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_117
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_120
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_124
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_121
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_123
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_122
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_125
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_129
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_126
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_128
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_127
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_130
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
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
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_144
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_141
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_143
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_142
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_145
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_149
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_146
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_148
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_147
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_150
            paddle.uniform([256, 256, 3, 3], dtype='float16', min=0, max=0.5),
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
            paddle.uniform([512, 256, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_164
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_161
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_163
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_162
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_165
            paddle.uniform([512, 512, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_169
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_166
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_168
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_167
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_170
            paddle.uniform([512, 256, 1, 1], dtype='float16', min=0, max=0.5),
            # parameter_174
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_171
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_173
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_172
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_175
            paddle.uniform([512, 512, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_179
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_176
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_178
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_177
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_180
            paddle.uniform([512, 512, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_184
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_181
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_183
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_182
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_185
            paddle.uniform([512, 512, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_189
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_186
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_188
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_187
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_190
            paddle.uniform([512, 512, 3, 3], dtype='float16', min=0, max=0.5),
            # parameter_194
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_191
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_193
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_192
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            # parameter_195
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            # parameter_196
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            # parameter_197
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            # parameter_198
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            # parameter_199
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            # parameter_200
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            # parameter_201
            paddle.uniform([1024, 512], dtype='float32', min=0, max=0.5),
            # parameter_202
            paddle.uniform([1024, 256], dtype='float32', min=0, max=0.5),
            # parameter_203
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_204
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_205
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_206
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_207
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_208
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_209
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_210
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            # parameter_211
            paddle.uniform([512, 37], dtype='float16', min=0, max=0.5),
            # parameter_212
            paddle.uniform([37], dtype='float16', min=0, max=0.5),
            # feed_0
            paddle.uniform([1, 3, 32, 100], dtype='float32', min=0, max=0.5),
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
            paddle.static.InputSpec(shape=[64, 64, 1, 1], dtype='float16'),
            # parameter_29
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_26
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_28
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_27
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_30
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float16'),
            # parameter_34
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_31
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_33
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_32
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_35
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float16'),
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
            paddle.static.InputSpec(shape=[64, 64, 3, 3], dtype='float16'),
            # parameter_49
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_46
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_48
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_47
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            # parameter_50
            paddle.static.InputSpec(shape=[128, 64, 3, 3], dtype='float16'),
            # parameter_54
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_51
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_53
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_52
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_55
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float16'),
            # parameter_59
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_56
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_58
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_57
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_60
            paddle.static.InputSpec(shape=[128, 64, 1, 1], dtype='float16'),
            # parameter_64
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_61
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_63
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_62
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_65
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float16'),
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
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float16'),
            # parameter_79
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_76
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_78
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_77
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_80
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float16'),
            # parameter_84
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_81
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_83
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_82
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            # parameter_85
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float16'),
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
            paddle.static.InputSpec(shape=[256, 128, 3, 3], dtype='float16'),
            # parameter_99
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_96
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_98
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_97
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_100
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_104
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_101
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_103
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_102
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_105
            paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float16'),
            # parameter_109
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_106
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_108
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_107
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_110
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_114
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_111
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_113
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_112
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_115
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_119
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_116
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_118
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_117
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_120
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_124
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_121
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_123
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_122
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_125
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_129
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_126
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_128
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_127
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_130
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
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
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_144
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_141
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_143
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_142
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_145
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
            # parameter_149
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_146
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_148
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_147
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_150
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float16'),
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
            paddle.static.InputSpec(shape=[512, 256, 3, 3], dtype='float16'),
            # parameter_164
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_161
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_163
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_162
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_165
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float16'),
            # parameter_169
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_166
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_168
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_167
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_170
            paddle.static.InputSpec(shape=[512, 256, 1, 1], dtype='float16'),
            # parameter_174
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_171
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_173
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_172
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_175
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float16'),
            # parameter_179
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_176
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_178
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_177
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_180
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float16'),
            # parameter_184
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_181
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_183
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_182
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_185
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float16'),
            # parameter_189
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_186
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_188
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_187
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_190
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float16'),
            # parameter_194
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_191
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_193
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_192
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            # parameter_195
            paddle.static.InputSpec(shape=[1024, 512], dtype='float32'),
            # parameter_196
            paddle.static.InputSpec(shape=[1024, 256], dtype='float32'),
            # parameter_197
            paddle.static.InputSpec(shape=[1024, 512], dtype='float32'),
            # parameter_198
            paddle.static.InputSpec(shape=[1024, 256], dtype='float32'),
            # parameter_199
            paddle.static.InputSpec(shape=[1024, 512], dtype='float32'),
            # parameter_200
            paddle.static.InputSpec(shape=[1024, 256], dtype='float32'),
            # parameter_201
            paddle.static.InputSpec(shape=[1024, 512], dtype='float32'),
            # parameter_202
            paddle.static.InputSpec(shape=[1024, 256], dtype='float32'),
            # parameter_203
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_204
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_205
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_206
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_207
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_208
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_209
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_210
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            # parameter_211
            paddle.static.InputSpec(shape=[512, 37], dtype='float16'),
            # parameter_212
            paddle.static.InputSpec(shape=[37], dtype='float16'),
            # feed_0
            paddle.static.InputSpec(shape=[None, 3, 32, None], dtype='float32'),
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