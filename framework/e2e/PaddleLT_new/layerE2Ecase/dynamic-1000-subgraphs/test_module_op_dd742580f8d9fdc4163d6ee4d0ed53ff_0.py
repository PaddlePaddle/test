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
    return [802][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_2261_0_0(self, parameter_0, data_0, data_1):

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [22, 196, 4, 64]

        # pd_op.reshape: (22x196x4x64xf32, 0x22x196x256xi64) <- (22x196x256xf32, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(data_0, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_1 = [16, 16, 32]

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('3'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.split: ([22x196x4x16xf32, 22x196x4x16xf32, 22x196x4x32xf32]) <- (22x196x4x64xf32, 3xi64, 1xi32)
        split_0 = paddle._C_ops.split(reshape_0, full_int_array_1, full_0)

        # builtin.split: (22x196x4x16xf32, 22x196x4x16xf32, 22x196x4x32xf32) <- ([22x196x4x16xf32, 22x196x4x16xf32, 22x196x4x32xf32])
        split_1, split_2, split_3, = split_0

        # pd_op.transpose: (22x4x196x16xf32) <- (22x196x4x16xf32)
        transpose_0 = paddle._C_ops.transpose(split_1, [0, 2, 1, 3])

        # pd_op.transpose: (22x4x196x16xf32) <- (22x196x4x16xf32)
        transpose_1 = paddle._C_ops.transpose(split_2, [0, 2, 1, 3])

        # pd_op.transpose: (22x4x196x32xf32) <- (22x196x4x32xf32)
        transpose_2 = paddle._C_ops.transpose(split_3, [0, 2, 1, 3])

        # pd_op.transpose: (22x4x16x196xf32) <- (22x4x196x16xf32)
        transpose_3 = paddle._C_ops.transpose(transpose_1, [0, 1, 3, 2])

        # pd_op.transpose: (196x4xf32) <- (4x196xf32)
        transpose_4 = paddle._C_ops.transpose(parameter_0, [1, 0])

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_2 = [0]

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [1]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_0 = paddle._C_ops.slice(data_1, [0], full_int_array_2, full_int_array_3, [1], [0])

        # pd_op.full: (1xi32) <- ()
        full_1 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_0 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_1 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_2 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_3 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_4 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_5 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_6 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_7 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_8 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_9 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_10 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_11 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_12 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_13 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_14 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_15 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_16 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_17 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_18 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_19 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_20 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_21 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_22 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_23 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_24 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_25 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_26 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_27 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_28 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_29 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_30 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_31 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_32 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_33 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_34 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_35 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_36 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_37 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_38 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_39 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_40 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_41 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_42 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_43 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_44 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_45 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_46 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_47 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_48 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_49 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_50 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_51 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_52 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_53 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_54 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_55 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_56 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_57 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_58 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_59 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_60 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_61 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_62 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_63 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_64 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_65 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_66 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_67 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_68 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_69 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_70 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_71 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_72 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_73 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_74 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_75 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_76 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_77 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_78 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_79 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_80 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_81 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_82 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_83 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_84 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_85 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_86 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_87 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_88 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_89 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_90 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_91 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_92 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_93 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_94 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_95 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_96 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_97 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_98 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_99 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_100 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_101 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_102 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_103 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_104 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_105 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_106 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_107 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_108 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_109 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_110 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_111 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_112 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_113 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_114 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_115 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_116 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_117 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_118 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_119 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_120 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_121 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_122 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_123 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_124 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_125 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_126 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_127 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_128 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_129 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_130 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_131 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_132 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_133 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_134 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_135 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_136 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_137 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_138 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_139 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_140 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_141 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_142 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_143 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_144 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_145 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_146 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_147 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_148 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_149 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_150 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_151 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_152 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_153 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_154 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_155 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_156 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_157 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_158 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_159 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_160 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_161 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_162 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_163 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_164 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_165 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_166 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_167 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_168 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_169 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_170 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_171 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_172 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_173 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_174 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_175 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_176 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_177 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_178 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_179 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_180 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_181 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_182 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_183 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_184 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_185 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_186 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_187 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_188 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_189 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_190 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_191 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_192 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_193 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_194 = full_1

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_195 = full_1

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_0 = paddle._C_ops.gather(transpose_4, slice_0, full_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [2]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_1 = paddle._C_ops.slice(data_1, [0], full_int_array_3, full_int_array_4, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_1 = paddle._C_ops.gather(transpose_4, slice_1, assign_195)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_5 = [3]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_2 = paddle._C_ops.slice(data_1, [0], full_int_array_4, full_int_array_5, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_2 = paddle._C_ops.gather(transpose_4, slice_2, assign_194)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_6 = [4]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_3 = paddle._C_ops.slice(data_1, [0], full_int_array_5, full_int_array_6, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_3 = paddle._C_ops.gather(transpose_4, slice_3, assign_193)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_7 = [5]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_4 = paddle._C_ops.slice(data_1, [0], full_int_array_6, full_int_array_7, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_4 = paddle._C_ops.gather(transpose_4, slice_4, assign_192)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_8 = [6]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_5 = paddle._C_ops.slice(data_1, [0], full_int_array_7, full_int_array_8, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_5 = paddle._C_ops.gather(transpose_4, slice_5, assign_191)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_9 = [7]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_6 = paddle._C_ops.slice(data_1, [0], full_int_array_8, full_int_array_9, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_6 = paddle._C_ops.gather(transpose_4, slice_6, assign_190)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_10 = [8]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_7 = paddle._C_ops.slice(data_1, [0], full_int_array_9, full_int_array_10, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_7 = paddle._C_ops.gather(transpose_4, slice_7, assign_189)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_11 = [9]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_8 = paddle._C_ops.slice(data_1, [0], full_int_array_10, full_int_array_11, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_8 = paddle._C_ops.gather(transpose_4, slice_8, assign_188)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_12 = [10]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_9 = paddle._C_ops.slice(data_1, [0], full_int_array_11, full_int_array_12, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_9 = paddle._C_ops.gather(transpose_4, slice_9, assign_187)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_13 = [11]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_10 = paddle._C_ops.slice(data_1, [0], full_int_array_12, full_int_array_13, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_10 = paddle._C_ops.gather(transpose_4, slice_10, assign_186)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_14 = [12]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_11 = paddle._C_ops.slice(data_1, [0], full_int_array_13, full_int_array_14, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_11 = paddle._C_ops.gather(transpose_4, slice_11, assign_185)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_15 = [13]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_12 = paddle._C_ops.slice(data_1, [0], full_int_array_14, full_int_array_15, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_12 = paddle._C_ops.gather(transpose_4, slice_12, assign_184)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_16 = [14]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_13 = paddle._C_ops.slice(data_1, [0], full_int_array_15, full_int_array_16, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_13 = paddle._C_ops.gather(transpose_4, slice_13, assign_183)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_17 = [15]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_14 = paddle._C_ops.slice(data_1, [0], full_int_array_16, full_int_array_17, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_14 = paddle._C_ops.gather(transpose_4, slice_14, assign_182)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_18 = [16]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_15 = paddle._C_ops.slice(data_1, [0], full_int_array_17, full_int_array_18, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_15 = paddle._C_ops.gather(transpose_4, slice_15, assign_181)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_19 = [17]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_16 = paddle._C_ops.slice(data_1, [0], full_int_array_18, full_int_array_19, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_16 = paddle._C_ops.gather(transpose_4, slice_16, assign_180)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_20 = [18]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_17 = paddle._C_ops.slice(data_1, [0], full_int_array_19, full_int_array_20, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_17 = paddle._C_ops.gather(transpose_4, slice_17, assign_179)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_21 = [19]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_18 = paddle._C_ops.slice(data_1, [0], full_int_array_20, full_int_array_21, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_18 = paddle._C_ops.gather(transpose_4, slice_18, assign_178)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_22 = [20]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_19 = paddle._C_ops.slice(data_1, [0], full_int_array_21, full_int_array_22, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_19 = paddle._C_ops.gather(transpose_4, slice_19, assign_177)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_23 = [21]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_20 = paddle._C_ops.slice(data_1, [0], full_int_array_22, full_int_array_23, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_20 = paddle._C_ops.gather(transpose_4, slice_20, assign_176)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_24 = [22]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_21 = paddle._C_ops.slice(data_1, [0], full_int_array_23, full_int_array_24, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_21 = paddle._C_ops.gather(transpose_4, slice_21, assign_175)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_25 = [23]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_22 = paddle._C_ops.slice(data_1, [0], full_int_array_24, full_int_array_25, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_22 = paddle._C_ops.gather(transpose_4, slice_22, assign_174)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_26 = [24]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_23 = paddle._C_ops.slice(data_1, [0], full_int_array_25, full_int_array_26, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_23 = paddle._C_ops.gather(transpose_4, slice_23, assign_173)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_27 = [25]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_24 = paddle._C_ops.slice(data_1, [0], full_int_array_26, full_int_array_27, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_24 = paddle._C_ops.gather(transpose_4, slice_24, assign_172)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_28 = [26]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_25 = paddle._C_ops.slice(data_1, [0], full_int_array_27, full_int_array_28, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_25 = paddle._C_ops.gather(transpose_4, slice_25, assign_171)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_29 = [27]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_26 = paddle._C_ops.slice(data_1, [0], full_int_array_28, full_int_array_29, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_26 = paddle._C_ops.gather(transpose_4, slice_26, assign_170)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_30 = [28]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_27 = paddle._C_ops.slice(data_1, [0], full_int_array_29, full_int_array_30, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_27 = paddle._C_ops.gather(transpose_4, slice_27, assign_169)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_31 = [29]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_28 = paddle._C_ops.slice(data_1, [0], full_int_array_30, full_int_array_31, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_28 = paddle._C_ops.gather(transpose_4, slice_28, assign_168)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_32 = [30]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_29 = paddle._C_ops.slice(data_1, [0], full_int_array_31, full_int_array_32, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_29 = paddle._C_ops.gather(transpose_4, slice_29, assign_167)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_33 = [31]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_30 = paddle._C_ops.slice(data_1, [0], full_int_array_32, full_int_array_33, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_30 = paddle._C_ops.gather(transpose_4, slice_30, assign_166)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_34 = [32]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_31 = paddle._C_ops.slice(data_1, [0], full_int_array_33, full_int_array_34, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_31 = paddle._C_ops.gather(transpose_4, slice_31, assign_165)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_35 = [33]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_32 = paddle._C_ops.slice(data_1, [0], full_int_array_34, full_int_array_35, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_32 = paddle._C_ops.gather(transpose_4, slice_32, assign_164)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_36 = [34]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_33 = paddle._C_ops.slice(data_1, [0], full_int_array_35, full_int_array_36, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_33 = paddle._C_ops.gather(transpose_4, slice_33, assign_163)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_37 = [35]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_34 = paddle._C_ops.slice(data_1, [0], full_int_array_36, full_int_array_37, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_34 = paddle._C_ops.gather(transpose_4, slice_34, assign_162)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_38 = [36]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_35 = paddle._C_ops.slice(data_1, [0], full_int_array_37, full_int_array_38, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_35 = paddle._C_ops.gather(transpose_4, slice_35, assign_161)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_39 = [37]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_36 = paddle._C_ops.slice(data_1, [0], full_int_array_38, full_int_array_39, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_36 = paddle._C_ops.gather(transpose_4, slice_36, assign_160)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_40 = [38]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_37 = paddle._C_ops.slice(data_1, [0], full_int_array_39, full_int_array_40, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_37 = paddle._C_ops.gather(transpose_4, slice_37, assign_159)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_41 = [39]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_38 = paddle._C_ops.slice(data_1, [0], full_int_array_40, full_int_array_41, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_38 = paddle._C_ops.gather(transpose_4, slice_38, assign_158)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_42 = [40]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_39 = paddle._C_ops.slice(data_1, [0], full_int_array_41, full_int_array_42, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_39 = paddle._C_ops.gather(transpose_4, slice_39, assign_157)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_43 = [41]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_40 = paddle._C_ops.slice(data_1, [0], full_int_array_42, full_int_array_43, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_40 = paddle._C_ops.gather(transpose_4, slice_40, assign_156)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_44 = [42]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_41 = paddle._C_ops.slice(data_1, [0], full_int_array_43, full_int_array_44, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_41 = paddle._C_ops.gather(transpose_4, slice_41, assign_155)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_45 = [43]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_42 = paddle._C_ops.slice(data_1, [0], full_int_array_44, full_int_array_45, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_42 = paddle._C_ops.gather(transpose_4, slice_42, assign_154)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_46 = [44]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_43 = paddle._C_ops.slice(data_1, [0], full_int_array_45, full_int_array_46, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_43 = paddle._C_ops.gather(transpose_4, slice_43, assign_153)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_47 = [45]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_44 = paddle._C_ops.slice(data_1, [0], full_int_array_46, full_int_array_47, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_44 = paddle._C_ops.gather(transpose_4, slice_44, assign_152)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_48 = [46]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_45 = paddle._C_ops.slice(data_1, [0], full_int_array_47, full_int_array_48, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_45 = paddle._C_ops.gather(transpose_4, slice_45, assign_151)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_49 = [47]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_46 = paddle._C_ops.slice(data_1, [0], full_int_array_48, full_int_array_49, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_46 = paddle._C_ops.gather(transpose_4, slice_46, assign_150)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_50 = [48]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_47 = paddle._C_ops.slice(data_1, [0], full_int_array_49, full_int_array_50, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_47 = paddle._C_ops.gather(transpose_4, slice_47, assign_149)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_51 = [49]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_48 = paddle._C_ops.slice(data_1, [0], full_int_array_50, full_int_array_51, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_48 = paddle._C_ops.gather(transpose_4, slice_48, assign_148)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_52 = [50]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_49 = paddle._C_ops.slice(data_1, [0], full_int_array_51, full_int_array_52, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_49 = paddle._C_ops.gather(transpose_4, slice_49, assign_147)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_53 = [51]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_50 = paddle._C_ops.slice(data_1, [0], full_int_array_52, full_int_array_53, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_50 = paddle._C_ops.gather(transpose_4, slice_50, assign_146)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_54 = [52]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_51 = paddle._C_ops.slice(data_1, [0], full_int_array_53, full_int_array_54, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_51 = paddle._C_ops.gather(transpose_4, slice_51, assign_145)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_55 = [53]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_52 = paddle._C_ops.slice(data_1, [0], full_int_array_54, full_int_array_55, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_52 = paddle._C_ops.gather(transpose_4, slice_52, assign_144)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_56 = [54]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_53 = paddle._C_ops.slice(data_1, [0], full_int_array_55, full_int_array_56, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_53 = paddle._C_ops.gather(transpose_4, slice_53, assign_143)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_57 = [55]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_54 = paddle._C_ops.slice(data_1, [0], full_int_array_56, full_int_array_57, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_54 = paddle._C_ops.gather(transpose_4, slice_54, assign_142)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_58 = [56]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_55 = paddle._C_ops.slice(data_1, [0], full_int_array_57, full_int_array_58, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_55 = paddle._C_ops.gather(transpose_4, slice_55, assign_141)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_59 = [57]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_56 = paddle._C_ops.slice(data_1, [0], full_int_array_58, full_int_array_59, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_56 = paddle._C_ops.gather(transpose_4, slice_56, assign_140)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_60 = [58]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_57 = paddle._C_ops.slice(data_1, [0], full_int_array_59, full_int_array_60, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_57 = paddle._C_ops.gather(transpose_4, slice_57, assign_139)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_61 = [59]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_58 = paddle._C_ops.slice(data_1, [0], full_int_array_60, full_int_array_61, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_58 = paddle._C_ops.gather(transpose_4, slice_58, assign_138)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_62 = [60]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_59 = paddle._C_ops.slice(data_1, [0], full_int_array_61, full_int_array_62, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_59 = paddle._C_ops.gather(transpose_4, slice_59, assign_137)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_63 = [61]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_60 = paddle._C_ops.slice(data_1, [0], full_int_array_62, full_int_array_63, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_60 = paddle._C_ops.gather(transpose_4, slice_60, assign_136)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_64 = [62]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_61 = paddle._C_ops.slice(data_1, [0], full_int_array_63, full_int_array_64, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_61 = paddle._C_ops.gather(transpose_4, slice_61, assign_135)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_65 = [63]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_62 = paddle._C_ops.slice(data_1, [0], full_int_array_64, full_int_array_65, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_62 = paddle._C_ops.gather(transpose_4, slice_62, assign_134)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_66 = [64]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_63 = paddle._C_ops.slice(data_1, [0], full_int_array_65, full_int_array_66, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_63 = paddle._C_ops.gather(transpose_4, slice_63, assign_133)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_67 = [65]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_64 = paddle._C_ops.slice(data_1, [0], full_int_array_66, full_int_array_67, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_64 = paddle._C_ops.gather(transpose_4, slice_64, assign_132)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_68 = [66]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_65 = paddle._C_ops.slice(data_1, [0], full_int_array_67, full_int_array_68, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_65 = paddle._C_ops.gather(transpose_4, slice_65, assign_131)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_69 = [67]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_66 = paddle._C_ops.slice(data_1, [0], full_int_array_68, full_int_array_69, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_66 = paddle._C_ops.gather(transpose_4, slice_66, assign_130)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_70 = [68]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_67 = paddle._C_ops.slice(data_1, [0], full_int_array_69, full_int_array_70, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_67 = paddle._C_ops.gather(transpose_4, slice_67, assign_129)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_71 = [69]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_68 = paddle._C_ops.slice(data_1, [0], full_int_array_70, full_int_array_71, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_68 = paddle._C_ops.gather(transpose_4, slice_68, assign_128)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_72 = [70]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_69 = paddle._C_ops.slice(data_1, [0], full_int_array_71, full_int_array_72, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_69 = paddle._C_ops.gather(transpose_4, slice_69, assign_127)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_73 = [71]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_70 = paddle._C_ops.slice(data_1, [0], full_int_array_72, full_int_array_73, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_70 = paddle._C_ops.gather(transpose_4, slice_70, assign_126)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_74 = [72]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_71 = paddle._C_ops.slice(data_1, [0], full_int_array_73, full_int_array_74, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_71 = paddle._C_ops.gather(transpose_4, slice_71, assign_125)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_75 = [73]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_72 = paddle._C_ops.slice(data_1, [0], full_int_array_74, full_int_array_75, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_72 = paddle._C_ops.gather(transpose_4, slice_72, assign_124)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_76 = [74]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_73 = paddle._C_ops.slice(data_1, [0], full_int_array_75, full_int_array_76, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_73 = paddle._C_ops.gather(transpose_4, slice_73, assign_123)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_77 = [75]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_74 = paddle._C_ops.slice(data_1, [0], full_int_array_76, full_int_array_77, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_74 = paddle._C_ops.gather(transpose_4, slice_74, assign_122)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_78 = [76]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_75 = paddle._C_ops.slice(data_1, [0], full_int_array_77, full_int_array_78, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_75 = paddle._C_ops.gather(transpose_4, slice_75, assign_121)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_79 = [77]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_76 = paddle._C_ops.slice(data_1, [0], full_int_array_78, full_int_array_79, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_76 = paddle._C_ops.gather(transpose_4, slice_76, assign_120)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_80 = [78]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_77 = paddle._C_ops.slice(data_1, [0], full_int_array_79, full_int_array_80, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_77 = paddle._C_ops.gather(transpose_4, slice_77, assign_119)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_81 = [79]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_78 = paddle._C_ops.slice(data_1, [0], full_int_array_80, full_int_array_81, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_78 = paddle._C_ops.gather(transpose_4, slice_78, assign_118)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_82 = [80]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_79 = paddle._C_ops.slice(data_1, [0], full_int_array_81, full_int_array_82, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_79 = paddle._C_ops.gather(transpose_4, slice_79, assign_117)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_83 = [81]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_80 = paddle._C_ops.slice(data_1, [0], full_int_array_82, full_int_array_83, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_80 = paddle._C_ops.gather(transpose_4, slice_80, assign_116)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_84 = [82]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_81 = paddle._C_ops.slice(data_1, [0], full_int_array_83, full_int_array_84, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_81 = paddle._C_ops.gather(transpose_4, slice_81, assign_115)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_85 = [83]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_82 = paddle._C_ops.slice(data_1, [0], full_int_array_84, full_int_array_85, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_82 = paddle._C_ops.gather(transpose_4, slice_82, assign_114)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_86 = [84]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_83 = paddle._C_ops.slice(data_1, [0], full_int_array_85, full_int_array_86, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_83 = paddle._C_ops.gather(transpose_4, slice_83, assign_113)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_87 = [85]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_84 = paddle._C_ops.slice(data_1, [0], full_int_array_86, full_int_array_87, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_84 = paddle._C_ops.gather(transpose_4, slice_84, assign_112)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_88 = [86]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_85 = paddle._C_ops.slice(data_1, [0], full_int_array_87, full_int_array_88, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_85 = paddle._C_ops.gather(transpose_4, slice_85, assign_111)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_89 = [87]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_86 = paddle._C_ops.slice(data_1, [0], full_int_array_88, full_int_array_89, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_86 = paddle._C_ops.gather(transpose_4, slice_86, assign_110)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_90 = [88]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_87 = paddle._C_ops.slice(data_1, [0], full_int_array_89, full_int_array_90, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_87 = paddle._C_ops.gather(transpose_4, slice_87, assign_109)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_91 = [89]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_88 = paddle._C_ops.slice(data_1, [0], full_int_array_90, full_int_array_91, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_88 = paddle._C_ops.gather(transpose_4, slice_88, assign_108)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_92 = [90]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_89 = paddle._C_ops.slice(data_1, [0], full_int_array_91, full_int_array_92, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_89 = paddle._C_ops.gather(transpose_4, slice_89, assign_107)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_93 = [91]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_90 = paddle._C_ops.slice(data_1, [0], full_int_array_92, full_int_array_93, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_90 = paddle._C_ops.gather(transpose_4, slice_90, assign_106)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_94 = [92]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_91 = paddle._C_ops.slice(data_1, [0], full_int_array_93, full_int_array_94, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_91 = paddle._C_ops.gather(transpose_4, slice_91, assign_105)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_95 = [93]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_92 = paddle._C_ops.slice(data_1, [0], full_int_array_94, full_int_array_95, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_92 = paddle._C_ops.gather(transpose_4, slice_92, assign_104)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_96 = [94]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_93 = paddle._C_ops.slice(data_1, [0], full_int_array_95, full_int_array_96, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_93 = paddle._C_ops.gather(transpose_4, slice_93, assign_103)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_97 = [95]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_94 = paddle._C_ops.slice(data_1, [0], full_int_array_96, full_int_array_97, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_94 = paddle._C_ops.gather(transpose_4, slice_94, assign_102)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_98 = [96]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_95 = paddle._C_ops.slice(data_1, [0], full_int_array_97, full_int_array_98, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_95 = paddle._C_ops.gather(transpose_4, slice_95, assign_101)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_99 = [97]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_96 = paddle._C_ops.slice(data_1, [0], full_int_array_98, full_int_array_99, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_96 = paddle._C_ops.gather(transpose_4, slice_96, assign_100)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_100 = [98]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_97 = paddle._C_ops.slice(data_1, [0], full_int_array_99, full_int_array_100, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_97 = paddle._C_ops.gather(transpose_4, slice_97, assign_99)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_101 = [99]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_98 = paddle._C_ops.slice(data_1, [0], full_int_array_100, full_int_array_101, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_98 = paddle._C_ops.gather(transpose_4, slice_98, assign_98)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_102 = [100]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_99 = paddle._C_ops.slice(data_1, [0], full_int_array_101, full_int_array_102, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_99 = paddle._C_ops.gather(transpose_4, slice_99, assign_97)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_103 = [101]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_100 = paddle._C_ops.slice(data_1, [0], full_int_array_102, full_int_array_103, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_100 = paddle._C_ops.gather(transpose_4, slice_100, assign_96)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_104 = [102]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_101 = paddle._C_ops.slice(data_1, [0], full_int_array_103, full_int_array_104, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_101 = paddle._C_ops.gather(transpose_4, slice_101, assign_95)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_105 = [103]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_102 = paddle._C_ops.slice(data_1, [0], full_int_array_104, full_int_array_105, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_102 = paddle._C_ops.gather(transpose_4, slice_102, assign_94)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_106 = [104]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_103 = paddle._C_ops.slice(data_1, [0], full_int_array_105, full_int_array_106, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_103 = paddle._C_ops.gather(transpose_4, slice_103, assign_93)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_107 = [105]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_104 = paddle._C_ops.slice(data_1, [0], full_int_array_106, full_int_array_107, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_104 = paddle._C_ops.gather(transpose_4, slice_104, assign_92)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_108 = [106]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_105 = paddle._C_ops.slice(data_1, [0], full_int_array_107, full_int_array_108, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_105 = paddle._C_ops.gather(transpose_4, slice_105, assign_91)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_109 = [107]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_106 = paddle._C_ops.slice(data_1, [0], full_int_array_108, full_int_array_109, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_106 = paddle._C_ops.gather(transpose_4, slice_106, assign_90)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_110 = [108]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_107 = paddle._C_ops.slice(data_1, [0], full_int_array_109, full_int_array_110, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_107 = paddle._C_ops.gather(transpose_4, slice_107, assign_89)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_111 = [109]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_108 = paddle._C_ops.slice(data_1, [0], full_int_array_110, full_int_array_111, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_108 = paddle._C_ops.gather(transpose_4, slice_108, assign_88)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_112 = [110]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_109 = paddle._C_ops.slice(data_1, [0], full_int_array_111, full_int_array_112, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_109 = paddle._C_ops.gather(transpose_4, slice_109, assign_87)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_113 = [111]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_110 = paddle._C_ops.slice(data_1, [0], full_int_array_112, full_int_array_113, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_110 = paddle._C_ops.gather(transpose_4, slice_110, assign_86)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_114 = [112]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_111 = paddle._C_ops.slice(data_1, [0], full_int_array_113, full_int_array_114, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_111 = paddle._C_ops.gather(transpose_4, slice_111, assign_85)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_115 = [113]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_112 = paddle._C_ops.slice(data_1, [0], full_int_array_114, full_int_array_115, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_112 = paddle._C_ops.gather(transpose_4, slice_112, assign_84)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_116 = [114]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_113 = paddle._C_ops.slice(data_1, [0], full_int_array_115, full_int_array_116, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_113 = paddle._C_ops.gather(transpose_4, slice_113, assign_83)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_117 = [115]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_114 = paddle._C_ops.slice(data_1, [0], full_int_array_116, full_int_array_117, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_114 = paddle._C_ops.gather(transpose_4, slice_114, assign_82)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_118 = [116]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_115 = paddle._C_ops.slice(data_1, [0], full_int_array_117, full_int_array_118, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_115 = paddle._C_ops.gather(transpose_4, slice_115, assign_81)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_119 = [117]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_116 = paddle._C_ops.slice(data_1, [0], full_int_array_118, full_int_array_119, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_116 = paddle._C_ops.gather(transpose_4, slice_116, assign_80)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_120 = [118]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_117 = paddle._C_ops.slice(data_1, [0], full_int_array_119, full_int_array_120, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_117 = paddle._C_ops.gather(transpose_4, slice_117, assign_79)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_121 = [119]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_118 = paddle._C_ops.slice(data_1, [0], full_int_array_120, full_int_array_121, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_118 = paddle._C_ops.gather(transpose_4, slice_118, assign_78)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_122 = [120]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_119 = paddle._C_ops.slice(data_1, [0], full_int_array_121, full_int_array_122, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_119 = paddle._C_ops.gather(transpose_4, slice_119, assign_77)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_123 = [121]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_120 = paddle._C_ops.slice(data_1, [0], full_int_array_122, full_int_array_123, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_120 = paddle._C_ops.gather(transpose_4, slice_120, assign_76)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_124 = [122]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_121 = paddle._C_ops.slice(data_1, [0], full_int_array_123, full_int_array_124, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_121 = paddle._C_ops.gather(transpose_4, slice_121, assign_75)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_125 = [123]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_122 = paddle._C_ops.slice(data_1, [0], full_int_array_124, full_int_array_125, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_122 = paddle._C_ops.gather(transpose_4, slice_122, assign_74)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_126 = [124]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_123 = paddle._C_ops.slice(data_1, [0], full_int_array_125, full_int_array_126, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_123 = paddle._C_ops.gather(transpose_4, slice_123, assign_73)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_127 = [125]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_124 = paddle._C_ops.slice(data_1, [0], full_int_array_126, full_int_array_127, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_124 = paddle._C_ops.gather(transpose_4, slice_124, assign_72)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_128 = [126]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_125 = paddle._C_ops.slice(data_1, [0], full_int_array_127, full_int_array_128, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_125 = paddle._C_ops.gather(transpose_4, slice_125, assign_71)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_129 = [127]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_126 = paddle._C_ops.slice(data_1, [0], full_int_array_128, full_int_array_129, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_126 = paddle._C_ops.gather(transpose_4, slice_126, assign_70)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_130 = [128]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_127 = paddle._C_ops.slice(data_1, [0], full_int_array_129, full_int_array_130, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_127 = paddle._C_ops.gather(transpose_4, slice_127, assign_69)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_131 = [129]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_128 = paddle._C_ops.slice(data_1, [0], full_int_array_130, full_int_array_131, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_128 = paddle._C_ops.gather(transpose_4, slice_128, assign_68)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_132 = [130]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_129 = paddle._C_ops.slice(data_1, [0], full_int_array_131, full_int_array_132, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_129 = paddle._C_ops.gather(transpose_4, slice_129, assign_67)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_133 = [131]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_130 = paddle._C_ops.slice(data_1, [0], full_int_array_132, full_int_array_133, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_130 = paddle._C_ops.gather(transpose_4, slice_130, assign_66)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_134 = [132]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_131 = paddle._C_ops.slice(data_1, [0], full_int_array_133, full_int_array_134, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_131 = paddle._C_ops.gather(transpose_4, slice_131, assign_65)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_135 = [133]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_132 = paddle._C_ops.slice(data_1, [0], full_int_array_134, full_int_array_135, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_132 = paddle._C_ops.gather(transpose_4, slice_132, assign_64)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_136 = [134]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_133 = paddle._C_ops.slice(data_1, [0], full_int_array_135, full_int_array_136, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_133 = paddle._C_ops.gather(transpose_4, slice_133, assign_63)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_137 = [135]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_134 = paddle._C_ops.slice(data_1, [0], full_int_array_136, full_int_array_137, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_134 = paddle._C_ops.gather(transpose_4, slice_134, assign_62)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_138 = [136]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_135 = paddle._C_ops.slice(data_1, [0], full_int_array_137, full_int_array_138, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_135 = paddle._C_ops.gather(transpose_4, slice_135, assign_61)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_139 = [137]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_136 = paddle._C_ops.slice(data_1, [0], full_int_array_138, full_int_array_139, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_136 = paddle._C_ops.gather(transpose_4, slice_136, assign_60)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_140 = [138]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_137 = paddle._C_ops.slice(data_1, [0], full_int_array_139, full_int_array_140, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_137 = paddle._C_ops.gather(transpose_4, slice_137, assign_59)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_141 = [139]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_138 = paddle._C_ops.slice(data_1, [0], full_int_array_140, full_int_array_141, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_138 = paddle._C_ops.gather(transpose_4, slice_138, assign_58)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_142 = [140]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_139 = paddle._C_ops.slice(data_1, [0], full_int_array_141, full_int_array_142, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_139 = paddle._C_ops.gather(transpose_4, slice_139, assign_57)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_143 = [141]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_140 = paddle._C_ops.slice(data_1, [0], full_int_array_142, full_int_array_143, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_140 = paddle._C_ops.gather(transpose_4, slice_140, assign_56)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_144 = [142]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_141 = paddle._C_ops.slice(data_1, [0], full_int_array_143, full_int_array_144, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_141 = paddle._C_ops.gather(transpose_4, slice_141, assign_55)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_145 = [143]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_142 = paddle._C_ops.slice(data_1, [0], full_int_array_144, full_int_array_145, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_142 = paddle._C_ops.gather(transpose_4, slice_142, assign_54)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_146 = [144]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_143 = paddle._C_ops.slice(data_1, [0], full_int_array_145, full_int_array_146, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_143 = paddle._C_ops.gather(transpose_4, slice_143, assign_53)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_147 = [145]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_144 = paddle._C_ops.slice(data_1, [0], full_int_array_146, full_int_array_147, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_144 = paddle._C_ops.gather(transpose_4, slice_144, assign_52)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_148 = [146]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_145 = paddle._C_ops.slice(data_1, [0], full_int_array_147, full_int_array_148, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_145 = paddle._C_ops.gather(transpose_4, slice_145, assign_51)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_149 = [147]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_146 = paddle._C_ops.slice(data_1, [0], full_int_array_148, full_int_array_149, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_146 = paddle._C_ops.gather(transpose_4, slice_146, assign_50)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_150 = [148]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_147 = paddle._C_ops.slice(data_1, [0], full_int_array_149, full_int_array_150, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_147 = paddle._C_ops.gather(transpose_4, slice_147, assign_49)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_151 = [149]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_148 = paddle._C_ops.slice(data_1, [0], full_int_array_150, full_int_array_151, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_148 = paddle._C_ops.gather(transpose_4, slice_148, assign_48)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_152 = [150]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_149 = paddle._C_ops.slice(data_1, [0], full_int_array_151, full_int_array_152, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_149 = paddle._C_ops.gather(transpose_4, slice_149, assign_47)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_153 = [151]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_150 = paddle._C_ops.slice(data_1, [0], full_int_array_152, full_int_array_153, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_150 = paddle._C_ops.gather(transpose_4, slice_150, assign_46)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_154 = [152]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_151 = paddle._C_ops.slice(data_1, [0], full_int_array_153, full_int_array_154, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_151 = paddle._C_ops.gather(transpose_4, slice_151, assign_45)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_155 = [153]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_152 = paddle._C_ops.slice(data_1, [0], full_int_array_154, full_int_array_155, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_152 = paddle._C_ops.gather(transpose_4, slice_152, assign_44)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_156 = [154]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_153 = paddle._C_ops.slice(data_1, [0], full_int_array_155, full_int_array_156, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_153 = paddle._C_ops.gather(transpose_4, slice_153, assign_43)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_157 = [155]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_154 = paddle._C_ops.slice(data_1, [0], full_int_array_156, full_int_array_157, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_154 = paddle._C_ops.gather(transpose_4, slice_154, assign_42)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_158 = [156]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_155 = paddle._C_ops.slice(data_1, [0], full_int_array_157, full_int_array_158, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_155 = paddle._C_ops.gather(transpose_4, slice_155, assign_41)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_159 = [157]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_156 = paddle._C_ops.slice(data_1, [0], full_int_array_158, full_int_array_159, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_156 = paddle._C_ops.gather(transpose_4, slice_156, assign_40)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_160 = [158]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_157 = paddle._C_ops.slice(data_1, [0], full_int_array_159, full_int_array_160, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_157 = paddle._C_ops.gather(transpose_4, slice_157, assign_39)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_161 = [159]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_158 = paddle._C_ops.slice(data_1, [0], full_int_array_160, full_int_array_161, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_158 = paddle._C_ops.gather(transpose_4, slice_158, assign_38)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_162 = [160]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_159 = paddle._C_ops.slice(data_1, [0], full_int_array_161, full_int_array_162, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_159 = paddle._C_ops.gather(transpose_4, slice_159, assign_37)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_163 = [161]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_160 = paddle._C_ops.slice(data_1, [0], full_int_array_162, full_int_array_163, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_160 = paddle._C_ops.gather(transpose_4, slice_160, assign_36)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_164 = [162]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_161 = paddle._C_ops.slice(data_1, [0], full_int_array_163, full_int_array_164, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_161 = paddle._C_ops.gather(transpose_4, slice_161, assign_35)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_165 = [163]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_162 = paddle._C_ops.slice(data_1, [0], full_int_array_164, full_int_array_165, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_162 = paddle._C_ops.gather(transpose_4, slice_162, assign_34)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_166 = [164]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_163 = paddle._C_ops.slice(data_1, [0], full_int_array_165, full_int_array_166, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_163 = paddle._C_ops.gather(transpose_4, slice_163, assign_33)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_167 = [165]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_164 = paddle._C_ops.slice(data_1, [0], full_int_array_166, full_int_array_167, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_164 = paddle._C_ops.gather(transpose_4, slice_164, assign_32)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_168 = [166]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_165 = paddle._C_ops.slice(data_1, [0], full_int_array_167, full_int_array_168, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_165 = paddle._C_ops.gather(transpose_4, slice_165, assign_31)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_169 = [167]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_166 = paddle._C_ops.slice(data_1, [0], full_int_array_168, full_int_array_169, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_166 = paddle._C_ops.gather(transpose_4, slice_166, assign_30)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_170 = [168]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_167 = paddle._C_ops.slice(data_1, [0], full_int_array_169, full_int_array_170, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_167 = paddle._C_ops.gather(transpose_4, slice_167, assign_29)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_171 = [169]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_168 = paddle._C_ops.slice(data_1, [0], full_int_array_170, full_int_array_171, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_168 = paddle._C_ops.gather(transpose_4, slice_168, assign_28)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_172 = [170]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_169 = paddle._C_ops.slice(data_1, [0], full_int_array_171, full_int_array_172, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_169 = paddle._C_ops.gather(transpose_4, slice_169, assign_27)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_173 = [171]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_170 = paddle._C_ops.slice(data_1, [0], full_int_array_172, full_int_array_173, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_170 = paddle._C_ops.gather(transpose_4, slice_170, assign_26)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_174 = [172]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_171 = paddle._C_ops.slice(data_1, [0], full_int_array_173, full_int_array_174, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_171 = paddle._C_ops.gather(transpose_4, slice_171, assign_25)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_175 = [173]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_172 = paddle._C_ops.slice(data_1, [0], full_int_array_174, full_int_array_175, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_172 = paddle._C_ops.gather(transpose_4, slice_172, assign_24)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_176 = [174]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_173 = paddle._C_ops.slice(data_1, [0], full_int_array_175, full_int_array_176, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_173 = paddle._C_ops.gather(transpose_4, slice_173, assign_23)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_177 = [175]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_174 = paddle._C_ops.slice(data_1, [0], full_int_array_176, full_int_array_177, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_174 = paddle._C_ops.gather(transpose_4, slice_174, assign_22)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_178 = [176]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_175 = paddle._C_ops.slice(data_1, [0], full_int_array_177, full_int_array_178, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_175 = paddle._C_ops.gather(transpose_4, slice_175, assign_21)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_179 = [177]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_176 = paddle._C_ops.slice(data_1, [0], full_int_array_178, full_int_array_179, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_176 = paddle._C_ops.gather(transpose_4, slice_176, assign_20)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_180 = [178]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_177 = paddle._C_ops.slice(data_1, [0], full_int_array_179, full_int_array_180, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_177 = paddle._C_ops.gather(transpose_4, slice_177, assign_19)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_181 = [179]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_178 = paddle._C_ops.slice(data_1, [0], full_int_array_180, full_int_array_181, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_178 = paddle._C_ops.gather(transpose_4, slice_178, assign_18)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_182 = [180]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_179 = paddle._C_ops.slice(data_1, [0], full_int_array_181, full_int_array_182, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_179 = paddle._C_ops.gather(transpose_4, slice_179, assign_17)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_183 = [181]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_180 = paddle._C_ops.slice(data_1, [0], full_int_array_182, full_int_array_183, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_180 = paddle._C_ops.gather(transpose_4, slice_180, assign_16)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_184 = [182]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_181 = paddle._C_ops.slice(data_1, [0], full_int_array_183, full_int_array_184, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_181 = paddle._C_ops.gather(transpose_4, slice_181, assign_15)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_185 = [183]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_182 = paddle._C_ops.slice(data_1, [0], full_int_array_184, full_int_array_185, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_182 = paddle._C_ops.gather(transpose_4, slice_182, assign_14)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_186 = [184]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_183 = paddle._C_ops.slice(data_1, [0], full_int_array_185, full_int_array_186, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_183 = paddle._C_ops.gather(transpose_4, slice_183, assign_13)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_187 = [185]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_184 = paddle._C_ops.slice(data_1, [0], full_int_array_186, full_int_array_187, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_184 = paddle._C_ops.gather(transpose_4, slice_184, assign_12)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_188 = [186]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_185 = paddle._C_ops.slice(data_1, [0], full_int_array_187, full_int_array_188, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_185 = paddle._C_ops.gather(transpose_4, slice_185, assign_11)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_189 = [187]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_186 = paddle._C_ops.slice(data_1, [0], full_int_array_188, full_int_array_189, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_186 = paddle._C_ops.gather(transpose_4, slice_186, assign_10)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_190 = [188]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_187 = paddle._C_ops.slice(data_1, [0], full_int_array_189, full_int_array_190, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_187 = paddle._C_ops.gather(transpose_4, slice_187, assign_9)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_191 = [189]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_188 = paddle._C_ops.slice(data_1, [0], full_int_array_190, full_int_array_191, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_188 = paddle._C_ops.gather(transpose_4, slice_188, assign_8)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_192 = [190]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_189 = paddle._C_ops.slice(data_1, [0], full_int_array_191, full_int_array_192, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_189 = paddle._C_ops.gather(transpose_4, slice_189, assign_7)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_193 = [191]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_190 = paddle._C_ops.slice(data_1, [0], full_int_array_192, full_int_array_193, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_190 = paddle._C_ops.gather(transpose_4, slice_190, assign_6)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_194 = [192]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_191 = paddle._C_ops.slice(data_1, [0], full_int_array_193, full_int_array_194, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_191 = paddle._C_ops.gather(transpose_4, slice_191, assign_5)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_195 = [193]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_192 = paddle._C_ops.slice(data_1, [0], full_int_array_194, full_int_array_195, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_192 = paddle._C_ops.gather(transpose_4, slice_192, assign_4)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_196 = [194]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_193 = paddle._C_ops.slice(data_1, [0], full_int_array_195, full_int_array_196, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_193 = paddle._C_ops.gather(transpose_4, slice_193, assign_3)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_197 = [195]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_194 = paddle._C_ops.slice(data_1, [0], full_int_array_196, full_int_array_197, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_194 = paddle._C_ops.gather(transpose_4, slice_194, assign_2)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_198 = [196]

        # pd_op.slice: (196xi64) <- (196x196xi64, 1xi64, 1xi64)
        slice_195 = paddle._C_ops.slice(data_1, [0], full_int_array_197, full_int_array_198, [1], [0])

        # pd_op.gather: (196x4xf32) <- (196x4xf32, 196xi64, 1xi32)
        gather_195 = paddle._C_ops.gather(transpose_4, slice_195, assign_1)

        # builtin.combine: ([196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32]) <- (196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32)
        combine_0 = [gather_0, gather_1, gather_2, gather_3, gather_4, gather_5, gather_6, gather_7, gather_8, gather_9, gather_10, gather_11, gather_12, gather_13, gather_14, gather_15, gather_16, gather_17, gather_18, gather_19, gather_20, gather_21, gather_22, gather_23, gather_24, gather_25, gather_26, gather_27, gather_28, gather_29, gather_30, gather_31, gather_32, gather_33, gather_34, gather_35, gather_36, gather_37, gather_38, gather_39, gather_40, gather_41, gather_42, gather_43, gather_44, gather_45, gather_46, gather_47, gather_48, gather_49, gather_50, gather_51, gather_52, gather_53, gather_54, gather_55, gather_56, gather_57, gather_58, gather_59, gather_60, gather_61, gather_62, gather_63, gather_64, gather_65, gather_66, gather_67, gather_68, gather_69, gather_70, gather_71, gather_72, gather_73, gather_74, gather_75, gather_76, gather_77, gather_78, gather_79, gather_80, gather_81, gather_82, gather_83, gather_84, gather_85, gather_86, gather_87, gather_88, gather_89, gather_90, gather_91, gather_92, gather_93, gather_94, gather_95, gather_96, gather_97, gather_98, gather_99, gather_100, gather_101, gather_102, gather_103, gather_104, gather_105, gather_106, gather_107, gather_108, gather_109, gather_110, gather_111, gather_112, gather_113, gather_114, gather_115, gather_116, gather_117, gather_118, gather_119, gather_120, gather_121, gather_122, gather_123, gather_124, gather_125, gather_126, gather_127, gather_128, gather_129, gather_130, gather_131, gather_132, gather_133, gather_134, gather_135, gather_136, gather_137, gather_138, gather_139, gather_140, gather_141, gather_142, gather_143, gather_144, gather_145, gather_146, gather_147, gather_148, gather_149, gather_150, gather_151, gather_152, gather_153, gather_154, gather_155, gather_156, gather_157, gather_158, gather_159, gather_160, gather_161, gather_162, gather_163, gather_164, gather_165, gather_166, gather_167, gather_168, gather_169, gather_170, gather_171, gather_172, gather_173, gather_174, gather_175, gather_176, gather_177, gather_178, gather_179, gather_180, gather_181, gather_182, gather_183, gather_184, gather_185, gather_186, gather_187, gather_188, gather_189, gather_190, gather_191, gather_192, gather_193, gather_194, gather_195]

        # pd_op.concat: (38416x4xf32) <- ([196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32, 196x4xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, assign_0)

        # pd_op.transpose: (4x38416xf32) <- (38416x4xf32)
        transpose_5 = paddle._C_ops.transpose(concat_0, [1, 0])

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_199 = [0, 196, 196]

        # pd_op.reshape: (4x196x196xf32, 0x4x38416xi64) <- (4x38416xf32, 3xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(transpose_5, full_int_array_199), lambda out: out if isinstance(out, (list, tuple)) else (out, None))
        return reshape_1, full_0, transpose_4, slice_0, full_1, gather_0, slice_1, assign_195, gather_1, slice_2, assign_194, gather_2, slice_3, assign_193, gather_3, slice_4, assign_192, gather_4, slice_5, assign_191, gather_5, slice_6, assign_190, gather_6, slice_7, assign_189, gather_7, slice_8, assign_188, gather_8, slice_9, assign_187, gather_9, slice_10, assign_186, gather_10, slice_11, assign_185, gather_11, slice_12, assign_184, gather_12, slice_13, assign_183, gather_13, slice_14, assign_182, gather_14, slice_15, assign_181, gather_15, slice_16, assign_180, gather_16, slice_17, assign_179, gather_17, slice_18, assign_178, gather_18, slice_19, assign_177, gather_19, slice_20, assign_176, gather_20, slice_21, assign_175, gather_21, slice_22, assign_174, gather_22, slice_23, assign_173, gather_23, slice_24, assign_172, gather_24, slice_25, assign_171, gather_25, slice_26, assign_170, gather_26, slice_27, assign_169, gather_27, slice_28, assign_168, gather_28, slice_29, assign_167, gather_29, slice_30, assign_166, gather_30, slice_31, assign_165, gather_31, slice_32, assign_164, gather_32, slice_33, assign_163, gather_33, slice_34, assign_162, gather_34, slice_35, assign_161, gather_35, slice_36, assign_160, gather_36, slice_37, assign_159, gather_37, slice_38, assign_158, gather_38, slice_39, assign_157, gather_39, slice_40, assign_156, gather_40, slice_41, assign_155, gather_41, slice_42, assign_154, gather_42, slice_43, assign_153, gather_43, slice_44, assign_152, gather_44, slice_45, assign_151, gather_45, slice_46, assign_150, gather_46, slice_47, assign_149, gather_47, slice_48, assign_148, gather_48, slice_49, assign_147, gather_49, slice_50, assign_146, gather_50, slice_51, assign_145, gather_51, slice_52, assign_144, gather_52, slice_53, assign_143, gather_53, slice_54, assign_142, gather_54, slice_55, assign_141, gather_55, slice_56, assign_140, gather_56, slice_57, assign_139, gather_57, slice_58, assign_138, gather_58, slice_59, assign_137, gather_59, slice_60, assign_136, gather_60, slice_61, assign_135, gather_61, slice_62, assign_134, gather_62, slice_63, assign_133, gather_63, slice_64, assign_132, gather_64, slice_65, assign_131, gather_65, slice_66, assign_130, gather_66, slice_67, assign_129, gather_67, slice_68, assign_128, gather_68, slice_69, assign_127, gather_69, slice_70, assign_126, gather_70, slice_71, assign_125, gather_71, slice_72, assign_124, gather_72, slice_73, assign_123, gather_73, slice_74, assign_122, gather_74, slice_75, assign_121, gather_75, slice_76, assign_120, gather_76, slice_77, assign_119, gather_77, slice_78, assign_118, gather_78, slice_79, assign_117, gather_79, slice_80, assign_116, gather_80, slice_81, assign_115, gather_81, slice_82, assign_114, gather_82, slice_83, assign_113, gather_83, slice_84, assign_112, gather_84, slice_85, assign_111, gather_85, slice_86, assign_110, gather_86, slice_87, assign_109, gather_87, slice_88, assign_108, gather_88, slice_89, assign_107, gather_89, slice_90, assign_106, gather_90, slice_91, assign_105, gather_91, slice_92, assign_104, gather_92, slice_93, assign_103, gather_93, slice_94, assign_102, gather_94, slice_95, assign_101, gather_95, slice_96, assign_100, gather_96, slice_97, assign_99, gather_97, slice_98, assign_98, gather_98, slice_99, assign_97, gather_99, slice_100, assign_96, gather_100, slice_101, assign_95, gather_101, slice_102, assign_94, gather_102, slice_103, assign_93, gather_103, slice_104, assign_92, gather_104, slice_105, assign_91, gather_105, slice_106, assign_90, gather_106, slice_107, assign_89, gather_107, slice_108, assign_88, gather_108, slice_109, assign_87, gather_109, slice_110, assign_86, gather_110, slice_111, assign_85, gather_111, slice_112, assign_84, gather_112, slice_113, assign_83, gather_113, slice_114, assign_82, gather_114, slice_115, assign_81, gather_115, slice_116, assign_80, gather_116, slice_117, assign_79, gather_117, slice_118, assign_78, gather_118, slice_119, assign_77, gather_119, slice_120, assign_76, gather_120, slice_121, assign_75, gather_121, slice_122, assign_74, gather_122, slice_123, assign_73, gather_123, slice_124, assign_72, gather_124, slice_125, assign_71, gather_125, slice_126, assign_70, gather_126, slice_127, assign_69, gather_127, slice_128, assign_68, gather_128, slice_129, assign_67, gather_129, slice_130, assign_66, gather_130, slice_131, assign_65, gather_131, slice_132, assign_64, gather_132, slice_133, assign_63, gather_133, slice_134, assign_62, gather_134, slice_135, assign_61, gather_135, slice_136, assign_60, gather_136, slice_137, assign_59, gather_137, slice_138, assign_58, gather_138, slice_139, assign_57, gather_139, slice_140, assign_56, gather_140, slice_141, assign_55, gather_141, slice_142, assign_54, gather_142, slice_143, assign_53, gather_143, slice_144, assign_52, gather_144, slice_145, assign_51, gather_145, slice_146, assign_50, gather_146, slice_147, assign_49, gather_147, slice_148, assign_48, gather_148, slice_149, assign_47, gather_149, slice_150, assign_46, gather_150, slice_151, assign_45, gather_151, slice_152, assign_44, gather_152, slice_153, assign_43, gather_153, slice_154, assign_42, gather_154, slice_155, assign_41, gather_155, slice_156, assign_40, gather_156, slice_157, assign_39, gather_157, slice_158, assign_38, gather_158, slice_159, assign_37, gather_159, slice_160, assign_36, gather_160, slice_161, assign_35, gather_161, slice_162, assign_34, gather_162, slice_163, assign_33, gather_163, slice_164, assign_32, gather_164, slice_165, assign_31, gather_165, slice_166, assign_30, gather_166, slice_167, assign_29, gather_167, slice_168, assign_28, gather_168, slice_169, assign_27, gather_169, slice_170, assign_26, gather_170, slice_171, assign_25, gather_171, slice_172, assign_24, gather_172, slice_173, assign_23, gather_173, slice_174, assign_22, gather_174, slice_175, assign_21, gather_175, slice_176, assign_20, gather_176, slice_177, assign_19, gather_177, slice_178, assign_18, gather_178, slice_179, assign_17, gather_179, slice_180, assign_16, gather_180, slice_181, assign_15, gather_181, slice_182, assign_14, gather_182, slice_183, assign_13, gather_183, slice_184, assign_12, gather_184, slice_185, assign_11, gather_185, slice_186, assign_10, gather_186, slice_187, assign_9, gather_187, slice_188, assign_8, gather_188, slice_189, assign_7, gather_189, slice_190, assign_6, gather_190, slice_191, assign_5, gather_191, slice_192, assign_4, gather_192, slice_193, assign_3, gather_193, slice_194, assign_2, gather_194, slice_195, assign_1, gather_195, assign_0, reshape_3, transpose_0, transpose_3, reshape_2, transpose_2



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

    def forward(self, parameter_0, data_0, data_1):
        return self.builtin_module_2261_0_0(parameter_0, data_0, data_1)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_2261_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_0
            paddle.uniform([4, 196], dtype='float32', min=0, max=0.5),
            # data_0
            paddle.uniform([22, 196, 256], dtype='float32', min=0, max=0.5),
            # data_1
            paddle.cast(paddle.randint(low=0, high=3, shape=[196, 196], dtype='int64'), 'int64'),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_0
            paddle.static.InputSpec(shape=[4, 196], dtype='float32'),
            # data_0
            paddle.static.InputSpec(shape=[22, 196, 256], dtype='float32'),
            # data_1
            paddle.static.InputSpec(shape=[196, 196], dtype='int64'),
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