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
    return [53][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_212_0_0(self, data_1, data_2, data_5, data_0, data_3, data_4):

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('0'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_0 = full_0

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_1 = full_0

        # builtin.combine: ([-1x-1xf32]) <- (-1x-1xf32)
        combine_0 = [data_0]

        # pd_op.concat: (-1x-1xf32) <- ([-1x-1xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        # pd_op.full: (xf32) <- ()
        full_1 = paddle._C_ops.full([], float('2'), paddle.float32, paddle.framework._current_expected_place())

        # pd_op.assign: (xf32) <- (xf32)
        assign_2 = full_1

        # pd_op.elementwise_pow: (-1x-1xf32) <- (-1x-1xf32, xf32)
        elementwise_pow_0 = paddle.pow(data_1, full_1)

        # pd_op.full: (1xf32) <- ()
        full_2 = paddle._C_ops.full([1], float('0.75'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(elementwise_pow_0, full_2, float('0'), True)

        # pd_op.full: (1xf32) <- ()
        full_3 = paddle._C_ops.full([1], float('-1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_3 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_4 = full_3

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_5 = full_3

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(data_1, full_3, float('1'), True)

        # pd_op.assign: (-1x-1xf32) <- (-1x-1xf32)
        assign_6 = scale_1

        # pd_op.full: (1xf32) <- ()
        full_4 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_7 = full_4

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(scale_1, full_4, float('1e-08'), True)

        # pd_op.log: (-1x-1xf32) <- (-1x-1xf32)
        log_0 = paddle._C_ops.log(scale_2)

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_3 = paddle._C_ops.scale(log_0, assign_5, float('0'), True)

        # pd_op.multiply: (-1x-1xf32) <- (-1x-1xf32, -1x-1xf32)
        multiply_0 = scale_0 * scale_3

        # pd_op.elementwise_pow: (-1x-1xf32) <- (-1x-1xf32, xf32)
        elementwise_pow_1 = paddle.pow(assign_6, assign_2)

        # pd_op.full: (1xf32) <- ()
        full_5 = paddle._C_ops.full([1], float('0.25'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_4 = paddle._C_ops.scale(elementwise_pow_1, full_5, float('0'), True)

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_5 = paddle._C_ops.scale(data_1, assign_7, float('1e-08'), True)

        # pd_op.log: (-1x-1xf32) <- (-1x-1xf32)
        log_1 = paddle._C_ops.log(scale_5)

        # pd_op.scale: (-1x-1xf32) <- (-1x-1xf32, 1xf32)
        scale_6 = paddle._C_ops.scale(log_1, assign_3, float('0'), True)

        # pd_op.multiply: (-1x-1xf32) <- (-1x-1xf32, -1x-1xf32)
        multiply_1 = scale_4 * scale_6

        # pd_op.full: (1xi32) <- ()
        full_6 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # pd_op.assign: (1xi32) <- (1xi32)
        assign_8 = full_6

        # pd_op.gather: (-1x-1xf32) <- (-1x-1xf32, -1xi32, 1xi32)
        gather_0 = paddle._C_ops.gather(multiply_1, data_2, full_6)

        # pd_op.gather: (-1x-1xf32) <- (-1x-1xf32, -1xi32, 1xi32)
        gather_1 = paddle._C_ops.gather(multiply_0, data_2, assign_8)

        # pd_op.subtract: (-1x-1xf32) <- (-1x-1xf32, -1x-1xf32)
        subtract_0 = gather_0 - gather_1

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_0 = [0]

        # pd_op.unsqueeze: (1x-1xf32, 0x-1xf32) <- (-1xf32, 1xi64)
        unsqueeze_0, unsqueeze_1 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(data_3, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([1x-1xf32]) <- (1x-1xf32)
        combine_1 = [unsqueeze_0]

        # pd_op.concat: (1x-1xf32) <- ([1x-1xf32], 1xi32)
        concat_1 = paddle._C_ops.concat(combine_1, assign_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_1 = [1]

        # pd_op.unsqueeze: (1x1x-1xf32, 0x1x-1xf32) <- (1x-1xf32, 1xi64)
        unsqueeze_2, unsqueeze_3 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(concat_1, full_int_array_1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.full_int_array: (3xi64) <- ()
        full_int_array_2 = [1, 100, 1]

        # pd_op.tile: (1x100x-1xf32) <- (1x1x-1xf32, 3xi64)
        tile_0 = paddle._C_ops.tile(unsqueeze_2, full_int_array_2)

        # pd_op.flatten: (100x-1xf32, 0x1x100x-1xf32) <- (1x100x-1xf32)
        flatten_0, flatten_1 = (lambda x, f: f(x))(paddle._C_ops.flatten(tile_0, 0, 1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # builtin.combine: ([-1x-1xf32]) <- (-1x-1xf32)
        combine_2 = [data_4]

        # pd_op.concat: (-1x-1xf32) <- ([-1x-1xf32], 1xi32)
        concat_2 = paddle._C_ops.concat(combine_2, assign_0)

        # pd_op.divide: (100x-1xf32) <- (-1x-1xf32, 100x-1xf32)
        divide_0 = data_5 / flatten_0

        # pd_op.divide: (-1x-1xf32) <- (-1x-1xf32, -1x-1xf32)
        divide_1 = concat_0 / concat_2

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_3 = [-2]

        # pd_op.unsqueeze: (100x1x-1xf32, 0x100x-1xf32) <- (100x-1xf32, 1xi64)
        unsqueeze_4, unsqueeze_5 = (lambda x, f: f(x))(paddle._C_ops.unsqueeze(divide_0, full_int_array_3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.subtract: (100x-1x-1xf32) <- (100x1x-1xf32, -1x-1xf32)
        subtract_1 = unsqueeze_4 - divide_1

        # pd_op.abs: (100x-1x-1xf32) <- (100x-1x-1xf32)
        abs_0 = paddle._C_ops.abs(subtract_1)

        # pd_op.full_int_array: (1xi64) <- ()
        full_int_array_4 = [-1]

        # pd_op.sum: (100x-1xf32) <- (100x-1x-1xf32, 1xi64)
        sum_0 = paddle._C_ops.sum(abs_0, full_int_array_4, None, False)
        return full_0, full_1, full_2, scale_0, full_3, full_4, scale_2, assign_5, scale_3, multiply_0, assign_4, assign_6, assign_2, full_5, scale_4, assign_7, scale_5, assign_3, scale_6, multiply_1, full_6, gather_0, assign_8, gather_1, full_int_array_0, unsqueeze_0, unsqueeze_1, assign_1, full_int_array_1, unsqueeze_2, unsqueeze_3, full_int_array_2, flatten_0, flatten_1, assign_0, concat_2, divide_0, divide_1, full_int_array_3, unsqueeze_4, unsqueeze_5, subtract_1, abs_0, full_int_array_4, concat_0, sum_0, subtract_0



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

    def forward(self, data_1, data_2, data_5, data_0, data_3, data_4):
        return self.builtin_module_212_0_0(data_1, data_2, data_5, data_0, data_3, data_4)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_212_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # data_1
            paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            # data_2
            paddle.to_tensor([5, 3], dtype='int32').reshape([2]),
            # data_5
            paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
            # data_0
            paddle.uniform([2, 4], dtype='float32', min=0, max=0.5),
            # data_3
            paddle.uniform([4], dtype='float32', min=0, max=0.5),
            # data_4
            paddle.uniform([2, 4], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # data_1
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            # data_2
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            # data_5
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            # data_0
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            # data_3
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            # data_4
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
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