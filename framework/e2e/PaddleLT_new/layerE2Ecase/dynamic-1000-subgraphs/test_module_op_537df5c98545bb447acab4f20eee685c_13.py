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
    return [33][block_idx] - 1 # number-of-ops-in-block

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
    def builtin_module_311_0_0(self, parameter_3, parameter_13, parameter_5, parameter_4, parameter_2, parameter_6, parameter_8, parameter_15, parameter_0, parameter_10, parameter_9, parameter_14, parameter_12, parameter_11, parameter_1, parameter_7, data_0, data_1, data_2, data_3):

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(data_0, parameter_0, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x256x1x1xf32, 0x256xi64) <- (256xf32, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_1, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_0 = conv2d_0 + reshape_0

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x512x-1x-1xf32, 256x512x1x1xf32)
        conv2d_1 = paddle._C_ops.conv2d(data_1, parameter_2, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x256x1x1xf32, 0x256xi64) <- (256xf32, 4xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_3, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_1 = conv2d_1 + reshape_2

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x1024x-1x-1xf32, 256x1024x1x1xf32)
        conv2d_2 = paddle._C_ops.conv2d(data_2, parameter_4, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x256x1x1xf32, 0x256xi64) <- (256xf32, 4xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_5, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_2 = conv2d_2 + reshape_4

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x2048x-1x-1xf32, 256x2048x1x1xf32)
        conv2d_3 = paddle._C_ops.conv2d(data_3, parameter_6, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x256x1x1xf32, 0x256xi64) <- (256xf32, 4xi64)
        reshape_6, reshape_7 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_7, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_3 = conv2d_3 + reshape_6

        # pd_op.nearest_interp: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, None, None, None)
        nearest_interp_0 = paddle._C_ops.nearest_interp(add_3, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 0)

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, -1x256x-1x-1xf32)
        add_4 = add_2 + nearest_interp_0

        # pd_op.nearest_interp: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, None, None, None)
        nearest_interp_1 = paddle._C_ops.nearest_interp(add_4, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 0)

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, -1x256x-1x-1xf32)
        add_5 = add_1 + nearest_interp_1

        # pd_op.nearest_interp: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, None, None, None)
        nearest_interp_2 = paddle._C_ops.nearest_interp(add_5, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'nearest', False, 0)

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, -1x256x-1x-1xf32)
        add_6 = add_0 + nearest_interp_2

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(add_6, parameter_8, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x256x1x1xf32, 0x256xi64) <- (256xf32, 4xi64)
        reshape_8, reshape_9 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_9, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_7 = conv2d_4 + reshape_8

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(add_5, parameter_10, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x256x1x1xf32, 0x256xi64) <- (256xf32, 4xi64)
        reshape_10, reshape_11 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_11, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_8 = conv2d_5 + reshape_10

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_6 = paddle._C_ops.conv2d(add_4, parameter_12, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x256x1x1xf32, 0x256xi64) <- (256xf32, 4xi64)
        reshape_12, reshape_13 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_13, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_9 = conv2d_6 + reshape_12

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_7 = paddle._C_ops.conv2d(add_3, parameter_14, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x256x1x1xf32, 0x256xi64) <- (256xf32, 4xi64)
        reshape_14, reshape_15 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_15, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x1x1xf32)
        add_10 = conv2d_7 + reshape_14

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [1, 1]

        # pd_op.pool2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(add_10, full_int_array_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')
        return conv2d_0, reshape_0, reshape_1, add_0, conv2d_1, reshape_2, reshape_3, add_1, conv2d_2, reshape_4, reshape_5, add_2, conv2d_3, reshape_6, reshape_7, add_3, nearest_interp_0, add_4, nearest_interp_1, add_5, nearest_interp_2, add_6, conv2d_4, reshape_8, reshape_9, conv2d_5, reshape_10, reshape_11, conv2d_6, reshape_12, reshape_13, conv2d_7, reshape_14, reshape_15, full_int_array_1, add_7, add_8, add_9, add_10, pool2d_0



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

    def forward(self, parameter_3, parameter_13, parameter_5, parameter_4, parameter_2, parameter_6, parameter_8, parameter_15, parameter_0, parameter_10, parameter_9, parameter_14, parameter_12, parameter_11, parameter_1, parameter_7, data_0, data_1, data_2, data_3):
        return self.builtin_module_311_0_0(parameter_3, parameter_13, parameter_5, parameter_4, parameter_2, parameter_6, parameter_8, parameter_15, parameter_0, parameter_10, parameter_9, parameter_14, parameter_12, parameter_11, parameter_1, parameter_7, data_0, data_1, data_2, data_3)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_311_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_3
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_13
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([256, 1024, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([256, 512, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([256, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_8
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_15
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_0
            paddle.uniform([256, 256, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_10
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_9
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_14
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_12
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_11
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # parameter_7
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            # data_0
            paddle.uniform([1, 256, 200, 272], dtype='float32', min=0, max=0.5),
            # data_1
            paddle.uniform([1, 512, 100, 136], dtype='float32', min=0, max=0.5),
            # data_2
            paddle.uniform([1, 1024, 50, 68], dtype='float32', min=0, max=0.5),
            # data_3
            paddle.uniform([1, 2048, 25, 34], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_3
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_13
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[256, 1024, 1, 1], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[256, 512, 1, 1], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[256, 2048, 1, 1], dtype='float32'),
            # parameter_8
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_15
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_0
            paddle.static.InputSpec(shape=[256, 256, 1, 1], dtype='float32'),
            # parameter_10
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_9
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_14
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_12
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_11
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # parameter_7
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            # data_0
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            # data_1
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            # data_2
            paddle.static.InputSpec(shape=[None, 1024, None, None], dtype='float32'),
            # data_3
            paddle.static.InputSpec(shape=[None, 2048, None, None], dtype='float32'),
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