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
import itertools

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
        return True, "last stage failed."
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
    if enable_cinn is None:
        return True
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

def ApplyToStatic(net, use_cinn):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(
        net,
        input_spec=net.get_input_spec(),
        build_strategy=build_strategy,
        full_graph=True,
    )

class InstanceTrait:

    @classmethod
    def instance(cls):
        if cls.instance_ is None:
            cls.instance_ = cls()
        return cls.instance_

    @classmethod
    def static_instance_with_cinn(cls):
        if cls.static_instance_with_cinn_ is None:
            cls.static_instance_with_cinn_ = ApplyToStatic(
                cls.instance(),
                use_cinn=True
            )
        return cls.static_instance_with_cinn_

    @classmethod
    def static_instance_without_cinn(cls):
        if cls.static_instance_without_cinn_ is None:
            cls.static_instance_without_cinn_ = ApplyToStatic(
                cls.instance(),
                use_cinn=False
            )
        return cls.static_instance_without_cinn_


class CinnTestBase:

    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def _test_entry(self):
        dy_outs = self.train(use_cinn=False)
        cinn_outs = self.train(use_cinn=GetEnvVarEnableCinn())

        for cinn_out, dy_out in zip(cinn_outs, dy_outs):
          if type(cinn_out) is list and type(dy_out) is list:
            for x, y in zip(cinn_out, dy_out):
              self.assert_all_close(x, y)
          else:
            self.assert_all_close(cinn_out, dy_out)

    def train(self, use_cinn):
        if GetEnvVarEnableJit():
            net = self.prepare_static_net(use_cinn)
        else:
            net = self.prepare_net()
        paddle.seed(2024)
        out = net(*self.inputs)
        return out
    
    def prepare_data(self):
        self.inputs = self.get_inputs()
        for input in self.inputs:
            input.stop_gradient = True

    def prepare_net(self):
        return self.get_test_class().instance()

    def prepare_static_net(self, use_cinn):
        if use_cinn:
            return self.get_test_class().static_instance_with_cinn()
        else:
            return self.get_test_class().static_instance_without_cinn()

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
class PrimitiveOp_ecc157d6968fd5a3f4fc982795e26dc6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3):
        arg_1 = paddle._C_ops.full_int_array([2147483647], paddle.int64, paddle.core.CPUPlace())
        arg_2 = paddle._C_ops.full_int_array([-1], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([-1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        return paddle._C_ops.strided_slice(input_0, [0], input_1, input_2, input_3)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='int32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b2e81f6483d0f873ba11e4ed4c41e41f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ecc157d6968fd5a3f4fc982795e26dc6
    def get_inputs(self):
        return [
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
            paddle.to_tensor([2147483647], dtype='int64').reshape([1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_90b8acafcc2e5d8159f2ac49ff18ecf0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ecc157d6968fd5a3f4fc982795e26dc6
    def get_inputs(self):
        return [
            paddle.to_tensor([256, 256], dtype='int32').reshape([2]),
            paddle.to_tensor([2147483647], dtype='int64').reshape([1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_97159a7fd03356c20552dec4900e25d6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3):
        arg_1 = paddle._C_ops.full_int_array([0, 0], paddle.int64, paddle.core.CPUPlace())
        arg_2 = paddle._C_ops.full_int_array([14, 14], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([2, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        return paddle._C_ops.strided_slice(input_0, [1, 2], input_1, input_2, input_3)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c4d4b9bf7648abae92910be6ddf8bfc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_97159a7fd03356c20552dec4900e25d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 14, 14, 128], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0, 0], dtype='int64').reshape([2]),
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e495a82c05cf21178be4ff0d25f77747(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3):
        arg_1 = paddle._C_ops.full_int_array([0, 0], paddle.int64, paddle.core.CPUPlace())
        arg_2 = paddle._C_ops.full_int_array([7, 7], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([2, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        return paddle._C_ops.strided_slice(input_0, [1, 2], input_1, input_2, input_3)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b04d11cc4e9ecd10c05858d2442f355c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e495a82c05cf21178be4ff0d25f77747
    def get_inputs(self):
        return [
            paddle.uniform([1, 7, 7, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0, 0], dtype='int64').reshape([2]),
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_50678e48892df32e5604a38dcd1df169(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3):
        arg_1 = paddle._C_ops.full_int_array([0, 0], paddle.int64, paddle.core.CPUPlace())
        arg_2 = paddle._C_ops.full_int_array([14, 14], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([2, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        return paddle._C_ops.strided_slice(input_0, [1, 2], input_1, input_2, input_3)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eb9c50ef7ea69e0d562ff8e2ed0d76af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50678e48892df32e5604a38dcd1df169
    def get_inputs(self):
        return [
            paddle.uniform([1, 14, 14, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 0], dtype='int64').reshape([2]),
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0875e0fa292ecd70bcdd76eee0e51425(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3):
        arg_1 = paddle._C_ops.full_int_array([0, 0], paddle.int64, paddle.core.CPUPlace())
        arg_2 = paddle._C_ops.full_int_array([7, 7], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([2, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        return paddle._C_ops.strided_slice(input_0, [1, 2], input_1, input_2, input_3)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cb07a9b53722c4eb1a1c53be277e75d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0875e0fa292ecd70bcdd76eee0e51425
    def get_inputs(self):
        return [
            paddle.uniform([1, 7, 7, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 0], dtype='int64').reshape([2]),
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0fc2e6a801debfcba31eae725ca57e84(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3):
        arg_1 = paddle._C_ops.full_int_array([2147483647], paddle.int64, paddle.core.CPUPlace())
        arg_2 = paddle._C_ops.full_int_array([-1], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([-1], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        return paddle._C_ops.strided_slice(input_0, [0], input_1, input_2, input_3)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[2], dtype='int32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6e785cea4077af9c60bc746f51f9a738(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fc2e6a801debfcba31eae725ca57e84
    def get_inputs(self):
        return [
            paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
            paddle.to_tensor([2147483647], dtype='int64').reshape([1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_457378df28f79d2cbd79be2895524f4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0fc2e6a801debfcba31eae725ca57e84
    def get_inputs(self):
        return [
            paddle.to_tensor([256, 256], dtype='int32').reshape([2]),
            paddle.to_tensor([2147483647], dtype='int64').reshape([1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
            paddle.to_tensor([-1], dtype='int64').reshape([1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_24fa92b5acc93fdd05789def8a916ed8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3):
        arg_1 = paddle._C_ops.full_int_array([0, 0], paddle.int64, paddle.core.CPUPlace())
        arg_2 = paddle._C_ops.full_int_array([14, 14], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([2, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        return paddle._C_ops.strided_slice(input_0, [1, 2], input_1, input_2, input_3)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 14, 14, 128], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_91ca3533a4cb7116d6b1d02feea04532(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24fa92b5acc93fdd05789def8a916ed8
    def get_inputs(self):
        return [
            paddle.uniform([1, 14, 14, 128], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0, 0], dtype='int64').reshape([2]),
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7904b5afcb2ce01f37b4bf0f5edf9dfb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3):
        arg_1 = paddle._C_ops.full_int_array([0, 0], paddle.int64, paddle.core.CPUPlace())
        arg_2 = paddle._C_ops.full_int_array([7, 7], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([2, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        return paddle._C_ops.strided_slice(input_0, [1, 2], input_1, input_2, input_3)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 7, 7, 256], dtype='float16'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e6c9f58ffd59e46a7733890b9e388d0f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7904b5afcb2ce01f37b4bf0f5edf9dfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 7, 7, 256], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0, 0], dtype='int64').reshape([2]),
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1b9881c2204c4b06a292f06f9da8e9f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3):
        arg_1 = paddle._C_ops.full_int_array([0, 0], paddle.int64, paddle.core.CPUPlace())
        arg_2 = paddle._C_ops.full_int_array([14, 14], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([2, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        return paddle._C_ops.strided_slice(input_0, [1, 2], input_1, input_2, input_3)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 14, 14, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a1e09029cbc37984c75a3c355274125f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1b9881c2204c4b06a292f06f9da8e9f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 14, 14, 128], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 0], dtype='int64').reshape([2]),
            paddle.to_tensor([14, 14], dtype='int64').reshape([2]),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9849d41924feef63eadcee9e880c2151(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3):
        arg_1 = paddle._C_ops.full_int_array([0, 0], paddle.int64, paddle.core.CPUPlace())
        arg_2 = paddle._C_ops.full_int_array([7, 7], paddle.int64, paddle.core.CPUPlace())
        arg_3 = paddle._C_ops.full_int_array([2, 2], paddle.int64, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        return paddle._C_ops.strided_slice(input_0, [1, 2], input_1, input_2, input_3)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 7, 7, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
            paddle.static.InputSpec(shape=[2], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_287a078a3b88c361d2e8e4fb665a15ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9849d41924feef63eadcee9e880c2151
    def get_inputs(self):
        return [
            paddle.uniform([1, 7, 7, 256], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 0], dtype='int64').reshape([2]),
            paddle.to_tensor([7, 7], dtype='int64').reshape([2]),
            paddle.to_tensor([2, 2], dtype='int64').reshape([2]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()


if __name__ == '__main__':
    unittest.main()