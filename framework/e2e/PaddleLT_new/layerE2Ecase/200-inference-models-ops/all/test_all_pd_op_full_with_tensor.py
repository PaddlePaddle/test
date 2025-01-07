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
class PrimitiveOp_dc18319b3ebf33a4e580bf28ac56e577(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 26, 512], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float16)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5e0d335475c6070a4001e8a4c0b5dfbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dc18319b3ebf33a4e580bf28ac56e577
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 26, 512], dtype='int32').reshape([3]),
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

class PrimitiveOp_772e318ee6c74d7b6927f3085056f487(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 26], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float16)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c627f27c045ae15c5be264ef229dd318(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772e318ee6c74d7b6927f3085056f487
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 26], dtype='int32').reshape([2]),
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
class TestPrimitiveOp_8241190f896102f68345124d59a720af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_772e318ee6c74d7b6927f3085056f487
    def get_inputs(self):
        return [
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 26], dtype='int32').reshape([2]),
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

class PrimitiveOp_a3d15bbb1bd469d87b22b73be70b4a1d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 256], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9a170a8c0d896080b172c46288c04b46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3d15bbb1bd469d87b22b73be70b4a1d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 256], dtype='int32').reshape([2]),
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

class PrimitiveOp_dbbf67fa4fcfbd2e2f0dcbe7414269a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 501, 30], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1307e2efde5a189c769fe7ad97a6506b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dbbf67fa4fcfbd2e2f0dcbe7414269a6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 501, 30], dtype='int32').reshape([3]),
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

class PrimitiveOp_462dad65dbd8a80c1f8231c7f2ed2aee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 501, 4], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4ea8191ac709fe62d40ce85f1edeb986(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_462dad65dbd8a80c1f8231c7f2ed2aee
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 501, 4], dtype='int32').reshape([3]),
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

class PrimitiveOp_bf72cf224d942c8964f165546753f951(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_78435d2e77426e7bc875e00c2a97cbd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bf72cf224d942c8964f165546753f951
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
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

class PrimitiveOp_d0fec40081b26a8162575218da0323b0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 40], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2ddcc95079f45b5b3cc72a3d3ab54d25(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0fec40081b26a8162575218da0323b0
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 40], dtype='int32').reshape([2]),
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

class PrimitiveOp_5a31cff529ac372f32211e4530d57719(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2, 6, 128], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float16)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fda6ccbb5881de49c219c202dc66ad94(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a31cff529ac372f32211e4530d57719
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([2, 6, 128], dtype='int32').reshape([3]),
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

class PrimitiveOp_69454090d2ecc1577f4070ebe97cf239(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2, 1, 128], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float16)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c183c073b6c9ae792dfed2d9533d66f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_69454090d2ecc1577f4070ebe97cf239
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([2, 1, 128], dtype='int32').reshape([3]),
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

class PrimitiveOp_57a7c7e80797928fa55f1d584fa214ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 1], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_00d9540c3ee01c24b3af12929e0475b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_57a7c7e80797928fa55f1d584fa214ef
    def get_inputs(self):
        return [
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 1], dtype='int32').reshape([2]),
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

class PrimitiveOp_cad3bb54f9d3243f6f965e429740d6bb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 1], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7286e182de484cde95017ab030687341(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cad3bb54f9d3243f6f965e429740d6bb
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 1], dtype='int32').reshape([2]),
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

class PrimitiveOp_8ef6dcdcb06c4f7ae19a49e27695da55(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 1], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float16)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8515067526624af0e0c97141c7d0a60e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ef6dcdcb06c4f7ae19a49e27695da55
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 1], dtype='int32').reshape([2]),
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
class TestPrimitiveOp_bf25bee5229c8579ef0eb75978267cc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8ef6dcdcb06c4f7ae19a49e27695da55
    def get_inputs(self):
        return [
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int32').reshape([2]),
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

class PrimitiveOp_4eefc5fc3b336dbab6d0cd9f91f8d04b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_164f30b11541280dcd4d1895a64e1201(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4eefc5fc3b336dbab6d0cd9f91f8d04b
    def get_inputs(self):
        return [
            paddle.to_tensor([3.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
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

class PrimitiveOp_31d384f167ee050b3a4433c6165f1b61(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2, 1, 256], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float16)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6201978f1a544f9b0a4f7e5eeb62bac0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31d384f167ee050b3a4433c6165f1b61
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([2, 1, 256], dtype='int32').reshape([3]),
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

class PrimitiveOp_b08277089547f8a3911a9c88495427c8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 256], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float16)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9eb56f29fac2072f9668130e65611be7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b08277089547f8a3911a9c88495427c8
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 256], dtype='int32').reshape([2]),
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

class PrimitiveOp_7fe6b29980f115d360d7826d4d889513(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([4, 1, 96], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f599ee8eed59b87a9acc8de10be1466a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7fe6b29980f115d360d7826d4d889513
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([4, 1, 96], dtype='int32').reshape([3]),
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

class PrimitiveOp_cd0d4c943bc398aca0211c28048af935(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 96], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_40f38501046d18d98b3f37f8dd5e63a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cd0d4c943bc398aca0211c28048af935
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 96], dtype='int32').reshape([2]),
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

class PrimitiveOp_7a69b857a6a663225a03f8056dd807c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 501, 256], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float16)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_51f07521f762c0b482f5035833c848fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7a69b857a6a663225a03f8056dd807c2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 501, 256], dtype='int32').reshape([3]),
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

class PrimitiveOp_91a7ac4b34bf20d08ba0d82cf1d72f98(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2, 6, 128], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_14e179b55b013acb11d500ac5e70a445(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91a7ac4b34bf20d08ba0d82cf1d72f98
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([2, 6, 128], dtype='int32').reshape([3]),
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

class PrimitiveOp_d84f79d603a00690bb757b602f22b8b2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2, 1, 128], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fbd2bb409b48ce8d50ce923d252b2be9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d84f79d603a00690bb757b602f22b8b2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([2, 1, 128], dtype='int32').reshape([3]),
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

class PrimitiveOp_f66a74b52c3b71a6a1ecc01b69d91de2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 501, 256], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4eff3f62990e937c60c1e89c628efab5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f66a74b52c3b71a6a1ecc01b69d91de2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 501, 256], dtype='int32').reshape([3]),
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
class TestPrimitiveOp_d1a9e658335387ed452f334ec6288f5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cad3bb54f9d3243f6f965e429740d6bb
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 1], dtype='int32').reshape([2]),
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
class TestPrimitiveOp_2d100059c63524d0c7d83a8a4b8ddcce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cad3bb54f9d3243f6f965e429740d6bb
    def get_inputs(self):
        return [
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int32').reshape([2]),
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

class PrimitiveOp_c3133293be2795ef52fd991b16400ede(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2, 1, 256], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b43df499188ba4a2b50af4a35f6df200(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c3133293be2795ef52fd991b16400ede
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([2, 1, 256], dtype='int32').reshape([3]),
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

class PrimitiveOp_2beb20405be6920f9da714cc470a6e38(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 26, 512], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d87b6293354c27c853404e8849342f56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2beb20405be6920f9da714cc470a6e38
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 26, 512], dtype='int32').reshape([3]),
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

class PrimitiveOp_38707d5957d06a1e47e5dfe88a2f39a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 26], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a22c21a7d2bbfa3b681b01423361b4e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38707d5957d06a1e47e5dfe88a2f39a2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 26], dtype='int32').reshape([2]),
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
class TestPrimitiveOp_c17a9e822c72bc736b3104c0dbfbbd57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38707d5957d06a1e47e5dfe88a2f39a2
    def get_inputs(self):
        return [
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 26], dtype='int32').reshape([2]),
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

class PrimitiveOp_bfb53b8e416395e1afde55577fb40211(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 501, 30], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float16)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f7555a6d73fa75d8e854a2c7d8c4c98d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bfb53b8e416395e1afde55577fb40211
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 501, 30], dtype='int32').reshape([3]),
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

class PrimitiveOp_950c687826a95d7cf30536f1df1d8e76(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 501, 4], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float16)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bb49555f452ca8fc41be7b77ff1c4900(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_950c687826a95d7cf30536f1df1d8e76
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 501, 4], dtype='int32').reshape([3]),
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

class PrimitiveOp_258770dcb7b58b093352264ef9f33340(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([4, 1, 96], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float16)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_73905b65ae6978781eea90b584639db6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_258770dcb7b58b093352264ef9f33340
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([4, 1, 96], dtype='int32').reshape([3]),
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

class PrimitiveOp_77bfe86e01a660e6775dede4bf9bd94e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 96], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float16)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_58fbd4689fefc5107e8985c83eabcbad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77bfe86e01a660e6775dede4bf9bd94e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 96], dtype='int32').reshape([2]),
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

class PrimitiveOp_fecba528afc70728059a3b4e627c6c68(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 26, 512], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float16)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_681898e317eb2b418c3e8ff0ffe91420(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fecba528afc70728059a3b4e627c6c68
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 26, 512], dtype='int32').reshape([3]),
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

class PrimitiveOp_d41ebc0e99952ff8c3b5baf8cc11cf86(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 26], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float16)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_724eed606002bcddaeb07d0bfa198f04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d41ebc0e99952ff8c3b5baf8cc11cf86
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 26], dtype='int32').reshape([2]),
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
class TestPrimitiveOp_fddb377bd3b87503a4737797361aad8b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d41ebc0e99952ff8c3b5baf8cc11cf86
    def get_inputs(self):
        return [
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 26], dtype='int32').reshape([2]),
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

class PrimitiveOp_58f911ece26c72c9f2de1578fd403c18(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 256], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_79b3689448a9fb54a8b0d5bf83a37f02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_58f911ece26c72c9f2de1578fd403c18
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 256], dtype='int32').reshape([2]),
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

class PrimitiveOp_1563239586ccbd39fda9bb467e99d92f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 501, 30], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_96260322cc0a385bbf128b9d1da014b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1563239586ccbd39fda9bb467e99d92f
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 501, 30], dtype='int32').reshape([3]),
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

class PrimitiveOp_2ec0ba63c47540f394f27fa084ca23af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 501, 4], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_02af444dfc0cf9cbeedda53f1aa2ba75(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ec0ba63c47540f394f27fa084ca23af
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 501, 4], dtype='int32').reshape([3]),
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

class PrimitiveOp_012b4103412d5ad908b3c9adfb5b973e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.int32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c56e2658e31804836644a98bcbf49d7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_012b4103412d5ad908b3c9adfb5b973e
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
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

class PrimitiveOp_0237246d1fd899d107839bdb7a34a3b2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 40], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_40cfcb6abe6571d19f08db1bc04267ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0237246d1fd899d107839bdb7a34a3b2
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 40], dtype='int32').reshape([2]),
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

class PrimitiveOp_f910302e05e440de6c9a6219c5f74813(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2, 6, 128], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float16)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5092c2dbd0a1d33bffc781a8c22fd0d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f910302e05e440de6c9a6219c5f74813
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([2, 6, 128], dtype='int32').reshape([3]),
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

class PrimitiveOp_9e23b9c830599f5d413a9c641296735c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2, 1, 128], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float16)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d8f39e5339446612ab1bd41b972b43c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e23b9c830599f5d413a9c641296735c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([2, 1, 128], dtype='int32').reshape([3]),
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

class PrimitiveOp_b2379f29491078d62ee2edaf83119930(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 1], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_34083000b48ee5310fe544a2e1638a63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b2379f29491078d62ee2edaf83119930
    def get_inputs(self):
        return [
            paddle.to_tensor([2.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 1], dtype='int32').reshape([2]),
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

class PrimitiveOp_410fde3f01982df5e0c6012add1e4203(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 1], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_96560198f2f7c9223f2ed9933de85fbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_410fde3f01982df5e0c6012add1e4203
    def get_inputs(self):
        return [
            paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 1], dtype='int32').reshape([2]),
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

class PrimitiveOp_b236571dbb646ffe715688845480714a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 1], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float16)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c1b20cb4fda9d03530d1525dd17f91ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b236571dbb646ffe715688845480714a
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 1], dtype='int32').reshape([2]),
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
class TestPrimitiveOp_be218e0b4bc3a1fae87391fc2a94b408(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b236571dbb646ffe715688845480714a
    def get_inputs(self):
        return [
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int32').reshape([2]),
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

class PrimitiveOp_5fa0d7c41d4df0e998cf143ca438237d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.int64)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3559375932a3a921a7d65d33423e8c86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fa0d7c41d4df0e998cf143ca438237d
    def get_inputs(self):
        return [
            paddle.to_tensor([3.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
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

class PrimitiveOp_5efb7c658cd956d5acee7bfa46a1e7d1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2, 1, 256], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float16)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e9c91ccf46f3451d480095b8c15e55ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5efb7c658cd956d5acee7bfa46a1e7d1
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([2, 1, 256], dtype='int32').reshape([3]),
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

class PrimitiveOp_85bcb25a9936ef4d4186ee71576de1e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 256], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float16)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_21a4a0b076e625e0d017d80b048f7fd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85bcb25a9936ef4d4186ee71576de1e2
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 256], dtype='int32').reshape([2]),
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

class PrimitiveOp_aa792f1f24facbea78a6c37e3332ca38(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([4, 1, 96], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4538d4b14ed3aae328deb4e974d7ce99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa792f1f24facbea78a6c37e3332ca38
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([4, 1, 96], dtype='int32').reshape([3]),
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

class PrimitiveOp_fdbecb8219fdbf9786fa363c8890d119(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 96], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4391f1adf062119bceca4b7b17184861(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fdbecb8219fdbf9786fa363c8890d119
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 96], dtype='int32').reshape([2]),
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

class PrimitiveOp_53562f64bfc5fed81093674f1f364790(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 501, 256], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float16)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_649544ff6ef933fd2c69914e660a155e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_53562f64bfc5fed81093674f1f364790
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 501, 256], dtype='int32').reshape([3]),
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

class PrimitiveOp_44c530e0796bc68a67c3ee316f80378f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2, 6, 128], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_743ff49c000e330f684c25a6b7f48dca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44c530e0796bc68a67c3ee316f80378f
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([2, 6, 128], dtype='int32').reshape([3]),
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

class PrimitiveOp_5c77fe0b8691f448e8edb9801fb420c4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2, 1, 128], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_59077af64c251c3ee79b56c54a3e9aa5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c77fe0b8691f448e8edb9801fb420c4
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([2, 1, 128], dtype='int32').reshape([3]),
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

class PrimitiveOp_0f90cb91d16e386673841f0369dd9d03(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 501, 256], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c61928fe629c8c2e3b91756f4de73a04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0f90cb91d16e386673841f0369dd9d03
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 501, 256], dtype='int32').reshape([3]),
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
class TestPrimitiveOp_693e5b6f7669d10b5e2efa6488356b27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_410fde3f01982df5e0c6012add1e4203
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 1], dtype='int32').reshape([2]),
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
class TestPrimitiveOp_761bca8b09c14d3e2c242801fc176fd0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_410fde3f01982df5e0c6012add1e4203
    def get_inputs(self):
        return [
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 1], dtype='int32').reshape([2]),
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

class PrimitiveOp_cdc76f67965975a55558fad924248354(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([2, 1, 256], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c1cdbc9b5dc47fd3dafe844fa68f0694(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cdc76f67965975a55558fad924248354
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([2, 1, 256], dtype='int32').reshape([3]),
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

class PrimitiveOp_24d8f075b34f423c868f7d0d17815541(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 26, 512], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_44bdbb7c46109f7df9c97c86ec76c692(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24d8f075b34f423c868f7d0d17815541
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 26, 512], dtype='int32').reshape([3]),
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

class PrimitiveOp_3ffcb7700fcf383d444db2c9d79ef06d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 26], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float32)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_414f775c32b4ca646e72a4e3d70bb5b9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ffcb7700fcf383d444db2c9d79ef06d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 26], dtype='int32').reshape([2]),
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
class TestPrimitiveOp_8e968319aecb7b889f60a0ac5e32befc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3ffcb7700fcf383d444db2c9d79ef06d
    def get_inputs(self):
        return [
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 26], dtype='int32').reshape([2]),
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

class PrimitiveOp_bb059134bc9830348ac21efbb0214112(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 501, 30], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float16)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b5e5fa01eba3100e5a2cd5b83b2d08a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb059134bc9830348ac21efbb0214112
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 501, 30], dtype='int32').reshape([3]),
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

class PrimitiveOp_9d82d3b70b327510e2767767ff144353(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 501, 4], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float16)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_479cda3477c32f66974b2d9ba3997283(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9d82d3b70b327510e2767767ff144353
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 501, 4], dtype='int32').reshape([3]),
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

class PrimitiveOp_a104aefd7cd59988f97ae905b945cb9f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([4, 1, 96], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float16)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[3], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c19f97120e86af07c44132d68407fce3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a104aefd7cd59988f97ae905b945cb9f
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([4, 1, 96], dtype='int32').reshape([3]),
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

class PrimitiveOp_31f593f5ffbd82671848f5b55bd01e4d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        arg_1 = paddle._C_ops.full_int_array([1, 96], paddle.int32, paddle.core.CPUPlace())
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.full_with_tensor(input_0, input_1, paddle.float16)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            paddle.static.InputSpec(shape=[2], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b519b21451a958181a3cb4b4bd4961e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_31f593f5ffbd82671848f5b55bd01e4d
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            paddle.to_tensor([1, 96], dtype='int32').reshape([2]),
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