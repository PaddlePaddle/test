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
class PrimitiveOp_84be7f92ba86be05f9d540e78e64db39(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_857787eb72ee082d08c011f7181b248a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_a0ed31833ba6b8dbc7414fb29e2fe947(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_1d240c8d00c8cc15b2570b9634ef7abd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_b88a5ee190f1d54d1d10a80b5ad6b9d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_2867bf338828974815f0e5f66c5c0b57(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_78ef16638319710aa4d91eb16f0639c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_4ff0d4630b58239a54d74379a7334a31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_58afd93cfeff248a124a77fefc13d035(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_c8726b4c86971cae670eeadb1202b215(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_0f0478ab090042a5264f2ecf1a388297(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_7e203369d5e51e5883aad49834ec4c55(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_fe84ddc42990983a94e2e8f48c783b2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_318927b1e7362cb4e2a2e9e839852c9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_4603f857147d64844efaeb88d8fdd205(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_8d806933688527771612f7d9abee6963(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_5e988d42426764c140ef23e33c93ff71(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_d8a2e5e9e7de923c12c24a2b36b5d4c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_a517bf658ee77b81908929452c6c34a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_bd263b935ca13de436d432b2680b8e36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_b7f9c3f3485993eaf888748d95812597(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_3f240f090af2077eabda573a3af20966(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_0d58ae81855938334f8e916313dfe603(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_64b1d2cac49e6852771657640f3c3bd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_dafc38b9a14a45799bf830887d09f837(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_963f902a582085826d0b768c69e9d6d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_92541f806ccce430e9d453a2369763a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_c75a810e47b62bae60bf4bcdc0ee9ba3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_2a7d45c6e315b91dfcad77d0b96bc0c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_71d2ba0cdcbedc0715b33809062688bd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_cf382723269024f0c4a8e8a240389032(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_a3092c10091eb8fc5a3f236e56c36bda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_29e74d7c64349de249000123370971c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_f88ee3ddd035143ed32806e2bee92893(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_e6a3d285e88b7e77fcca93af775f8283(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_27f1a27384e2025fbdfb14e89d3b6ee8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_35fcc13c617ad8993aa4bf02b55ab514(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_5171dfa152136f5cf664f88461cc3f56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_bc31aeda536403a1e81010c7da617cac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_e7c3d00e41557a0d3707b8f25e08c42d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84be7f92ba86be05f9d540e78e64db39
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_e6845c6a1c7a4a396c48e981069fb530(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f4bb430ccd6c9b1a7cae39959a20f63f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e6845c6a1c7a4a396c48e981069fb530
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 384], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 196, 384], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2147483647], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_4648ad1888b75d7243234bdde29c1e76(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_04807b79de338c530e9a29f0456b312c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4648ad1888b75d7243234bdde29c1e76
    def get_inputs(self):
        return [
            paddle.uniform([1, 38, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 27, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_a53492b67c0c5b0d3ad23c58d9855523(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4648ad1888b75d7243234bdde29c1e76
    def get_inputs(self):
        return [
            paddle.uniform([1, 61, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 50, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([50], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_ca1dfaabb8838fe94524b4aa618ecb07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4648ad1888b75d7243234bdde29c1e76
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([72], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_33d20eb0fb7a27c04e395108b346be5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4648ad1888b75d7243234bdde29c1e76
    def get_inputs(self):
        return [
            paddle.uniform([1, 95, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 84, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([84], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_ddeb0dd5ee9b73eed8d381232d4ba842(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4648ad1888b75d7243234bdde29c1e76
    def get_inputs(self):
        return [
            paddle.uniform([1, 106, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 95, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([95], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_4f8ff116eabb8ba8b45aeced2434ed09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4648ad1888b75d7243234bdde29c1e76
    def get_inputs(self):
        return [
            paddle.uniform([1, 117, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 106, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([106], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_be3ac9321c60b21eb1e91b1ff1913075(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4648ad1888b75d7243234bdde29c1e76
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 117, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([117], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_76f149ddc53b52adb03368fcb982832b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4648ad1888b75d7243234bdde29c1e76
    def get_inputs(self):
        return [
            paddle.uniform([1, 151, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 140, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([140], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_734ace675b5b0f2833992a0b5a903c5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4648ad1888b75d7243234bdde29c1e76
    def get_inputs(self):
        return [
            paddle.uniform([1, 162, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 151, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([151], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_77374b7e87e3ec619f2ab555f66bb521(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4648ad1888b75d7243234bdde29c1e76
    def get_inputs(self):
        return [
            paddle.uniform([1, 174, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 162, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([162], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_5d24103ff589e6ecb59f913a4544fd59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4648ad1888b75d7243234bdde29c1e76
    def get_inputs(self):
        return [
            paddle.uniform([1, 185, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 174, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([174], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_334b9fcc5a393b1ff3ac27bd3f6ba4ac(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d366a1aba996c54865adad6f4021a5ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_334b9fcc5a393b1ff3ac27bd3f6ba4ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_7d71d0c3a554918e5a9483057d0d2b14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_334b9fcc5a393b1ff3ac27bd3f6ba4ac
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_8c7dc5b39e2964299148c78cf9e6692c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ec18d2390aef32b94280a0695e2efce9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c7dc5b39e2964299148c78cf9e6692c
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2147483647], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_bb240ad836879c820ff4fd785057dd99(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_95311466de0d6fef5f9e0e1106f7c2a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb240ad836879c820ff4fd785057dd99
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_eb2d8846588f6e87f3ca32371a51dda3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb240ad836879c820ff4fd785057dd99
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_b198699f4840869c484ca35aca5afd93(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cde2481b5a999a5c6a2ef44115c0136f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b198699f4840869c484ca35aca5afd93
    def get_inputs(self):
        return [
            paddle.uniform([1, 38, 56, 56], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 27, 56, 56], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_a9e0338b9e39136b1d49d14d3a078e44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b198699f4840869c484ca35aca5afd93
    def get_inputs(self):
        return [
            paddle.uniform([1, 61, 28, 28], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 50, 28, 28], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([50], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_da45a95ee356826f1ec71f856da40cc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b198699f4840869c484ca35aca5afd93
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 72, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([72], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_0b36774a9db76a23c0ad46397f95a91a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b198699f4840869c484ca35aca5afd93
    def get_inputs(self):
        return [
            paddle.uniform([1, 95, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 84, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([84], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_30d4db3e7031627a749531f3dd4cae9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b198699f4840869c484ca35aca5afd93
    def get_inputs(self):
        return [
            paddle.uniform([1, 106, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 95, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([95], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_b231f18ef5de7c417c52019b784689ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b198699f4840869c484ca35aca5afd93
    def get_inputs(self):
        return [
            paddle.uniform([1, 117, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 106, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([106], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_dab8f327321d10137799e0fb9e1e92ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b198699f4840869c484ca35aca5afd93
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 117, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([117], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_6b84c6f96fb9bd60d998adeb65a06f8a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b198699f4840869c484ca35aca5afd93
    def get_inputs(self):
        return [
            paddle.uniform([1, 151, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 140, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([140], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_89922a6e38b3d4b84acd62729c0d776d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b198699f4840869c484ca35aca5afd93
    def get_inputs(self):
        return [
            paddle.uniform([1, 162, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 151, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([151], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_c402739661026b8f89f5ead0835af7d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b198699f4840869c484ca35aca5afd93
    def get_inputs(self):
        return [
            paddle.uniform([1, 174, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 162, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([162], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_8adbdbd83d7a6fdcf9cd1d68b8ac828e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b198699f4840869c484ca35aca5afd93
    def get_inputs(self):
        return [
            paddle.uniform([1, 185, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 174, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([174], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_2881c736340e2bbd70cf2778867292ae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 40], dtype='int64'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_86446f8dc969d0f006dbfb3d44049937(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_d35d4bc21601db6059ff7f530689cc4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_be4de3c1fb66b5a92832e921a1d44c74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([3], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_75783aba885713b898807ac550deafc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([4], dtype='int64').reshape([1]),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_d86e316a2bee038cbac10306694ac23f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([5], dtype='int64').reshape([1]),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_db7d5bad00c68410f87b1e57d86fa66c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([6], dtype='int64').reshape([1]),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_a3137d8f6296c85a4723e08b2ac380bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([7], dtype='int64').reshape([1]),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_af990f1febfb7c9d62080285d05e11d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([8], dtype='int64').reshape([1]),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_5fdad93a44cf9eb0f7838207f3bdd66f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([9], dtype='int64').reshape([1]),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_357bdf527966a529c481a66aa582edf3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([10], dtype='int64').reshape([1]),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_b154de29fb33f495d43a99d1908d6a20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([11], dtype='int64').reshape([1]),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_1c5568809739097f8a587d51ff67e6d7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([12], dtype='int64').reshape([1]),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_08b1d5ed55182d84c768855d414befe3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([13], dtype='int64').reshape([1]),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_960a444a9c915ff0fbc023bd80d35466(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([14], dtype='int64').reshape([1]),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_9401b827929b9357460be222e48bf288(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([15], dtype='int64').reshape([1]),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_338e2e41cbd05acdc12bf110a38b3e0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([16], dtype='int64').reshape([1]),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_a349767758b6bf7f50a473aa03eb4d87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([17], dtype='int64').reshape([1]),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_2888c0201dc77c4b267fc0c923d691ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([18], dtype='int64').reshape([1]),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_8cd605b4e82c0087720ab6c34b1b3774(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([19], dtype='int64').reshape([1]),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_d57ee05f41c9526e6251ff22176bceb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([20], dtype='int64').reshape([1]),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_a8c2bfd4ff3c38dc221a55f6ab4d10e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([21], dtype='int64').reshape([1]),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_d9987d2c814043e5f093a72ed7bbc19a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([22], dtype='int64').reshape([1]),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_48bfa8c7005c0ada32da573df871aa24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([23], dtype='int64').reshape([1]),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_f4491f0c34ff5464cc09875d91c5d24e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([24], dtype='int64').reshape([1]),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_4e5425f839f9e8dd24e094615c5f7272(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([25], dtype='int64').reshape([1]),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_fce9ff473f2861fd25d65cacc90d9829(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([26], dtype='int64').reshape([1]),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_1e62059049fa9c47b28e75b31bd031a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_4d92e027a19c456c639bb334ceed7540(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([28], dtype='int64').reshape([1]),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_1a3f4b7dded3993e8798a9215c6af6a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([29], dtype='int64').reshape([1]),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_c785f858a9f3b4dbc6d5d9ce42af5ca0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([30], dtype='int64').reshape([1]),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_1b714aca8a52d3a1d5f9ef48ddf66f12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([31], dtype='int64').reshape([1]),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_4471e9a6559ae789aee16eeb18b21657(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([32], dtype='int64').reshape([1]),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_d881096564a5ffb3f8352aa1c511d546(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([33], dtype='int64').reshape([1]),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_ad3353772c564830277aa28b45622e8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([34], dtype='int64').reshape([1]),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_1df44c27f7eb0ce490d4a5988df53ac1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([35], dtype='int64').reshape([1]),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_cc685e92a4f6f455001bed44c6267271(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([36], dtype='int64').reshape([1]),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_d64ab3dbd0bbd2bc0382ff93486b3888(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([37], dtype='int64').reshape([1]),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_3e52a1973b9a7d70588bca670859ddc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([38], dtype='int64').reshape([1]),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_08adecd15e64ccc295f75e17d84a084b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2881c736340e2bbd70cf2778867292ae
    def get_inputs(self):
        return [
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 40], dtype='int64'), 'int64'),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([39], dtype='int64').reshape([1]),
            paddle.to_tensor([40], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_0bdabb63d5d8db5563420236ceb23def(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 197, 384], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 196, 384], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_96facb91e3b37fd4ac3b3fb56bec512f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bdabb63d5d8db5563420236ceb23def
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 384], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 196, 384], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2147483647], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_b448b0ef293ea1154868f2504a59538f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 38, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 27, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4f953001aced3db30ff5a562489ef95c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b448b0ef293ea1154868f2504a59538f
    def get_inputs(self):
        return [
            paddle.uniform([1, 38, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 27, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_7b8c07b312ff641f05d908c0e3e079bc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 61, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 50, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_747f9e83067355626158748cfa235574(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7b8c07b312ff641f05d908c0e3e079bc
    def get_inputs(self):
        return [
            paddle.uniform([1, 61, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 50, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([50], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_fcfd21cd7ea27448a1832c94a14aa413(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 84, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 72, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_efbd2f04f2e3fb726f72f8ac2abb6bb6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fcfd21cd7ea27448a1832c94a14aa413
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 72, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([72], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_6a9ca7f46ef1b06b9493cf3c45a384ce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 95, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 84, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3c554ba353d93bd945fa32d34487161f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a9ca7f46ef1b06b9493cf3c45a384ce
    def get_inputs(self):
        return [
            paddle.uniform([1, 95, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 84, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([84], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_88f0820cd07752baed97beafabc0ddbe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 106, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 95, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bd939d3735f894ec577326d4a09e0888(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88f0820cd07752baed97beafabc0ddbe
    def get_inputs(self):
        return [
            paddle.uniform([1, 106, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 95, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([95], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_3d4e3bdee0a219cbbfc40ecff8effe67(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 117, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 106, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a07d8448957c0f8396201ecb3599a1c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d4e3bdee0a219cbbfc40ecff8effe67
    def get_inputs(self):
        return [
            paddle.uniform([1, 117, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 106, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([106], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_436cbdec8ac4bb4fe7d52da332084f3b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 117, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d4c30daeca39a3832284d589496a8374(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_436cbdec8ac4bb4fe7d52da332084f3b
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 117, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([117], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_3a268dea5e349ea269520e14a64a96ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 151, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 140, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ca4d537b022614261ecf66b646b02f1f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a268dea5e349ea269520e14a64a96ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 151, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 140, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([140], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_fc552433e685d486f834a8062be206b8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 162, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 151, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9c1d0f4f3d8e85018aa293761317bf70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc552433e685d486f834a8062be206b8
    def get_inputs(self):
        return [
            paddle.uniform([1, 162, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 151, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([151], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_ba2ec0889d44be42a6557c04a539d6c0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 174, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 162, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cfadb3b93ea4d127cb4c3d663c87eaa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ba2ec0889d44be42a6557c04a539d6c0
    def get_inputs(self):
        return [
            paddle.uniform([1, 174, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 162, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([162], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_09c5c301b2a1be134de8f26cf8555a88(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 185, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 174, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bf67dac6efbfad8ff389d1331f65c6a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09c5c301b2a1be134de8f26cf8555a88
    def get_inputs(self):
        return [
            paddle.uniform([1, 185, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 174, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([174], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_9f660474257435c6434c4e7a77346ca5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 180, 320], dtype='float16'),
            paddle.static.InputSpec(shape=[1, 180, 320], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_02abd92390392e37018a4c14739cd6e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f660474257435c6434c4e7a77346ca5
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_fca97f4e6008cfc63af0045456da368f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9f660474257435c6434c4e7a77346ca5
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 180, 320], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_d45d050fb7edf440768313bcebbbc424(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 197, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 196, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_59d49f55929deef26019b95c7565845f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d45d050fb7edf440768313bcebbbc424
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2147483647], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_508be2b8e9530231da66f1b5f84b97b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [1], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[1, 2, 180, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[1, 180, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_39a9368df6046af4f9bc859f14cf4e0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_508be2b8e9530231da66f1b5f84b97b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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
class TestPrimitiveOp_bbe9354b29c9c55d47614ac91b9d7b02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_508be2b8e9530231da66f1b5f84b97b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 2, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 180, 320], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
            paddle.to_tensor([2], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_7ccd48f80c2df902e4651e5fdcd31412(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 38, 56, 56], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 27, 56, 56], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_700479989efd37e025ccdd201e02f31b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7ccd48f80c2df902e4651e5fdcd31412
    def get_inputs(self):
        return [
            paddle.uniform([1, 38, 56, 56], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 27, 56, 56], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([27], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_04376814eb425d0786d51c316882b15f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 61, 28, 28], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 50, 28, 28], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ed87b732d28eb09f4f515c1be4ae8faf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04376814eb425d0786d51c316882b15f
    def get_inputs(self):
        return [
            paddle.uniform([1, 61, 28, 28], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 50, 28, 28], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([50], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_2f4bd4102b6393a01b171ac688309e62(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 84, 14, 14], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 72, 14, 14], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d0026dd1b5094711aad0201a3d47b377(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f4bd4102b6393a01b171ac688309e62
    def get_inputs(self):
        return [
            paddle.uniform([1, 84, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 72, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([72], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_022794f87b7fa7a3d6371013306b7019(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 95, 14, 14], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 84, 14, 14], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_27a22d389d5133aa96bd128a01da7f2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_022794f87b7fa7a3d6371013306b7019
    def get_inputs(self):
        return [
            paddle.uniform([1, 95, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 84, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([84], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_7304a43dde7945e1843ce39c8d400a3a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 106, 14, 14], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 95, 14, 14], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_20bb4a131027ca411d00b103350508b3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7304a43dde7945e1843ce39c8d400a3a
    def get_inputs(self):
        return [
            paddle.uniform([1, 106, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 95, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([95], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_be6d9f579331d34070d79c3192879b2c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 117, 14, 14], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 106, 14, 14], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1e44f940108ff9b73b3a00b094bd49b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be6d9f579331d34070d79c3192879b2c
    def get_inputs(self):
        return [
            paddle.uniform([1, 117, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 106, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([106], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_3a9022bfb8d1d502486b193566b882a9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 14, 14], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 117, 14, 14], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_006a49354d8f25fdb0abcdf8ff3a4df6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3a9022bfb8d1d502486b193566b882a9
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 117, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([117], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_c8b624e441bc1f156d9c87ca81246840(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 151, 7, 7], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 140, 7, 7], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_afd6b15f43c6a80ee90a94983ca0071d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c8b624e441bc1f156d9c87ca81246840
    def get_inputs(self):
        return [
            paddle.uniform([1, 151, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 140, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([140], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_2c9c577888dc143666a1974d92b7efe2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 162, 7, 7], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 151, 7, 7], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5b65f02a5ff0691519ceb2b8540e3362(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2c9c577888dc143666a1974d92b7efe2
    def get_inputs(self):
        return [
            paddle.uniform([1, 162, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 151, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([151], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_06ccd2a26fa12e403369391fd287ac04(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 174, 7, 7], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 162, 7, 7], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0cb0bc66c48079b6451573b65306e78a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_06ccd2a26fa12e403369391fd287ac04
    def get_inputs(self):
        return [
            paddle.uniform([1, 174, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 162, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([162], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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

class PrimitiveOp_10d909129f6767cee7bbabc43bb4ec2d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3, arg_4):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        input_4 = arg_4
        return paddle._C_ops.set_value_with_tensor_(input_0, input_1, input_2, input_3, input_4, [1], [], [])

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 185, 7, 7], dtype='float16'),
            paddle.static.InputSpec(shape=[None, 174, 7, 7], dtype='float16'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
            paddle.static.InputSpec(shape=[1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_be3fe5dde6beac7dc24f47b2ecaef122(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10d909129f6767cee7bbabc43bb4ec2d
    def get_inputs(self):
        return [
            paddle.uniform([1, 185, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.uniform([1, 174, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0], dtype='int64').reshape([1]),
            paddle.to_tensor([174], dtype='int64').reshape([1]),
            paddle.to_tensor([1], dtype='int64').reshape([1]),
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