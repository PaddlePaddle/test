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
class PrimitiveOp_105cc40438ac8027928fc639a25c56bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0c326c792233ad3eadee46148c80a2f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_105cc40438ac8027928fc639a25c56bf
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([16, 1]),
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
class TestPrimitiveOp_b93ec1abbe7577a7bdee48c712a50d17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_105cc40438ac8027928fc639a25c56bf
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([16, 1]),
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

class PrimitiveOp_5a67b113cd890a95214feacc62f30cd6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ed55b33b4a0c3253545b3122908347d8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a67b113cd890a95214feacc62f30cd6
    def get_inputs(self):
        return [
            paddle.uniform([1843, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[1843, 4, 1], dtype='int64'), 'int64'),
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
class TestPrimitiveOp_3ef41c29f0aba664bf4c7dcdc37c1acf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a67b113cd890a95214feacc62f30cd6
    def get_inputs(self):
        return [
            paddle.uniform([5583, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[5583, 4, 1], dtype='int64'), 'int64'),
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
class TestPrimitiveOp_ad771ff6bd035addf04fb3a9322acc6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_105cc40438ac8027928fc639a25c56bf
    def get_inputs(self):
        return [
            paddle.uniform([36, 17], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[36, 1], dtype='int64'), 'int64'),
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
class TestPrimitiveOp_e1f3bf2be8e7f5822fadb9de3c232af3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a67b113cd890a95214feacc62f30cd6
    def get_inputs(self):
        return [
            paddle.uniform([1724, 4, 19], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[1724, 4, 1], dtype='int64'), 'int64'),
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
class TestPrimitiveOp_0a994550fbd94ebcd3598c16865f9f2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_105cc40438ac8027928fc639a25c56bf
    def get_inputs(self):
        return [
            paddle.uniform([24, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([24, 1]),
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
class TestPrimitiveOp_357cc9db35bbfaa4e01dbac2f8775e56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_105cc40438ac8027928fc639a25c56bf
    def get_inputs(self):
        return [
            paddle.uniform([24, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([24, 1]),
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
class TestPrimitiveOp_a657b3123451a30c164d506b266ebe3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a67b113cd890a95214feacc62f30cd6
    def get_inputs(self):
        return [
            paddle.uniform([1492, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[1492, 4, 1], dtype='int64'), 'int64'),
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
class TestPrimitiveOp_1bde9e34de32ebce30faefe4b1fcafeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_105cc40438ac8027928fc639a25c56bf
    def get_inputs(self):
        return [
            paddle.uniform([4, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [0], [0], [0]], dtype='int64').reshape([4, 1]),
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
class TestPrimitiveOp_e129bf81d938f44c3743998c0fc80c4c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_105cc40438ac8027928fc639a25c56bf
    def get_inputs(self):
        return [
            paddle.uniform([4, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [1], [1], [1]], dtype='int64').reshape([4, 1]),
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
class TestPrimitiveOp_7e8a4556097713406670339ff2b1a032(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a67b113cd890a95214feacc62f30cd6
    def get_inputs(self):
        return [
            paddle.uniform([2079, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[2079, 4, 1], dtype='int64'), 'int64'),
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
class TestPrimitiveOp_96b57f0a28cf249c66cd99005e8a54a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a67b113cd890a95214feacc62f30cd6
    def get_inputs(self):
        return [
            paddle.uniform([4526, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[4526, 4, 1], dtype='int64'), 'int64'),
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
class TestPrimitiveOp_8846710296824e6320442cd7afa431e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a67b113cd890a95214feacc62f30cd6
    def get_inputs(self):
        return [
            paddle.uniform([1, 2434, 81], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 2434, 1], dtype='int64'), 'int64'),
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
class TestPrimitiveOp_352935d60439a2035ba074dcbd537c03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a67b113cd890a95214feacc62f30cd6
    def get_inputs(self):
        return [
            paddle.uniform([1046, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[1046, 4, 1], dtype='int64'), 'int64'),
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
class TestPrimitiveOp_1167bf75f2975cc080e2e03bfa3ae0ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a67b113cd890a95214feacc62f30cd6
    def get_inputs(self):
        return [
            paddle.uniform([2335, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[2335, 4, 1], dtype='int64'), 'int64'),
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
class TestPrimitiveOp_87677b0600877efd45d572f3ed2cd0c9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a67b113cd890a95214feacc62f30cd6
    def get_inputs(self):
        return [
            paddle.uniform([2986, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[2986, 4, 1], dtype='int64'), 'int64'),
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
class TestPrimitiveOp_4b4e1a2d4117f8c26275ea9aebb1be69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a67b113cd890a95214feacc62f30cd6
    def get_inputs(self):
        return [
            paddle.uniform([3783, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[3783, 4, 1], dtype='int64'), 'int64'),
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
class TestPrimitiveOp_2e78ce8b08c5e028201c8cc06b7bcf34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_105cc40438ac8027928fc639a25c56bf
    def get_inputs(self):
        return [
            paddle.uniform([20, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([20, 1]),
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
class TestPrimitiveOp_c7399820bfdd65f06fec1767f2ef195c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_105cc40438ac8027928fc639a25c56bf
    def get_inputs(self):
        return [
            paddle.uniform([20, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([20, 1]),
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
class TestPrimitiveOp_0cb34d4df17ec1f4642f42085662075c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a67b113cd890a95214feacc62f30cd6
    def get_inputs(self):
        return [
            paddle.uniform([1, 8732, 21], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 8732, 1], dtype='int64'), 'int64'),
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
class TestPrimitiveOp_1735f09324b1c358b45cca02d3049900(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a67b113cd890a95214feacc62f30cd6
    def get_inputs(self):
        return [
            paddle.uniform([2041, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[2041, 4, 1], dtype='int64'), 'int64'),
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
class TestPrimitiveOp_3dce9dea33b14db673a41e9b64c1c334(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a67b113cd890a95214feacc62f30cd6
    def get_inputs(self):
        return [
            paddle.uniform([4210, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[4210, 4, 1], dtype='int64'), 'int64'),
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

class PrimitiveOp_91665e1816c0c72da91465e23d6f4fab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6617e5f44f13e52547619b8ed7ac771a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91665e1816c0c72da91465e23d6f4fab
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([16, 1]),
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
class TestPrimitiveOp_df113341b9d5898ac0b76e8f883a3aee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91665e1816c0c72da91465e23d6f4fab
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([16, 1]),
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

class PrimitiveOp_108890b611d94460938ee9ce491b990e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 17], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_872ee615a562f8a57761f78fdcc75813(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_108890b611d94460938ee9ce491b990e
    def get_inputs(self):
        return [
            paddle.uniform([1843, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[1843, 4, 1], dtype='int64'), 'int64'),
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
class TestPrimitiveOp_7cc4fbc72e84044ec46c5eec26a93d3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_108890b611d94460938ee9ce491b990e
    def get_inputs(self):
        return [
            paddle.uniform([5583, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[5583, 4, 1], dtype='int64'), 'int64'),
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
class TestPrimitiveOp_d994c8ec4d63636c2b387f56087ce5a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91665e1816c0c72da91465e23d6f4fab
    def get_inputs(self):
        return [
            paddle.uniform([36, 17], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[36, 1], dtype='int64'), 'int64'),
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

class PrimitiveOp_051bd2d2b291acc230b762a6a5378148(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 19], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 4, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a831639eac27fb2a33890c2f69bc9fda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_051bd2d2b291acc230b762a6a5378148
    def get_inputs(self):
        return [
            paddle.uniform([1724, 4, 19], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[1724, 4, 1], dtype='int64'), 'int64'),
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
class TestPrimitiveOp_bc3ec71759e550d98878bbf74a6328c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91665e1816c0c72da91465e23d6f4fab
    def get_inputs(self):
        return [
            paddle.uniform([24, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([24, 1]),
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
class TestPrimitiveOp_e90f816d811061de68f5754194a1a820(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91665e1816c0c72da91465e23d6f4fab
    def get_inputs(self):
        return [
            paddle.uniform([24, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([24, 1]),
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
class TestPrimitiveOp_4109d1990e81c2f18425b6e1e455728b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_108890b611d94460938ee9ce491b990e
    def get_inputs(self):
        return [
            paddle.uniform([1492, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[1492, 4, 1], dtype='int64'), 'int64'),
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
class TestPrimitiveOp_ae80d169df96e3a15f1c23d436383cc6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91665e1816c0c72da91465e23d6f4fab
    def get_inputs(self):
        return [
            paddle.uniform([4, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [0], [0], [0]], dtype='int64').reshape([4, 1]),
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
class TestPrimitiveOp_fec18607ca03d8c2b2e6e0833105a972(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91665e1816c0c72da91465e23d6f4fab
    def get_inputs(self):
        return [
            paddle.uniform([4, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [1], [1], [1]], dtype='int64').reshape([4, 1]),
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
class TestPrimitiveOp_e0072b7344104b7f633fed8b47d13353(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_108890b611d94460938ee9ce491b990e
    def get_inputs(self):
        return [
            paddle.uniform([2079, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[2079, 4, 1], dtype='int64'), 'int64'),
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
class TestPrimitiveOp_dbed0bc1d3600fab6d14cb4800e25e11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_108890b611d94460938ee9ce491b990e
    def get_inputs(self):
        return [
            paddle.uniform([4526, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[4526, 4, 1], dtype='int64'), 'int64'),
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

class PrimitiveOp_bb4d74ffa3c7d5c0aee2c65da6435f5b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return (lambda x, f: f(x))(paddle._C_ops.cross_entropy_with_softmax(input_0, input_1, False, True, True, -100, -1), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, 1], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_434d1249aee664b23995275f9de2d4d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb4d74ffa3c7d5c0aee2c65da6435f5b
    def get_inputs(self):
        return [
            paddle.uniform([1, 2434, 81], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 2434, 1], dtype='int64'), 'int64'),
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
class TestPrimitiveOp_361232b71b1a9f333110366dda7dda9b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_108890b611d94460938ee9ce491b990e
    def get_inputs(self):
        return [
            paddle.uniform([1046, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[1046, 4, 1], dtype='int64'), 'int64'),
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
class TestPrimitiveOp_835500d5e3769a9d09bdebc69b55212b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_108890b611d94460938ee9ce491b990e
    def get_inputs(self):
        return [
            paddle.uniform([2335, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[2335, 4, 1], dtype='int64'), 'int64'),
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
class TestPrimitiveOp_54936f8dd228a2379b5c5571e5aebd24(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_108890b611d94460938ee9ce491b990e
    def get_inputs(self):
        return [
            paddle.uniform([2986, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[2986, 4, 1], dtype='int64'), 'int64'),
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
class TestPrimitiveOp_721baab64713f315f5998aa64de8d7f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_108890b611d94460938ee9ce491b990e
    def get_inputs(self):
        return [
            paddle.uniform([3783, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[3783, 4, 1], dtype='int64'), 'int64'),
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
class TestPrimitiveOp_96f80c9cd482814fd8c495135b86329a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91665e1816c0c72da91465e23d6f4fab
    def get_inputs(self):
        return [
            paddle.uniform([20, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype='int64').reshape([20, 1]),
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
class TestPrimitiveOp_d7b2062729b8a573d0a06d4410263349(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_91665e1816c0c72da91465e23d6f4fab
    def get_inputs(self):
        return [
            paddle.uniform([20, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]], dtype='int64').reshape([20, 1]),
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
class TestPrimitiveOp_a724000dc796f79f3b6b5bf9ff483c89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb4d74ffa3c7d5c0aee2c65da6435f5b
    def get_inputs(self):
        return [
            paddle.uniform([1, 8732, 21], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[1, 8732, 1], dtype='int64'), 'int64'),
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
class TestPrimitiveOp_39fd17cb52e9b470fe5085d1ad3a90ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_108890b611d94460938ee9ce491b990e
    def get_inputs(self):
        return [
            paddle.uniform([2041, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[2041, 4, 1], dtype='int64'), 'int64'),
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
class TestPrimitiveOp_5d53857c6bef51f84bf58b7863ffdb68(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_108890b611d94460938ee9ce491b990e
    def get_inputs(self):
        return [
            paddle.uniform([4210, 4, 17], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[4210, 4, 1], dtype='int64'), 'int64'),
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