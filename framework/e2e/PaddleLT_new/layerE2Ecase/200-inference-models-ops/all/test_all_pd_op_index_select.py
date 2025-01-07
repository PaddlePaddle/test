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
class PrimitiveOp_f78e2a0efb830f33b0549dcdfd4a74fe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.index_select(input_0, input_1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8249699d1591898abce86429e6bf898f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f78e2a0efb830f33b0549dcdfd4a74fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 56, 56], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([5, 6, 7, 4, 1, 2, 3, 0], dtype='int64').reshape([8]),
        ]


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
class TestPrimitiveOp_1135b77b6b10339f97fa5d27c22115fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f78e2a0efb830f33b0549dcdfd4a74fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 28, 28], dtype='float16', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[32], dtype='int64'), 'int64'),
        ]


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
class TestPrimitiveOp_5cde5201b1f67ac20be2659bd8fa95a0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f78e2a0efb830f33b0549dcdfd4a74fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 28, 28], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([4, 5, 3, 7, 8, 6, 10, 11, 9, 1, 2, 0], dtype='int64').reshape([12]),
        ]


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
class TestPrimitiveOp_2172de04dc7e38e34824e23b5d3bf7d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f78e2a0efb830f33b0549dcdfd4a74fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[48], dtype='int64'), 'int64'),
        ]


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
class TestPrimitiveOp_91cf1d2052966c4e22edc2740b1edf5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f78e2a0efb830f33b0549dcdfd4a74fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12, 1, 2, 3, 0], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_c5df3ea04b9cc8edc3f57f729026c3c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f78e2a0efb830f33b0549dcdfd4a74fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[64], dtype='int64'), 'int64'),
        ]


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
class TestPrimitiveOp_65dcbab2259ce90953e00aea94efda97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f78e2a0efb830f33b0549dcdfd4a74fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[32], dtype='int64'), 'int64'),
        ]


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
class TestPrimitiveOp_12198d89e23d7c72c13a2ed234cf2246(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f78e2a0efb830f33b0549dcdfd4a74fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[128], dtype='int64'), 'int64'),
        ]


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
class TestPrimitiveOp_f601bfbb1c7bba3f637af7e216e12a1b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f78e2a0efb830f33b0549dcdfd4a74fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[128], dtype='int64'), 'int64'),
        ]


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
class TestPrimitiveOp_67472344c4e57b20826d98efef9ccd78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f78e2a0efb830f33b0549dcdfd4a74fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[64], dtype='int64'), 'int64'),
        ]


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
class TestPrimitiveOp_e669a724f9dc64be79247919c38cd2ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f78e2a0efb830f33b0549dcdfd4a74fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[256], dtype='int64'), 'int64'),
        ]


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
class TestPrimitiveOp_b0f805397b5e19caf5ca45a018732a45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f78e2a0efb830f33b0549dcdfd4a74fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[96], dtype='int64'), 'int64'),
        ]


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
class TestPrimitiveOp_04070089638c4e17f609f3a165973903(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f78e2a0efb830f33b0549dcdfd4a74fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[384], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_df392a200a2991025eb95fced58823cb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.index_select(input_0, input_1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_11d66709e19de4385de4c3cc96435375(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df392a200a2991025eb95fced58823cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 6, 7, 4, 1, 2, 3, 0], dtype='int64').reshape([8]),
        ]


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
class TestPrimitiveOp_ff72ca746c8946b9be6a583bae7a8390(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df392a200a2991025eb95fced58823cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[32], dtype='int64'), 'int64'),
        ]


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
class TestPrimitiveOp_531b13eda956b01913467d934d9b8ea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df392a200a2991025eb95fced58823cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([4, 5, 3, 7, 8, 6, 10, 11, 9, 1, 2, 0], dtype='int64').reshape([12]),
        ]


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
class TestPrimitiveOp_c334182ff13783654d4194f14e3a750a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df392a200a2991025eb95fced58823cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[48], dtype='int64'), 'int64'),
        ]


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
class TestPrimitiveOp_3519e00645e6dd5e06478bf2ac2c34fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df392a200a2991025eb95fced58823cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12, 1, 2, 3, 0], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_9c9921402ce6352a360de7ee326262be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df392a200a2991025eb95fced58823cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[64], dtype='int64'), 'int64'),
        ]


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
class TestPrimitiveOp_fef4c2c0877a9a1209c1efbbf4a4a5a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df392a200a2991025eb95fced58823cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[32], dtype='int64'), 'int64'),
        ]


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
class TestPrimitiveOp_95abebb67e8e0959c4a8a9a44e24af32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df392a200a2991025eb95fced58823cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[128], dtype='int64'), 'int64'),
        ]


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
class TestPrimitiveOp_feb9e9f7cafcf30bdcd3232749449af5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df392a200a2991025eb95fced58823cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[128], dtype='int64'), 'int64'),
        ]


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
class TestPrimitiveOp_8b6f60bd582e2e733301ab9c1b131b32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df392a200a2991025eb95fced58823cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[64], dtype='int64'), 'int64'),
        ]


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
class TestPrimitiveOp_4be25e548a028768b72476671417944c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df392a200a2991025eb95fced58823cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[256], dtype='int64'), 'int64'),
        ]


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
class TestPrimitiveOp_7979a8450849276071ff2b38a84136a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df392a200a2991025eb95fced58823cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[96], dtype='int64'), 'int64'),
        ]


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
class TestPrimitiveOp_3fbb3ea7d00ee59e35b71b20986cf99e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df392a200a2991025eb95fced58823cb
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[384], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_01f53cc47a84dc5ca349904a555455c3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.index_select(input_0, input_1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 56, 56], dtype='float16'),
            paddle.static.InputSpec(shape=[8], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_69bc8b99d9fdf19c7d3081e6a5806c4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_01f53cc47a84dc5ca349904a555455c3
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 56, 56], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([5, 6, 7, 4, 1, 2, 3, 0], dtype='int64').reshape([8]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_38ec677d8cfccbc37c6ec42d1c39e3e9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.index_select(input_0, input_1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 28, 28], dtype='float16'),
            paddle.static.InputSpec(shape=[32], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aa34d94be6cc2aaca8d7d1e8616a4539(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38ec677d8cfccbc37c6ec42d1c39e3e9
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 28, 28], dtype='float16', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[32], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_09e4d614a3e8fd11dfbd73e2a3b364b9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.index_select(input_0, input_1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 12, 28, 28], dtype='float16'),
            paddle.static.InputSpec(shape=[12], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9913113723a29160f6ae5ad32919af53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_09e4d614a3e8fd11dfbd73e2a3b364b9
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 28, 28], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([4, 5, 3, 7, 8, 6, 10, 11, 9, 1, 2, 0], dtype='int64').reshape([12]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3e679d304161c6dfe7e6969b242f2655(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.index_select(input_0, input_1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, 14, 14], dtype='float16'),
            paddle.static.InputSpec(shape=[48], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1fb61fd8c927b104d26ee8b499bc140f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3e679d304161c6dfe7e6969b242f2655
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[48], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_02cd67089d47b84e8f1e2a46f3b819a1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.index_select(input_0, input_1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 14, 14], dtype='float16'),
            paddle.static.InputSpec(shape=[16], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_19c80eb5a7476a20b884652faebce253(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02cd67089d47b84e8f1e2a46f3b819a1
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12, 1, 2, 3, 0], dtype='int64').reshape([16]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_24046655602089495de72879c4d1b96d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.index_select(input_0, input_1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 14, 14], dtype='float16'),
            paddle.static.InputSpec(shape=[64], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_79e77c1f086c596bc9ce2340850524e3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24046655602089495de72879c4d1b96d
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[64], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1be33ac384ec68366e99899cde54cea9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.index_select(input_0, input_1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 14, 14], dtype='float16'),
            paddle.static.InputSpec(shape=[32], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3cf72a3e596890d0daca7f425ef7eae2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1be33ac384ec68366e99899cde54cea9
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[32], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d2e0005866386f31db108d2ee2f5ce11(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.index_select(input_0, input_1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 14, 14], dtype='float16'),
            paddle.static.InputSpec(shape=[128], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_261c163047918bf9ca9e3d2907732552(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d2e0005866386f31db108d2ee2f5ce11
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 14, 14], dtype='float16', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[128], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1781e4fb925e688079dfd5cac981d5a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.index_select(input_0, input_1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 7, 7], dtype='float16'),
            paddle.static.InputSpec(shape=[128], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_06af73f3c3f487d58079ae3bd6f012d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1781e4fb925e688079dfd5cac981d5a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[128], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_150bf426e3d01c11d676b19f0174a6b1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.index_select(input_0, input_1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 7, 7], dtype='float16'),
            paddle.static.InputSpec(shape=[64], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a40cf62e6ffe2620bbeba22ff11a9881(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_150bf426e3d01c11d676b19f0174a6b1
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[64], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d00dd3196898c79b763c78b789c46f60(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.index_select(input_0, input_1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 7, 7], dtype='float16'),
            paddle.static.InputSpec(shape=[256], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_83cd603365884288db66e6cf2b4e89a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d00dd3196898c79b763c78b789c46f60
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[256], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_de435bf51422d1688c13b6d24447860d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.index_select(input_0, input_1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 7, 7], dtype='float16'),
            paddle.static.InputSpec(shape=[96], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2aa6ac5a16d12887eb6b6d94ccdcdcd9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_de435bf51422d1688c13b6d24447860d
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[96], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4976cde45908f89167ad0584edf18b7e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.index_select(input_0, input_1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, 7, 7], dtype='float16'),
            paddle.static.InputSpec(shape=[384], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_61ed919fa0081a3ca6d41eea0de4fe32(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4976cde45908f89167ad0584edf18b7e
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 7, 7], dtype='float16', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[384], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b6adf567ccdcbcaa7a31c437177e8c26(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.index_select(input_0, input_1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 56, 56], dtype='float32'),
            paddle.static.InputSpec(shape=[8], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b8044dcf192f202a7d5a18daa97a74cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6adf567ccdcbcaa7a31c437177e8c26
    def get_inputs(self):
        return [
            paddle.uniform([1, 8, 56, 56], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 6, 7, 4, 1, 2, 3, 0], dtype='int64').reshape([8]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1ee87ea2ac280016bdac140d156f5623(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.index_select(input_0, input_1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[32], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_20e5c1c1c2224d81253868d93cef7153(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1ee87ea2ac280016bdac140d156f5623
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[32], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_026e77e2018e5447bcbba7996b85ee6c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.index_select(input_0, input_1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 12, 28, 28], dtype='float32'),
            paddle.static.InputSpec(shape=[12], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_48e6a3e01830f2cb46e991695b62e2f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_026e77e2018e5447bcbba7996b85ee6c
    def get_inputs(self):
        return [
            paddle.uniform([1, 12, 28, 28], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([4, 5, 3, 7, 8, 6, 10, 11, 9, 1, 2, 0], dtype='int64').reshape([12]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_bcf73c61605bd45c6cd70ce681f47dab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.index_select(input_0, input_1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[48], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1019bd03cfd1bacb7cd9d0dbe7df5afb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcf73c61605bd45c6cd70ce681f47dab
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[48], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_375cf2ea9838c94f570387a12074f2ba(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.index_select(input_0, input_1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[16], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cd5eba03964426e6c1f27a32a406f63b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_375cf2ea9838c94f570387a12074f2ba
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12, 1, 2, 3, 0], dtype='int64').reshape([16]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1f2b3fcad9f0cb3655d6dac8c2117858(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.index_select(input_0, input_1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ef51145e3780d61eda351bd48e942144(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1f2b3fcad9f0cb3655d6dac8c2117858
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[64], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_215f0c795cf1bc773ef4c8dc5efcb07e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.index_select(input_0, input_1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[32], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_201b36a0c5772c062f1a9960f7618e97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_215f0c795cf1bc773ef4c8dc5efcb07e
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[32], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b41ff62dcdfd22070353b77fe47b3b92(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.index_select(input_0, input_1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 14, 14], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_78a1edc6c6857fdd50d416cc101f6771(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b41ff62dcdfd22070353b77fe47b3b92
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 14, 14], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[128], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ec69a8d733e1085f6a7bc417d1aec7b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.index_select(input_0, input_1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5feb24a56e05c0501d7cea0fad062a28(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec69a8d733e1085f6a7bc417d1aec7b6
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[128], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d45fe82fbba5a3852295c5bb8d42c7a6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.index_select(input_0, input_1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_99f560dab94d6af0ca609d00bd442d6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d45fe82fbba5a3852295c5bb8d42c7a6
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[64], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_bddcbb3d71b317844c3017129974e3b0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.index_select(input_0, input_1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5816d76bfe859ba49c2950b9293c925a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bddcbb3d71b317844c3017129974e3b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[256], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f61baaeff7e6cb65fd938398159e4141(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.index_select(input_0, input_1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[96], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_063b5d331f6df68fa61c73a894d169fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f61baaeff7e6cb65fd938398159e4141
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[96], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_318b1c8712a4bb2c21f20ea4c97130b2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.index_select(input_0, input_1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, 7, 7], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2848dcf88b49285765a44682271f2fe5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_318b1c8712a4bb2c21f20ea4c97130b2
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 7, 7], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[384], dtype='int64'), 'int64'),
        ]


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