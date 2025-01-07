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
class PrimitiveOp_c4aece58fce9426b3c374ec70011d2a5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bb242b21f1f42794557aec83fc4f30a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4aece58fce9426b3c374ec70011d2a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 258, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 24, 36], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_e24b5a9809ce7b745a46b4b2c13e9114(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4aece58fce9426b3c374ec70011d2a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 112, 160], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_baa47e7e7b9f27eba64cee5491fbd5b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4aece58fce9426b3c374ec70011d2a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_bf8666293c1c0f2e197b711d3700e15a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4aece58fce9426b3c374ec70011d2a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_969ee86241915dfa29ba745c3cd56110(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4aece58fce9426b3c374ec70011d2a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_512c547658197a518880a29afff515a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4aece58fce9426b3c374ec70011d2a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_a5780026c21382d9405e297595569206(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4aece58fce9426b3c374ec70011d2a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_388f2877308c2f5f2d88fee9fdcd7b3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4aece58fce9426b3c374ec70011d2a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_ba1900f94eb8c2ab23ae78fe7228098b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4aece58fce9426b3c374ec70011d2a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_18f64d51f2d027be1d0697b9e6379d97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4aece58fce9426b3c374ec70011d2a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_cd6f9071f44fc649381192307aa16ee2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4aece58fce9426b3c374ec70011d2a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 7, 10], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_3bd8cedf0a8375a80ae6312c1907f3fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4aece58fce9426b3c374ec70011d2a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_21c9f2939b2fac5e3703333c260e811d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4aece58fce9426b3c374ec70011d2a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_17dcb7132e964f2dfb00145b893d7939(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4aece58fce9426b3c374ec70011d2a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_3c7786e3e23d310d73c9cf0d6c5a0fc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4aece58fce9426b3c374ec70011d2a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_be54199ed686189338eea086f27a7455(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4aece58fce9426b3c374ec70011d2a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_cc23b99374bb8294efb323b12b11c2c7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4aece58fce9426b3c374ec70011d2a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_7923379336610740ad7d18ce8e3e3793(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4aece58fce9426b3c374ec70011d2a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_0e7c90618b511bf235ec349aab028f53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4aece58fce9426b3c374ec70011d2a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_9c40e2161843d43873aeff092d44ea74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4aece58fce9426b3c374ec70011d2a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 15, 25], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 15, 25], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 258, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 15, 25], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_996d50bcbdbb1995f634df0b25adcc44(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4aece58fce9426b3c374ec70011d2a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_168b18547fe3766e4ef889ecf39f5ffc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4aece58fce9426b3c374ec70011d2a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 192, 288], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_0b5f38934ffae5ceebfdbac545586945(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4aece58fce9426b3c374ec70011d2a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_47f28473c50c7e2fbaa9d693683ddf0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4aece58fce9426b3c374ec70011d2a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_6d7e338943e50b100c6a6f5ca1f25fa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4aece58fce9426b3c374ec70011d2a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 28, 40], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_4ead8e9b19fe16047e60750347c527f0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4aece58fce9426b3c374ec70011d2a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 120, 200], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_96bb56fbe9136101bfa9929385d65a24(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [0, 0], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5178aa4b82fe29f86c276bd0cef25b1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96bb56fbe9136101bfa9929385d65a24
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 120, 200], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_d28693c22bc3a0f8ab62781466b96588(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4aece58fce9426b3c374ec70011d2a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_6b3a55d836120ad67c0046637d1b71ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4aece58fce9426b3c374ec70011d2a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_649f42b1162ab8990dd98908cfcd9669(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4aece58fce9426b3c374ec70011d2a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 56, 80], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_4f8ec6b8e640628a9b612b30930f66a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4aece58fce9426b3c374ec70011d2a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_fea13c5d870eec15096241c48e5f89d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4aece58fce9426b3c374ec70011d2a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_54c22f981a50e9d5947ee71b0ab8b89a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c4aece58fce9426b3c374ec70011d2a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 14, 20], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_cbf8eb800fffe87af15ae6152178312b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96bb56fbe9136101bfa9929385d65a24
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 192, 288], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3024728c6f104c46f4a168b8307418d3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 258, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 18, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 258, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 9, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_18e853b85f49bb0774144ec3be960d93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3024728c6f104c46f4a168b8307418d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 24, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 258, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 24, 36], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_833d44e31fcb7334fca614baaa382f8e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 18, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 9, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c9e662c5c103617c90cb367878775b53(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_833d44e31fcb7334fca614baaa382f8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 112, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 112, 160], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_7210b27d5a709eb2be05a90a479e441f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_833d44e31fcb7334fca614baaa382f8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_39a54b27aa9a80cdda589f12114e9bf8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 18, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 256, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 9, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_76ca7a8b4bc9405d6bb81acc946d12ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39a54b27aa9a80cdda589f12114e9bf8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_62c38fd61f5eac6e6e506e9bbafdae1c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 18, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 512, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 9, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b47d3edce7ffe5f0555fcb79808e8edc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62c38fd61f5eac6e6e506e9bbafdae1c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_36f2d09fed936d49781f79a3ee5b0529(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 258, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 18, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[512, 258, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 9, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7111d9b2ff0e26291f34a1e3ef63a9b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36f2d09fed936d49781f79a3ee5b0529
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b6f54a52dcb8e7bc60d4c0bca1d8ac85(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [1, 1], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 18, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 128, 3, 3], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 9, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6f48e05e55f8eabd393603373e307cf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6f54a52dcb8e7bc60d4c0bca1d8ac85
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_c1a0d7428fad6494260f4d0ef2ec05b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6f54a52dcb8e7bc60d4c0bca1d8ac85
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_1d0db6345553d9f1ce30db206f7445fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62c38fd61f5eac6e6e506e9bbafdae1c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_add69b9328cf4d660f289e58612d09ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39a54b27aa9a80cdda589f12114e9bf8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 60, 100], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_fa9fad110e4384ab0f986e3e2a31689a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_833d44e31fcb7334fca614baaa382f8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 7, 10], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 7, 10], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_4791b5f24b7bea21a9e14141b1dd35fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62c38fd61f5eac6e6e506e9bbafdae1c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_564bc4a8975cc2a4e868d12cf61be38f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_833d44e31fcb7334fca614baaa382f8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_55f3aace0ec6a98fcb0bc6be908bef30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62c38fd61f5eac6e6e506e9bbafdae1c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_6a36dc7b6295ba014032a842c39b94cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_62c38fd61f5eac6e6e506e9bbafdae1c
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 512, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_072fdfedbaf20ff2dea6c630e8fbfad5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_833d44e31fcb7334fca614baaa382f8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_47ad5412b38dd6eb5d2b68eb8c89fb7d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36f2d09fed936d49781f79a3ee5b0529
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_4d533e61428683610fa957984c27d338(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39a54b27aa9a80cdda589f12114e9bf8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 48, 72], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_d1a25db9617a4c37dab0e1737113d697(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36f2d09fed936d49781f79a3ee5b0529
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_a57789409e6db1cafa2976fe348d8824(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3024728c6f104c46f4a168b8307418d3
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 15, 25], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 15, 25], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 258, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 15, 25], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_3e3471a3da284a33ad98778b7142f479(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6f54a52dcb8e7bc60d4c0bca1d8ac85
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_21861e2d0890f4ac926f4bafecfed999(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39a54b27aa9a80cdda589f12114e9bf8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 192, 288], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_ecef2558d520022f60d4aee8197dbec2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39a54b27aa9a80cdda589f12114e9bf8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 96, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_7efd4b35c2d1323377aa6d694f46b81b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36f2d09fed936d49781f79a3ee5b0529
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_9c70592e534d818bec886d6f34cb48f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_833d44e31fcb7334fca614baaa382f8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 28, 40], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 28, 40], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_7777abb13d415b5318668054b1de2c65(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_39a54b27aa9a80cdda589f12114e9bf8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 120, 200], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_880c490b07720e503c0e7c4ed0bb1f46(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [0, 0], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[128, 128, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9d6950e790a336953f493b9b9ee04e34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_880c490b07720e503c0e7c4ed0bb1f46
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 120, 200], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 120, 200], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_812259ab1cdd3a98e2c097d5ed51cf06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_36f2d09fed936d49781f79a3ee5b0529
    def get_inputs(self):
        return [
            paddle.uniform([1, 258, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([512, 258, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_e63191a104b26ac699c12b27056d1949(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_833d44e31fcb7334fca614baaa382f8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_36c45a4c2bfe18a05942875da0453b06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_833d44e31fcb7334fca614baaa382f8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 56, 80], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 56, 80], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_fa69341a7f9744b208a71c767cac614b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_833d44e31fcb7334fca614baaa382f8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_9ae844f62a321e991d3b2db9fe47d855(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b6f54a52dcb8e7bc60d4c0bca1d8ac85
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 30, 50], dtype='float32', min=0, max=0.5),
            paddle.uniform([128, 128, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
        ]


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
class TestPrimitiveOp_54d99883cdfa368a2c97717e5d31c963(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_833d44e31fcb7334fca614baaa382f8e
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 18, 14, 20], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 9, 14, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4fa9ba73c877b3e2398a037fc806f932(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2, arg_3):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        input_3 = arg_3
        return paddle._C_ops.deformable_conv(input_0, input_1, input_2, input_3, [1, 1], [0, 0], [1, 1], 1, 1, 1)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[256, 128, 1, 1], dtype='float32'),
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d7cf8bf2dab4e43d337118e0e71c0917(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4fa9ba73c877b3e2398a037fc806f932
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 2, 192, 288], dtype='float32', min=0, max=0.5),
            paddle.uniform([256, 128, 1, 1], dtype='float32', min=0, max=0.5),
            paddle.uniform([1, 1, 192, 288], dtype='float32', min=0, max=0.5),
        ]


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