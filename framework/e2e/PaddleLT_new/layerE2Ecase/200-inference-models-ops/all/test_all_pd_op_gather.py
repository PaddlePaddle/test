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
class PrimitiveOp_2786ce783eef054821f0c749996f3164(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = 0
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1768f576745230d89b17db9f613bda54(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2786ce783eef054821f0c749996f3164
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float16', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[196], dtype='int64'), 'int64'),
        ]


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
class TestPrimitiveOp_aa0050c08c5594bff8652224abec94bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2786ce783eef054821f0c749996f3164
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4], dtype='int64').reshape([16]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4a8836dc3e04c113e83b2b0f59219fae(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = 0
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_40de93dea9f7af0443cf6c7b7ac906fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a8836dc3e04c113e83b2b0f59219fae
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[196], dtype='int64'), 'int64'),
        ]


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
class TestPrimitiveOp_c6003238050e546b3bbdcc8f0e1f03a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2786ce783eef054821f0c749996f3164
    def get_inputs(self):
        return [
            paddle.uniform([49, 6], dtype='float16', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[49], dtype='int64'), 'int64'),
        ]


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
class TestPrimitiveOp_c08f665646b1e8276f65d9bca13f0a8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a8836dc3e04c113e83b2b0f59219fae
    def get_inputs(self):
        return [
            paddle.uniform([49, 6], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[49], dtype='int64'), 'int64'),
        ]


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
class TestPrimitiveOp_08f3f1864ad9109fdc3866a228d67239(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2786ce783eef054821f0c749996f3164
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float16', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[196], dtype='int64'), 'int64'),
        ]


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
class TestPrimitiveOp_5a8f82c2e8548f26279b5965fbb8142a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a8836dc3e04c113e83b2b0f59219fae
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[196], dtype='int64'), 'int64'),
        ]


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
class TestPrimitiveOp_12e8c94cb5ada4270d1b2240fbd15efb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2786ce783eef054821f0c749996f3164
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_073150f06d57290071618eed3b3d5612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a8836dc3e04c113e83b2b0f59219fae
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_442472cc10bf2ea480ce11caddb481f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a8836dc3e04c113e83b2b0f59219fae
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 12, 13, 14, 9, 8, 9, 10, 5, 4, 5, 6, 1, 0, 1, 2], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_3494db4ac1323276b972cd127e61f186(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2786ce783eef054821f0c749996f3164
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float16', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[49], dtype='int64'), 'int64'),
        ]


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
class TestPrimitiveOp_73763e89e8313c42bdad649bb5f2fdcc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2786ce783eef054821f0c749996f3164
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([13, 12, 13, 14, 9, 8, 9, 10, 5, 4, 5, 6, 1, 0, 1, 2], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_87979913c7c79ec01cccf612eb661133(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2786ce783eef054821f0c749996f3164
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_f0083b4e084aa753cbd7951cb49a50a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a8836dc3e04c113e83b2b0f59219fae
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_2923fdf2c4f2ae3aeb19187df469a79b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2786ce783eef054821f0c749996f3164
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_34193284823a5e4154bb06c5a69ca954(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2786ce783eef054821f0c749996f3164
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_5ca781014f8d05ca38cc142095490ede(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a8836dc3e04c113e83b2b0f59219fae
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[49], dtype='int64'), 'int64'),
        ]


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
class TestPrimitiveOp_c31c19d007a5081e4e2120038618ca2e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a8836dc3e04c113e83b2b0f59219fae
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_aa6b588b1683146ba2d4ba573b9380f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a8836dc3e04c113e83b2b0f59219fae
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([6, 5, 4, 5, 2, 1, 0, 1, 6, 5, 4, 5, 10, 9, 8, 9], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_b945a648db16832085bb731800c7069f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a8836dc3e04c113e83b2b0f59219fae
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 8, 9, 10, 5, 4, 5, 6, 1, 0, 1, 2, 5, 4, 5, 6], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_8d45dcf79ab1a63e854dffdbca0c02a4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2786ce783eef054821f0c749996f3164
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([9, 8, 9, 10, 5, 4, 5, 6, 1, 0, 1, 2, 5, 4, 5, 6], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_857da3cdb31c46c33fe90f4e82cf09be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a8836dc3e04c113e83b2b0f59219fae
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 9, 8, 9, 6, 5, 4, 5, 2, 1, 0, 1, 6, 5, 4, 5], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_a745143ed37364a72d85c88e51fb9c18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a8836dc3e04c113e83b2b0f59219fae
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_476c41aa01b9e29a3dfd05ab07b00be4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a8836dc3e04c113e83b2b0f59219fae
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_67c3744c39083e22e2f6c69ac2a50223(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2786ce783eef054821f0c749996f3164
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([10, 9, 8, 9, 6, 5, 4, 5, 2, 1, 0, 1, 6, 5, 4, 5], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_36b3727ae627a4fff1f8abc93bef7d70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a8836dc3e04c113e83b2b0f59219fae
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 0, 1, 2, 5, 4, 5, 6, 9, 8, 9, 10, 13, 12, 13, 14], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_473caba6c4b58131f3e464c768b317a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2786ce783eef054821f0c749996f3164
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_bfb1df16eba8a226f3be4aa89461258d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a8836dc3e04c113e83b2b0f59219fae
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 1, 0, 1, 6, 5, 4, 5, 10, 9, 8, 9, 14, 13, 12, 13], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_ef15a999f6cf8462656a969f492124fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a8836dc3e04c113e83b2b0f59219fae
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 4, 5, 6, 1, 0, 1, 2, 5, 4, 5, 6, 9, 8, 9, 10], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_a0b6686f9aade00ae9873ed483123ab7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2786ce783eef054821f0c749996f3164
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1, 0, 1, 2, 5, 4, 5, 6, 9, 8, 9, 10, 13, 12, 13, 14], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_7fca0d204c712ef130274ee16250aafe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a8836dc3e04c113e83b2b0f59219fae
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_6f5c746a64ed812af2926824b07a9cda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2786ce783eef054821f0c749996f3164
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2, 1, 0, 1, 6, 5, 4, 5, 10, 9, 8, 9, 14, 13, 12, 13], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_7571d8e15c7d98ebbf3a5b28b8078f70(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2786ce783eef054821f0c749996f3164
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_6b4b93ce542001a3a55f05603b3b93ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a8836dc3e04c113e83b2b0f59219fae
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([14, 13, 12, 13, 10, 9, 8, 9, 6, 5, 4, 5, 2, 1, 0, 1], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_518be35625c526ce30a84efe25757c12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a8836dc3e04c113e83b2b0f59219fae
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_7ac2814f2dc030c4b8c5b93813e8cc60(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2786ce783eef054821f0c749996f3164
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([6, 5, 4, 5, 2, 1, 0, 1, 6, 5, 4, 5, 10, 9, 8, 9], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_78b21d8e110f6fd7efc5aa8bf20b5569(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2786ce783eef054821f0c749996f3164
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([14, 13, 12, 13, 10, 9, 8, 9, 6, 5, 4, 5, 2, 1, 0, 1], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_17ba8a695b1b0f66c5482e8f5f42ecba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2786ce783eef054821f0c749996f3164
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([5, 4, 5, 6, 1, 0, 1, 2, 5, 4, 5, 6, 9, 8, 9, 10], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_e1e30f82b0b8904390a4ba2a3c4f39bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2786ce783eef054821f0c749996f3164
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_c280a36f57b167adc89e3984e4283dbd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a8836dc3e04c113e83b2b0f59219fae
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3], dtype='int64').reshape([16]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9eebb7d99fed5d36ce04a02a6f8bea22(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = 0
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 4], dtype='float16'),
            paddle.static.InputSpec(shape=[196], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_76eb885f7ac5ab54460980cb416b2a6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9eebb7d99fed5d36ce04a02a6f8bea22
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float16', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[196], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_21cb96f19ba41b1e12f052ad2ed3c196(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = 0
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16, 8], dtype='float16'),
            paddle.static.InputSpec(shape=[16], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_80383ebb442a27414cc15ab921c3139a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21cb96f19ba41b1e12f052ad2ed3c196
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4], dtype='int64').reshape([16]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6b3465e3b203b5be5fcfcc771c20ac50(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = 0
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[196], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_476389bac8cd9d352be377fa826a8b4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b3465e3b203b5be5fcfcc771c20ac50
    def get_inputs(self):
        return [
            paddle.uniform([196, 4], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[196], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2cd40091f0b577e6b9199f7b3e5a47dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = 0
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 6], dtype='float16'),
            paddle.static.InputSpec(shape=[49], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ef895ce20adbc2ef9320e68e52cdb22f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cd40091f0b577e6b9199f7b3e5a47dd
    def get_inputs(self):
        return [
            paddle.uniform([49, 6], dtype='float16', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[49], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_804f3351fdbf10b6bf99a3fe9c0ce6b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = 0
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 6], dtype='float32'),
            paddle.static.InputSpec(shape=[49], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a968e2799040402bd066ca0a21fd8ffe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_804f3351fdbf10b6bf99a3fe9c0ce6b3
    def get_inputs(self):
        return [
            paddle.uniform([49, 6], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[49], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_278aa511716fb0b535af74c704b61fb0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = 0
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 8], dtype='float16'),
            paddle.static.InputSpec(shape=[196], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4856f6c4a68ca44ec142376c46201a10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_278aa511716fb0b535af74c704b61fb0
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float16', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[196], dtype='int64'), 'int64'),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b66194313bbec1479a5e196a1aec9b4f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = 0
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[196, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[196], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e674e77197f6c8c1f1dfee31f580aeab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b66194313bbec1479a5e196a1aec9b4f
    def get_inputs(self):
        return [
            paddle.uniform([196, 8], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[196], dtype='int64'), 'int64'),
        ]


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
class TestPrimitiveOp_6d31547b272117d11db6b4df8562920a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21cb96f19ba41b1e12f052ad2ed3c196
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype='int64').reshape([16]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f31902dce6deb9ea5f25c565db36a921(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = 0
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16, 8], dtype='float32'),
            paddle.static.InputSpec(shape=[16], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cdad24340296548d253f27444302b7e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f31902dce6deb9ea5f25c565db36a921
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_ac56e7cf8627c281fe5a7e1c581d803a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f31902dce6deb9ea5f25c565db36a921
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([13, 12, 13, 14, 9, 8, 9, 10, 5, 4, 5, 6, 1, 0, 1, 2], dtype='int64').reshape([16]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_af60c492c3efe510c99f3d2b1046fd50(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = 0
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 16], dtype='float16'),
            paddle.static.InputSpec(shape=[49], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f9ee2252a70e4a22da377f14335e63dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af60c492c3efe510c99f3d2b1046fd50
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float16', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[49], dtype='int64'), 'int64'),
        ]


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
class TestPrimitiveOp_2ae5abf08fe166750771c89a841c8654(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21cb96f19ba41b1e12f052ad2ed3c196
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([13, 12, 13, 14, 9, 8, 9, 10, 5, 4, 5, 6, 1, 0, 1, 2], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_cae872b16afcda4aaf8f9b0e2a235edf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21cb96f19ba41b1e12f052ad2ed3c196
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_5094c0f58980e73533408c31cfa6d991(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f31902dce6deb9ea5f25c565db36a921
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_192746d98322be78c7dd4bddb0895570(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21cb96f19ba41b1e12f052ad2ed3c196
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_cc8297ac77a873f302faaaff76265430(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21cb96f19ba41b1e12f052ad2ed3c196
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7], dtype='int64').reshape([16]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b394cfe2daa450c93478745415e61c6f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = 0
        return paddle._C_ops.gather(input_0, input_1, input_2)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[49, 16], dtype='float32'),
            paddle.static.InputSpec(shape=[49], dtype='int64'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5c22d3096b4e2e2299c85b97172faf8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b394cfe2daa450c93478745415e61c6f
    def get_inputs(self):
        return [
            paddle.uniform([49, 16], dtype='float32', min=0, max=0.5),
            paddle.cast(paddle.randint(low=0, high=3, shape=[49], dtype='int64'), 'int64'),
        ]


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
class TestPrimitiveOp_aafde152b0398fa22f486cb80d7e2c82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f31902dce6deb9ea5f25c565db36a921
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_55f3977b1dac2df15de4208aa5ef2cdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f31902dce6deb9ea5f25c565db36a921
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([6, 5, 4, 5, 2, 1, 0, 1, 6, 5, 4, 5, 10, 9, 8, 9], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_e64eeaf8103e7ec33d3b065cd44f38c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f31902dce6deb9ea5f25c565db36a921
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([9, 8, 9, 10, 5, 4, 5, 6, 1, 0, 1, 2, 5, 4, 5, 6], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_966860d97b8c5225711df55fd500992a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21cb96f19ba41b1e12f052ad2ed3c196
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([9, 8, 9, 10, 5, 4, 5, 6, 1, 0, 1, 2, 5, 4, 5, 6], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_37b023282e5c192a33dcd7bbf5234549(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f31902dce6deb9ea5f25c565db36a921
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([10, 9, 8, 9, 6, 5, 4, 5, 2, 1, 0, 1, 6, 5, 4, 5], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_a591f484be7e4a090cc66b058132aa81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f31902dce6deb9ea5f25c565db36a921
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_55f24713d99cd713514620c33d932bb4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f31902dce6deb9ea5f25c565db36a921
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_3ee46280b030195233f5cfb99e25d584(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21cb96f19ba41b1e12f052ad2ed3c196
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([10, 9, 8, 9, 6, 5, 4, 5, 2, 1, 0, 1, 6, 5, 4, 5], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_46c122cf730f4b3630416c67326ca22d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f31902dce6deb9ea5f25c565db36a921
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1, 0, 1, 2, 5, 4, 5, 6, 9, 8, 9, 10, 13, 12, 13, 14], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_bcf02cf8ba51f2cf8be44f61de3b21e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21cb96f19ba41b1e12f052ad2ed3c196
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_920f73f8e8820d3a97c1cfe0e56e9035(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f31902dce6deb9ea5f25c565db36a921
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2, 1, 0, 1, 6, 5, 4, 5, 10, 9, 8, 9, 14, 13, 12, 13], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_8d84d66ea67a428079c52864f4c2e249(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f31902dce6deb9ea5f25c565db36a921
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5, 4, 5, 6, 1, 0, 1, 2, 5, 4, 5, 6, 9, 8, 9, 10], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_e0940e7e14d3a719cf4815285312ea1c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21cb96f19ba41b1e12f052ad2ed3c196
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([1, 0, 1, 2, 5, 4, 5, 6, 9, 8, 9, 10, 13, 12, 13, 14], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_e9194cb0711596e05bbf720f597e8b6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f31902dce6deb9ea5f25c565db36a921
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_664c91e74ccddbff8ac75371664b565f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21cb96f19ba41b1e12f052ad2ed3c196
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([2, 1, 0, 1, 6, 5, 4, 5, 10, 9, 8, 9, 14, 13, 12, 13], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_8528a84c839e9f9bd9640b95d177256b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21cb96f19ba41b1e12f052ad2ed3c196
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_83451fef63c36b30b7da56e7a2ce421a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f31902dce6deb9ea5f25c565db36a921
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([14, 13, 12, 13, 10, 9, 8, 9, 6, 5, 4, 5, 2, 1, 0, 1], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_6ce539f679c442828ce626ba6e361eeb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f31902dce6deb9ea5f25c565db36a921
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_55c85ec6368f9b48532602a864eeb32d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21cb96f19ba41b1e12f052ad2ed3c196
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([6, 5, 4, 5, 2, 1, 0, 1, 6, 5, 4, 5, 10, 9, 8, 9], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_98a39208e942ba42ed2d7a1d7558d52b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21cb96f19ba41b1e12f052ad2ed3c196
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([14, 13, 12, 13, 10, 9, 8, 9, 6, 5, 4, 5, 2, 1, 0, 1], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_b46598d33ac7d6727fffb47903315930(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21cb96f19ba41b1e12f052ad2ed3c196
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([5, 4, 5, 6, 1, 0, 1, 2, 5, 4, 5, 6, 9, 8, 9, 10], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_744ca36505f96167e1d6e80ae0537627(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_21cb96f19ba41b1e12f052ad2ed3c196
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12], dtype='int64').reshape([16]),
        ]


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
class TestPrimitiveOp_d1d00ac59bf70b63c71daa49ff77e71c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f31902dce6deb9ea5f25c565db36a921
    def get_inputs(self):
        return [
            paddle.uniform([16, 8], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3], dtype='int64').reshape([16]),
        ]


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