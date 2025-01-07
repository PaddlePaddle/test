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
class PrimitiveOp_bcc827b79bd840d26c0466333bfa67de(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_effd3743acd3301ecf95bd7e84e234e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 384], dtype='float16', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0e91964e6f3a7600c14098554165fc98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_83342861c7ffaebc9c9a3ab08c6494f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 384], dtype='float16', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7f8977e56d7d0bd5ed4b3e0ded365bc2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0308ebe4c5701543f2b94ee28f1e5043(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2aca2a88b69d22f5737a5b316c167916(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 384], dtype='float16', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5e10c294f1eab30dd0c9f06b8dcdbe07(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_687cd369fdd3cf9dc499093ab2d492c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4f00206646f3be2f4de3821744d7a979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 26, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_81d2664aee4a61236284940b97246272(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 512], dtype='float16', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b40192c1e65b36dea8f489ebf100b854(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e0f75b9c8daa9da2cbf957d0f4258fc9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_65d079b8ad81a9662942ac4f40fd954a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ca9ec997f152c52b4600c876ed330577(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4274570345878601, 0.43177613615989685, 0.20555298030376434, 0.39094802737236023, 0.49809014797210693, 0.04659171402454376, 0.43223974108695984, 0.27359694242477417, 0.34672853350639343, 0.29964154958724976, 0.06038147956132889, 0.45957058668136597, 0.313096821308136, 0.07567092776298523, 0.13435299694538116, 0.20235183835029602, 0.074857696890831, 0.3765570819377899, 0.39376622438430786, 0.07235901057720184, 0.03240571171045303, 0.42651236057281494, 0.29713869094848633, 0.25388598442077637], dtype='float32').reshape([24]),
            paddle.to_tensor([0.008527816273272038, 0.3680625557899475, 0.19607119262218475, 0.2619350254535675, 0.05993141978979111, 0.4876129627227783, 0.45239174365997314, 0.08012320101261139, 0.2630710005760193, 0.24455560743808746, 0.17314158380031586, 0.4521302282810211, 0.3052957057952881, 0.07388428598642349, 0.32233917713165283, 0.15124258399009705, 0.40676814317703247, 0.23380887508392334, 0.20318107306957245, 0.4708256423473358, 0.2285236418247223, 0.49071204662323, 0.23710103332996368, 0.09393852949142456], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6aadf27a816de5badeb9c95e1f0273c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([4, 64, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6f20aa71779fde912d5eec52102c9eb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a254031597d8288871b0422f0fa8e363(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float16'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9289ae0ebcd444d1ef3d27a94f344315(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float16', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9a767c8c1de3063eef200012e67eb0a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b0bbca2b4b85b255a35f1f5d169e4f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 768], dtype='float16', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_04ebe9bb6ba15cb7c6a3968b62c01530(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 512], dtype='float16', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4107a7be698cf39c1a9594518d36a877(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 384], dtype='float16', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a1bf8870d2da577eda88381bc273949f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3950336b1c0088f1d835dc0a2fbee9e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1938f81f7e9ef403b9b7ee161b1f9c66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 768], dtype='float16', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fc2895c63d7b792d9e5951fceda08c59(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 320], dtype='float16', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4c697ae9d77861bf2a38924914465eac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9904ff3fc9ed46a9364a1a50e3fa8c27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_59e39f5d3004abcdda5546b4d19bde93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 9216, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e8bc50355e2b9c1d33e2e76635beb9cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e003c55fba6b7ef2403e145f42433d3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_667997ee8fc6261db030cd15ba679d06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 160], dtype='float16', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_068181e1320356cb5f55f144f7a24875(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6ba2c6e94bc4f3d0fde967a17a21d6eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 25, 512], dtype='float16', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c479bd7d8a3ea86b7fc351570fa13503(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9f2c7fdd899a5d5dafcc25c695f884df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 256], dtype='float16', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0d7b41405c7f3fcf758103340f8cb211(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 192], dtype='float16', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_04a7fa6710f3bd8e1c69c804da35ea09(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 320], dtype='float16', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_debd4858042661a0c4b942216d741d31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6c0a50e1d9d3ea979b57b3641ef4d88a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8eae3b4d7191255835c177c1f57e3297(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ac186de53eee3328cf2e88dcc7562c3b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6a442445212ee5d8b8e912cadb9b421e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 128], dtype='float16', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d64fe56cc7dd031f803d3aba96d9d8cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 128], dtype='float16', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9bc3d61d0989da60b928e4edaf8c8a88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([4, 64, 192], dtype='float16', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_35d0326978897cc39a7b689e3d7faf2d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 26, 512], dtype='float16', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_163bb28289c79e8c323140afe5001cf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_168566e56e904df1298e659419eb0b8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([4, 256, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ecaedb8b0672bcddd8a6a6576485e792(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 192], dtype='float16', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_30dd10d0c07f20d38e050d6f695094d3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([4, 16, 240], dtype='float16', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c21559071a9b79f6e2482ff903ff5ed1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([4, 16, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b862d93ee38e3c975ef018336d38328c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_460f3318fdbb8ee5ebecb7761d45ded2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 9216, 128], dtype='float16', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9432716958c98ab24546417f96b1b852(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_784b769e33775a1d4958f6d54695136d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 160], dtype='float16', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c56c9b412cf8d9593cd06a2720284011(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 96], dtype='float16', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f3d42daaa27c36c3a593e99b474e2753(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3291398882865906, 0.2799290418624878, 0.24255892634391785, 0.022600550204515457, 0.17959235608577728, 0.24647726118564606, 0.38651296496391296, 0.27117931842803955, 0.14117272198200226, 0.15848414599895477, 0.34334030747413635, 0.35698118805885315, 0.36005890369415283, 0.18805812299251556, 0.48915889859199524, 0.36336442828178406, 0.4558719992637634, 0.2793504297733307, 0.24825513362884521, 0.47472038865089417, 0.09363708645105362, 0.45508119463920593, 0.011177970096468925, 0.39462152123451233], dtype='float32').reshape([24]),
            paddle.to_tensor([0.29301589727401733, 0.2085038423538208, 0.03606534004211426, 0.35489121079444885, 0.11573629081249237, 0.10927984118461609, 0.43879151344299316, 0.12053073197603226, 0.020735815167427063, 0.4717411994934082, 0.23745466768741608, 0.35614898800849915, 0.37861767411231995, 0.4362887740135193, 0.35406142473220825, 0.03167526796460152, 0.18890176713466644, 0.49303510785102844, 0.4234236478805542, 0.14414286613464355, 0.43344712257385254, 0.12832221388816833, 0.029874257743358612, 0.20786912739276886], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_02d36333bb5974994bb9512b479c0726(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3210602402687073, 0.1652502715587616, 0.38910743594169617, 0.0852208137512207, 0.1875229775905609, 0.06697116792201996, 0.40072891116142273, 0.2111017405986786, 0.288394957780838, 0.44486507773399353, 0.47676968574523926, 0.4502319395542145, 0.3937016427516937, 0.20148120820522308, 0.4022679626941681, 0.005294621456414461, 0.4023727774620056, 0.0890527069568634, 0.2519589364528656, 0.20851051807403564, 0.28808578848838806, 0.4886685609817505, 0.08577290177345276, 0.4194251298904419], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4824790358543396, 0.28178420662879944, 0.11231352388858795, 0.05267416685819626, 0.2506307363510132, 0.3153255879878998, 0.1649695485830307, 0.2177359163761139, 0.40498632192611694, 0.008085126988589764, 0.4202231466770172, 0.47174975275993347, 0.12288246303796768, 0.4858485162258148, 0.06832252442836761, 0.17681068181991577, 0.3441885709762573, 0.2808915376663208, 0.32924842834472656, 0.0750841498374939, 0.44072580337524414, 0.11699662357568741, 0.4559996426105499, 0.11988300830125809], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0b221a77726a2cfc6791c27b8d77cec5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 512], dtype='float16', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9bb5eac8214120bb406299350520c7d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bcb20e3316390a97eb14252c47224ab3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b7396b367a995cb31a2e44fe241f1b81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e049124f4268ecdb6e87fc6abfe97675(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2aaab0c65f891a5338ae23c9649b92db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4495331943035126, 0.2390548437833786, 0.19318528473377228, 0.132700577378273, 0.0465238057076931, 0.3378826975822449, 0.33943110704421997, 0.42073583602905273, 0.38162028789520264, 0.39442095160484314, 0.21441227197647095, 0.009111440740525723, 0.10047465562820435, 0.13634875416755676, 0.2825288772583008, 0.13066412508487701, 0.4245243966579437, 0.3586450517177582, 0.21571151912212372, 0.2083481401205063, 0.29532063007354736, 0.3010399043560028, 0.03234931826591492, 0.4178794324398041], dtype='float32').reshape([24]),
            paddle.to_tensor([0.12118066847324371, 0.4409113824367523, 0.4618769586086273, 0.13432025909423828, 0.02096180059015751, 0.1489228904247284, 0.4852922558784485, 0.3460778295993805, 0.17764738202095032, 0.3244779407978058, 0.37616482377052307, 0.3215655982494354, 0.06428605318069458, 0.23318053781986237, 0.3563835918903351, 0.11045725643634796, 0.3157748878002167, 0.4086260497570038, 0.007421903777867556, 0.26640671491622925, 0.13232295215129852, 0.27705997228622437, 0.3172267973423004, 0.40885913372039795], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b845c90b460d5eeec7f3d2c4aea0318d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 2304, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e4d168ceffed1560bd10c7a01d2af642(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cbdf61b0c4ffebf66ef7bb9e85d2635a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 25, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6462bfde0cc71a73817a33afc57befc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.28441187739372253, 0.10656187683343887, 0.21883845329284668, 0.3915109932422638, 0.2528392970561981, 0.06576692312955856, 0.05956346169114113, 0.03799008950591087, 0.4183785021305084, 0.22229468822479248, 0.2085215449333191, 0.23389792442321777, 0.3521454334259033, 0.14446155726909637, 0.3218226730823517, 0.047325752675533295, 0.26577284932136536, 0.3744446635246277, 0.19749882817268372, 0.3558489978313446, 0.2746719717979431, 0.38763442635536194, 0.11997835338115692, 0.028146913275122643], dtype='float32').reshape([24]),
            paddle.to_tensor([0.22275647521018982, 0.12340947240591049, 0.02118116430938244, 0.411749005317688, 0.4021325409412384, 0.399991512298584, 0.3278912305831909, 0.12510359287261963, 0.18763622641563416, 0.29445427656173706, 0.3866947293281555, 0.3403390347957611, 0.41840213537216187, 0.13720844686031342, 0.12938237190246582, 0.32747989892959595, 0.3018067181110382, 0.4771754741668701, 0.4534560441970825, 0.4147586226463318, 0.12004035711288452, 0.3550736606121063, 0.17861953377723694, 0.38331589102745056], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1787fe966317706317d7c88386d3c421(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 256], dtype='float16', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7d798bd68ea3ce60c19c4ca5804a4f51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([4, 256, 144], dtype='float16', min=0, max=0.5),
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aabf0a08eced233cd94e12d0d878c8d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 2304, 256], dtype='float16', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8e3910b29aa529fd0121f7d524e57102(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1024], dtype='float16', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f4f4ac0b1548bd48471dd6654bec7cd3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_234e8939b6f7b81d981d178debeb11b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 128], dtype='float16', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fc361031adc0ab7e7e4a62b958dd1dd6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1024], dtype='float16', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ee791e0d496884ee4e79377606a60288(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_84d05576fcd2882cba2d842114f48766(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 320], dtype='float16', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bfa81efa831236c1011a3fbaf7c7c608(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([4, 256, 144], dtype='float16', min=0, max=0.5),
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0697324beff840eda046dc156c74faee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_45f042834f8efee9797be5b0c54afe9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([4, 64, 192], dtype='float16', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eeacf2af6ccd79983a9aae3af6a3b72d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c14954bd589a11e15bb22a55833316d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_58f1847d93159e6bb79926b7ef706d3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0033353352919220924, 0.47618368268013, 0.197532057762146, 0.4897301495075226, 0.31167104840278625, 0.34787553548812866, 0.1985020786523819, 0.05122765526175499, 0.19276872277259827, 0.3168918192386627, 0.3845776617527008, 0.2888976037502289, 0.29651880264282227, 0.0008442751714028418, 0.457075834274292, 0.3543223440647125, 0.3077501654624939, 0.4268065392971039, 0.3943912982940674, 0.2577085494995117, 0.38496899604797363, 0.29549700021743774, 0.1865539848804474, 0.3796634376049042], dtype='float32').reshape([24]),
            paddle.to_tensor([0.180519700050354, 0.10383564233779907, 0.2558480501174927, 0.0013169227167963982, 0.11466995626688004, 0.14514382183551788, 0.2604365944862366, 0.4266960024833679, 0.41506364941596985, 0.2930034399032593, 0.2402764856815338, 0.06055678799748421, 0.07846041768789291, 0.43997666239738464, 0.3923080265522003, 0.4841328263282776, 0.22388023138046265, 0.2722157835960388, 0.22711409628391266, 0.3444993495941162, 0.3947950005531311, 0.4641958177089691, 0.2309851050376892, 0.33825892210006714], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dbd3147bf3edf6f2132a409aeb9d6a3d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3b63948a1844f93e58f6ca22a37c37a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1dc0f453a5d0eb9207c4fd160f146f7c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1973121017217636, 0.17461606860160828, 0.3475719392299652, 0.082064189016819, 0.2503150403499603, 0.4508446753025055, 0.2503358721733093, 0.3479492962360382, 0.10011161118745804, 0.1858212947845459, 0.43693807721138, 0.29668566584587097, 0.4360484182834625, 0.1190735474228859, 0.12189847230911255, 0.012037292122840881, 0.28188276290893555, 0.30776211619377136, 0.20944051444530487, 0.2792387008666992, 0.23679988086223602, 0.08849692344665527, 0.25976940989494324, 0.4305691421031952], dtype='float32').reshape([24]),
            paddle.to_tensor([0.1640278697013855, 0.1621425896883011, 0.38214758038520813, 0.19742800295352936, 0.34483060240745544, 0.32480672001838684, 0.3432570993900299, 0.3183292746543884, 0.18073347210884094, 0.3267594873905182, 0.2084232121706009, 0.019398842006921768, 0.3618757724761963, 0.10920897871255875, 0.49014297127723694, 0.16398555040359497, 0.4082464873790741, 0.28947269916534424, 0.4446847140789032, 0.0029546874575316906, 0.2877699136734009, 0.29461097717285156, 0.02090122550725937, 0.024387184530496597], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_13133281d4f1b869fd71350db0059266(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 96], dtype='float16', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_edb257f880a904ec0604d047c24ce43e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.43621549010276794, 0.1463736891746521, 0.2666698694229126, 0.44631755352020264, 0.2960793375968933, 0.22160761058330536, 0.487611323595047, 0.17121650278568268, 0.39789727330207825, 0.3016062080860138, 0.052289608865976334, 0.12913253903388977, 0.11650820076465607, 0.42367085814476013, 0.09788241982460022, 0.4506460726261139, 0.07656759768724442, 0.40071722865104675, 0.15001536905765533, 0.2118317037820816, 0.3744438588619232, 0.33575284481048584, 0.47277015447616577, 0.3351629376411438], dtype='float32').reshape([24]),
            paddle.to_tensor([0.3291267454624176, 0.02947189286351204, 0.08261172473430634, 0.01127054076641798, 0.2825303077697754, 0.2548500895500183, 0.2174159586429596, 0.23359008133411407, 0.44517913460731506, 0.3823173940181732, 0.30236542224884033, 0.29040539264678955, 0.3124516010284424, 0.03885800018906593, 0.31476736068725586, 0.1154744103550911, 0.11883702874183655, 0.25659430027008057, 0.24349141120910645, 0.08828964084386826, 0.21199722588062286, 0.3730752766132355, 0.48400285840034485, 0.09242087602615356], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8673a0b0daba627cbc078c5691d903e1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f23ec4eafbf7a3258de35eb8f79baf78(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7ed4bf24daf2d494dfdf343eb38b23e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 128], dtype='float16', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2da5970fd032c8a9fb759947676d4f19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 96], dtype='float16', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_578b5b0d43c98ead33a2ccab23e286af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f29e5e959f516ec0ee28f7dbe3215b6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b6e24ac6f5ba560174e5a9e21dd779c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 128], dtype='float16', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_71ad6673c740b6e41df1f6f6b9cf5335(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 2048], dtype='float16', min=0, max=0.5),
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_50399217fdf7209a51e0cecf5214afa0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d90eea07f23b57be5f9ff68a6d62521f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_42b851f955c30bd182626893e8e27d74(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.46114641427993774, 0.07426156848669052, 0.4666295647621155, 0.35491183400154114, 0.030394725501537323, 0.2541913092136383, 0.2923702299594879, 0.3210584223270416, 0.1203196793794632, 0.4155246317386627, 0.02784230001270771, 0.1949152797460556, 0.3059973418712616, 0.4791969656944275, 0.07841590791940689, 0.477334201335907, 0.2740705907344818, 0.02989884652197361, 0.39064106345176697, 0.34095969796180725, 0.16401034593582153, 0.08200228959321976, 0.4023308753967285, 0.4756181538105011], dtype='float32').reshape([24]),
            paddle.to_tensor([0.2856740355491638, 0.36230796575546265, 0.4513092339038849, 0.3251866400241852, 0.39540255069732666, 0.32235607504844666, 0.43886807560920715, 0.3593394160270691, 0.3579729497432709, 0.4287498891353607, 0.34552475810050964, 0.09864765405654907, 0.03783891722559929, 0.4013940989971161, 0.007578900083899498, 0.25015920400619507, 0.20387496054172516, 0.12132887542247772, 0.07793361693620682, 0.3131977617740631, 0.06049670651555061, 0.2516135573387146, 0.21102569997310638, 0.24982479214668274], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8aff9b3db87ecc0f80859752c8d1af34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 768], dtype='float16', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_420702d25a9d162c1a1eceeaf7845bef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_79187b3b158eb1dc09e593c5b19f4178(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([4, 16, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e02e19b7b0234bb5c071078f0073931c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.11771857738494873, 0.32901522517204285, 0.43469908833503723, 0.0518282987177372, 0.19801312685012817, 0.3224130868911743, 0.19311955571174622, 0.03456486389040947, 0.3933168649673462, 0.16783148050308228, 0.4415842890739441, 0.2783028781414032, 0.292898952960968, 0.03837459534406662, 0.23894594609737396, 0.07346932590007782, 0.0796826034784317, 0.03692133352160454, 0.15529470145702362, 0.11405026167631149, 0.3957921266555786, 0.29471057653427124, 0.17031031847000122, 0.1990957111120224], dtype='float32').reshape([24]),
            paddle.to_tensor([0.25832879543304443, 0.4931113123893738, 0.18722683191299438, 0.022597914561629295, 0.2763192355632782, 0.0614265613257885, 0.04740649089217186, 0.08078627288341522, 0.2758050560951233, 0.09211423993110657, 0.4565351903438568, 0.1683322936296463, 0.0028722784481942654, 0.32548317313194275, 0.14181718230247498, 0.437023401260376, 0.3877274692058563, 0.2050875723361969, 0.2360878884792328, 0.26332393288612366, 0.37715286016464233, 0.2514707148075104, 0.007052071392536163, 0.04978630691766739], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_77acfab37a0f0bf710b1eb68dce59f8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_13988ec92b0b104f309844e8fd603752(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3456335663795471, 0.3952365219593048, 0.24798202514648438, 0.2325054109096527, 0.17840445041656494, 0.06605525314807892, 0.172715961933136, 0.2820567786693573, 0.24686597287654877, 0.3920708894729614, 0.21232019364833832, 0.29470473527908325, 0.27448299527168274, 0.3338477313518524, 0.42042791843414307, 0.1684396117925644, 0.12548065185546875, 0.43801331520080566, 0.3677135407924652, 0.31036290526390076, 0.2199028730392456, 0.40418750047683716, 0.4175221621990204, 0.3173573315143585], dtype='float32').reshape([24]),
            paddle.to_tensor([0.3102054297924042, 0.3570568561553955, 0.43766143918037415, 0.14387300610542297, 0.3451296091079712, 0.07791547477245331, 0.15333037078380585, 0.03412756323814392, 0.3246271014213562, 0.3086414337158203, 0.05776789039373398, 0.43148788809776306, 0.2942861020565033, 0.07223251461982727, 0.1622968465089798, 0.18738383054733276, 0.07832445949316025, 0.4212600886821747, 0.0860716924071312, 0.17707887291908264, 0.04824107140302658, 0.12659397721290588, 0.4985545873641968, 0.44871801137924194], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a2c6291ff5004ac47e4ff96dcd544979(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1985253095626831, 0.45667925477027893, 0.3018195629119873, 0.19589181244373322, 0.25143489241600037, 0.07409659773111343, 0.21123021841049194, 0.17175306379795074, 0.4288332164287567, 0.23377065360546112, 0.11329129338264465, 0.15842397511005402, 0.1774071305990219, 0.10280224680900574, 0.07773150503635406, 0.2935408353805542, 0.24118661880493164, 0.4110824763774872, 0.2182559370994568, 0.3329613208770752, 0.4691450595855713, 0.3909914791584015, 0.26234397292137146, 0.06722228229045868], dtype='float32').reshape([24]),
            paddle.to_tensor([0.049227021634578705, 0.4759061932563782, 0.2724014222621918, 0.3841787874698639, 0.16340342164039612, 0.30459460616111755, 0.11845973134040833, 0.4246612787246704, 0.42658576369285583, 0.4780181646347046, 0.27172282338142395, 0.44673728942871094, 0.00405121361836791, 0.45625001192092896, 0.14672964811325073, 0.30459949374198914, 0.41687309741973877, 0.39910057187080383, 0.03664478659629822, 0.13953712582588196, 0.31201842427253723, 0.24736545979976654, 0.19617009162902832, 0.142278254032135], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_939ea0b6e80bd0bf42deec2698f19722(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([4, 256, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0e2d9edb0f28a78d854793f9858c8660(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.055094800889492035, 0.37313616275787354, 0.16536080837249756, 0.4353681206703186, 0.1005638912320137, 0.09350959211587906, 0.001203414285555482, 0.4297681748867035, 0.4439481794834137, 0.2922578454017639, 0.12797638773918152, 0.3204362392425537, 0.17904186248779297, 0.4908580482006073, 0.36302876472473145, 0.39271265268325806, 0.42487093806266785, 0.13703276216983795, 0.4736252427101135, 0.3023057281970978, 0.4670041501522064, 0.04072602838277817, 0.08456756919622421, 0.19652710855007172], dtype='float32').reshape([24]),
            paddle.to_tensor([0.14108552038669586, 0.32060131430625916, 0.1491626501083374, 0.48202118277549744, 0.4138736128807068, 0.009161756373941898, 0.10162550956010818, 0.15046727657318115, 0.11680741608142853, 0.41517016291618347, 0.16995950043201447, 0.31623271107673645, 0.28821897506713867, 0.2355450838804245, 0.16864117980003357, 0.32969239354133606, 0.26796066761016846, 0.3035222291946411, 0.32983559370040894, 0.31064915657043457, 0.35407254099845886, 0.18587519228458405, 0.1315876543521881, 0.03467690572142601], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ddccf71a9d24de2712006dd32014aaa9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 768], dtype='float16', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b75e3738c82bb5489706f7e28561ff18(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_979766a1042e4133050847b8cc30ed91(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.09823548048734665, 0.47838714718818665, 0.11724691838026047, 0.10808845609426498, 0.39418816566467285, 0.23837871849536896, 0.09598349779844284, 0.3074903190135956, 0.13679452240467072, 0.35758092999458313, 0.20383164286613464, 0.31422656774520874, 0.11314160376787186, 0.293411523103714, 0.1421324461698532, 0.4042380154132843, 0.44725847244262695, 0.056149087846279144, 0.15730471909046173, 0.3959125578403473, 0.3254162073135376, 0.21248623728752136, 0.19642086327075958, 0.1760583519935608], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4864860773086548, 0.25115394592285156, 0.043536558747291565, 0.07683107256889343, 0.1662353277206421, 0.14700046181678772, 0.13580895960330963, 0.26541194319725037, 0.18187929689884186, 0.2111029326915741, 0.41531258821487427, 0.42303693294525146, 0.2888645529747009, 0.18637269735336304, 0.2386856973171234, 0.033667322248220444, 0.4948573708534241, 0.340821772813797, 0.2237222045660019, 0.3811515271663666, 0.36648210883140564, 0.007210808806121349, 0.3156265318393707, 0.11826980113983154], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dbf33eea0c25d6fc99ebe198870c7b16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4769151508808136, 0.008158857002854347, 0.4917914867401123, 0.49117112159729004, 0.06479177623987198, 0.393295556306839, 0.2697933316230774, 0.4985416829586029, 0.46857184171676636, 0.29198071360588074, 0.30213606357574463, 0.2349623590707779, 0.42220428586006165, 0.4138154983520508, 0.2541247010231018, 0.2591824531555176, 0.0999550074338913, 0.01961411163210869, 0.33799442648887634, 0.30161264538764954, 0.1246507465839386, 0.43829742074012756, 0.09040495753288269, 0.10719519853591919], dtype='float32').reshape([24]),
            paddle.to_tensor([0.2391877919435501, 0.3737815022468567, 0.1289060264825821, 0.08537900447845459, 0.29051750898361206, 0.35171228647232056, 0.03218435123562813, 0.18995121121406555, 0.27258914709091187, 0.010567101649940014, 0.44685351848602295, 0.44253531098365784, 0.22592973709106445, 0.1847885102033615, 0.019909050315618515, 0.11204899102449417, 0.4909094572067261, 0.13680391013622284, 0.4597402513027191, 0.12017976492643356, 0.33845311403274536, 0.17003022134304047, 0.4787217676639557, 0.29335784912109375], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cf11fa79551289fb706d55550876b5cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.07141704112291336, 0.2793707251548767, 0.2827867567539215, 0.04335314780473709, 0.10892792791128159, 0.39086300134658813, 0.30681997537612915, 0.1558820754289627, 0.26720330119132996, 0.3384144604206085, 0.4373806118965149, 0.19455914199352264, 0.054969143122434616, 0.08056273311376572, 0.4426751434803009, 0.14130644500255585, 0.40365806221961975, 0.012007717043161392, 0.3433373272418976, 0.11088936775922775, 0.309093713760376, 0.023922933265566826, 0.22095416486263275, 0.3902665674686432], dtype='float32').reshape([24]),
            paddle.to_tensor([0.11661326140165329, 0.1886414736509323, 0.11401382088661194, 0.45248258113861084, 0.19017833471298218, 0.4890216290950775, 0.457439124584198, 0.17805473506450653, 0.03263169154524803, 0.3352476954460144, 0.2398412972688675, 0.2101982682943344, 0.37956905364990234, 0.10066847503185272, 0.46803611516952515, 0.34879422187805176, 0.06290063261985779, 0.10800700634717941, 0.4783109724521637, 0.1266653835773468, 0.41118860244750977, 0.2866671681404114, 0.09591562300920486, 0.01160537451505661], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_983cf177e7203a222c63fa668ee59f49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_40c6429424535ac10edf695ab1c49576(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3905773460865021, 0.47255751490592957, 0.06827348470687866, 0.3897691071033478, 0.2544814944267273, 0.35140788555145264, 0.45024076104164124, 0.39821434020996094, 0.184661865234375, 0.3832377791404724, 0.3933756351470947, 0.17552097141742706, 0.1412210464477539, 0.03268500789999962, 0.1432354897260666, 0.4405647814273834, 0.08373763412237167, 0.31229478120803833, 0.27684828639030457, 0.11988204717636108, 0.0763169378042221, 0.04725104942917824, 0.4849715232849121, 0.05370239168405533], dtype='float32').reshape([24]),
            paddle.to_tensor([0.18731768429279327, 0.06912380456924438, 0.4691333770751953, 0.1881413459777832, 0.1494733989238739, 0.3702240586280823, 0.3775033950805664, 0.0677185207605362, 0.007905966602265835, 0.3199048936367035, 0.08473286777734756, 0.31208425760269165, 0.4825545847415924, 0.4732895493507385, 0.38520902395248413, 0.4892708361148834, 0.48557931184768677, 0.3618151843547821, 0.20597906410694122, 0.4474101960659027, 0.12635764479637146, 0.351006418466568, 0.24386991560459137, 0.22308635711669922], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f9b3e9f3d94923916bcb34376cd88756(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 512], dtype='float16', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f80f1885323eb4b6b6a5782b41201434(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.18176643550395966, 0.293670117855072, 0.2624484896659851, 0.4361293911933899, 0.07585692405700684, 0.06858580559492111, 0.12120606005191803, 0.11128450930118561, 0.05744192376732826, 0.40900009870529175, 0.2146269679069519, 0.27555376291275024, 0.4207175374031067, 0.36192575097084045, 0.2787052392959595, 0.26047801971435547, 0.11711111664772034, 0.15214577317237854, 0.08547452837228775, 0.4935280978679657, 0.14669598639011383, 0.011698123998939991, 0.33363938331604004, 0.2845083177089691], dtype='float32').reshape([24]),
            paddle.to_tensor([0.44516196846961975, 0.3163892924785614, 0.23835355043411255, 0.27797743678092957, 0.1787179857492447, 0.2777225971221924, 0.3618502616882324, 0.41591179370880127, 0.38624534010887146, 0.1391330510377884, 0.0550733357667923, 0.49920082092285156, 0.47869768738746643, 0.4868645668029785, 0.11592158675193787, 0.34415993094444275, 0.03951523080468178, 0.09975859522819519, 0.23246128857135773, 0.32906413078308105, 0.3633597493171692, 0.38302287459373474, 0.32489871978759766, 0.44802775979042053], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f834e5811e46f5da82070c3c4f4a543e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a252d558d05c006dc62b3eebbb72d1d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_45332d89826183473f80d3a4a6e48c16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2739688456058502, 0.35141420364379883, 0.15971989929676056, 0.31889134645462036, 0.24460992217063904, 0.27155646681785583, 0.31388652324676514, 0.3078359067440033, 0.16364701092243195, 0.45672547817230225, 0.12903429567813873, 0.3904195725917816, 0.42172160744667053, 0.4748568534851074, 0.19573597609996796, 0.26933568716049194, 0.1165073961019516, 0.19645756483078003, 0.4910942614078522, 0.17037662863731384, 0.34213700890541077, 0.2966890037059784, 0.3765350878238678, 0.02019944041967392], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4578011929988861, 0.21994835138320923, 0.14889274537563324, 0.4537711441516876, 0.22798968851566315, 0.16223593056201935, 0.2246052324771881, 0.47284629940986633, 0.27551379799842834, 0.2863596975803375, 0.2144504189491272, 0.3153936564922333, 0.4178737998008728, 0.16110862791538239, 0.3688730001449585, 0.14729738235473633, 0.17677165567874908, 0.17660313844680786, 0.46386268734931946, 0.13384221494197845, 0.3307311534881592, 0.24721048772335052, 0.47821661829948425, 0.33737441897392273], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a15a63ccd3ba6a7c452b22c08ee43e47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 256], dtype='float16', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5431e34bb94bed4fc3c774e4edd4c350(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.12926340103149414, 0.3746367394924164, 0.0966346338391304, 0.48589155077934265, 0.005977271124720573, 0.053024400025606155, 0.16533605754375458, 0.26235973834991455, 0.3109891712665558, 0.026151619851589203, 0.43614351749420166, 0.18745367228984833, 0.23843912780284882, 0.4210590422153473, 0.08469489961862564, 0.43124309182167053, 0.4239633083343506, 0.3147006630897522, 0.351948618888855, 0.40203309059143066, 0.12918326258659363, 0.2036353200674057, 0.19623592495918274, 0.3300062417984009], dtype='float32').reshape([24]),
            paddle.to_tensor([0.028716526925563812, 0.052008677273988724, 0.25178372859954834, 0.4622236490249634, 0.3794941008090973, 0.44839930534362793, 0.019807705655694008, 0.3140290379524231, 0.005965252872556448, 0.13115064799785614, 0.42736563086509705, 0.2064211219549179, 0.4952631890773773, 0.29872438311576843, 0.2807587683200836, 0.17627611756324768, 0.19078217446804047, 0.15894819796085358, 0.48415806889533997, 0.1591634303331375, 0.32323431968688965, 0.4438753128051758, 0.3799758851528168, 0.3279326558113098], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d6552586b651882f89033cf9f2fc8cdd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8cdec4a9d0381e097b5592872b5cc817(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4737900495529175, 0.016287220641970634, 0.4447653293609619, 0.24136938154697418, 0.3814895749092102, 0.12673434615135193, 0.38548627495765686, 0.2312469184398651, 0.49276092648506165, 0.09035108983516693, 0.47155842185020447, 0.17014248669147491, 0.2294987142086029, 0.39062392711639404, 0.47454673051834106, 0.053361523896455765, 0.43074747920036316, 0.37570759654045105, 0.043803874403238297, 0.1925482600927353, 0.2910689413547516, 0.34281936287879944, 0.48895928263664246, 0.46691572666168213], dtype='float32').reshape([24]),
            paddle.to_tensor([0.1545064002275467, 0.16438405215740204, 0.49940791726112366, 0.1299777328968048, 0.2455441802740097, 0.2921180725097656, 0.09486763924360275, 0.19775164127349854, 0.13376471400260925, 0.48334771394729614, 0.41669464111328125, 0.2740554213523865, 0.4707385301589966, 0.40687793493270874, 0.298538476228714, 0.34233173727989197, 0.06177215278148651, 0.03756299987435341, 0.3395310342311859, 0.41933736205101013, 0.37930285930633545, 0.038709212094545364, 0.3911648690700531, 0.35464227199554443], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9e6130b20f139429ad2c46ac349f4a45(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4e64327ff89668c30a4bf77d44e70307(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.07557713240385056, 0.3558233082294464, 0.42243364453315735, 0.4306146800518036, 0.43811923265457153, 0.05486283451318741, 0.3844211995601654, 0.07491683214902878, 0.06300283223390579, 0.4082828164100647, 0.1770239770412445, 0.3924751579761505, 0.25515016913414, 0.4513915181159973, 0.4416095018386841, 0.40787339210510254, 0.43840399384498596, 0.37976089119911194, 0.32013317942619324, 0.32050690054893494, 0.2619212567806244, 0.10713458061218262, 0.2494690716266632, 0.4395127296447754], dtype='float32').reshape([24]),
            paddle.to_tensor([0.027584651485085487, 0.026302579790353775, 0.23662877082824707, 0.1326027512550354, 0.20365245640277863, 0.23568670451641083, 0.2173229604959488, 0.0010855309665203094, 0.41290244460105896, 0.43040579557418823, 0.44403910636901855, 0.3929780125617981, 0.08812742680311203, 0.17284369468688965, 0.29252997040748596, 0.4727805256843567, 0.3039507567882538, 0.10545778274536133, 0.3486534357070923, 0.3999795913696289, 0.04018956050276756, 0.44150444865226746, 0.07359124720096588, 0.4029289484024048], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_791c969fe7eeec42ef258c421b089380(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.08470150083303452, 0.11682290583848953, 0.14982552826404572, 0.11840085685253143, 0.28003719449043274, 0.14028280973434448, 0.29519957304000854, 0.31474918127059937, 0.24839916825294495, 0.041571516543626785, 0.18337000906467438, 0.35485830903053284, 0.18036144971847534, 0.32456502318382263, 0.1485738605260849, 0.03197994828224182, 0.42962315678596497, 0.3214710056781769, 0.23568417131900787, 0.3074137270450592, 0.4908376932144165, 0.2721588909626007, 0.4266371428966522, 0.036100633442401886], dtype='float32').reshape([24]),
            paddle.to_tensor([0.3304949998855591, 0.44384172558784485, 0.3347543478012085, 0.43953436613082886, 0.45144498348236084, 0.39297837018966675, 0.3075687885284424, 0.2458404004573822, 0.42095494270324707, 0.38859954476356506, 0.23835857212543488, 0.08090169727802277, 0.26767849922180176, 0.11635294556617737, 0.05923840031027794, 0.001141236163675785, 0.474603533744812, 0.15980187058448792, 0.29345011711120605, 0.3268604874610901, 0.19323277473449707, 0.35638538002967834, 0.07396213710308075, 0.1427922397851944], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_92256550d6343f2cd102b69f2d72e614(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3205685019493103, 0.3178185224533081, 0.2832992970943451, 0.42627134919166565, 0.18488532304763794, 0.2937871813774109, 0.1958288848400116, 0.24533402919769287, 0.46788474917411804, 0.387450635433197, 0.1728552281856537, 0.41208121180534363, 0.07871252298355103, 0.3964398503303528, 0.3614954948425293, 0.10986866056919098, 0.37517237663269043, 0.4539951682090759, 0.15348729491233826, 0.4768764078617096, 0.06612689048051834, 0.42901498079299927, 0.20638227462768555, 0.0009711519232951105], dtype='float32').reshape([24]),
            paddle.to_tensor([0.2281855195760727, 0.3718682825565338, 0.1964660882949829, 0.18728835880756378, 0.11395839601755142, 0.23615674674510956, 0.1815652996301651, 0.3390391767024994, 0.3278261721134186, 0.4212964177131653, 0.289099782705307, 0.3604222536087036, 0.4060812294483185, 0.28670623898506165, 0.38802000880241394, 0.08675705641508102, 0.4644566476345062, 0.01908094435930252, 0.050156120210886, 0.3207481801509857, 0.281728595495224, 0.051431991159915924, 0.31013697385787964, 0.48861098289489746], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_05039c3329edede44d1e8990518d8d56(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 2304, 512], dtype='float16', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_90254402f8d31f2bf1325fe100ba91b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 2304, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a761a98445958715716c74f35e08c12e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.14205768704414368, 0.4415822923183441, 0.23854197561740875, 0.3292657732963562, 0.14784906804561615, 0.018509870395064354, 0.12917637825012207, 0.29400134086608887, 0.23513856530189514, 0.25394970178604126, 0.15431971848011017, 0.3330668807029724, 0.12729501724243164, 0.04253244772553444, 0.05267944186925888, 0.009348760358989239, 0.06349314004182816, 0.3159849941730499, 0.41451936960220337, 0.19024299085140228, 0.2127906084060669, 0.31246519088745117, 0.07096941024065018, 0.14675413072109222], dtype='float32').reshape([24]),
            paddle.to_tensor([0.08113555610179901, 0.013244487345218658, 0.3367671072483063, 0.21360637247562408, 0.48038187623023987, 0.4228726029396057, 0.06821106374263763, 0.1842879056930542, 0.19122567772865295, 0.28619566559791565, 0.17702403664588928, 0.3620232045650482, 0.358939528465271, 0.2499011754989624, 0.4523940682411194, 0.30654534697532654, 0.3462390601634979, 0.11081027239561081, 0.057840969413518906, 0.4870637059211731, 0.03272996097803116, 0.3537355959415436, 0.49751219153404236, 0.20614276826381683], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_405a41dafda22494a682d582ffe6be4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 256], dtype='float16', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ec2f36a317d8dc95d242e5edfe98e61d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.45181822776794434, 0.15644659101963043, 0.41599833965301514, 0.4687879979610443, 0.1417054980993271, 0.2063867151737213, 0.21506008505821228, 0.38842394948005676, 0.41120296716690063, 0.16470792889595032, 0.285857230424881, 0.038601044565439224, 0.4585890471935272, 0.2383735477924347, 0.10306701809167862, 0.47930487990379333, 0.3253852128982544, 0.06533379852771759, 0.09188483655452728, 0.33919185400009155, 0.3822097182273865, 0.24318350851535797, 0.3876914083957672, 0.16744489967823029], dtype='float32').reshape([24]),
            paddle.to_tensor([0.12142302095890045, 0.18080535531044006, 0.11763933300971985, 0.33988332748413086, 0.343924880027771, 0.03538785129785538, 0.12519605457782745, 0.20895296335220337, 0.14743110537528992, 0.1517002433538437, 0.4648142158985138, 0.2057114690542221, 0.2446841448545456, 0.46642202138900757, 0.4943891763687134, 0.3995833694934845, 0.33166688680648804, 0.2197190523147583, 0.34716513752937317, 0.36663302779197693, 0.4549614191055298, 0.16004899144172668, 0.17169106006622314, 0.3339395225048065], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_32c80d45f30a19c3d9ce4785182e4c10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 160], dtype='float16', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6b6af2d30d053cdd4129fd3af99d6b5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 768], dtype='float16', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2173fe57189fce78b472d5539fd0b18d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a5ff97e3d7fdecf7ba93a17327573ed6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.30761468410491943, 0.25545886158943176, 0.11495378613471985, 0.17797860503196716, 0.4689677357673645, 0.4000888764858246, 0.0020925207063555717, 0.04494655877351761, 0.264802485704422, 0.2899845838546753, 0.2785765826702118, 0.2719224989414215, 0.07882795482873917, 0.1616554856300354, 0.1692863404750824, 0.25682011246681213, 0.38755348324775696, 0.4533008933067322, 0.17495843768119812, 0.22069837152957916, 0.15079163014888763, 0.4184850752353668, 0.23045295476913452, 0.04470523074269295], dtype='float32').reshape([24]),
            paddle.to_tensor([0.11151678115129471, 0.4131118357181549, 0.2148454636335373, 0.4352688789367676, 0.3119615912437439, 0.4573203921318054, 0.04482884332537651, 0.07706257700920105, 0.3026610016822815, 0.07344765961170197, 0.15054576098918915, 0.044502269476652145, 0.08810137957334518, 0.4785158634185791, 0.4546124041080475, 0.0761626586318016, 0.3586846888065338, 0.036855924874544144, 0.23320068418979645, 0.4175979197025299, 0.09095178544521332, 0.4313460886478424, 0.46131259202957153, 0.48826664686203003], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dba092ea447ecb281584a14038679d2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6acdaa4aa8726b7a142ebb48d7ffe613(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.05392275005578995, 0.2818715274333954, 0.03615570068359375, 0.20967014133930206, 0.14658932387828827, 0.34495145082473755, 0.40726515650749207, 0.37677526473999023, 0.07277308404445648, 0.30851444602012634, 0.42304614186286926, 0.18453237414360046, 0.2879643142223358, 0.26125243306159973, 0.3647204339504242, 0.39734500646591187, 0.38414183259010315, 0.01990112103521824, 0.14504720270633698, 0.1533571183681488, 0.37138065695762634, 0.0779024064540863, 0.4185749590396881, 0.058583684265613556], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4617677628993988, 0.23614075779914856, 0.0859203115105629, 0.4824765920639038, 0.4810549318790436, 0.16216987371444702, 0.18264314532279968, 0.44300296902656555, 0.07724204659461975, 0.4090736508369446, 0.2143612504005432, 0.1444007009267807, 0.3600376546382904, 0.2866038978099823, 0.06198585033416748, 0.48612532019615173, 0.1343386471271515, 0.14052188396453857, 0.49764618277549744, 0.2540387511253357, 0.4850650131702423, 0.24434922635555267, 0.4234316945075989, 0.3348560631275177], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_46287c0a66a942517380cfbd77df6f7a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.01105545461177826, 0.45600083470344543, 0.1472207009792328, 0.2544405162334442, 0.43499475717544556, 0.1663062423467636, 0.37715381383895874, 0.08432743698358536, 0.008724329993128777, 0.15259897708892822, 0.4038931131362915, 0.28620994091033936, 0.0890762060880661, 0.23181991279125214, 0.4317178428173065, 0.45366808772087097, 0.44179409742355347, 0.14664272964000702, 0.332912415266037, 0.04833918437361717, 0.2811003625392914, 0.21524909138679504, 0.47788986563682556, 0.10210508108139038], dtype='float32').reshape([24]),
            paddle.to_tensor([0.29892483353614807, 0.00020900979870930314, 0.049991875886917114, 0.026884499937295914, 0.4150582551956177, 0.22972820699214935, 0.3861006498336792, 0.3100590109825134, 0.4489043951034546, 0.495871901512146, 0.12742160260677338, 0.010572493076324463, 0.4621841013431549, 0.2164771556854248, 0.12355801463127136, 0.09486736357212067, 0.28200891613960266, 0.18363486230373383, 0.07001034915447235, 0.3041735291481018, 0.24856653809547424, 0.024174576625227928, 0.4064213037490845, 0.28615763783454895], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e448bc7fedfbde58e4d6892a8a5f7d58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.23474448919296265, 0.08997555077075958, 0.33955928683280945, 0.40124720335006714, 0.3704860508441925, 0.4738033711910248, 0.2780499756336212, 0.13458776473999023, 0.43936723470687866, 0.10048970580101013, 0.2777318060398102, 0.44833943247795105, 0.17911478877067566, 0.06459233164787292, 0.49738913774490356, 0.33132410049438477, 0.4555908441543579, 0.16767114400863647, 0.2422560602426529, 0.2416214495897293, 0.22913630306720734, 0.17069366574287415, 0.2416754513978958, 0.1942424476146698], dtype='float32').reshape([24]),
            paddle.to_tensor([0.35250532627105713, 0.2005789428949356, 0.32223692536354065, 0.42975544929504395, 0.4608476459980011, 0.04539401829242706, 0.10758612304925919, 0.40506601333618164, 0.33503568172454834, 0.2947435677051544, 0.37529292702674866, 0.05912544205784798, 0.40782251954078674, 0.2556036412715912, 0.1256026029586792, 0.15748092532157898, 0.16714735329151154, 0.23512257635593414, 0.43873095512390137, 0.2348024994134903, 0.06220410391688347, 0.10202769935131073, 0.36546432971954346, 0.21086686849594116], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5177cd3c986b5c28fd78ad70811cadc4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.04341757670044899, 0.17153790593147278, 0.1251583695411682, 0.2766187787055969, 0.13866543769836426, 0.4495866298675537, 0.2950727045536041, 0.047320734709501266, 0.2981654405593872, 0.018491068854928017, 0.20255699753761292, 0.11563907563686371, 0.25632721185684204, 0.45663735270500183, 0.1606222242116928, 0.3366738259792328, 0.4341822564601898, 0.1583838164806366, 0.3315364718437195, 0.025674160569906235, 0.13253335654735565, 0.0920831710100174, 0.3664415180683136, 0.23842822015285492], dtype='float32').reshape([24]),
            paddle.to_tensor([0.16132861375808716, 0.3060794770717621, 0.30159449577331543, 0.44834211468696594, 0.16444161534309387, 0.0052471221424639225, 0.24062617123126984, 0.1866290271282196, 0.26007893681526184, 0.36656081676483154, 0.15327471494674683, 0.010063513182103634, 0.0948360487818718, 0.22680942714214325, 0.07861218601465225, 0.43750330805778503, 0.04992927610874176, 0.30160877108573914, 0.16938968002796173, 0.15771284699440002, 0.15416525304317474, 0.30159902572631836, 0.18859180808067322, 0.1481732726097107], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0e98cecefd5132369df8caff325c5677(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4258430004119873, 0.4043937921524048, 0.08537891507148743, 0.05949791520833969, 0.2487742155790329, 0.40879178047180176, 0.36262497305870056, 0.4746846556663513, 0.4779752492904663, 0.2871939539909363, 0.06054530665278435, 0.08476774394512177, 0.4751846194267273, 0.4916658401489258, 0.08174728602170944, 0.20136471092700958, 0.4613259732723236, 0.11537114530801773, 0.13589829206466675, 0.4434391260147095, 0.3364270329475403, 0.09092148393392563, 0.2093760222196579, 0.11300299316644669], dtype='float32').reshape([24]),
            paddle.to_tensor([0.2358911633491516, 0.2933349609375, 0.13400079309940338, 0.125348761677742, 0.13178804516792297, 0.46917495131492615, 0.25751635432243347, 0.3299882411956787, 0.018248556181788445, 0.4731366038322449, 0.49038833379745483, 0.05257653072476387, 0.23448865115642548, 0.3030720353126526, 0.07812820374965668, 0.02519453689455986, 0.07034394145011902, 0.178263321518898, 0.4488944113254547, 0.1642979085445404, 0.028249628841876984, 0.13841675221920013, 0.38248637318611145, 0.4505709111690521], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0180669372fc84e7433c19da17bf6546(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.04228634014725685, 0.3086622655391693, 0.13276219367980957, 0.30395007133483887, 0.004799717105925083, 0.2970888316631317, 0.07038295269012451, 0.4458417594432831, 0.22578908503055573, 0.14197063446044922, 0.013123948127031326, 0.0666479840874672, 0.05203132703900337, 0.42245620489120483, 0.48057305812835693, 0.4700353741645813, 0.284076064825058, 0.4608613848686218, 0.23168109357357025, 0.06484774500131607, 0.4914454221725464, 0.05354077368974686, 0.1024472564458847, 0.19013266265392303], dtype='float32').reshape([24]),
            paddle.to_tensor([0.36631155014038086, 0.23084913194179535, 0.21997793018817902, 0.47608256340026855, 0.10233937203884125, 0.16964471340179443, 0.03537775203585625, 0.25137224793434143, 0.3494671881198883, 0.4114452600479126, 0.023448318243026733, 0.19525642693042755, 0.39177054166793823, 0.4076698124408722, 0.31069034337997437, 0.3819114863872528, 0.04235578700900078, 0.09784754365682602, 0.49831750988960266, 0.47350364923477173, 0.4132387638092041, 0.4302612543106079, 0.2860724627971649, 0.3112300932407379], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7b298efaf36b77ac7da06fb582b1f908(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6d1cf1b74b0f30ee2948603dbe21a396(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1035740002989769, 0.1295202374458313, 0.14366847276687622, 0.49217090010643005, 0.10821545869112015, 0.10661082714796066, 0.4754040539264679, 0.12215252220630646, 0.038166921585798264, 0.2644746005535126, 0.13129262626171112, 0.39217284321784973, 0.42081138491630554, 0.42129525542259216, 0.2439534217119217, 0.261929452419281, 0.294368177652359, 0.2997193932533264, 0.2187143713235855, 0.3327438533306122, 0.27129560708999634, 0.03936881944537163, 0.30701062083244324, 0.37757521867752075], dtype='float32').reshape([24]),
            paddle.to_tensor([0.25314557552337646, 0.2843272387981415, 0.28727027773857117, 0.06917640566825867, 0.45657777786254883, 0.27277663350105286, 0.38662365078926086, 0.092476487159729, 0.11354199051856995, 0.4542045295238495, 0.16675810515880585, 0.05748770385980606, 0.40060460567474365, 0.003968958742916584, 0.2902595102787018, 0.37619245052337646, 0.4679041802883148, 0.23376911878585815, 0.05942581221461296, 0.29923492670059204, 0.24800780415534973, 0.3901248574256897, 0.3345220685005188, 0.06277187168598175], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_01681137c9b40879bf85c7228a7090e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1824134737253189, 0.49814826250076294, 0.0069441478699445724, 0.48965024948120117, 0.4977090060710907, 0.15261100232601166, 0.042410243302583694, 0.1910579651594162, 0.06030287966132164, 0.3869251608848572, 0.47306588292121887, 0.4250328838825226, 0.27883395552635193, 0.2332274615764618, 0.3894924521446228, 0.10761517286300659, 0.17170391976833344, 0.26800787448883057, 0.2943716049194336, 0.35984307527542114, 0.34916049242019653, 0.12699183821678162, 0.40980416536331177, 0.04790949076414108], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4194541275501251, 0.30980581045150757, 0.23269504308700562, 0.2916938066482544, 0.2173774093389511, 0.2624126970767975, 0.30144003033638, 0.2126806229352951, 0.10852868854999542, 0.1543165147304535, 0.15847784280776978, 0.13367633521556854, 0.46247756481170654, 0.15025699138641357, 0.26561209559440613, 0.286119282245636, 0.21557362377643585, 0.4007219076156616, 0.11764578521251678, 0.15842530131340027, 0.1848953813314438, 0.3914937973022461, 0.11386066675186157, 0.3431903123855591], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7ddf319ba5c8823fbba947f0a7bf3827(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a4aac0c75e545df6e5b5c4b6e7b8d44d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.39517417550086975, 0.3536563515663147, 0.4822602868080139, 0.21384558081626892, 0.3235523998737335, 0.4825739562511444, 0.13019821047782898, 0.01079529244452715, 0.22576725482940674, 0.4476866126060486, 0.29225435853004456, 0.2099895179271698, 0.026658814400434494, 0.01382521167397499, 0.475036084651947, 0.4646579623222351, 0.018700728192925453, 0.1769464910030365, 0.4254573881626129, 0.2752411365509033, 0.46598365902900696, 0.2485819160938263, 0.013161174021661282, 0.09954072535037994], dtype='float32').reshape([24]),
            paddle.to_tensor([0.03876975178718567, 0.21373344957828522, 0.3756926357746124, 0.17542016506195068, 0.19525600969791412, 0.19853855669498444, 0.18526841700077057, 0.3543507158756256, 0.13645541667938232, 0.059873033314943314, 0.23291927576065063, 0.044369764626026154, 0.1702864170074463, 0.29753437638282776, 0.46651172637939453, 0.11118439584970474, 0.189055398106575, 0.19702266156673431, 0.017139531672000885, 0.13025830686092377, 0.16606055200099945, 0.3478761315345764, 0.22786249220371246, 0.1389087587594986], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_364c08f19a1f5f67e436560d6bf31835(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3718504011631012, 0.4057827591896057, 0.08825507014989853, 0.3191535472869873, 0.3273610770702362, 0.09889256954193115, 0.34657755494117737, 0.08334571868181229, 0.3149149417877197, 0.05698605999350548, 0.4020753800868988, 0.47568681836128235, 0.05964718386530876, 0.0746929794549942, 0.18867149949073792, 0.0264153853058815, 0.1237323209643364, 0.35682469606399536, 0.13458889722824097, 0.34701645374298096, 0.0067862761206924915, 0.46045929193496704, 0.3511141240596771, 0.4836772084236145], dtype='float32').reshape([24]),
            paddle.to_tensor([0.3772055506706238, 0.11499578505754471, 0.24551093578338623, 0.4761704206466675, 0.41531121730804443, 0.1077539473772049, 0.11296368390321732, 0.037412069737911224, 0.0006786211743019521, 0.4879702925682068, 0.10229720920324326, 0.37581220269203186, 0.490165114402771, 0.3041853904724121, 0.29955312609672546, 0.027473805472254753, 0.0033867796882987022, 0.2763216495513916, 0.006897625979036093, 0.05040846765041351, 0.16725175082683563, 0.3542760908603668, 0.14458028972148895, 0.022288020700216293], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_be1eebe0a5cf5e0352cbaef41d035d88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([4, 64, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_18685871512fa959904ce09b5a475d34(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.19510102272033691, 0.20686554908752441, 0.31951838731765747, 0.15141570568084717, 0.1971461921930313, 0.23452912271022797, 0.008153560571372509, 0.2686106562614441, 0.28337332606315613, 0.11421645432710648, 0.3937045931816101, 0.26021596789360046, 0.43100133538246155, 0.16396458446979523, 0.3214974105358124, 0.2177264243364334, 0.4210642874240875, 0.2913675308227539, 0.289941668510437, 0.3929293155670166, 0.07857263833284378, 0.19846592843532562, 0.3294351398944855, 0.2203688770532608], dtype='float32').reshape([24]),
            paddle.to_tensor([0.27907368540763855, 0.09010409563779831, 0.11294716596603394, 0.0027453519869595766, 0.3846855163574219, 0.17290423810482025, 0.18456506729125977, 0.3076610863208771, 0.40406158566474915, 0.10581346601247787, 0.44341564178466797, 0.12190552055835724, 0.4910966753959656, 0.2187657505273819, 0.48935776948928833, 0.4100140631198883, 0.04153335466980934, 0.3354561924934387, 0.13319899141788483, 0.16179582476615906, 0.14329266548156738, 0.07169412821531296, 0.04286563768982887, 0.4193059802055359], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_31cefdebde78080edfe4a6256b3e4fc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_422c7f9bfc21d18e9e509b18f74613e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 192], dtype='float16', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6a4925293b013edba095dcff05bfacf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3349670171737671, 0.46534308791160583, 0.3643534183502197, 0.41702523827552795, 0.19068734347820282, 0.05611376091837883, 0.4849598705768585, 0.14781707525253296, 0.42353081703186035, 0.4401087760925293, 0.311707466840744, 0.04038776457309723, 0.15079443156719208, 0.014491400681436062, 0.10487615317106247, 0.4530875086784363, 0.1493648886680603, 0.46924588084220886, 0.051501184701919556, 0.304236501455307, 0.3389779031276703, 0.16972464323043823, 0.13079366087913513, 0.26695775985717773], dtype='float32').reshape([24]),
            paddle.to_tensor([0.31890788674354553, 0.12577171623706818, 0.13630738854408264, 0.4791688024997711, 0.21668066084384918, 0.4769456386566162, 0.49684375524520874, 0.20623788237571716, 0.46564146876335144, 0.46870407462120056, 0.3254479467868805, 0.43968111276626587, 0.17941807210445404, 0.35473185777664185, 0.23073816299438477, 0.4777703881263733, 0.23228968679904938, 0.39240074157714844, 0.49448615312576294, 0.04849475249648094, 0.19518831372261047, 0.28630977869033813, 0.4168829321861267, 0.4603678584098816], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_90a8f02835a888191b7fc375b5d4703e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.05121523141860962, 0.21934635937213898, 0.15172675251960754, 0.46700021624565125, 0.4412868022918701, 0.03750431165099144, 0.0622614361345768, 0.2170773148536682, 0.19763746857643127, 0.32561802864074707, 0.04589257016777992, 0.03163204342126846, 0.05635915696620941, 0.2435961365699768, 0.07495290040969849, 0.06594714522361755, 0.39458167552948, 0.19440683722496033, 0.30179962515830994, 0.31341537833213806, 0.34694650769233704, 0.07910668849945068, 0.16503579914569855, 0.18005156517028809], dtype='float32').reshape([24]),
            paddle.to_tensor([0.02027488872408867, 0.4624885618686676, 0.07281754165887833, 0.24458208680152893, 0.3375331163406372, 0.3931454122066498, 0.17199786007404327, 0.22989076375961304, 0.2197123020887375, 0.19207079708576202, 0.010000357404351234, 0.006858407985419035, 0.09518784284591675, 0.010860231705009937, 0.16182701289653778, 0.41861408948898315, 0.039846502244472504, 0.07708526402711868, 0.22896268963813782, 0.049331407994031906, 0.0629739761352539, 0.24582064151763916, 0.04951018467545509, 0.047755368053913116], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_693e1b003f18ece28f27b82b315818bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.14240507781505585, 0.2461661994457245, 0.3206366300582886, 0.19137898087501526, 0.43590784072875977, 0.34423115849494934, 0.16656647622585297, 0.16306623816490173, 0.4494502544403076, 0.24616478383541107, 0.3063179552555084, 0.22665995359420776, 0.07646908611059189, 0.3839546740055084, 0.232854962348938, 0.26128092408180237, 0.4535377323627472, 0.2770805358886719, 0.47044259309768677, 0.2856404483318329, 0.2464005947113037, 0.3421778082847595, 0.07225311547517776, 0.26087984442710876], dtype='float32').reshape([24]),
            paddle.to_tensor([0.19987092912197113, 0.3576652407646179, 0.11787109076976776, 0.3325967788696289, 0.288742333650589, 0.4972230792045593, 0.48721984028816223, 0.49964529275894165, 0.19424232840538025, 0.09946565330028534, 0.4673880934715271, 0.445083349943161, 0.09163814038038254, 0.011729656718671322, 0.3958911895751953, 0.2483629584312439, 0.06255633383989334, 0.29839298129081726, 0.13364702463150024, 0.1678139567375183, 0.34484246373176575, 0.009993314743041992, 0.36321166157722473, 0.3363507390022278], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f58c29cc65afdf6028dafba769a6327f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.034891657531261444, 0.25244250893592834, 0.49686574935913086, 0.3393307626247406, 0.2291957437992096, 0.008383574895560741, 0.058248378336429596, 0.2189389020204544, 0.08078977465629578, 0.08699306845664978, 0.21978430449962616, 0.4298100173473358, 0.41483643651008606, 0.020389322191476822, 0.34036436676979065, 0.10749401152133942, 0.06748127192258835, 0.3839569389820099, 0.02750573679804802, 0.28205761313438416, 0.3634272813796997, 0.19779933989048004, 0.11216151714324951, 0.49488502740859985], dtype='float32').reshape([24]),
            paddle.to_tensor([0.17641766369342804, 0.010071221739053726, 0.016195164993405342, 0.050734374672174454, 0.18561075627803802, 0.0179679524153471, 0.19758334755897522, 0.3588763177394867, 0.27565568685531616, 0.011221563443541527, 0.14906294643878937, 0.20636238157749176, 0.059937965124845505, 0.34043920040130615, 0.3125981092453003, 0.10364650934934616, 0.2571318447589874, 0.4433269798755646, 0.014500022865831852, 0.3173336684703827, 0.37322044372558594, 0.3926061987876892, 0.1845085471868515, 0.3290131390094757], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_35f8f728f303c9ce68658f4aa6b02a49(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b40192c1e65b36dea8f489ebf100b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_35b5045c59069402b1ef803ff07db12a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2000417560338974, 0.43612444400787354, 0.10320629924535751, 0.3864510655403137, 0.300361305475235, 0.39112409949302673, 0.45817074179649353, 0.3014487624168396, 0.27445268630981445, 0.24538226425647736, 0.24052831530570984, 0.4098356366157532, 0.18186643719673157, 0.006081525702029467, 0.16125011444091797, 0.41375982761383057, 0.3779347538948059, 0.3756195306777954, 0.3533840477466583, 0.050032224506139755, 0.3142375349998474, 0.39956992864608765, 0.20750340819358826, 0.14826947450637817], dtype='float32').reshape([24]),
            paddle.to_tensor([0.34167858958244324, 0.3799011707305908, 0.13324213027954102, 0.04921453446149826, 0.32886746525764465, 0.19674910604953766, 0.08400838822126389, 0.4864448010921478, 0.14022554457187653, 0.2453107386827469, 0.09226436913013458, 0.07580164074897766, 0.2612718939781189, 0.12942540645599365, 0.43427303433418274, 0.3478774130344391, 0.3271027207374573, 0.20842719078063965, 0.3521370589733124, 0.43646863102912903, 0.26970207691192627, 0.008656446821987629, 0.3533826768398285, 0.23955699801445007], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dab9bb8e91ba55e62668bc6550291d47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.26550623774528503, 0.31317174434661865, 0.2199205905199051, 0.18995782732963562, 0.25097888708114624, 0.13643352687358856, 0.22136832773685455, 0.43960127234458923, 0.4574490487575531, 0.33221688866615295, 0.2499828189611435, 0.46667593717575073, 0.04879075661301613, 0.4514358937740326, 0.2205541878938675, 0.0220150426030159, 0.04371117055416107, 0.41575679183006287, 0.13118937611579895, 0.4692220687866211, 0.05059470236301422, 0.22142383456230164, 0.09091667830944061, 0.4025753438472748], dtype='float32').reshape([24]),
            paddle.to_tensor([0.11101546138525009, 0.335085928440094, 0.28303608298301697, 0.410030335187912, 0.3223246932029724, 0.40834444761276245, 0.231110081076622, 0.4958789348602295, 0.4362051486968994, 0.36121881008148193, 0.46931642293930054, 0.023132525384426117, 0.2836085259914398, 0.09515850991010666, 0.0554984025657177, 0.247451514005661, 0.24981847405433655, 0.39132487773895264, 0.048574574291706085, 0.04890216886997223, 0.11277665942907333, 0.37451615929603577, 0.45408895611763, 0.301052063703537], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_30a04418c7a140355ef6be2df3090622(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_130c2e3140d4ec0a5fc439e705de4670(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.13091669976711273, 0.40552303194999695, 0.39937496185302734, 0.49862000346183777, 0.38601189851760864, 0.12211894243955612, 0.17948594689369202, 0.2911418080329895, 0.2626224458217621, 0.4496499300003052, 0.28549137711524963, 0.18957547843456268, 0.3354063332080841, 0.10536371171474457, 0.4517870247364044, 0.27307987213134766, 0.026530519127845764, 0.3480890393257141, 0.15569426119327545, 0.32755985856056213, 0.326185405254364, 0.3942374587059021, 0.0774802416563034, 0.4810302257537842], dtype='float32').reshape([24]),
            paddle.to_tensor([0.3104518949985504, 0.30517590045928955, 0.4897959530353546, 0.28241443634033203, 0.32304996252059937, 0.0023260777816176414, 0.1564987599849701, 0.19034205377101898, 0.06846291571855545, 0.20177443325519562, 0.2427491545677185, 0.11338825523853302, 0.09213965386152267, 0.0379730649292469, 0.08227625489234924, 0.47217857837677, 0.2307276576757431, 0.20990292727947235, 0.24079908430576324, 0.3523121476173401, 0.3563387095928192, 0.4621063768863678, 0.4361419677734375, 0.03155728429555893], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b5740e06b324122e623da5f0602df669(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 2048], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_23646148445c623664a0407af350fff8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3325293958187103, 0.17666979134082794, 0.31898054480552673, 0.21584579348564148, 0.04341138154268265, 0.4263390004634857, 0.10768727213144302, 0.1624918133020401, 0.1501302272081375, 0.46592506766319275, 0.08752952516078949, 0.4479202330112457, 0.09978561103343964, 0.38107606768608093, 0.43186691403388977, 0.2065444439649582, 0.2996521294116974, 0.3966462314128876, 0.3860372304916382, 0.25005075335502625, 0.13984428346157074, 0.282341331243515, 0.1683175414800644, 0.10451474040746689], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4624478816986084, 0.36710506677627563, 0.3395622968673706, 0.3909805417060852, 0.11182694137096405, 0.24047018587589264, 0.14647217094898224, 0.1858019381761551, 0.16573773324489594, 0.09910175949335098, 0.4861185848712921, 0.2190948873758316, 0.08021880686283112, 0.04225517064332962, 0.0299625676125288, 0.27269625663757324, 0.22387155890464783, 0.33000442385673523, 0.4295661747455597, 0.31865277886390686, 0.021751921623945236, 0.3084564507007599, 0.10799778997898102, 0.249933123588562], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_192f9667ef49a38315a77c1a914556ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.24688509106636047, 0.17250116169452667, 0.2432025820016861, 0.2606157660484314, 0.41954201459884644, 0.2613685429096222, 0.22679725289344788, 0.045032985508441925, 0.2605045437812805, 0.4567176103591919, 0.2659311890602112, 0.13469034433364868, 0.28229042887687683, 0.3167077898979187, 0.36961105465888977, 0.014117349870502949, 0.4851415455341339, 0.021073240786790848, 0.15424425899982452, 0.16487419605255127, 0.0053559597581624985, 0.38903504610061646, 0.030328867956995964, 0.49262383580207825], dtype='float32').reshape([24]),
            paddle.to_tensor([0.37008392810821533, 0.043361105024814606, 0.35771626234054565, 0.030832350254058838, 0.32057854533195496, 0.25181934237480164, 0.030989358201622963, 0.07841219753026962, 0.1291738748550415, 0.057169973850250244, 0.24631285667419434, 0.3054234981536865, 0.4609898626804352, 0.1715894192457199, 0.49217599630355835, 0.36563172936439514, 0.19789007306098938, 0.135758176445961, 0.04906543344259262, 0.15772047638893127, 0.36431390047073364, 0.025460606440901756, 0.09055653214454651, 0.31386837363243103], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_db1a7a1149486d1d2677f4fed6458810(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([4, 16, 240], dtype='float16', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_636a8ffa9baf161b787bb8c882c4ddf4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_be2f5356e0e705cc5fe7da416f4ced99
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ca014865213ebb2ed793442be13fe352(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bcc827b79bd840d26c0466333bfa67de
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3136044144630432, 0.40865516662597656, 0.07151678949594498, 0.24499498307704926, 0.1565580517053604, 0.45366188883781433, 0.16632704436779022, 0.3836362659931183, 0.3116975426673889, 0.30352872610092163, 0.1864796131849289, 0.08439808338880539, 0.15562450885772705, 0.1277054399251938, 0.27769604325294495, 0.4765385389328003, 0.42150402069091797, 0.47910937666893005, 0.2691156268119812, 0.24255825579166412, 0.05393240228295326, 0.172566756606102, 0.49462807178497314, 0.24584709107875824], dtype='float32').reshape([24]),
            paddle.to_tensor([0.14712202548980713, 0.26961320638656616, 0.30474522709846497, 0.43793705105781555, 0.25082191824913025, 0.2535148561000824, 0.2118612676858902, 0.1316499412059784, 0.044650040566921234, 0.459414005279541, 0.3043159246444702, 0.16374902427196503, 0.2391333431005478, 0.22803060710430145, 0.26731616258621216, 0.03101859800517559, 0.3141729235649109, 0.2126837819814682, 0.42483556270599365, 0.4464513957500458, 0.1091708093881607, 0.010207617655396461, 0.025683632120490074, 0.2227029800415039], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_18cb7f96be46c4ada6233283b85a6cdb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a254031597d8288871b0422f0fa8e363
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6442ebb8a4f6dd1e51ac6b4895ca1ac6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 197, 384], dtype='float16'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8b9902d93743ebcb63aa215da65bdd89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6442ebb8a4f6dd1e51ac6b4895ca1ac6
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 384], dtype='float16', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c1810c147e9790f0c38d76040d9a876f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 197, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0558f21e8763d70103846b66b4c48263(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1810c147e9790f0c38d76040d9a876f
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d05efc18ba61e8edb2f620938f463652(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 196, 384], dtype='float16'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0fb814464ca664f81c94e9aef647bec8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d05efc18ba61e8edb2f620938f463652
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 384], dtype='float16', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_98076a412c77a91daadd5b6e87472d20(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            paddle.static.InputSpec(shape=[320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fffb331858a3b2f0a0efd546116408f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_98076a412c77a91daadd5b6e87472d20
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_bdec3fbaf88ffee720f6c968e9f14283(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 576, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a95186096a2daf3591659df92cd3633d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bdec3fbaf88ffee720f6c968e9f14283
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_fece71f44bda3a9c3728d72c018f20c2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 384], dtype='float16'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_34785e4a167b9b7c579a8dd92110ff02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fece71f44bda3a9c3728d72c018f20c2
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 384], dtype='float16', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f180a2767fb438719a209475fc419192(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 197, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aaf9c901f800ebabed54730feaa5dd04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f180a2767fb438719a209475fc419192
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_937f99ba5d70cd6325f7d53e0a367306(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_68982d1dabf2b78308598f8f129d610e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_937f99ba5d70cd6325f7d53e0a367306
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b246d33eedb7e4e85c06d4af9d9f5222(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 26, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0af144ad16a653566f9f02b25dfce5fb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b246d33eedb7e4e85c06d4af9d9f5222
    def get_inputs(self):
        return [
            paddle.uniform([1, 26, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8849179f5d074ef8df50d7e13dc2683f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, 512], dtype='float16'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5cad56c365cc8b4ad959ec3e6163a443(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8849179f5d074ef8df50d7e13dc2683f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 512], dtype='float16', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_84ece0043cb40c018adc993a409dbe42(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 196, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            paddle.static.InputSpec(shape=[320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f9b4b8df54a80bac677389875e07fcc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_84ece0043cb40c018adc993a409dbe42
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f5f81246e867c3e233889d0268bf2278(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 196, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6d9077eb01ab74a54b64654816f50a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5f81246e867c3e233889d0268bf2278
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 24], dtype='float32'),
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            paddle.static.InputSpec(shape=[24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6de48d07bd9107b0e1c2761b549ca3e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4274570345878601, 0.43177613615989685, 0.20555298030376434, 0.39094802737236023, 0.49809014797210693, 0.04659171402454376, 0.43223974108695984, 0.27359694242477417, 0.34672853350639343, 0.29964154958724976, 0.06038147956132889, 0.45957058668136597, 0.313096821308136, 0.07567092776298523, 0.13435299694538116, 0.20235183835029602, 0.074857696890831, 0.3765570819377899, 0.39376622438430786, 0.07235901057720184, 0.03240571171045303, 0.42651236057281494, 0.29713869094848633, 0.25388598442077637], dtype='float32').reshape([24]),
            paddle.to_tensor([0.008527816273272038, 0.3680625557899475, 0.19607119262218475, 0.2619350254535675, 0.05993141978979111, 0.4876129627227783, 0.45239174365997314, 0.08012320101261139, 0.2630710005760193, 0.24455560743808746, 0.17314158380031586, 0.4521302282810211, 0.3052957057952881, 0.07388428598642349, 0.32233917713165283, 0.15124258399009705, 0.40676814317703247, 0.23380887508392334, 0.20318107306957245, 0.4708256423473358, 0.2285236418247223, 0.49071204662323, 0.23710103332996368, 0.09393852949142456], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1db26a12080162c71898598c11e7c804(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dba234e3449ed32e0cee53b18cd784e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1db26a12080162c71898598c11e7c804
    def get_inputs(self):
        return [
            paddle.uniform([4, 64, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_771af089da2fc564774a095a0bf6734b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 50, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_97e48df8b215732131f747f2abfca91e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_771af089da2fc564774a095a0bf6734b
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d7d9c6d92ea8322849caa9afdedf5626(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 577, 768], dtype='float16'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_374705c2cb260a8c321b06df561cf6d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d7d9c6d92ea8322849caa9afdedf5626
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float16', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_25aaecce78402ab576c487feb7a35cb7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 577, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2bc61d442fdf33966af18cf3a6d8e22f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_25aaecce78402ab576c487feb7a35cb7
    def get_inputs(self):
        return [
            paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_120a985ec451d79cd6130d6ce016a247(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 197, 768], dtype='float16'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_373e64be0923f3c10d6be646d32901d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_120a985ec451d79cd6130d6ce016a247
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 768], dtype='float16', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c9f087c5bb03024d10c0dd030843b854(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 576, 512], dtype='float16'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d6a3f08ba5637fdc2fa751974deb7d4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9f087c5bb03024d10c0dd030843b854
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 512], dtype='float16', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_38658ec03168625761e5584f2e9d7d22(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 196, 384], dtype='float16'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_53f3c40c1423714b0618e7afe76964aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38658ec03168625761e5584f2e9d7d22
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 384], dtype='float16', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_68a63a83e8dd554783b3c9361f9867a1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 196, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8b6665dd964fc9364809c1619aabf914(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_68a63a83e8dd554783b3c9361f9867a1
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9762c724605bd2f40bd18d7fccaaf833(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 197, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_77b39f0161c9bdb3369c754427eba03f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9762c724605bd2f40bd18d7fccaaf833
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b8073412405de3409153fccd43d92989(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 197, 768], dtype='float16'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_38c814fcbaea8a15b21b3765d14cebd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b8073412405de3409153fccd43d92989
    def get_inputs(self):
        return [
            paddle.uniform([1, 197, 768], dtype='float16', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3f2cfce654eb8814c3e8361e62d73ceb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 196, 320], dtype='float16'),
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            paddle.static.InputSpec(shape=[320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e0a0ce9d9a04d732e4b632379b57a6a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3f2cfce654eb8814c3e8361e62d73ceb
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 320], dtype='float16', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4e31b4942e66845fa18cfc19a0dc94ad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2083c7051605975db0fb104b0f20d390(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e31b4942e66845fa18cfc19a0dc94ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_826bdcbe8204f31a0e9e0400de04a075(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 50, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ea26cc94e15736388466fecdcea01129(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_826bdcbe8204f31a0e9e0400de04a075
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b9cb1c401643c1dff62f03931bdd1f80(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 9216, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9691510bd14d036976644682d569861a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b9cb1c401643c1dff62f03931bdd1f80
    def get_inputs(self):
        return [
            paddle.uniform([1, 9216, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d75498a38aadad78d5e110232b1b0b05(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            paddle.static.InputSpec(shape=[160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0a4aae9e024a0c3d7ca2a7f03805dd15(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d75498a38aadad78d5e110232b1b0b05
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0ea392b61ab64b9e90e6cf691bc44b23(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 784, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1df6378c570481bbe6c03a3c5bc3eb11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0ea392b61ab64b9e90e6cf691bc44b23
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_996b58ad674d876e7812f873b8eccb82(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 196, 160], dtype='float16'),
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            paddle.static.InputSpec(shape=[160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b39799f77df00bd199083d35f78a1a0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_996b58ad674d876e7812f873b8eccb82
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 160], dtype='float16', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_46f278bf3300d8623c5e81c61377a770(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b4a2b427829fecadb92f3d45377b04b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_46f278bf3300d8623c5e81c61377a770
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b5bcf592ec43184bd25bb879dfb78e0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8849179f5d074ef8df50d7e13dc2683f
    def get_inputs(self):
        return [
            paddle.uniform([1, 25, 512], dtype='float16', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_eb68b6cc42aff4c0f4d6edf34ab233ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 200, 64], dtype='float16'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a263a4aaeec00172cc9579a0a2dfbb84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb68b6cc42aff4c0f4d6edf34ab233ee
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_60517ba961807371eabb8b5c57c07c29(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 50, 256], dtype='float16'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_234c3cacbf037368a437f9a0656f5405(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60517ba961807371eabb8b5c57c07c29
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 256], dtype='float16', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4e44010dfbba62ca79757686b03cd206(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 784, 192], dtype='float16'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8d9093d93802e2f46fa5987129c2291a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e44010dfbba62ca79757686b03cd206
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 192], dtype='float16', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_24fc18f2d05b090df58f251d42345b99(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 320], dtype='float16'),
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            paddle.static.InputSpec(shape=[320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_661d5c734b828c9a043de7321f6d1c04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_24fc18f2d05b090df58f251d42345b99
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 320], dtype='float16', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_dd1a8ccd6ce65a97d9c5683f556229f9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
            paddle.static.InputSpec(shape=[384], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_09bb260a1b92580bcacd7fe4d4aac712(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd1a8ccd6ce65a97d9c5683f556229f9
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
            paddle.uniform([384], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a81fb319cf0944ebabb6b2dc2352deef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_764a0d09955e55d9d63164eada6942be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a81fb319cf0944ebabb6b2dc2352deef
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d07565080e76228d3518844748f235d9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 512], dtype='float16'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d00b2d7e52808baef8e8b8296c16565f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d07565080e76228d3518844748f235d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_243bda4b16991324f1d0281b6b42595c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 196, 320], dtype='float32'),
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            paddle.static.InputSpec(shape=[320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9007a1d50fa04d2bfc88b0c92ffc53ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_243bda4b16991324f1d0281b6b42595c
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_92ce833570191359ba1946438e0ec3a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 100, 128], dtype='float16'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_57704545ec2ecfcc3bdfc25bde0aa28a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_92ce833570191359ba1946438e0ec3a2
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 128], dtype='float16', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_124a7efc51219f8dc1dec25adb89b860(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 784, 128], dtype='float16'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4f624fd8eaab11720d31f91cacd8bc22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_124a7efc51219f8dc1dec25adb89b860
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 128], dtype='float16', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5c53405c45b63972cf6b8f355c048412(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 192], dtype='float16'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6f3ca6dd284f49f091946e9745769682(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5c53405c45b63972cf6b8f355c048412
    def get_inputs(self):
        return [
            paddle.uniform([4, 64, 192], dtype='float16', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9260a9eb5d2a9a519b745defa3dc7af6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 26, 512], dtype='float16'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fd6e8508a84f9776f43334831e43522c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9260a9eb5d2a9a519b745defa3dc7af6
    def get_inputs(self):
        return [
            paddle.uniform([1, 26, 512], dtype='float16', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d62776885a616123801c290bc97a8b3c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 100, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_57e3eb3702c2487d416f73b7b4d3bce1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d62776885a616123801c290bc97a8b3c
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1a33f016f927643dc5e246f0362c3b4c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 144], dtype='float32'),
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            paddle.static.InputSpec(shape=[144], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4d36d4dc0ec21d8d742f023718de1f33(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1a33f016f927643dc5e246f0362c3b4c
    def get_inputs(self):
        return [
            paddle.uniform([4, 256, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_fc2a09014f97164837a99be3577b88b1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 784, 192], dtype='float16'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9df88732b0ff117196377447cfff70d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc2a09014f97164837a99be3577b88b1
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 192], dtype='float16', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_948579d48ed42627af97258c6d530789(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 240], dtype='float16'),
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            paddle.static.InputSpec(shape=[240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e422825340eb44d98f32080678ca09b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_948579d48ed42627af97258c6d530789
    def get_inputs(self):
        return [
            paddle.uniform([4, 16, 240], dtype='float16', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_88a9245c5f4bdb9919540edc0f87bec7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            paddle.static.InputSpec(shape=[240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7e27956e17dc9690f4c1f87fca4dfa06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_88a9245c5f4bdb9919540edc0f87bec7
    def get_inputs(self):
        return [
            paddle.uniform([4, 16, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b43b09367eb75a46443762c359b6dd41(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 784, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cea316924377eb4bf29f1ee4d339e641(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b43b09367eb75a46443762c359b6dd41
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_61b9819214790b354856d066b5254fda(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 9216, 128], dtype='float16'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4ccf819d1588d65277ea81156f54acf1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_61b9819214790b354856d066b5254fda
    def get_inputs(self):
        return [
            paddle.uniform([1, 9216, 128], dtype='float16', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7cdfab677827819afac50a8c90c94fbe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3136, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            paddle.static.InputSpec(shape=[32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f26829d6ef511f96cbaa7c1e688f0925(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cdfab677827819afac50a8c90c94fbe
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_44355986b76d9d1ec744346261b82e00(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 160], dtype='float16'),
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            paddle.static.InputSpec(shape=[160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4c2b3b38a7a0a0901c74d1ed0acdaf00(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_44355986b76d9d1ec744346261b82e00
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 160], dtype='float16', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_22ac592ec353803653cc8d60859390b6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 96], dtype='float16'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f02a41e7cc1b5ed1381771a556697d9e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_22ac592ec353803653cc8d60859390b6
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 96], dtype='float16', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dd6c6c67f541085287a858cae58dd66e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3291398882865906, 0.2799290418624878, 0.24255892634391785, 0.022600550204515457, 0.17959235608577728, 0.24647726118564606, 0.38651296496391296, 0.27117931842803955, 0.14117272198200226, 0.15848414599895477, 0.34334030747413635, 0.35698118805885315, 0.36005890369415283, 0.18805812299251556, 0.48915889859199524, 0.36336442828178406, 0.4558719992637634, 0.2793504297733307, 0.24825513362884521, 0.47472038865089417, 0.09363708645105362, 0.45508119463920593, 0.011177970096468925, 0.39462152123451233], dtype='float32').reshape([24]),
            paddle.to_tensor([0.29301589727401733, 0.2085038423538208, 0.03606534004211426, 0.35489121079444885, 0.11573629081249237, 0.10927984118461609, 0.43879151344299316, 0.12053073197603226, 0.020735815167427063, 0.4717411994934082, 0.23745466768741608, 0.35614898800849915, 0.37861767411231995, 0.4362887740135193, 0.35406142473220825, 0.03167526796460152, 0.18890176713466644, 0.49303510785102844, 0.4234236478805542, 0.14414286613464355, 0.43344712257385254, 0.12832221388816833, 0.029874257743358612, 0.20786912739276886], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 24], dtype='float16'),
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            paddle.static.InputSpec(shape=[24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d0d6df997384ba699ed9a66a18ce65f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3210602402687073, 0.1652502715587616, 0.38910743594169617, 0.0852208137512207, 0.1875229775905609, 0.06697116792201996, 0.40072891116142273, 0.2111017405986786, 0.288394957780838, 0.44486507773399353, 0.47676968574523926, 0.4502319395542145, 0.3937016427516937, 0.20148120820522308, 0.4022679626941681, 0.005294621456414461, 0.4023727774620056, 0.0890527069568634, 0.2519589364528656, 0.20851051807403564, 0.28808578848838806, 0.4886685609817505, 0.08577290177345276, 0.4194251298904419], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4824790358543396, 0.28178420662879944, 0.11231352388858795, 0.05267416685819626, 0.2506307363510132, 0.3153255879878998, 0.1649695485830307, 0.2177359163761139, 0.40498632192611694, 0.008085126988589764, 0.4202231466770172, 0.47174975275993347, 0.12288246303796768, 0.4858485162258148, 0.06832252442836761, 0.17681068181991577, 0.3441885709762573, 0.2808915376663208, 0.32924842834472656, 0.0750841498374939, 0.44072580337524414, 0.11699662357568741, 0.4559996426105499, 0.11988300830125809], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5966d97bb4ebaea9e5998cfcfc3ce794(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 512], dtype='float16'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0d31cd484019bea74a26a6f98aadb1cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5966d97bb4ebaea9e5998cfcfc3ce794
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 512], dtype='float16', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_75e7e64732c7c6ccf15cd528dfd4fef9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3136, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_496d1e80f8dfe3ae82879f7cc1bbc71a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75e7e64732c7c6ccf15cd528dfd4fef9
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_76ab68a4f64cd208aaed96067e47bd84(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 784, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8712000af22202a45e6527cc6dcd6ee7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_76ab68a4f64cd208aaed96067e47bd84
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9fe5a14c8756c432c3c85131da9acc90(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3136, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fcba6a4ce9bb6abcd449b03d3fd4981a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9fe5a14c8756c432c3c85131da9acc90
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_945573908e2e1a35f58f5888b59e049e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 784, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3193f6791fed5ac6756d31b3e6d3de04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_945573908e2e1a35f58f5888b59e049e
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dff7b76d5aa149b5deab8db88e4d907a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4495331943035126, 0.2390548437833786, 0.19318528473377228, 0.132700577378273, 0.0465238057076931, 0.3378826975822449, 0.33943110704421997, 0.42073583602905273, 0.38162028789520264, 0.39442095160484314, 0.21441227197647095, 0.009111440740525723, 0.10047465562820435, 0.13634875416755676, 0.2825288772583008, 0.13066412508487701, 0.4245243966579437, 0.3586450517177582, 0.21571151912212372, 0.2083481401205063, 0.29532063007354736, 0.3010399043560028, 0.03234931826591492, 0.4178794324398041], dtype='float32').reshape([24]),
            paddle.to_tensor([0.12118066847324371, 0.4409113824367523, 0.4618769586086273, 0.13432025909423828, 0.02096180059015751, 0.1489228904247284, 0.4852922558784485, 0.3460778295993805, 0.17764738202095032, 0.3244779407978058, 0.37616482377052307, 0.3215655982494354, 0.06428605318069458, 0.23318053781986237, 0.3563835918903351, 0.11045725643634796, 0.3157748878002167, 0.4086260497570038, 0.007421903777867556, 0.26640671491622925, 0.13232295215129852, 0.27705997228622437, 0.3172267973423004, 0.40885913372039795], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3d62404dbd033bdc163e060c3ce9a389(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2304, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e3623517c364a51de478dd9fb247070a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3d62404dbd033bdc163e060c3ce9a389
    def get_inputs(self):
        return [
            paddle.uniform([1, 2304, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ca64d98122be9855566edd7ce39d6f38(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 100, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_13a6f3387658e413f005155897945851(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca64d98122be9855566edd7ce39d6f38
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3be0823de8a21cc4b10112bb2054c5b8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_937f99ba5d70cd6325f7d53e0a367306
    def get_inputs(self):
        return [
            paddle.uniform([1, 25, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0fe43c5d9a95ae092b1ad71342e03a1e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.28441187739372253, 0.10656187683343887, 0.21883845329284668, 0.3915109932422638, 0.2528392970561981, 0.06576692312955856, 0.05956346169114113, 0.03799008950591087, 0.4183785021305084, 0.22229468822479248, 0.2085215449333191, 0.23389792442321777, 0.3521454334259033, 0.14446155726909637, 0.3218226730823517, 0.047325752675533295, 0.26577284932136536, 0.3744446635246277, 0.19749882817268372, 0.3558489978313446, 0.2746719717979431, 0.38763442635536194, 0.11997835338115692, 0.028146913275122643], dtype='float32').reshape([24]),
            paddle.to_tensor([0.22275647521018982, 0.12340947240591049, 0.02118116430938244, 0.411749005317688, 0.4021325409412384, 0.399991512298584, 0.3278912305831909, 0.12510359287261963, 0.18763622641563416, 0.29445427656173706, 0.3866947293281555, 0.3403390347957611, 0.41840213537216187, 0.13720844686031342, 0.12938237190246582, 0.32747989892959595, 0.3018067181110382, 0.4771754741668701, 0.4534560441970825, 0.4147586226463318, 0.12004035711288452, 0.3550736606121063, 0.17861953377723694, 0.38331589102745056], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_188986d6741f24baaf0a3d19b5e200bf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 256], dtype='float16'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_75ecd1c27ada6a8bd84b094da74a95eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_188986d6741f24baaf0a3d19b5e200bf
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 256], dtype='float16', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c975aced4862620259d590acb0b59e57(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 144], dtype='float16'),
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            paddle.static.InputSpec(shape=[144], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c33a486797bc534f7a35da0ef38d140e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c975aced4862620259d590acb0b59e57
    def get_inputs(self):
        return [
            paddle.uniform([4, 256, 144], dtype='float16', min=0, max=0.5),
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4e4a9e383b7df49d051e6b8c5b585349(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2304, 256], dtype='float16'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7fdc4504ea4caf3421f698ed87c2d1ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4e4a9e383b7df49d051e6b8c5b585349
    def get_inputs(self):
        return [
            paddle.uniform([1, 2304, 256], dtype='float16', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_80d1bff3499b380e40b9d4e8d64755b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 1024], dtype='float16'),
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2e97dae47081523a290116c1504e921a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_80d1bff3499b380e40b9d4e8d64755b5
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1024], dtype='float16', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4f66f6d8a77466a94167b5fb5b0bd3d8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_85ab3d82e1bf2867806338266e688366(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f66f6d8a77466a94167b5fb5b0bd3d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0bd9f48ad5a1ba28632691345c23ee5b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 128], dtype='float16'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ae08b8cc06bbe27ee73e5fd643d26099(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0bd9f48ad5a1ba28632691345c23ee5b
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 128], dtype='float16', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9e095bc50434233938a51734fc6647af(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 576, 1024], dtype='float16'),
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d746ef79142ade9d6b68f5ae5d9f7a35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9e095bc50434233938a51734fc6647af
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1024], dtype='float16', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_fab0a321a75b6de4e9a12bb27282f84f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_95f03321ab43afb15b86a0ae8d7198e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fab0a321a75b6de4e9a12bb27282f84f
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_67fef6d9e76154297e791695a35775e6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 196, 320], dtype='float16'),
            paddle.static.InputSpec(shape=[320], dtype='float32'),
            paddle.static.InputSpec(shape=[320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fffc9730be6594b4c692ece84af3a64b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_67fef6d9e76154297e791695a35775e6
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 320], dtype='float16', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
            paddle.uniform([320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8c271728cf0f315e87b62e614a45b46d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 144], dtype='float16'),
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            paddle.static.InputSpec(shape=[144], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_07a562da43fb637acebe7fc154f96556(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8c271728cf0f315e87b62e614a45b46d
    def get_inputs(self):
        return [
            paddle.uniform([4, 256, 144], dtype='float16', min=0, max=0.5),
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_032235d3bbb8d048478a35ac82cc344b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 64], dtype='float16'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4f49231131dc1acd6075854902b39ef9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_032235d3bbb8d048478a35ac82cc344b
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_eee3955791de9461dd0ba26a54ba6149(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 192], dtype='float16'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_45a59321492d9595e46dc246722499b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eee3955791de9461dd0ba26a54ba6149
    def get_inputs(self):
        return [
            paddle.uniform([4, 64, 192], dtype='float16', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0a38eac31623f166b869c7d0cf3df78d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3136, 32], dtype='float16'),
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            paddle.static.InputSpec(shape=[32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_235673f38fb88d0745508d8e4f3c4591(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a38eac31623f166b869c7d0cf3df78d
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4a3e458f3bc9adb63425d460898264eb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 196, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            paddle.static.InputSpec(shape=[160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ebaaa18c887d9bbafbf4595bc23a15ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a3e458f3bc9adb63425d460898264eb
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_10b17e7abb55075acc6227a5ec4e2615(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.0033353352919220924, 0.47618368268013, 0.197532057762146, 0.4897301495075226, 0.31167104840278625, 0.34787553548812866, 0.1985020786523819, 0.05122765526175499, 0.19276872277259827, 0.3168918192386627, 0.3845776617527008, 0.2888976037502289, 0.29651880264282227, 0.0008442751714028418, 0.457075834274292, 0.3543223440647125, 0.3077501654624939, 0.4268065392971039, 0.3943912982940674, 0.2577085494995117, 0.38496899604797363, 0.29549700021743774, 0.1865539848804474, 0.3796634376049042], dtype='float32').reshape([24]),
            paddle.to_tensor([0.180519700050354, 0.10383564233779907, 0.2558480501174927, 0.0013169227167963982, 0.11466995626688004, 0.14514382183551788, 0.2604365944862366, 0.4266960024833679, 0.41506364941596985, 0.2930034399032593, 0.2402764856815338, 0.06055678799748421, 0.07846041768789291, 0.43997666239738464, 0.3923080265522003, 0.4841328263282776, 0.22388023138046265, 0.2722157835960388, 0.22711409628391266, 0.3444993495941162, 0.3947950005531311, 0.4641958177089691, 0.2309851050376892, 0.33825892210006714], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_fd76dde7897eb5024349a84835b9a895(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fef4c9111d47c30eaa0cb1559aea643b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fd76dde7897eb5024349a84835b9a895
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e65c2b3383197e131b1ab0ffd13efb05(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3136, 64], dtype='float16'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_682ef60e5d0343b62f44125fccf9c12b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e65c2b3383197e131b1ab0ffd13efb05
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_790801b2b62f9a09f21439184c0da35e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1973121017217636, 0.17461606860160828, 0.3475719392299652, 0.082064189016819, 0.2503150403499603, 0.4508446753025055, 0.2503358721733093, 0.3479492962360382, 0.10011161118745804, 0.1858212947845459, 0.43693807721138, 0.29668566584587097, 0.4360484182834625, 0.1190735474228859, 0.12189847230911255, 0.012037292122840881, 0.28188276290893555, 0.30776211619377136, 0.20944051444530487, 0.2792387008666992, 0.23679988086223602, 0.08849692344665527, 0.25976940989494324, 0.4305691421031952], dtype='float32').reshape([24]),
            paddle.to_tensor([0.1640278697013855, 0.1621425896883011, 0.38214758038520813, 0.19742800295352936, 0.34483060240745544, 0.32480672001838684, 0.3432570993900299, 0.3183292746543884, 0.18073347210884094, 0.3267594873905182, 0.2084232121706009, 0.019398842006921768, 0.3618757724761963, 0.10920897871255875, 0.49014297127723694, 0.16398555040359497, 0.4082464873790741, 0.28947269916534424, 0.4446847140789032, 0.0029546874575316906, 0.2877699136734009, 0.29461097717285156, 0.02090122550725937, 0.024387184530496597], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0c43932f3ae55dd8d54965e0cc3122b0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3136, 96], dtype='float16'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c519010e36e7eb88cb2cab9e798815a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c43932f3ae55dd8d54965e0cc3122b0
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 96], dtype='float16', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c347918cf04581828d0ea64cc1ab9d27(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.43621549010276794, 0.1463736891746521, 0.2666698694229126, 0.44631755352020264, 0.2960793375968933, 0.22160761058330536, 0.487611323595047, 0.17121650278568268, 0.39789727330207825, 0.3016062080860138, 0.052289608865976334, 0.12913253903388977, 0.11650820076465607, 0.42367085814476013, 0.09788241982460022, 0.4506460726261139, 0.07656759768724442, 0.40071722865104675, 0.15001536905765533, 0.2118317037820816, 0.3744438588619232, 0.33575284481048584, 0.47277015447616577, 0.3351629376411438], dtype='float32').reshape([24]),
            paddle.to_tensor([0.3291267454624176, 0.02947189286351204, 0.08261172473430634, 0.01127054076641798, 0.2825303077697754, 0.2548500895500183, 0.2174159586429596, 0.23359008133411407, 0.44517913460731506, 0.3823173940181732, 0.30236542224884033, 0.29040539264678955, 0.3124516010284424, 0.03885800018906593, 0.31476736068725586, 0.1154744103550911, 0.11883702874183655, 0.25659430027008057, 0.24349141120910645, 0.08828964084386826, 0.21199722588062286, 0.3730752766132355, 0.48400285840034485, 0.09242087602615356], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6b88fdbcb6d4d141e1c1e8040abc4d42(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 784, 64], dtype='float16'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f34106299165f600c5eb717ebd3d3f6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b88fdbcb6d4d141e1c1e8040abc4d42
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d0d0c8960e5ba257281bc058e06cb5ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3d0c2e6b91a8b89baaf2c0128d0c3563(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d0d0c8960e5ba257281bc058e06cb5ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4d172ad2aaa384bff102069d7010dace(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 100, 128], dtype='float16'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_94623ce016b9acd19edbdb3dec020faf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4d172ad2aaa384bff102069d7010dace
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 128], dtype='float16', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1cb3f21028b66566615e879ed67fa628(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3136, 96], dtype='float16'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3a3142ac9878643354485e1334978d0b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1cb3f21028b66566615e879ed67fa628
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 96], dtype='float16', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b44fdbf136291acd96b634ff68cdefe1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 784, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_adb5ee1637c98d8d9930d2a888a2f11c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b44fdbf136291acd96b634ff68cdefe1
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_351d71e9591bcb7783c07e4f4f380d17(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3136, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1ffca9908b8f4b3f20889fccbd9af31b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_351d71e9591bcb7783c07e4f4f380d17
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1dab2a3c19dc0534e9ec6db315e8a9d9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 784, 128], dtype='float16'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aa3acdbb12e2d4425489391767df80df(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1dab2a3c19dc0534e9ec6db315e8a9d9
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 128], dtype='float16', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6a868288f22e057105c6a7169fd949a5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 2048], dtype='float16'),
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4292c669d339aa703f96afdd43801532(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a868288f22e057105c6a7169fd949a5
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 2048], dtype='float16', min=0, max=0.5),
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_add1c87818b8747db67c0ca3be9aabfb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3136, 32], dtype='float16'),
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            paddle.static.InputSpec(shape=[32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_57196d7785f2addc0fe02eefdb1f3d66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_add1c87818b8747db67c0ca3be9aabfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_50bb534aaf6a4c8cd698cdf1890ba8c1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 200, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e84a52aaddd9df4ef2aaf580dfd567ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50bb534aaf6a4c8cd698cdf1890ba8c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 200, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cf958a19bb3700bfa988f6cbf1864071(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.46114641427993774, 0.07426156848669052, 0.4666295647621155, 0.35491183400154114, 0.030394725501537323, 0.2541913092136383, 0.2923702299594879, 0.3210584223270416, 0.1203196793794632, 0.4155246317386627, 0.02784230001270771, 0.1949152797460556, 0.3059973418712616, 0.4791969656944275, 0.07841590791940689, 0.477334201335907, 0.2740705907344818, 0.02989884652197361, 0.39064106345176697, 0.34095969796180725, 0.16401034593582153, 0.08200228959321976, 0.4023308753967285, 0.4756181538105011], dtype='float32').reshape([24]),
            paddle.to_tensor([0.2856740355491638, 0.36230796575546265, 0.4513092339038849, 0.3251866400241852, 0.39540255069732666, 0.32235607504844666, 0.43886807560920715, 0.3593394160270691, 0.3579729497432709, 0.4287498891353607, 0.34552475810050964, 0.09864765405654907, 0.03783891722559929, 0.4013940989971161, 0.007578900083899498, 0.25015920400619507, 0.20387496054172516, 0.12132887542247772, 0.07793361693620682, 0.3131977617740631, 0.06049670651555061, 0.2516135573387146, 0.21102569997310638, 0.24982479214668274], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ff6c86c1e8a2353c0da603ba5beb31fd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 768], dtype='float16'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5b26a731b12d9df778a7b91c70b872c4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff6c86c1e8a2353c0da603ba5beb31fd
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 768], dtype='float16', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_efe9e779d1a71a6021c7417f96fd76da(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3136, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a596b640ed8e661efde224c0ee93c7fa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_efe9e779d1a71a6021c7417f96fd76da
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_fe1c1cf1fc694579fa16e824e4cfc7f3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 240], dtype='float32'),
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            paddle.static.InputSpec(shape=[240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_17269d35ff18adb892994be171a7a28e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fe1c1cf1fc694579fa16e824e4cfc7f3
    def get_inputs(self):
        return [
            paddle.uniform([4, 16, 240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0622361d60cebc0615c5cbf9e497c75e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.11771857738494873, 0.32901522517204285, 0.43469908833503723, 0.0518282987177372, 0.19801312685012817, 0.3224130868911743, 0.19311955571174622, 0.03456486389040947, 0.3933168649673462, 0.16783148050308228, 0.4415842890739441, 0.2783028781414032, 0.292898952960968, 0.03837459534406662, 0.23894594609737396, 0.07346932590007782, 0.0796826034784317, 0.03692133352160454, 0.15529470145702362, 0.11405026167631149, 0.3957921266555786, 0.29471057653427124, 0.17031031847000122, 0.1990957111120224], dtype='float32').reshape([24]),
            paddle.to_tensor([0.25832879543304443, 0.4931113123893738, 0.18722683191299438, 0.022597914561629295, 0.2763192355632782, 0.0614265613257885, 0.04740649089217186, 0.08078627288341522, 0.2758050560951233, 0.09211423993110657, 0.4565351903438568, 0.1683322936296463, 0.0028722784481942654, 0.32548317313194275, 0.14181718230247498, 0.437023401260376, 0.3877274692058563, 0.2050875723361969, 0.2360878884792328, 0.26332393288612366, 0.37715286016464233, 0.2514707148075104, 0.007052071392536163, 0.04978630691766739], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5d90f9210ad2f5e7f4099981446e8fce(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 96], dtype='float32'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ee317ff3499c59b5ee90605b121ba48d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5d90f9210ad2f5e7f4099981446e8fce
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
            paddle.uniform([96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_982fc7dcd561744b5d7eab3bad06c306(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3456335663795471, 0.3952365219593048, 0.24798202514648438, 0.2325054109096527, 0.17840445041656494, 0.06605525314807892, 0.172715961933136, 0.2820567786693573, 0.24686597287654877, 0.3920708894729614, 0.21232019364833832, 0.29470473527908325, 0.27448299527168274, 0.3338477313518524, 0.42042791843414307, 0.1684396117925644, 0.12548065185546875, 0.43801331520080566, 0.3677135407924652, 0.31036290526390076, 0.2199028730392456, 0.40418750047683716, 0.4175221621990204, 0.3173573315143585], dtype='float32').reshape([24]),
            paddle.to_tensor([0.3102054297924042, 0.3570568561553955, 0.43766143918037415, 0.14387300610542297, 0.3451296091079712, 0.07791547477245331, 0.15333037078380585, 0.03412756323814392, 0.3246271014213562, 0.3086414337158203, 0.05776789039373398, 0.43148788809776306, 0.2942861020565033, 0.07223251461982727, 0.1622968465089798, 0.18738383054733276, 0.07832445949316025, 0.4212600886821747, 0.0860716924071312, 0.17707887291908264, 0.04824107140302658, 0.12659397721290588, 0.4985545873641968, 0.44871801137924194], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7df63b98258e2d4d48b86744193cc10f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1985253095626831, 0.45667925477027893, 0.3018195629119873, 0.19589181244373322, 0.25143489241600037, 0.07409659773111343, 0.21123021841049194, 0.17175306379795074, 0.4288332164287567, 0.23377065360546112, 0.11329129338264465, 0.15842397511005402, 0.1774071305990219, 0.10280224680900574, 0.07773150503635406, 0.2935408353805542, 0.24118661880493164, 0.4110824763774872, 0.2182559370994568, 0.3329613208770752, 0.4691450595855713, 0.3909914791584015, 0.26234397292137146, 0.06722228229045868], dtype='float32').reshape([24]),
            paddle.to_tensor([0.049227021634578705, 0.4759061932563782, 0.2724014222621918, 0.3841787874698639, 0.16340342164039612, 0.30459460616111755, 0.11845973134040833, 0.4246612787246704, 0.42658576369285583, 0.4780181646347046, 0.27172282338142395, 0.44673728942871094, 0.00405121361836791, 0.45625001192092896, 0.14672964811325073, 0.30459949374198914, 0.41687309741973877, 0.39910057187080383, 0.03664478659629822, 0.13953712582588196, 0.31201842427253723, 0.24736545979976654, 0.19617009162902832, 0.142278254032135], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_72e46c64704abb74b1cba7fd943c1f73(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 144], dtype='float32'),
            paddle.static.InputSpec(shape=[144], dtype='float32'),
            paddle.static.InputSpec(shape=[144], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ae05866749ee1d417b14a639eb81fb77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72e46c64704abb74b1cba7fd943c1f73
    def get_inputs(self):
        return [
            paddle.uniform([4, 256, 144], dtype='float32', min=0, max=0.5),
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
            paddle.uniform([144], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bfea8cd27f02dc274431dd5acff606f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.055094800889492035, 0.37313616275787354, 0.16536080837249756, 0.4353681206703186, 0.1005638912320137, 0.09350959211587906, 0.001203414285555482, 0.4297681748867035, 0.4439481794834137, 0.2922578454017639, 0.12797638773918152, 0.3204362392425537, 0.17904186248779297, 0.4908580482006073, 0.36302876472473145, 0.39271265268325806, 0.42487093806266785, 0.13703276216983795, 0.4736252427101135, 0.3023057281970978, 0.4670041501522064, 0.04072602838277817, 0.08456756919622421, 0.19652710855007172], dtype='float32').reshape([24]),
            paddle.to_tensor([0.14108552038669586, 0.32060131430625916, 0.1491626501083374, 0.48202118277549744, 0.4138736128807068, 0.009161756373941898, 0.10162550956010818, 0.15046727657318115, 0.11680741608142853, 0.41517016291618347, 0.16995950043201447, 0.31623271107673645, 0.28821897506713867, 0.2355450838804245, 0.16864117980003357, 0.32969239354133606, 0.26796066761016846, 0.3035222291946411, 0.32983559370040894, 0.31064915657043457, 0.35407254099845886, 0.18587519228458405, 0.1315876543521881, 0.03467690572142601], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5fae025ad456e53ffd0c900e721931d6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 768], dtype='float16'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ad7113ea1d2945d13903f87b5e0a68dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5fae025ad456e53ffd0c900e721931d6
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 768], dtype='float16', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_fb06dea1c2e2abee1dc3b14d3edf6867(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 32], dtype='float16'),
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            paddle.static.InputSpec(shape=[32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_47fd55605b5e8a82203ce3b1c2c6e811(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fb06dea1c2e2abee1dc3b14d3edf6867
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 32], dtype='float16', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bdb4419b437c58a28eb95c71abc99fe1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.09823548048734665, 0.47838714718818665, 0.11724691838026047, 0.10808845609426498, 0.39418816566467285, 0.23837871849536896, 0.09598349779844284, 0.3074903190135956, 0.13679452240467072, 0.35758092999458313, 0.20383164286613464, 0.31422656774520874, 0.11314160376787186, 0.293411523103714, 0.1421324461698532, 0.4042380154132843, 0.44725847244262695, 0.056149087846279144, 0.15730471909046173, 0.3959125578403473, 0.3254162073135376, 0.21248623728752136, 0.19642086327075958, 0.1760583519935608], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4864860773086548, 0.25115394592285156, 0.043536558747291565, 0.07683107256889343, 0.1662353277206421, 0.14700046181678772, 0.13580895960330963, 0.26541194319725037, 0.18187929689884186, 0.2111029326915741, 0.41531258821487427, 0.42303693294525146, 0.2888645529747009, 0.18637269735336304, 0.2386856973171234, 0.033667322248220444, 0.4948573708534241, 0.340821772813797, 0.2237222045660019, 0.3811515271663666, 0.36648210883140564, 0.007210808806121349, 0.3156265318393707, 0.11826980113983154], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_61de41da15b43820ef267d3e9b8699c0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.4769151508808136, 0.008158857002854347, 0.4917914867401123, 0.49117112159729004, 0.06479177623987198, 0.393295556306839, 0.2697933316230774, 0.4985416829586029, 0.46857184171676636, 0.29198071360588074, 0.30213606357574463, 0.2349623590707779, 0.42220428586006165, 0.4138154983520508, 0.2541247010231018, 0.2591824531555176, 0.0999550074338913, 0.01961411163210869, 0.33799442648887634, 0.30161264538764954, 0.1246507465839386, 0.43829742074012756, 0.09040495753288269, 0.10719519853591919], dtype='float32').reshape([24]),
            paddle.to_tensor([0.2391877919435501, 0.3737815022468567, 0.1289060264825821, 0.08537900447845459, 0.29051750898361206, 0.35171228647232056, 0.03218435123562813, 0.18995121121406555, 0.27258914709091187, 0.010567101649940014, 0.44685351848602295, 0.44253531098365784, 0.22592973709106445, 0.1847885102033615, 0.019909050315618515, 0.11204899102449417, 0.4909094572067261, 0.13680391013622284, 0.4597402513027191, 0.12017976492643356, 0.33845311403274536, 0.17003022134304047, 0.4787217676639557, 0.29335784912109375], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ec6be145baae9c695371d61466ac6cd1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.07141704112291336, 0.2793707251548767, 0.2827867567539215, 0.04335314780473709, 0.10892792791128159, 0.39086300134658813, 0.30681997537612915, 0.1558820754289627, 0.26720330119132996, 0.3384144604206085, 0.4373806118965149, 0.19455914199352264, 0.054969143122434616, 0.08056273311376572, 0.4426751434803009, 0.14130644500255585, 0.40365806221961975, 0.012007717043161392, 0.3433373272418976, 0.11088936775922775, 0.309093713760376, 0.023922933265566826, 0.22095416486263275, 0.3902665674686432], dtype='float32').reshape([24]),
            paddle.to_tensor([0.11661326140165329, 0.1886414736509323, 0.11401382088661194, 0.45248258113861084, 0.19017833471298218, 0.4890216290950775, 0.457439124584198, 0.17805473506450653, 0.03263169154524803, 0.3352476954460144, 0.2398412972688675, 0.2101982682943344, 0.37956905364990234, 0.10066847503185272, 0.46803611516952515, 0.34879422187805176, 0.06290063261985779, 0.10800700634717941, 0.4783109724521637, 0.1266653835773468, 0.41118860244750977, 0.2866671681404114, 0.09591562300920486, 0.01160537451505661], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a080a38ac7b4d267e648290f6711c8c9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 784, 64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3d1c9579c3a72604cd34400c30e91027(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a080a38ac7b4d267e648290f6711c8c9
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3a90f88fc85c5c152e21635687518102(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3905773460865021, 0.47255751490592957, 0.06827348470687866, 0.3897691071033478, 0.2544814944267273, 0.35140788555145264, 0.45024076104164124, 0.39821434020996094, 0.184661865234375, 0.3832377791404724, 0.3933756351470947, 0.17552097141742706, 0.1412210464477539, 0.03268500789999962, 0.1432354897260666, 0.4405647814273834, 0.08373763412237167, 0.31229478120803833, 0.27684828639030457, 0.11988204717636108, 0.0763169378042221, 0.04725104942917824, 0.4849715232849121, 0.05370239168405533], dtype='float32').reshape([24]),
            paddle.to_tensor([0.18731768429279327, 0.06912380456924438, 0.4691333770751953, 0.1881413459777832, 0.1494733989238739, 0.3702240586280823, 0.3775033950805664, 0.0677185207605362, 0.007905966602265835, 0.3199048936367035, 0.08473286777734756, 0.31208425760269165, 0.4825545847415924, 0.4732895493507385, 0.38520902395248413, 0.4892708361148834, 0.48557931184768677, 0.3618151843547821, 0.20597906410694122, 0.4474101960659027, 0.12635764479637146, 0.351006418466568, 0.24386991560459137, 0.22308635711669922], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ed11e7e2542cb43d4b3a4ef724efe0b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 512], dtype='float16'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2e3a1b114d9633874735522cadceb8d1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed11e7e2542cb43d4b3a4ef724efe0b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 512], dtype='float16', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7d0f4d671aa759f47741f5473b7afe89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.18176643550395966, 0.293670117855072, 0.2624484896659851, 0.4361293911933899, 0.07585692405700684, 0.06858580559492111, 0.12120606005191803, 0.11128450930118561, 0.05744192376732826, 0.40900009870529175, 0.2146269679069519, 0.27555376291275024, 0.4207175374031067, 0.36192575097084045, 0.2787052392959595, 0.26047801971435547, 0.11711111664772034, 0.15214577317237854, 0.08547452837228775, 0.4935280978679657, 0.14669598639011383, 0.011698123998939991, 0.33363938331604004, 0.2845083177089691], dtype='float32').reshape([24]),
            paddle.to_tensor([0.44516196846961975, 0.3163892924785614, 0.23835355043411255, 0.27797743678092957, 0.1787179857492447, 0.2777225971221924, 0.3618502616882324, 0.41591179370880127, 0.38624534010887146, 0.1391330510377884, 0.0550733357667923, 0.49920082092285156, 0.47869768738746643, 0.4868645668029785, 0.11592158675193787, 0.34415993094444275, 0.03951523080468178, 0.09975859522819519, 0.23246128857135773, 0.32906413078308105, 0.3633597493171692, 0.38302287459373474, 0.32489871978759766, 0.44802775979042053], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_96d1ee1fb3ae0e22e962da3414e565ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eb39d5dfbf86af5f748b3665d381dbae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_96d1ee1fb3ae0e22e962da3414e565ee
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ac73b851feaab6eec671131b794cc11e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 196, 160], dtype='float32'),
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            paddle.static.InputSpec(shape=[160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5b96ed047c744ebcd3e7e4433b29887b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac73b851feaab6eec671131b794cc11e
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2a733413f78e70f995a2e22b54ace68d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2739688456058502, 0.35141420364379883, 0.15971989929676056, 0.31889134645462036, 0.24460992217063904, 0.27155646681785583, 0.31388652324676514, 0.3078359067440033, 0.16364701092243195, 0.45672547817230225, 0.12903429567813873, 0.3904195725917816, 0.42172160744667053, 0.4748568534851074, 0.19573597609996796, 0.26933568716049194, 0.1165073961019516, 0.19645756483078003, 0.4910942614078522, 0.17037662863731384, 0.34213700890541077, 0.2966890037059784, 0.3765350878238678, 0.02019944041967392], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4578011929988861, 0.21994835138320923, 0.14889274537563324, 0.4537711441516876, 0.22798968851566315, 0.16223593056201935, 0.2246052324771881, 0.47284629940986633, 0.27551379799842834, 0.2863596975803375, 0.2144504189491272, 0.3153936564922333, 0.4178737998008728, 0.16110862791538239, 0.3688730001449585, 0.14729738235473633, 0.17677165567874908, 0.17660313844680786, 0.46386268734931946, 0.13384221494197845, 0.3307311534881592, 0.24721048772335052, 0.47821661829948425, 0.33737441897392273], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_86c40ede5ca74b70f3666f068e562bfd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 50, 256], dtype='float16'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2e519f2a063f62d63f5de578ecfa07ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86c40ede5ca74b70f3666f068e562bfd
    def get_inputs(self):
        return [
            paddle.uniform([1, 50, 256], dtype='float16', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_71ede6aec39c04e66408a2c198c162e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.12926340103149414, 0.3746367394924164, 0.0966346338391304, 0.48589155077934265, 0.005977271124720573, 0.053024400025606155, 0.16533605754375458, 0.26235973834991455, 0.3109891712665558, 0.026151619851589203, 0.43614351749420166, 0.18745367228984833, 0.23843912780284882, 0.4210590422153473, 0.08469489961862564, 0.43124309182167053, 0.4239633083343506, 0.3147006630897522, 0.351948618888855, 0.40203309059143066, 0.12918326258659363, 0.2036353200674057, 0.19623592495918274, 0.3300062417984009], dtype='float32').reshape([24]),
            paddle.to_tensor([0.028716526925563812, 0.052008677273988724, 0.25178372859954834, 0.4622236490249634, 0.3794941008090973, 0.44839930534362793, 0.019807705655694008, 0.3140290379524231, 0.005965252872556448, 0.13115064799785614, 0.42736563086509705, 0.2064211219549179, 0.4952631890773773, 0.29872438311576843, 0.2807587683200836, 0.17627611756324768, 0.19078217446804047, 0.15894819796085358, 0.48415806889533997, 0.1591634303331375, 0.32323431968688965, 0.4438753128051758, 0.3799758851528168, 0.3279326558113098], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_727d751a20cd38b7c6e1f9893adad5ab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_19f1610724e23fcfaf179c4660e639a9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_727d751a20cd38b7c6e1f9893adad5ab
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_10acd6a6a4eb37d6c09bab72eeddbd0d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4737900495529175, 0.016287220641970634, 0.4447653293609619, 0.24136938154697418, 0.3814895749092102, 0.12673434615135193, 0.38548627495765686, 0.2312469184398651, 0.49276092648506165, 0.09035108983516693, 0.47155842185020447, 0.17014248669147491, 0.2294987142086029, 0.39062392711639404, 0.47454673051834106, 0.053361523896455765, 0.43074747920036316, 0.37570759654045105, 0.043803874403238297, 0.1925482600927353, 0.2910689413547516, 0.34281936287879944, 0.48895928263664246, 0.46691572666168213], dtype='float32').reshape([24]),
            paddle.to_tensor([0.1545064002275467, 0.16438405215740204, 0.49940791726112366, 0.1299777328968048, 0.2455441802740097, 0.2921180725097656, 0.09486763924360275, 0.19775164127349854, 0.13376471400260925, 0.48334771394729614, 0.41669464111328125, 0.2740554213523865, 0.4707385301589966, 0.40687793493270874, 0.298538476228714, 0.34233173727989197, 0.06177215278148651, 0.03756299987435341, 0.3395310342311859, 0.41933736205101013, 0.37930285930633545, 0.038709212094545364, 0.3911648690700531, 0.35464227199554443], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_034b3071a2dd9cdd40d97b86df36fc31(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3136, 64], dtype='float16'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aa219f413293acee1079f66bab1df3f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_034b3071a2dd9cdd40d97b86df36fc31
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eed7811981a5f04679099fa1650bc1f6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.07557713240385056, 0.3558233082294464, 0.42243364453315735, 0.4306146800518036, 0.43811923265457153, 0.05486283451318741, 0.3844211995601654, 0.07491683214902878, 0.06300283223390579, 0.4082828164100647, 0.1770239770412445, 0.3924751579761505, 0.25515016913414, 0.4513915181159973, 0.4416095018386841, 0.40787339210510254, 0.43840399384498596, 0.37976089119911194, 0.32013317942619324, 0.32050690054893494, 0.2619212567806244, 0.10713458061218262, 0.2494690716266632, 0.4395127296447754], dtype='float32').reshape([24]),
            paddle.to_tensor([0.027584651485085487, 0.026302579790353775, 0.23662877082824707, 0.1326027512550354, 0.20365245640277863, 0.23568670451641083, 0.2173229604959488, 0.0010855309665203094, 0.41290244460105896, 0.43040579557418823, 0.44403910636901855, 0.3929780125617981, 0.08812742680311203, 0.17284369468688965, 0.29252997040748596, 0.4727805256843567, 0.3039507567882538, 0.10545778274536133, 0.3486534357070923, 0.3999795913696289, 0.04018956050276756, 0.44150444865226746, 0.07359124720096588, 0.4029289484024048], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_04c408b090113e96c36fa131db21fa8e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.08470150083303452, 0.11682290583848953, 0.14982552826404572, 0.11840085685253143, 0.28003719449043274, 0.14028280973434448, 0.29519957304000854, 0.31474918127059937, 0.24839916825294495, 0.041571516543626785, 0.18337000906467438, 0.35485830903053284, 0.18036144971847534, 0.32456502318382263, 0.1485738605260849, 0.03197994828224182, 0.42962315678596497, 0.3214710056781769, 0.23568417131900787, 0.3074137270450592, 0.4908376932144165, 0.2721588909626007, 0.4266371428966522, 0.036100633442401886], dtype='float32').reshape([24]),
            paddle.to_tensor([0.3304949998855591, 0.44384172558784485, 0.3347543478012085, 0.43953436613082886, 0.45144498348236084, 0.39297837018966675, 0.3075687885284424, 0.2458404004573822, 0.42095494270324707, 0.38859954476356506, 0.23835857212543488, 0.08090169727802277, 0.26767849922180176, 0.11635294556617737, 0.05923840031027794, 0.001141236163675785, 0.474603533744812, 0.15980187058448792, 0.29345011711120605, 0.3268604874610901, 0.19323277473449707, 0.35638538002967834, 0.07396213710308075, 0.1427922397851944], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8b214005277be47d692df3fa611b36ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3205685019493103, 0.3178185224533081, 0.2832992970943451, 0.42627134919166565, 0.18488532304763794, 0.2937871813774109, 0.1958288848400116, 0.24533402919769287, 0.46788474917411804, 0.387450635433197, 0.1728552281856537, 0.41208121180534363, 0.07871252298355103, 0.3964398503303528, 0.3614954948425293, 0.10986866056919098, 0.37517237663269043, 0.4539951682090759, 0.15348729491233826, 0.4768764078617096, 0.06612689048051834, 0.42901498079299927, 0.20638227462768555, 0.0009711519232951105], dtype='float32').reshape([24]),
            paddle.to_tensor([0.2281855195760727, 0.3718682825565338, 0.1964660882949829, 0.18728835880756378, 0.11395839601755142, 0.23615674674510956, 0.1815652996301651, 0.3390391767024994, 0.3278261721134186, 0.4212964177131653, 0.289099782705307, 0.3604222536087036, 0.4060812294483185, 0.28670623898506165, 0.38802000880241394, 0.08675705641508102, 0.4644566476345062, 0.01908094435930252, 0.050156120210886, 0.3207481801509857, 0.281728595495224, 0.051431991159915924, 0.31013697385787964, 0.48861098289489746], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0c1bc01a9019f47f2a80aabf0c65dc2b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2304, 512], dtype='float16'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_60519f6399dd37c86ed426b5ab6ec1ff(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c1bc01a9019f47f2a80aabf0c65dc2b
    def get_inputs(self):
        return [
            paddle.uniform([1, 2304, 512], dtype='float16', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0aa9e55029dab4942371cb2922fb3ab0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 2304, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9826b58bbc6b2b1baae545d8f6130a02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0aa9e55029dab4942371cb2922fb3ab0
    def get_inputs(self):
        return [
            paddle.uniform([1, 2304, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a9f05f8f53f824613f404a95fcdd2ba7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.14205768704414368, 0.4415822923183441, 0.23854197561740875, 0.3292657732963562, 0.14784906804561615, 0.018509870395064354, 0.12917637825012207, 0.29400134086608887, 0.23513856530189514, 0.25394970178604126, 0.15431971848011017, 0.3330668807029724, 0.12729501724243164, 0.04253244772553444, 0.05267944186925888, 0.009348760358989239, 0.06349314004182816, 0.3159849941730499, 0.41451936960220337, 0.19024299085140228, 0.2127906084060669, 0.31246519088745117, 0.07096941024065018, 0.14675413072109222], dtype='float32').reshape([24]),
            paddle.to_tensor([0.08113555610179901, 0.013244487345218658, 0.3367671072483063, 0.21360637247562408, 0.48038187623023987, 0.4228726029396057, 0.06821106374263763, 0.1842879056930542, 0.19122567772865295, 0.28619566559791565, 0.17702403664588928, 0.3620232045650482, 0.358939528465271, 0.2499011754989624, 0.4523940682411194, 0.30654534697532654, 0.3462390601634979, 0.11081027239561081, 0.057840969413518906, 0.4870637059211731, 0.03272996097803116, 0.3537355959415436, 0.49751219153404236, 0.20614276826381683], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_79422893ac9670c7400ec9a14471c3aa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 256], dtype='float16'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
            paddle.static.InputSpec(shape=[256], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cedff8586305abe4daad0ab667304d5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_79422893ac9670c7400ec9a14471c3aa
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 256], dtype='float16', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
            paddle.uniform([256], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bbc9af7b22b9290ec7dbaba2f164d24f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.45181822776794434, 0.15644659101963043, 0.41599833965301514, 0.4687879979610443, 0.1417054980993271, 0.2063867151737213, 0.21506008505821228, 0.38842394948005676, 0.41120296716690063, 0.16470792889595032, 0.285857230424881, 0.038601044565439224, 0.4585890471935272, 0.2383735477924347, 0.10306701809167862, 0.47930487990379333, 0.3253852128982544, 0.06533379852771759, 0.09188483655452728, 0.33919185400009155, 0.3822097182273865, 0.24318350851535797, 0.3876914083957672, 0.16744489967823029], dtype='float32').reshape([24]),
            paddle.to_tensor([0.12142302095890045, 0.18080535531044006, 0.11763933300971985, 0.33988332748413086, 0.343924880027771, 0.03538785129785538, 0.12519605457782745, 0.20895296335220337, 0.14743110537528992, 0.1517002433538437, 0.4648142158985138, 0.2057114690542221, 0.2446841448545456, 0.46642202138900757, 0.4943891763687134, 0.3995833694934845, 0.33166688680648804, 0.2197190523147583, 0.34716513752937317, 0.36663302779197693, 0.4549614191055298, 0.16004899144172668, 0.17169106006622314, 0.3339395225048065], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5f9d11a6ca1b1451d1023ece1eca04b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 196, 160], dtype='float16'),
            paddle.static.InputSpec(shape=[160], dtype='float32'),
            paddle.static.InputSpec(shape=[160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_20f6bbfd9e2e906ebffbdbff00d4f46a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5f9d11a6ca1b1451d1023ece1eca04b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 160], dtype='float16', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
            paddle.uniform([160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8dd42229734a93b51f6df67a38866d14(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 196, 768], dtype='float16'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c33f2b302021795042d84cce791b0133(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8dd42229734a93b51f6df67a38866d14
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 768], dtype='float16', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e3fe96e45a2ce21b2d43587c27e4ccfb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
            paddle.static.InputSpec(shape=[128], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9172a939d9eeb5b33886ccde64d3e5dd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3fe96e45a2ce21b2d43587c27e4ccfb
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
            paddle.uniform([128], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_45c96e5cd40f440eb3a0636984123bc5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.30761468410491943, 0.25545886158943176, 0.11495378613471985, 0.17797860503196716, 0.4689677357673645, 0.4000888764858246, 0.0020925207063555717, 0.04494655877351761, 0.264802485704422, 0.2899845838546753, 0.2785765826702118, 0.2719224989414215, 0.07882795482873917, 0.1616554856300354, 0.1692863404750824, 0.25682011246681213, 0.38755348324775696, 0.4533008933067322, 0.17495843768119812, 0.22069837152957916, 0.15079163014888763, 0.4184850752353668, 0.23045295476913452, 0.04470523074269295], dtype='float32').reshape([24]),
            paddle.to_tensor([0.11151678115129471, 0.4131118357181549, 0.2148454636335373, 0.4352688789367676, 0.3119615912437439, 0.4573203921318054, 0.04482884332537651, 0.07706257700920105, 0.3026610016822815, 0.07344765961170197, 0.15054576098918915, 0.044502269476652145, 0.08810137957334518, 0.4785158634185791, 0.4546124041080475, 0.0761626586318016, 0.3586846888065338, 0.036855924874544144, 0.23320068418979645, 0.4175979197025299, 0.09095178544521332, 0.4313460886478424, 0.46131259202957153, 0.48826664686203003], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_41cd05d31cc9a73ae8b5261a24a032dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            paddle.static.InputSpec(shape=[32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cb848613d6366a316f13bb4b83db18ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41cd05d31cc9a73ae8b5261a24a032dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f4f49edcd7f99d43387be9a121699a88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.05392275005578995, 0.2818715274333954, 0.03615570068359375, 0.20967014133930206, 0.14658932387828827, 0.34495145082473755, 0.40726515650749207, 0.37677526473999023, 0.07277308404445648, 0.30851444602012634, 0.42304614186286926, 0.18453237414360046, 0.2879643142223358, 0.26125243306159973, 0.3647204339504242, 0.39734500646591187, 0.38414183259010315, 0.01990112103521824, 0.14504720270633698, 0.1533571183681488, 0.37138065695762634, 0.0779024064540863, 0.4185749590396881, 0.058583684265613556], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4617677628993988, 0.23614075779914856, 0.0859203115105629, 0.4824765920639038, 0.4810549318790436, 0.16216987371444702, 0.18264314532279968, 0.44300296902656555, 0.07724204659461975, 0.4090736508369446, 0.2143612504005432, 0.1444007009267807, 0.3600376546382904, 0.2866038978099823, 0.06198585033416748, 0.48612532019615173, 0.1343386471271515, 0.14052188396453857, 0.49764618277549744, 0.2540387511253357, 0.4850650131702423, 0.24434922635555267, 0.4234316945075989, 0.3348560631275177], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_10feef1a485025008de287958d98353b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.01105545461177826, 0.45600083470344543, 0.1472207009792328, 0.2544405162334442, 0.43499475717544556, 0.1663062423467636, 0.37715381383895874, 0.08432743698358536, 0.008724329993128777, 0.15259897708892822, 0.4038931131362915, 0.28620994091033936, 0.0890762060880661, 0.23181991279125214, 0.4317178428173065, 0.45366808772087097, 0.44179409742355347, 0.14664272964000702, 0.332912415266037, 0.04833918437361717, 0.2811003625392914, 0.21524909138679504, 0.47788986563682556, 0.10210508108139038], dtype='float32').reshape([24]),
            paddle.to_tensor([0.29892483353614807, 0.00020900979870930314, 0.049991875886917114, 0.026884499937295914, 0.4150582551956177, 0.22972820699214935, 0.3861006498336792, 0.3100590109825134, 0.4489043951034546, 0.495871901512146, 0.12742160260677338, 0.010572493076324463, 0.4621841013431549, 0.2164771556854248, 0.12355801463127136, 0.09486736357212067, 0.28200891613960266, 0.18363486230373383, 0.07001034915447235, 0.3041735291481018, 0.24856653809547424, 0.024174576625227928, 0.4064213037490845, 0.28615763783454895], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c5c83d2f19b268ccfb605d6b66d3d91a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.23474448919296265, 0.08997555077075958, 0.33955928683280945, 0.40124720335006714, 0.3704860508441925, 0.4738033711910248, 0.2780499756336212, 0.13458776473999023, 0.43936723470687866, 0.10048970580101013, 0.2777318060398102, 0.44833943247795105, 0.17911478877067566, 0.06459233164787292, 0.49738913774490356, 0.33132410049438477, 0.4555908441543579, 0.16767114400863647, 0.2422560602426529, 0.2416214495897293, 0.22913630306720734, 0.17069366574287415, 0.2416754513978958, 0.1942424476146698], dtype='float32').reshape([24]),
            paddle.to_tensor([0.35250532627105713, 0.2005789428949356, 0.32223692536354065, 0.42975544929504395, 0.4608476459980011, 0.04539401829242706, 0.10758612304925919, 0.40506601333618164, 0.33503568172454834, 0.2947435677051544, 0.37529292702674866, 0.05912544205784798, 0.40782251954078674, 0.2556036412715912, 0.1256026029586792, 0.15748092532157898, 0.16714735329151154, 0.23512257635593414, 0.43873095512390137, 0.2348024994134903, 0.06220410391688347, 0.10202769935131073, 0.36546432971954346, 0.21086686849594116], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8af68f73ed9e4e54e0fadd7d02065db8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.04341757670044899, 0.17153790593147278, 0.1251583695411682, 0.2766187787055969, 0.13866543769836426, 0.4495866298675537, 0.2950727045536041, 0.047320734709501266, 0.2981654405593872, 0.018491068854928017, 0.20255699753761292, 0.11563907563686371, 0.25632721185684204, 0.45663735270500183, 0.1606222242116928, 0.3366738259792328, 0.4341822564601898, 0.1583838164806366, 0.3315364718437195, 0.025674160569906235, 0.13253335654735565, 0.0920831710100174, 0.3664415180683136, 0.23842822015285492], dtype='float32').reshape([24]),
            paddle.to_tensor([0.16132861375808716, 0.3060794770717621, 0.30159449577331543, 0.44834211468696594, 0.16444161534309387, 0.0052471221424639225, 0.24062617123126984, 0.1866290271282196, 0.26007893681526184, 0.36656081676483154, 0.15327471494674683, 0.010063513182103634, 0.0948360487818718, 0.22680942714214325, 0.07861218601465225, 0.43750330805778503, 0.04992927610874176, 0.30160877108573914, 0.16938968002796173, 0.15771284699440002, 0.15416525304317474, 0.30159902572631836, 0.18859180808067322, 0.1481732726097107], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e1f9ee37c6233815dbcd688e66126e6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.4258430004119873, 0.4043937921524048, 0.08537891507148743, 0.05949791520833969, 0.2487742155790329, 0.40879178047180176, 0.36262497305870056, 0.4746846556663513, 0.4779752492904663, 0.2871939539909363, 0.06054530665278435, 0.08476774394512177, 0.4751846194267273, 0.4916658401489258, 0.08174728602170944, 0.20136471092700958, 0.4613259732723236, 0.11537114530801773, 0.13589829206466675, 0.4434391260147095, 0.3364270329475403, 0.09092148393392563, 0.2093760222196579, 0.11300299316644669], dtype='float32').reshape([24]),
            paddle.to_tensor([0.2358911633491516, 0.2933349609375, 0.13400079309940338, 0.125348761677742, 0.13178804516792297, 0.46917495131492615, 0.25751635432243347, 0.3299882411956787, 0.018248556181788445, 0.4731366038322449, 0.49038833379745483, 0.05257653072476387, 0.23448865115642548, 0.3030720353126526, 0.07812820374965668, 0.02519453689455986, 0.07034394145011902, 0.178263321518898, 0.4488944113254547, 0.1642979085445404, 0.028249628841876984, 0.13841675221920013, 0.38248637318611145, 0.4505709111690521], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_82fbbaa85069298a1e16ef4fb39d4fb8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.04228634014725685, 0.3086622655391693, 0.13276219367980957, 0.30395007133483887, 0.004799717105925083, 0.2970888316631317, 0.07038295269012451, 0.4458417594432831, 0.22578908503055573, 0.14197063446044922, 0.013123948127031326, 0.0666479840874672, 0.05203132703900337, 0.42245620489120483, 0.48057305812835693, 0.4700353741645813, 0.284076064825058, 0.4608613848686218, 0.23168109357357025, 0.06484774500131607, 0.4914454221725464, 0.05354077368974686, 0.1024472564458847, 0.19013266265392303], dtype='float32').reshape([24]),
            paddle.to_tensor([0.36631155014038086, 0.23084913194179535, 0.21997793018817902, 0.47608256340026855, 0.10233937203884125, 0.16964471340179443, 0.03537775203585625, 0.25137224793434143, 0.3494671881198883, 0.4114452600479126, 0.023448318243026733, 0.19525642693042755, 0.39177054166793823, 0.4076698124408722, 0.31069034337997437, 0.3819114863872528, 0.04235578700900078, 0.09784754365682602, 0.49831750988960266, 0.47350364923477173, 0.4132387638092041, 0.4302612543106079, 0.2860724627971649, 0.3112300932407379], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0a0e8b036a643006abac343bacb204e0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 3136, 32], dtype='float32'),
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            paddle.static.InputSpec(shape=[32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_47145eb9f17e8ec62235966e58c195a2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a0e8b036a643006abac343bacb204e0
    def get_inputs(self):
        return [
            paddle.uniform([1, 3136, 32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
            paddle.uniform([32], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d4e99b9683f7b6a477528e89a159143b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.1035740002989769, 0.1295202374458313, 0.14366847276687622, 0.49217090010643005, 0.10821545869112015, 0.10661082714796066, 0.4754040539264679, 0.12215252220630646, 0.038166921585798264, 0.2644746005535126, 0.13129262626171112, 0.39217284321784973, 0.42081138491630554, 0.42129525542259216, 0.2439534217119217, 0.261929452419281, 0.294368177652359, 0.2997193932533264, 0.2187143713235855, 0.3327438533306122, 0.27129560708999634, 0.03936881944537163, 0.30701062083244324, 0.37757521867752075], dtype='float32').reshape([24]),
            paddle.to_tensor([0.25314557552337646, 0.2843272387981415, 0.28727027773857117, 0.06917640566825867, 0.45657777786254883, 0.27277663350105286, 0.38662365078926086, 0.092476487159729, 0.11354199051856995, 0.4542045295238495, 0.16675810515880585, 0.05748770385980606, 0.40060460567474365, 0.003968958742916584, 0.2902595102787018, 0.37619245052337646, 0.4679041802883148, 0.23376911878585815, 0.05942581221461296, 0.29923492670059204, 0.24800780415534973, 0.3901248574256897, 0.3345220685005188, 0.06277187168598175], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d732e1964ed5f34779c533078c3c8e3c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.1824134737253189, 0.49814826250076294, 0.0069441478699445724, 0.48965024948120117, 0.4977090060710907, 0.15261100232601166, 0.042410243302583694, 0.1910579651594162, 0.06030287966132164, 0.3869251608848572, 0.47306588292121887, 0.4250328838825226, 0.27883395552635193, 0.2332274615764618, 0.3894924521446228, 0.10761517286300659, 0.17170391976833344, 0.26800787448883057, 0.2943716049194336, 0.35984307527542114, 0.34916049242019653, 0.12699183821678162, 0.40980416536331177, 0.04790949076414108], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4194541275501251, 0.30980581045150757, 0.23269504308700562, 0.2916938066482544, 0.2173774093389511, 0.2624126970767975, 0.30144003033638, 0.2126806229352951, 0.10852868854999542, 0.1543165147304535, 0.15847784280776978, 0.13367633521556854, 0.46247756481170654, 0.15025699138641357, 0.26561209559440613, 0.286119282245636, 0.21557362377643585, 0.4007219076156616, 0.11764578521251678, 0.15842530131340027, 0.1848953813314438, 0.3914937973022461, 0.11386066675186157, 0.3431903123855591], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7cb413a72c1ae67d0ece2c769e1468be(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 784, 64], dtype='float16'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9c5d928e6dd8f5b2214243381000394b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7cb413a72c1ae67d0ece2c769e1468be
    def get_inputs(self):
        return [
            paddle.uniform([1, 784, 64], dtype='float16', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
            paddle.uniform([64], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_983ac8fb5ca8d8c7e84fd8f312d06f5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.39517417550086975, 0.3536563515663147, 0.4822602868080139, 0.21384558081626892, 0.3235523998737335, 0.4825739562511444, 0.13019821047782898, 0.01079529244452715, 0.22576725482940674, 0.4476866126060486, 0.29225435853004456, 0.2099895179271698, 0.026658814400434494, 0.01382521167397499, 0.475036084651947, 0.4646579623222351, 0.018700728192925453, 0.1769464910030365, 0.4254573881626129, 0.2752411365509033, 0.46598365902900696, 0.2485819160938263, 0.013161174021661282, 0.09954072535037994], dtype='float32').reshape([24]),
            paddle.to_tensor([0.03876975178718567, 0.21373344957828522, 0.3756926357746124, 0.17542016506195068, 0.19525600969791412, 0.19853855669498444, 0.18526841700077057, 0.3543507158756256, 0.13645541667938232, 0.059873033314943314, 0.23291927576065063, 0.044369764626026154, 0.1702864170074463, 0.29753437638282776, 0.46651172637939453, 0.11118439584970474, 0.189055398106575, 0.19702266156673431, 0.017139531672000885, 0.13025830686092377, 0.16606055200099945, 0.3478761315345764, 0.22786249220371246, 0.1389087587594986], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3f9555c8e192d922dc5f6a11f6b97c4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3718504011631012, 0.4057827591896057, 0.08825507014989853, 0.3191535472869873, 0.3273610770702362, 0.09889256954193115, 0.34657755494117737, 0.08334571868181229, 0.3149149417877197, 0.05698605999350548, 0.4020753800868988, 0.47568681836128235, 0.05964718386530876, 0.0746929794549942, 0.18867149949073792, 0.0264153853058815, 0.1237323209643364, 0.35682469606399536, 0.13458889722824097, 0.34701645374298096, 0.0067862761206924915, 0.46045929193496704, 0.3511141240596771, 0.4836772084236145], dtype='float32').reshape([24]),
            paddle.to_tensor([0.3772055506706238, 0.11499578505754471, 0.24551093578338623, 0.4761704206466675, 0.41531121730804443, 0.1077539473772049, 0.11296368390321732, 0.037412069737911224, 0.0006786211743019521, 0.4879702925682068, 0.10229720920324326, 0.37581220269203186, 0.490165114402771, 0.3041853904724121, 0.29955312609672546, 0.027473805472254753, 0.0033867796882987022, 0.2763216495513916, 0.006897625979036093, 0.05040846765041351, 0.16725175082683563, 0.3542760908603668, 0.14458028972148895, 0.022288020700216293], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4a03124c9258ab8dbad9b09aac642661(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9b9a79471f2248e0c5c2599b679c86c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4a03124c9258ab8dbad9b09aac642661
    def get_inputs(self):
        return [
            paddle.uniform([4, 64, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_77efcf7066f464b11800917cc7a6e7c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.19510102272033691, 0.20686554908752441, 0.31951838731765747, 0.15141570568084717, 0.1971461921930313, 0.23452912271022797, 0.008153560571372509, 0.2686106562614441, 0.28337332606315613, 0.11421645432710648, 0.3937045931816101, 0.26021596789360046, 0.43100133538246155, 0.16396458446979523, 0.3214974105358124, 0.2177264243364334, 0.4210642874240875, 0.2913675308227539, 0.289941668510437, 0.3929293155670166, 0.07857263833284378, 0.19846592843532562, 0.3294351398944855, 0.2203688770532608], dtype='float32').reshape([24]),
            paddle.to_tensor([0.27907368540763855, 0.09010409563779831, 0.11294716596603394, 0.0027453519869595766, 0.3846855163574219, 0.17290423810482025, 0.18456506729125977, 0.3076610863208771, 0.40406158566474915, 0.10581346601247787, 0.44341564178466797, 0.12190552055835724, 0.4910966753959656, 0.2187657505273819, 0.48935776948928833, 0.4100140631198883, 0.04153335466980934, 0.3354561924934387, 0.13319899141788483, 0.16179582476615906, 0.14329266548156738, 0.07169412821531296, 0.04286563768982887, 0.4193059802055359], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5077cb31a25157d88979de23e00f4793(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 196, 768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
            paddle.static.InputSpec(shape=[768], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4d10de05e75bd5b25a8a0c8e3fdcdc99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5077cb31a25157d88979de23e00f4793
    def get_inputs(self):
        return [
            paddle.uniform([1, 196, 768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
            paddle.uniform([768], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2ef84b6b4ad597fb38bc99418e87778d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 192], dtype='float16'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1a84a210ae96f6729f8eadec0adfb1cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2ef84b6b4ad597fb38bc99418e87778d
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 192], dtype='float16', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dd13214e68ef0629ecefe52548a034a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3349670171737671, 0.46534308791160583, 0.3643534183502197, 0.41702523827552795, 0.19068734347820282, 0.05611376091837883, 0.4849598705768585, 0.14781707525253296, 0.42353081703186035, 0.4401087760925293, 0.311707466840744, 0.04038776457309723, 0.15079443156719208, 0.014491400681436062, 0.10487615317106247, 0.4530875086784363, 0.1493648886680603, 0.46924588084220886, 0.051501184701919556, 0.304236501455307, 0.3389779031276703, 0.16972464323043823, 0.13079366087913513, 0.26695775985717773], dtype='float32').reshape([24]),
            paddle.to_tensor([0.31890788674354553, 0.12577171623706818, 0.13630738854408264, 0.4791688024997711, 0.21668066084384918, 0.4769456386566162, 0.49684375524520874, 0.20623788237571716, 0.46564146876335144, 0.46870407462120056, 0.3254479467868805, 0.43968111276626587, 0.17941807210445404, 0.35473185777664185, 0.23073816299438477, 0.4777703881263733, 0.23228968679904938, 0.39240074157714844, 0.49448615312576294, 0.04849475249648094, 0.19518831372261047, 0.28630977869033813, 0.4168829321861267, 0.4603678584098816], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3a8bddebb891b417fd53d125ed2f0d5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.05121523141860962, 0.21934635937213898, 0.15172675251960754, 0.46700021624565125, 0.4412868022918701, 0.03750431165099144, 0.0622614361345768, 0.2170773148536682, 0.19763746857643127, 0.32561802864074707, 0.04589257016777992, 0.03163204342126846, 0.05635915696620941, 0.2435961365699768, 0.07495290040969849, 0.06594714522361755, 0.39458167552948, 0.19440683722496033, 0.30179962515830994, 0.31341537833213806, 0.34694650769233704, 0.07910668849945068, 0.16503579914569855, 0.18005156517028809], dtype='float32').reshape([24]),
            paddle.to_tensor([0.02027488872408867, 0.4624885618686676, 0.07281754165887833, 0.24458208680152893, 0.3375331163406372, 0.3931454122066498, 0.17199786007404327, 0.22989076375961304, 0.2197123020887375, 0.19207079708576202, 0.010000357404351234, 0.006858407985419035, 0.09518784284591675, 0.010860231705009937, 0.16182701289653778, 0.41861408948898315, 0.039846502244472504, 0.07708526402711868, 0.22896268963813782, 0.049331407994031906, 0.0629739761352539, 0.24582064151763916, 0.04951018467545509, 0.047755368053913116], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ec1db65e424e69fff8ab2b741d517461(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.14240507781505585, 0.2461661994457245, 0.3206366300582886, 0.19137898087501526, 0.43590784072875977, 0.34423115849494934, 0.16656647622585297, 0.16306623816490173, 0.4494502544403076, 0.24616478383541107, 0.3063179552555084, 0.22665995359420776, 0.07646908611059189, 0.3839546740055084, 0.232854962348938, 0.26128092408180237, 0.4535377323627472, 0.2770805358886719, 0.47044259309768677, 0.2856404483318329, 0.2464005947113037, 0.3421778082847595, 0.07225311547517776, 0.26087984442710876], dtype='float32').reshape([24]),
            paddle.to_tensor([0.19987092912197113, 0.3576652407646179, 0.11787109076976776, 0.3325967788696289, 0.288742333650589, 0.4972230792045593, 0.48721984028816223, 0.49964529275894165, 0.19424232840538025, 0.09946565330028534, 0.4673880934715271, 0.445083349943161, 0.09163814038038254, 0.011729656718671322, 0.3958911895751953, 0.2483629584312439, 0.06255633383989334, 0.29839298129081726, 0.13364702463150024, 0.1678139567375183, 0.34484246373176575, 0.009993314743041992, 0.36321166157722473, 0.3363507390022278], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_30f6c2d6766d87c4f8d1a817f1b4f1ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.034891657531261444, 0.25244250893592834, 0.49686574935913086, 0.3393307626247406, 0.2291957437992096, 0.008383574895560741, 0.058248378336429596, 0.2189389020204544, 0.08078977465629578, 0.08699306845664978, 0.21978430449962616, 0.4298100173473358, 0.41483643651008606, 0.020389322191476822, 0.34036436676979065, 0.10749401152133942, 0.06748127192258835, 0.3839569389820099, 0.02750573679804802, 0.28205761313438416, 0.3634272813796997, 0.19779933989048004, 0.11216151714324951, 0.49488502740859985], dtype='float32').reshape([24]),
            paddle.to_tensor([0.17641766369342804, 0.010071221739053726, 0.016195164993405342, 0.050734374672174454, 0.18561075627803802, 0.0179679524153471, 0.19758334755897522, 0.3588763177394867, 0.27565568685531616, 0.011221563443541527, 0.14906294643878937, 0.20636238157749176, 0.059937965124845505, 0.34043920040130615, 0.3125981092453003, 0.10364650934934616, 0.2571318447589874, 0.4433269798755646, 0.014500022865831852, 0.3173336684703827, 0.37322044372558594, 0.3926061987876892, 0.1845085471868515, 0.3290131390094757], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0e08952df2309ddf778fd4e3ba94ed04(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6726fc0e42cc37e6738f07b81b887696(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e08952df2309ddf778fd4e3ba94ed04
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8b11f62946dad8748e9cda6e1247b4bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.2000417560338974, 0.43612444400787354, 0.10320629924535751, 0.3864510655403137, 0.300361305475235, 0.39112409949302673, 0.45817074179649353, 0.3014487624168396, 0.27445268630981445, 0.24538226425647736, 0.24052831530570984, 0.4098356366157532, 0.18186643719673157, 0.006081525702029467, 0.16125011444091797, 0.41375982761383057, 0.3779347538948059, 0.3756195306777954, 0.3533840477466583, 0.050032224506139755, 0.3142375349998474, 0.39956992864608765, 0.20750340819358826, 0.14826947450637817], dtype='float32').reshape([24]),
            paddle.to_tensor([0.34167858958244324, 0.3799011707305908, 0.13324213027954102, 0.04921453446149826, 0.32886746525764465, 0.19674910604953766, 0.08400838822126389, 0.4864448010921478, 0.14022554457187653, 0.2453107386827469, 0.09226436913013458, 0.07580164074897766, 0.2612718939781189, 0.12942540645599365, 0.43427303433418274, 0.3478774130344391, 0.3271027207374573, 0.20842719078063965, 0.3521370589733124, 0.43646863102912903, 0.26970207691192627, 0.008656446821987629, 0.3533826768398285, 0.23955699801445007], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2839fca4e50b40fc0b5c0cd238b1ece5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.26550623774528503, 0.31317174434661865, 0.2199205905199051, 0.18995782732963562, 0.25097888708114624, 0.13643352687358856, 0.22136832773685455, 0.43960127234458923, 0.4574490487575531, 0.33221688866615295, 0.2499828189611435, 0.46667593717575073, 0.04879075661301613, 0.4514358937740326, 0.2205541878938675, 0.0220150426030159, 0.04371117055416107, 0.41575679183006287, 0.13118937611579895, 0.4692220687866211, 0.05059470236301422, 0.22142383456230164, 0.09091667830944061, 0.4025753438472748], dtype='float32').reshape([24]),
            paddle.to_tensor([0.11101546138525009, 0.335085928440094, 0.28303608298301697, 0.410030335187912, 0.3223246932029724, 0.40834444761276245, 0.231110081076622, 0.4958789348602295, 0.4362051486968994, 0.36121881008148193, 0.46931642293930054, 0.023132525384426117, 0.2836085259914398, 0.09515850991010666, 0.0554984025657177, 0.247451514005661, 0.24981847405433655, 0.39132487773895264, 0.048574574291706085, 0.04890216886997223, 0.11277665942907333, 0.37451615929603577, 0.45408895611763, 0.301052063703537], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_656d73e975ded93491678fc0452b556b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 576, 1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
            paddle.static.InputSpec(shape=[1024], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b3fa6e79764adc3626ad7ca452a7a8c5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_656d73e975ded93491678fc0452b556b
    def get_inputs(self):
        return [
            paddle.uniform([1, 576, 1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            paddle.uniform([1024], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c380c8af7e8330bbd017676c63acd5f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.13091669976711273, 0.40552303194999695, 0.39937496185302734, 0.49862000346183777, 0.38601189851760864, 0.12211894243955612, 0.17948594689369202, 0.2911418080329895, 0.2626224458217621, 0.4496499300003052, 0.28549137711524963, 0.18957547843456268, 0.3354063332080841, 0.10536371171474457, 0.4517870247364044, 0.27307987213134766, 0.026530519127845764, 0.3480890393257141, 0.15569426119327545, 0.32755985856056213, 0.326185405254364, 0.3942374587059021, 0.0774802416563034, 0.4810302257537842], dtype='float32').reshape([24]),
            paddle.to_tensor([0.3104518949985504, 0.30517590045928955, 0.4897959530353546, 0.28241443634033203, 0.32304996252059937, 0.0023260777816176414, 0.1564987599849701, 0.19034205377101898, 0.06846291571855545, 0.20177443325519562, 0.2427491545677185, 0.11338825523853302, 0.09213965386152267, 0.0379730649292469, 0.08227625489234924, 0.47217857837677, 0.2307276576757431, 0.20990292727947235, 0.24079908430576324, 0.3523121476173401, 0.3563387095928192, 0.4621063768863678, 0.4361419677734375, 0.03155728429555893], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_aac44a73b00bf30ed87610d317d20ec9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 2048], dtype='float32'),
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
            paddle.static.InputSpec(shape=[2048], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6a9f6993a7571d41e28b318525abcca2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aac44a73b00bf30ed87610d317d20ec9
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 2048], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
            paddle.uniform([2048], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8bc15446d129642aefa160c9338dfe16(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.3325293958187103, 0.17666979134082794, 0.31898054480552673, 0.21584579348564148, 0.04341138154268265, 0.4263390004634857, 0.10768727213144302, 0.1624918133020401, 0.1501302272081375, 0.46592506766319275, 0.08752952516078949, 0.4479202330112457, 0.09978561103343964, 0.38107606768608093, 0.43186691403388977, 0.2065444439649582, 0.2996521294116974, 0.3966462314128876, 0.3860372304916382, 0.25005075335502625, 0.13984428346157074, 0.282341331243515, 0.1683175414800644, 0.10451474040746689], dtype='float32').reshape([24]),
            paddle.to_tensor([0.4624478816986084, 0.36710506677627563, 0.3395622968673706, 0.3909805417060852, 0.11182694137096405, 0.24047018587589264, 0.14647217094898224, 0.1858019381761551, 0.16573773324489594, 0.09910175949335098, 0.4861185848712921, 0.2190948873758316, 0.08021880686283112, 0.04225517064332962, 0.0299625676125288, 0.27269625663757324, 0.22387155890464783, 0.33000442385673523, 0.4295661747455597, 0.31865277886390686, 0.021751921623945236, 0.3084564507007599, 0.10799778997898102, 0.249933123588562], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8d5f0b78ea35512581ca53198d94a491(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d529a8355f7340f6a8d1dc00c7e174ee
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([0.24688509106636047, 0.17250116169452667, 0.2432025820016861, 0.2606157660484314, 0.41954201459884644, 0.2613685429096222, 0.22679725289344788, 0.045032985508441925, 0.2605045437812805, 0.4567176103591919, 0.2659311890602112, 0.13469034433364868, 0.28229042887687683, 0.3167077898979187, 0.36961105465888977, 0.014117349870502949, 0.4851415455341339, 0.021073240786790848, 0.15424425899982452, 0.16487419605255127, 0.0053559597581624985, 0.38903504610061646, 0.030328867956995964, 0.49262383580207825], dtype='float32').reshape([24]),
            paddle.to_tensor([0.37008392810821533, 0.043361105024814606, 0.35771626234054565, 0.030832350254058838, 0.32057854533195496, 0.25181934237480164, 0.030989358201622963, 0.07841219753026962, 0.1291738748550415, 0.057169973850250244, 0.24631285667419434, 0.3054234981536865, 0.4609898626804352, 0.1715894192457199, 0.49217599630355835, 0.36563172936439514, 0.19789007306098938, 0.135758176445961, 0.04906543344259262, 0.15772047638893127, 0.36431390047073364, 0.025460606440901756, 0.09055653214454651, 0.31386837363243103], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_63c9116429cf16d6e20f420c1d4ccc9b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 240], dtype='float16'),
            paddle.static.InputSpec(shape=[240], dtype='float32'),
            paddle.static.InputSpec(shape=[240], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b21e023a353c56a15175677d6015ce37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_63c9116429cf16d6e20f420c1d4ccc9b
    def get_inputs(self):
        return [
            paddle.uniform([4, 16, 240], dtype='float16', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
            paddle.uniform([240], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_adba015e4075934edea2be396e519460(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-05'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 49, 192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
            paddle.static.InputSpec(shape=[192], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c96e2de8d6f3122368aef5de69773acf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adba015e4075934edea2be396e519460
    def get_inputs(self):
        return [
            paddle.uniform([1, 49, 192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
            paddle.uniform([192], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_447e7ad0f48ee670d77b10b73bd1d447(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e78afc332b98817e6e0e0ece8dd0611
    def get_inputs(self):
        return [
            paddle.uniform([196, 16, 24], dtype='float16', min=0, max=0.5),
            paddle.to_tensor([0.3136044144630432, 0.40865516662597656, 0.07151678949594498, 0.24499498307704926, 0.1565580517053604, 0.45366188883781433, 0.16632704436779022, 0.3836362659931183, 0.3116975426673889, 0.30352872610092163, 0.1864796131849289, 0.08439808338880539, 0.15562450885772705, 0.1277054399251938, 0.27769604325294495, 0.4765385389328003, 0.42150402069091797, 0.47910937666893005, 0.2691156268119812, 0.24255825579166412, 0.05393240228295326, 0.172566756606102, 0.49462807178497314, 0.24584709107875824], dtype='float32').reshape([24]),
            paddle.to_tensor([0.14712202548980713, 0.26961320638656616, 0.30474522709846497, 0.43793705105781555, 0.25082191824913025, 0.2535148561000824, 0.2118612676858902, 0.1316499412059784, 0.044650040566921234, 0.459414005279541, 0.3043159246444702, 0.16374902427196503, 0.2391333431005478, 0.22803060710430145, 0.26731616258621216, 0.03101859800517559, 0.3141729235649109, 0.2126837819814682, 0.42483556270599365, 0.4464513957500458, 0.1091708093881607, 0.010207617655396461, 0.025683632120490074, 0.2227029800415039], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_655616a8789ba66b34a6c297ec8ce0d8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1, arg_2):
        input_0 = arg_0
        input_1 = arg_1
        input_2 = arg_2
        return (lambda x, f: f(x))(paddle._C_ops.layer_norm(input_0, input_1, input_2, float('1e-06'), 2), lambda out: out if isinstance(out, (list, tuple)) else (out, None,None))

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 512], dtype='float16'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
            paddle.static.InputSpec(shape=[512], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_31cd1cdac5f2a17d3b48119f9dd72f30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_655616a8789ba66b34a6c297ec8ce0d8
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 512], dtype='float16', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
            paddle.uniform([512], dtype='float32', min=0, max=0.5),
        ]


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