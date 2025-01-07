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
class PrimitiveOp_347f98521c547dd9f0da4f650873cd6a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.meshgrid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f6f8e1c7741ff60a7ff56440ef9c8932(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347f98521c547dd9f0da4f650873cd6a
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.9130434989929199, -0.8260869383811951, -0.739130437374115, -0.6521739363670349, -0.5652173757553101, -0.47826087474823, -0.3913043439388275, -0.30434781312942505, -0.21739129722118378, -0.1304347813129425, -0.043478261679410934, 0.043478261679410934, 0.1304347813129425, 0.21739129722118378, 0.30434781312942505, 0.3913043439388275, 0.47826087474823, 0.5652173757553101, 0.6521739363670349, 0.739130437374115, 0.8260869383811951, 0.9130434989929199, 1.0], dtype='float32').reshape([24]),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a5346a0698148cbfe7dd3e716e6305e8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347f98521c547dd9f0da4f650873cd6a
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_24858d5ed164ff0b8532f8839ed3463b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347f98521c547dd9f0da4f650873cd6a
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_95d5829b73efb4a20e790e78af9dcb77(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347f98521c547dd9f0da4f650873cd6a
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_79a7f7166aa0f125f0faad974cc62308(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347f98521c547dd9f0da4f650873cd6a
    def get_inputs(self):
        return [
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
class TestPrimitiveOp_5089b3f73deab0f37f40e7fca738b75f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347f98521c547dd9f0da4f650873cd6a
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e7f8e89fd6a9b1ba440b250f1198c0c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347f98521c547dd9f0da4f650873cd6a
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0, 544.0, 576.0, 608.0, 640.0, 672.0, 704.0, 736.0], dtype='float32').reshape([24]),
            paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0, 544.0, 576.0, 608.0, 640.0, 672.0, 704.0, 736.0], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2e238ae2cb7c237fe18d6f6473b77785(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347f98521c547dd9f0da4f650873cd6a
    def get_inputs(self):
        return [
            paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0, 560.0, 592.0, 624.0, 656.0, 688.0, 720.0, 752.0], dtype='float32').reshape([24]),
            paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0, 560.0, 592.0, 624.0, 656.0, 688.0, 720.0, 752.0], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_378f4e7b6724c3771b20ba6ed5494b8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347f98521c547dd9f0da4f650873cd6a
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.9166666865348816, -0.8333333134651184, -0.75, -0.6666666865348816, -0.5833333134651184, -0.5, -0.4166666567325592, -0.3333333432674408, -0.25, -0.1666666716337204, -0.0833333358168602, 5.551115123125783e-17, 0.0833333358168602, 0.1666666716337204, 0.25, 0.3333333432674408, 0.4166666567325592, 0.5, 0.5833333134651184, 0.6666666865348816, 0.75, 0.8333333134651184, 0.9166666865348816, 1.0], dtype='float32').reshape([25]),
            paddle.uniform([38], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_32afc5a9edf3c13f4ff7ee18d752033d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347f98521c547dd9f0da4f650873cd6a
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.8947368264198303, -0.7894737124443054, -0.6842105388641357, -0.5789473652839661, -0.4736842215061188, -0.3684210479259491, -0.2631579041481018, -0.15789473056793213, -0.05263157933950424, 0.05263157933950424, 0.15789473056793213, 0.2631579041481018, 0.3684210479259491, 0.4736842215061188, 0.5789473652839661, 0.6842105388641357, 0.7894737124443054, 0.8947368264198303, 1.0], dtype='float32').reshape([20]),
            paddle.to_tensor([-1.0, -0.931034505367279, -0.8620689511299133, -0.7931034564971924, -0.7241379022598267, -0.6551724076271057, -0.5862069129943848, -0.517241358757019, -0.4482758641242981, -0.37931033968925476, -0.3103448152542114, -0.24137930572032928, -0.17241379618644714, -0.1034482792019844, -0.03448275849223137, 0.03448275849223137, 0.1034482792019844, 0.17241379618644714, 0.24137930572032928, 0.3103448152542114, 0.37931033968925476, 0.4482758641242981, 0.517241358757019, 0.5862069129943848, 0.6551724076271057, 0.7241379022598267, 0.7931034564971924, 0.8620689511299133, 0.931034505367279, 1.0], dtype='float32').reshape([30]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f8c8829367d850a7eb2f0f589ff3dce1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347f98521c547dd9f0da4f650873cd6a
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0], dtype='float32').reshape([16]),
            paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0], dtype='float32').reshape([16]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5d2fb1fdb156c9ef4ec5b20cd0001a96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347f98521c547dd9f0da4f650873cd6a
    def get_inputs(self):
        return [
            paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0], dtype='float32').reshape([16]),
            paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0], dtype='float32').reshape([16]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_866241f345de03d44fb2a9dec87553b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347f98521c547dd9f0da4f650873cd6a
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_806d9390afab00d19b5f6ea0c7175109(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347f98521c547dd9f0da4f650873cd6a
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5eb654119c18e73534950f2311eca568(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347f98521c547dd9f0da4f650873cd6a
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0, 544.0, 576.0, 608.0], dtype='float32').reshape([20]),
            paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0, 544.0, 576.0, 608.0], dtype='float32').reshape([20]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dbf960e514f765d18e512b416bb0f055(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347f98521c547dd9f0da4f650873cd6a
    def get_inputs(self):
        return [
            paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0, 560.0, 592.0, 624.0], dtype='float32').reshape([20]),
            paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0, 560.0, 592.0, 624.0], dtype='float32').reshape([20]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e14efba5b5ae1784d4935088e23bd1b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347f98521c547dd9f0da4f650873cd6a
    def get_inputs(self):
        return [
            paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0], dtype='float32').reshape([14]),
            paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0], dtype='float32').reshape([14]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5ec02d7f281f220eccf44e94c6dad7c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347f98521c547dd9f0da4f650873cd6a
    def get_inputs(self):
        return [
            paddle.to_tensor([8.0, 24.0, 40.0, 56.0, 72.0, 88.0, 104.0, 120.0, 136.0, 152.0, 168.0, 184.0, 200.0, 216.0, 232.0, 248.0, 264.0, 280.0, 296.0, 312.0, 328.0, 344.0, 360.0, 376.0, 392.0, 408.0, 424.0, 440.0], dtype='float32').reshape([28]),
            paddle.to_tensor([8.0, 24.0, 40.0, 56.0, 72.0, 88.0, 104.0, 120.0, 136.0, 152.0, 168.0, 184.0, 200.0, 216.0, 232.0, 248.0, 264.0, 280.0, 296.0, 312.0, 328.0, 344.0, 360.0, 376.0, 392.0, 408.0, 424.0, 440.0], dtype='float32').reshape([28]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ccc424d143fb7e64e36f94a891a0a44d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347f98521c547dd9f0da4f650873cd6a
    def get_inputs(self):
        return [
            paddle.uniform([56], dtype='float32', min=0, max=0.5),
            paddle.uniform([56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bb4ae6dd0e5c42b7c1f80043bc03e3cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347f98521c547dd9f0da4f650873cd6a
    def get_inputs(self):
        return [
            paddle.to_tensor([8.0, 24.0, 40.0, 56.0, 72.0, 88.0, 104.0, 120.0, 136.0, 152.0, 168.0, 184.0, 200.0, 216.0, 232.0, 248.0, 264.0, 280.0, 296.0, 312.0, 328.0, 344.0, 360.0, 376.0], dtype='float32').reshape([24]),
            paddle.to_tensor([8.0, 24.0, 40.0, 56.0, 72.0, 88.0, 104.0, 120.0, 136.0, 152.0, 168.0, 184.0, 200.0, 216.0, 232.0, 248.0, 264.0, 280.0, 296.0, 312.0, 328.0, 344.0, 360.0, 376.0], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6c7f3cb828b8dd6afa51c467c826435c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347f98521c547dd9f0da4f650873cd6a
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a0bc28058be56373c300022fb372f8a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347f98521c547dd9f0da4f650873cd6a
    def get_inputs(self):
        return [
            paddle.uniform([34], dtype='float32', min=0, max=0.5),
            paddle.uniform([34], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_035b1c8e7a98734f1230a3f4234a45dc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347f98521c547dd9f0da4f650873cd6a
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0], dtype='float32').reshape([17]),
            paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0], dtype='float32').reshape([17]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fda32006c452a3ed1c3b624c0d899987(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347f98521c547dd9f0da4f650873cd6a
    def get_inputs(self):
        return [
            paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0], dtype='float32').reshape([17]),
            paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0], dtype='float32').reshape([17]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3e1f40861371bfa72bfd5f878fce07b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347f98521c547dd9f0da4f650873cd6a
    def get_inputs(self):
        return [
            paddle.to_tensor([32.0, 96.0, 160.0, 224.0, 288.0, 352.0, 416.0, 480.0, 544.0, 608.0, 672.0, 736.0, 800.0, 864.0, 928.0, 992.0], dtype='float32').reshape([16]),
            paddle.to_tensor([32.0, 96.0, 160.0, 224.0, 288.0, 352.0, 416.0, 480.0, 544.0, 608.0, 672.0, 736.0, 800.0, 864.0, 928.0, 992.0], dtype='float32').reshape([16]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a1fd1370390d428890bda9281aa6df85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347f98521c547dd9f0da4f650873cd6a
    def get_inputs(self):
        return [
            paddle.to_tensor([64.0, 192.0, 320.0, 448.0, 576.0, 704.0, 832.0, 960.0], dtype='float32').reshape([8]),
            paddle.to_tensor([64.0, 192.0, 320.0, 448.0, 576.0, 704.0, 832.0, 960.0], dtype='float32').reshape([8]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_641b1bb128044493616299ea6633df58(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347f98521c547dd9f0da4f650873cd6a
    def get_inputs(self):
        return [
            paddle.to_tensor([-1.0, -0.8571428656578064, -0.7142857313156128, -0.5714285969734192, -0.4285714328289032, -0.2857142984867096, -0.1428571492433548, 5.551115123125783e-17, 0.1428571492433548, 0.2857142984867096, 0.4285714328289032, 0.5714285969734192, 0.7142857313156128, 0.8571428656578064, 1.0], dtype='float32').reshape([15]),
            paddle.to_tensor([-1.0, -0.9166666865348816, -0.8333333134651184, -0.75, -0.6666666865348816, -0.5833333134651184, -0.5, -0.4166666567325592, -0.3333333432674408, -0.25, -0.1666666716337204, -0.0833333358168602, 5.551115123125783e-17, 0.0833333358168602, 0.1666666716337204, 0.25, 0.3333333432674408, 0.4166666567325592, 0.5, 0.5833333134651184, 0.6666666865348816, 0.75, 0.8333333134651184, 0.9166666865348816, 1.0], dtype='float32').reshape([25]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_402ec19e0483abbf314d72f5dc389bfb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347f98521c547dd9f0da4f650873cd6a
    def get_inputs(self):
        return [
            paddle.uniform([100], dtype='float32', min=0, max=0.5),
            paddle.uniform([152], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_38a7362b24d11de9aeea12ea4ccbfe42(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347f98521c547dd9f0da4f650873cd6a
    def get_inputs(self):
        return [
            paddle.uniform([50], dtype='float32', min=0, max=0.5),
            paddle.uniform([76], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_70ba726e1969548d6ae638c7501c5b83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347f98521c547dd9f0da4f650873cd6a
    def get_inputs(self):
        return [
            paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0, 560.0, 592.0, 624.0, 656.0, 688.0, 720.0, 752.0, 784.0], dtype='float32').reshape([25]),
            paddle.uniform([38], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_81c9a3a6932090c70065f0d40921181c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347f98521c547dd9f0da4f650873cd6a
    def get_inputs(self):
        return [
            paddle.to_tensor([32.0, 96.0, 160.0, 224.0, 288.0, 352.0, 416.0, 480.0, 544.0, 608.0, 672.0, 736.0, 800.0], dtype='float32').reshape([13]),
            paddle.to_tensor([32.0, 96.0, 160.0, 224.0, 288.0, 352.0, 416.0, 480.0, 544.0, 608.0, 672.0, 736.0, 800.0, 864.0, 928.0, 992.0, 1056.0, 1120.0, 1184.0], dtype='float32').reshape([19]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e8831a72b482e8bbd3e6b5269a266c4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347f98521c547dd9f0da4f650873cd6a
    def get_inputs(self):
        return [
            paddle.to_tensor([64.0, 192.0, 320.0, 448.0, 576.0, 704.0, 832.0], dtype='float32').reshape([7]),
            paddle.to_tensor([64.0, 192.0, 320.0, 448.0, 576.0, 704.0, 832.0, 960.0, 1088.0, 1216.0], dtype='float32').reshape([10]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cebcd250715cdd335e3553efe20aa163(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347f98521c547dd9f0da4f650873cd6a
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_daf571bbc7e88d8c993e0389c8dcc7b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347f98521c547dd9f0da4f650873cd6a
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_147a5e72a799a7187a7370209be7f9cc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347f98521c547dd9f0da4f650873cd6a
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0, 544.0], dtype='float32').reshape([18]),
            paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0, 544.0], dtype='float32').reshape([18]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_53fe7a59315e7e2d76bed3b1e1236af0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_347f98521c547dd9f0da4f650873cd6a
    def get_inputs(self):
        return [
            paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0, 560.0], dtype='float32').reshape([18]),
            paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0, 560.0], dtype='float32').reshape([18]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e10ffb95a0044f7a91021bb80c3f02b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.meshgrid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[96], dtype='float32'),
            paddle.static.InputSpec(shape=[96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_303508e87a20446e37632b4dc11cf9fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e10ffb95a0044f7a91021bb80c3f02b4
    def get_inputs(self):
        return [
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

class PrimitiveOp_2a518985e188178585bc86a30b14112b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.meshgrid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[48], dtype='float32'),
            paddle.static.InputSpec(shape=[48], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1b68c038c53aa082a159149c6019aac2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a518985e188178585bc86a30b14112b
    def get_inputs(self):
        return [
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
            paddle.uniform([48], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ebc151b28b9d28c70ff68348541e7f22(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.meshgrid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[24], dtype='float32'),
            paddle.static.InputSpec(shape=[24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ded52bd4f620b251af76535b49d43355(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebc151b28b9d28c70ff68348541e7f22
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0, 544.0, 576.0, 608.0, 640.0, 672.0, 704.0, 736.0], dtype='float32').reshape([24]),
            paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0, 544.0, 576.0, 608.0, 640.0, 672.0, 704.0, 736.0], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_926597e9f45a07b7c7b66ee1f23d67f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebc151b28b9d28c70ff68348541e7f22
    def get_inputs(self):
        return [
            paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0, 560.0, 592.0, 624.0, 656.0, 688.0, 720.0, 752.0], dtype='float32').reshape([24]),
            paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0, 560.0, 592.0, 624.0, 656.0, 688.0, 720.0, 752.0], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5ea282e283312b9efb7d0df339ba5033(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.meshgrid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[64], dtype='float32'),
            paddle.static.InputSpec(shape=[64], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8fc32478e9053084a3401ace9f6a0bcd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5ea282e283312b9efb7d0df339ba5033
    def get_inputs(self):
        return [
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

class PrimitiveOp_860838ffe5b5e88a510b1064fa819f49(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.meshgrid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[32], dtype='float32'),
            paddle.static.InputSpec(shape=[32], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e03c1164d3ceb4c06fdf3820947d5a10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_860838ffe5b5e88a510b1064fa819f49
    def get_inputs(self):
        return [
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

class PrimitiveOp_51be175592073f55c60e959694a783a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.meshgrid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[16], dtype='float32'),
            paddle.static.InputSpec(shape=[16], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e27792f58824ae355f3d712669309424(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51be175592073f55c60e959694a783a7
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0], dtype='float32').reshape([16]),
            paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0], dtype='float32').reshape([16]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9727d38bf777e8b1e5a90265ffb42f83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_51be175592073f55c60e959694a783a7
    def get_inputs(self):
        return [
            paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0], dtype='float32').reshape([16]),
            paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0], dtype='float32').reshape([16]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_225cb7e37bbc5fbb0057d09ad0e84b76(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.meshgrid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[80], dtype='float32'),
            paddle.static.InputSpec(shape=[80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e183951898bc2dd707e25b17be5929a3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_225cb7e37bbc5fbb0057d09ad0e84b76
    def get_inputs(self):
        return [
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
            paddle.uniform([80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_eb3e46f0a8eecf9ace4759e724471257(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.meshgrid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[40], dtype='float32'),
            paddle.static.InputSpec(shape=[40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_48510ba22ef15e2a9382fe0bd783c3b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_eb3e46f0a8eecf9ace4759e724471257
    def get_inputs(self):
        return [
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
            paddle.uniform([40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2d55e6e387bfa30b495babf668f04f3c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.meshgrid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[20], dtype='float32'),
            paddle.static.InputSpec(shape=[20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_824260e8588c52c62fb1c1527997addb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d55e6e387bfa30b495babf668f04f3c
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0, 544.0, 576.0, 608.0], dtype='float32').reshape([20]),
            paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0, 544.0, 576.0, 608.0], dtype='float32').reshape([20]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8c452bda5d2efb9aa61d821f06db8059(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2d55e6e387bfa30b495babf668f04f3c
    def get_inputs(self):
        return [
            paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0, 560.0, 592.0, 624.0], dtype='float32').reshape([20]),
            paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0, 560.0, 592.0, 624.0], dtype='float32').reshape([20]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5e92d0768e2b0bdea542d790026fb27d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.meshgrid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[14], dtype='float32'),
            paddle.static.InputSpec(shape=[14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d28db764fddce80b0a7b5778fb06d448(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e92d0768e2b0bdea542d790026fb27d
    def get_inputs(self):
        return [
            paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0], dtype='float32').reshape([14]),
            paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0], dtype='float32').reshape([14]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_823024acd8d07f267a1b9720487e1ea1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.meshgrid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[28], dtype='float32'),
            paddle.static.InputSpec(shape=[28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5001e514496bf5ca2f6bb15df8d4ec12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_823024acd8d07f267a1b9720487e1ea1
    def get_inputs(self):
        return [
            paddle.to_tensor([8.0, 24.0, 40.0, 56.0, 72.0, 88.0, 104.0, 120.0, 136.0, 152.0, 168.0, 184.0, 200.0, 216.0, 232.0, 248.0, 264.0, 280.0, 296.0, 312.0, 328.0, 344.0, 360.0, 376.0, 392.0, 408.0, 424.0, 440.0], dtype='float32').reshape([28]),
            paddle.to_tensor([8.0, 24.0, 40.0, 56.0, 72.0, 88.0, 104.0, 120.0, 136.0, 152.0, 168.0, 184.0, 200.0, 216.0, 232.0, 248.0, 264.0, 280.0, 296.0, 312.0, 328.0, 344.0, 360.0, 376.0, 392.0, 408.0, 424.0, 440.0], dtype='float32').reshape([28]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_df0140c12eddf10b145febdcb9f6bbca(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.meshgrid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[56], dtype='float32'),
            paddle.static.InputSpec(shape=[56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_388334b82a4d4b96782043f239b94ca3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_df0140c12eddf10b145febdcb9f6bbca
    def get_inputs(self):
        return [
            paddle.uniform([56], dtype='float32', min=0, max=0.5),
            paddle.uniform([56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8655010fe322357084d8f348018e4b46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebc151b28b9d28c70ff68348541e7f22
    def get_inputs(self):
        return [
            paddle.to_tensor([8.0, 24.0, 40.0, 56.0, 72.0, 88.0, 104.0, 120.0, 136.0, 152.0, 168.0, 184.0, 200.0, 216.0, 232.0, 248.0, 264.0, 280.0, 296.0, 312.0, 328.0, 344.0, 360.0, 376.0], dtype='float32').reshape([24]),
            paddle.to_tensor([8.0, 24.0, 40.0, 56.0, 72.0, 88.0, 104.0, 120.0, 136.0, 152.0, 168.0, 184.0, 200.0, 216.0, 232.0, 248.0, 264.0, 280.0, 296.0, 312.0, 328.0, 344.0, 360.0, 376.0], dtype='float32').reshape([24]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d5f5359b95c2738974458bb657914ea2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.meshgrid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[68], dtype='float32'),
            paddle.static.InputSpec(shape=[68], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6d80c10aaa1f71bac6845b4bd0b51a92(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d5f5359b95c2738974458bb657914ea2
    def get_inputs(self):
        return [
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
            paddle.uniform([68], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e79cd4978874c2399ae5eddbe7ad80e8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.meshgrid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[34], dtype='float32'),
            paddle.static.InputSpec(shape=[34], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_edeb0d2c3514d0027f9ace13ce2e65f9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e79cd4978874c2399ae5eddbe7ad80e8
    def get_inputs(self):
        return [
            paddle.uniform([34], dtype='float32', min=0, max=0.5),
            paddle.uniform([34], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_946587d37f5711dd2fcb11076eb62e9b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.meshgrid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[17], dtype='float32'),
            paddle.static.InputSpec(shape=[17], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b656a543285aa38ba195933818d07ac1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_946587d37f5711dd2fcb11076eb62e9b
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0], dtype='float32').reshape([17]),
            paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0], dtype='float32').reshape([17]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6479d4c73a7258411ed6c1289cd84ef7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_946587d37f5711dd2fcb11076eb62e9b
    def get_inputs(self):
        return [
            paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0], dtype='float32').reshape([17]),
            paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0], dtype='float32').reshape([17]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_483cd7219618f378389eb80aa8661b90(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.meshgrid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100], dtype='float32'),
            paddle.static.InputSpec(shape=[152], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8bdd2156a0f9a0f8cb53c4454881e75d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_483cd7219618f378389eb80aa8661b90
    def get_inputs(self):
        return [
            paddle.uniform([100], dtype='float32', min=0, max=0.5),
            paddle.uniform([152], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b27661ee446b50289ad99e1b28cbacc1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.meshgrid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[50], dtype='float32'),
            paddle.static.InputSpec(shape=[76], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_39e5719c140c50c391306f086b63ac51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b27661ee446b50289ad99e1b28cbacc1
    def get_inputs(self):
        return [
            paddle.uniform([50], dtype='float32', min=0, max=0.5),
            paddle.uniform([76], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d4077cccae8920a4d0576cf43fee99a7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.meshgrid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[25], dtype='float32'),
            paddle.static.InputSpec(shape=[38], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ba72a0eb5c026b6d5a10b6a908f6f1c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d4077cccae8920a4d0576cf43fee99a7
    def get_inputs(self):
        return [
            paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0, 560.0, 592.0, 624.0, 656.0, 688.0, 720.0, 752.0, 784.0], dtype='float32').reshape([25]),
            paddle.uniform([38], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_adb5610e37367aa32d14186a8431d771(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.meshgrid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[13], dtype='float32'),
            paddle.static.InputSpec(shape=[19], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c9d71f2b90b4807c3470a7e550aa2f36(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_adb5610e37367aa32d14186a8431d771
    def get_inputs(self):
        return [
            paddle.to_tensor([32.0, 96.0, 160.0, 224.0, 288.0, 352.0, 416.0, 480.0, 544.0, 608.0, 672.0, 736.0, 800.0], dtype='float32').reshape([13]),
            paddle.to_tensor([32.0, 96.0, 160.0, 224.0, 288.0, 352.0, 416.0, 480.0, 544.0, 608.0, 672.0, 736.0, 800.0, 864.0, 928.0, 992.0, 1056.0, 1120.0, 1184.0], dtype='float32').reshape([19]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_49f5092f2d7d70788bf5d268126a512e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.meshgrid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[7], dtype='float32'),
            paddle.static.InputSpec(shape=[10], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_08803e32acbd12ddaa87014ae7d632d6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49f5092f2d7d70788bf5d268126a512e
    def get_inputs(self):
        return [
            paddle.to_tensor([64.0, 192.0, 320.0, 448.0, 576.0, 704.0, 832.0], dtype='float32').reshape([7]),
            paddle.to_tensor([64.0, 192.0, 320.0, 448.0, 576.0, 704.0, 832.0, 960.0, 1088.0, 1216.0], dtype='float32').reshape([10]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8961bf63d82949f7e60ca41b81143d50(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.meshgrid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[72], dtype='float32'),
            paddle.static.InputSpec(shape=[72], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ae3ec571dea4c144b88f4c413c16e2a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8961bf63d82949f7e60ca41b81143d50
    def get_inputs(self):
        return [
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
            paddle.uniform([72], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_742a536927cbd62125e6bf36d963f90d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.meshgrid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[36], dtype='float32'),
            paddle.static.InputSpec(shape=[36], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9e019057ac306f797364030fee4935a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_742a536927cbd62125e6bf36d963f90d
    def get_inputs(self):
        return [
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
            paddle.uniform([36], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_902930176f158f84013299fa5f90aac6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0_0, arg_0_1):
        input_0 = [arg_0_0, arg_0_1]
        return paddle._C_ops.meshgrid(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[18], dtype='float32'),
            paddle.static.InputSpec(shape=[18], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_16a886981c9f8bef2afb5595c74e9f6b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_902930176f158f84013299fa5f90aac6
    def get_inputs(self):
        return [
            paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0, 544.0], dtype='float32').reshape([18]),
            paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0, 544.0], dtype='float32').reshape([18]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a6fc3b724aada08548e3bacdba89f73a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_902930176f158f84013299fa5f90aac6
    def get_inputs(self):
        return [
            paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0, 560.0], dtype='float32').reshape([18]),
            paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0, 560.0], dtype='float32').reshape([18]),
        ]


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