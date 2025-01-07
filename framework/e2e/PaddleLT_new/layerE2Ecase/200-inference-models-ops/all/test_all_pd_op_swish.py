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
class PrimitiveOp_cb60f9c200b2237364d4f861ea777605(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e4b2497f64b1d19f05319266a53d3049(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 96, 96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d1b463e4e2c75b277142b3b5c406e964(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.084073543548584]], [[2.1860599517822266]], [[4.302435398101807]], [[5.035457134246826]], [[3.3365373611450195]], [[6.1132659912109375]], [[3.9774508476257324]], [[2.044682025909424]]]], dtype='float32').reshape([1, 8, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_97f40fb9308b821e733e352d237cd143(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 96, 96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_386100d4a1bc5aab091e511ead60c242(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 48, 48], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f41f35ba2d46699d691ead397eb393da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[120.87162780761719]], [[93.590087890625]], [[121.81399536132812]], [[102.97189331054688]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_28a2cdca5623b01f5daa417e8cb362b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 48, 48], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9e8a0fd6eecf341eb183c4fc41a632cf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[38169.98046875]], [[29501.55859375]], [[29287.126953125]], [[28427.654296875]], [[33067.99609375]], [[28659.333984375]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8082188590dd89960ac8badec558bc5e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 24, 24], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_503a585e2fa8db450961e43b6e7fc605(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[12704672.0]], [[13112854.0]], [[11213443.0]], [[10963389.0]], [[11479708.0]], [[12300290.0]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_075738bb0d5742f15f96bd739c473061(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 24, 24], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fe444b4794227c853a5e15daf10aeb35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[20794421248.0]], [[21243885568.0]], [[14570157056.0]], [[21341642752.0]], [[17447108608.0]], [[17596098560.0]], [[17947897856.0]], [[17306509312.0]], [[20433899520.0]], [[17335146496.0]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f6bfcb4a1960faacedd38b8005b84bce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 12, 12], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3d14307ca588e24d49fb6f5cd28e9b0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4136647524352.0]], [[4266075881472.0]], [[4329928916992.0]], [[4151151689728.0]], [[4188881813504.0]], [[3968265617408.0]], [[4221484138496.0]], [[4134500564992.0]], [[4134837682176.0]], [[4516995923968.0]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a577c8778766e40194d2c71fe5a6df06(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 12, 12], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9246b1cef3a0ce384a155895120047ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[9250891404148736.0]], [[9848080836853760.0]], [[9425116752510976.0]], [[9266629238063104.0]], [[9590617948553216.0]], [[7944887412457472.0]], [[9306087102611456.0]], [[8054471355531264.0]], [[8635626601578496.0]], [[7865451455447040.0]], [[9412944815194112.0]], [[8805285325963264.0]], [[9036523143954432.0]], [[8954717438738432.0]], [[1.0360265350578176e+16]], [[9684357757272064.0]], [[9426586705068032.0]], [[8379586421194752.0]], [[9036406106095616.0]], [[8308818781929472.0]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3ff370bc15a919b1e52a493094855da3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.2921134094670103e+19]], [[1.2300644498744017e+19]], [[1.3194345141986394e+19]], [[1.2899814564207526e+19]], [[1.1796499725711049e+19]], [[1.2174788900270637e+19]], [[1.2061600775260865e+19]], [[1.0648668959940805e+19]], [[1.2359384807946322e+19]], [[1.1939772687880028e+19]], [[1.2545779516645048e+19]], [[1.2074128610747744e+19]], [[1.2337226350111752e+19]], [[1.266942949479311e+19]], [[1.2902584233997894e+19]], [[1.0900813364468908e+19]], [[1.3282225807859646e+19]], [[1.1602068686524908e+19]], [[1.173952633169458e+19]], [[1.2873751740582724e+19]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7486a3e8d1f74c8f48e9a6f706cf5efe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.939725129306423e+22]], [[1.838752849400907e+22]], [[2.083941198013932e+22]], [[1.9013116762847665e+22]], [[1.6683789733377048e+22]], [[1.8566440744605805e+22]], [[1.7757571733531927e+22]], [[1.9442102641753214e+22]], [[1.7698620740309554e+22]], [[1.835695805973848e+22]], [[1.704057165125678e+22]], [[1.9874858033546873e+22]], [[1.8094821539827753e+22]], [[1.8804735208089232e+22]], [[1.730277122156229e+22]], [[1.9713656188484773e+22]], [[1.898762638895675e+22]], [[1.5995058745962905e+22]], [[1.838375560342124e+22]], [[1.9749617431509327e+22]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2b5993dda68991e0dc322e967fb82d4b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.488690339274873e+26]], [[1.3495483934012892e+26]], [[1.535659346801632e+26]], [[1.47407424607551e+26]], [[1.5288283330036965e+26]], [[1.453183308412034e+26]], [[1.4048603098702648e+26]], [[1.3664479168158164e+26]], [[1.540907629958043e+26]], [[1.4500199685045538e+26]], [[1.5328436357862209e+26]], [[1.5949385122848198e+26]], [[1.3089760710832502e+26]], [[1.5025457809823566e+26]], [[1.556153126065201e+26]], [[1.5752117486398355e+26]], [[1.845169149047012e+26]], [[1.3481031832368345e+26]], [[1.2154525543690687e+26]], [[1.7846199252343493e+26]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2e83d4fce6ebfae58262238e83813c6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 12, 12], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_22bb6473146b797234d4693a193a998d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.1985650231090769e+30]], [[1.395382680014162e+30]], [[1.3322687430967235e+30]], [[1.3293154884351324e+30]], [[1.1862243083424508e+30]], [[1.310513669625576e+30]], [[1.2080782114993519e+30]], [[1.3944915505693786e+30]], [[1.3278696131548733e+30]], [[1.2736695396526357e+30]], [[1.310833430504864e+30]], [[1.24965785502345e+30]], [[1.2040855828643471e+30]], [[1.284046354425298e+30]], [[1.1335086446309725e+30]], [[1.2995287651656476e+30]], [[1.2816282005546137e+30]], [[1.3003801511741112e+30]], [[1.2974355101089848e+30]], [[1.2641011940218409e+30]], [[1.2597174778841908e+30]], [[1.2594038371918645e+30]], [[1.2292064569330293e+30]], [[1.2706612788662522e+30]], [[1.2443006513695552e+30]], [[1.3341079726155397e+30]], [[1.2467568864035572e+30]], [[1.3736818570891697e+30]]]], dtype='float32').reshape([1, 28, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0d89313fc4dcf95539c3be835f203ed2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.1391893147635776e+34]], [[1.157838633867404e+34]], [[1.1935227506758208e+34]], [[1.1442386245958148e+34]], [[1.1695650208895348e+34]], [[1.0834960017181794e+34]], [[1.1631392455276162e+34]], [[1.0310266559170922e+34]], [[1.0896489350954396e+34]], [[1.116867894121219e+34]], [[1.1071952496242627e+34]], [[1.067699886816898e+34]], [[1.1161365191460092e+34]], [[1.120800210656009e+34]], [[1.1540729440619018e+34]], [[1.2425662212121897e+34]], [[1.0848501843271537e+34]], [[1.1433559733478044e+34]], [[1.041805276045146e+34]], [[1.123234000773244e+34]], [[1.1930481244647588e+34]], [[1.2710680575006806e+34]], [[1.1998231227117598e+34]], [[1.1085177409682313e+34]], [[1.1112477701368673e+34]], [[1.239194567721192e+34]], [[1.1546212277053013e+34]], [[1.0397557425161052e+34]]]], dtype='float32').reshape([1, 28, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_435464b0c5d1e2ce5463ad5fc173c5ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.1169612211333231e+38]], [[1.2678051521894092e+38]], [[1.1245902466816406e+38]], [[1.1046493928557145e+38]], [[1.1644399081263192e+38]], [[1.0964417101614047e+38]], [[1.0214675802374103e+38]], [[1.1379036203454776e+38]], [[1.271786082134366e+38]], [[1.1065979239463373e+38]], [[1.1815876713257265e+38]], [[1.3380350236830135e+38]], [[1.3105300480195015e+38]], [[1.219681383158969e+38]], [[1.1979884333794313e+38]], [[1.1445890067869852e+38]], [[1.2567041837651386e+38]], [[1.221420295546338e+38]], [[1.2002837936742766e+38]], [[1.3002880396419455e+38]], [[1.3507934706200945e+38]], [[1.2858157290933559e+38]], [[1.2282487743875994e+38]], [[1.2222447754967265e+38]], [[1.3418792501872416e+38]], [[1.2436865276333229e+38]], [[1.2064914279575702e+38]], [[1.1935161620618261e+38]]]], dtype='float32').reshape([1, 28, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aba219735175354d35c4fff940853712(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 6, 6], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1881558ceddc82769ea5d5a5bbb0e34e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 28, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ac0f4694975c8e128ddaf1c9ffdecfc1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 1152, 6, 6], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f2e7beee64e846114758588dcb9fc00a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_af8ddb0337bdb5393a51d1c3a36fa10a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 6, 6], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a313ee4fed4cf271a5112f50bb8ddf88(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 112, 112], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ce71d0658d0d6296eafb480a58c13f6d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 112, 112], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_75765ae7916920c84bcb0b114b4f09e6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 162, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7f85100d68e7fa59230f2f759c4b5e11(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 228, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_07a3c6f25fdd18a32a7462c82a8b1c4f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eb0a224864c8e2f18d65b1221fdaa5db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 366, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b0100da7452e57244315bf785a73bede(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 432, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1f4d9cbe05a4d464de117f28f5c091e7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 504, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0f3bc4e1909da1977a6c505a31379129(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 570, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_56327f6d550e008b1029e34d8132703a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 636, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7284dd3ee7b7e76713bad812a89be39a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 702, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_97561ed07fed7d3636ed22df28246365(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4376b1fffc215b9af535c4ae1262e580(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 840, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_beab909d0eb20820023a9b8326423f2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 906, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6ed608ebde16c040d022bb240bf0a11a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 972, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4f18e66da62f4dfb380934b26a7880fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 1044, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_187c89aa996c36a61a6a20521cfa35f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a7f3374ece94d8aa2197285ccc6281fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 320, 320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_860f017d6b136e8584ff686711a056e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 320, 320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d302e02db1eca90cbd40f16885e9059d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 160, 160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_87d95732e7be8db5b34ec0bf43236518(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 160, 160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3fd498e78dfb1430e4a140e8984fef3a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 160, 160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d7ef78ad82e5d5a5a55db93c66d3f3cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 80, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c0458ec0b888a5fca12cb193f756fb19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c771ddf27147f9316e98a3dd4c95d968(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 80, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b964b234b0b58fdf63a930e3dc9cdb5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0f16744fa8a74cfad4e7203413765d04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f15fe6d4aa848f4608416bdad89575ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a1dbd64b80f5c66a72e717d29a5f56be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4476535a2df32edd71c5a96743591dec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_29b0e5ceb0df64deb47a82410b99c10c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dd37862d66ae7629153669757b0a839a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8456f9525585f452ae5381e1d0b987be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0b2c086d991baa15f1d85e1ffb50380e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[13980526.0]], [[16786818.0]], [[13421100.0]], [[14328887.0]], [[15565914.0]], [[14893876.0]], [[17302300.0]], [[13505882.0]], [[13839678.0]], [[16651823.0]], [[16340616.0]], [[14594916.0]], [[14825827.0]], [[15898333.0]], [[13663712.0]], [[13155664.0]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cf88881ffb0c0f4b11f9c90be63da20d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e1baaafa1969287cf0cf5bc431439ad5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6589385728.0]], [[7747439616.0]], [[6090580992.0]], [[6799010304.0]], [[7744267264.0]], [[6030663680.0]], [[6990911488.0]], [[7763949056.0]], [[7313573888.0]], [[6446460928.0]], [[5292515840.0]], [[8529415168.0]], [[7481018880.0]], [[8095347712.0]], [[9393774592.0]], [[8455483392.0]], [[7120087040.0]], [[5546908672.0]], [[8256373760.0]], [[7469214720.0]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_277d7abe5d2cafeed3a97aa60b79f4af(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1721964691456.0]], [[1814429827072.0]], [[1740761464832.0]], [[2057639559168.0]], [[2237306765312.0]], [[2160493592576.0]], [[2041806585856.0]], [[2195535167488.0]], [[2102716530688.0]], [[2022248677376.0]], [[1794602827776.0]], [[1983108349952.0]], [[2021920210944.0]], [[2177277624320.0]], [[2015597953024.0]], [[1641698951168.0]], [[2085865127936.0]], [[2026762928128.0]], [[1806169276416.0]], [[1769575809024.0]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_548630302890bd69c6615598dca4ff30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[679419131723776.0]], [[771216407265280.0]], [[628409482870784.0]], [[745873751408640.0]], [[826519412801536.0]], [[682559960776704.0]], [[530898692866048.0]], [[573434639679488.0]], [[435650075557888.0]], [[632708678025216.0]], [[588598424371200.0]], [[632598418161664.0]], [[607354412335104.0]], [[593554481086464.0]], [[897616489480192.0]], [[717797751521280.0]], [[742576088940544.0]], [[853886675976192.0]], [[502872286429184.0]], [[766375979122688.0]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_35ebec7af544d7e114d49386d6cc88ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_831fdaaf66a2d80ca122742e7fd58010(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.1746728054882304e+17]], [[4.908796806399263e+17]], [[4.6199805279889e+17]], [[4.765287517949788e+17]], [[4.533135945666396e+17]], [[4.772737396422738e+17]], [[4.879870029862011e+17]], [[4.719863256632525e+17]], [[4.7954024542398054e+17]], [[4.660497875069829e+17]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a35a2665489fe61029f1bbef0851b509(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ae37d8c1c7de11e12a29c0ef3fdc6ccf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.0448465533682665e+21]], [[2.5647132934946706e+21]], [[3.3636206236677817e+21]], [[3.7591757195894063e+21]], [[3.0866759867072837e+21]], [[3.1825041425284265e+21]], [[2.6460874277867445e+21]], [[2.8330625290913553e+21]], [[3.354648327310153e+21]], [[3.096355066731433e+21]], [[3.383155268526478e+21]], [[3.679187005132702e+21]], [[3.2077096637429124e+21]], [[3.2808312331928066e+21]], [[3.4659677399495433e+21]], [[2.705366902306939e+21]], [[3.4397756489416867e+21]], [[2.646977170188127e+21]], [[3.266049856265823e+21]], [[2.8648283875880364e+21]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8d5488601c5950e8076b167c261b2ea7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.3341877434942511e+25]], [[1.3362224193655813e+25]], [[1.1710643385719422e+25]], [[8.747087481165137e+24]], [[1.3129383625349928e+25]], [[1.327094278645007e+25]], [[7.299406605925989e+24]], [[9.356594999015858e+24]], [[1.2165826019110242e+25]], [[7.734146788118625e+24]], [[1.4094151799169451e+25]], [[8.013293872196292e+24]], [[1.224833484658743e+25]], [[1.2853865361708011e+25]], [[1.2648768696488985e+25]], [[1.2101833110997039e+25]], [[7.842301201544289e+24]], [[8.735102862124748e+24]], [[1.1806317423857716e+25]], [[1.3840458339609742e+25]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_395048e622ae7f60cdb32488f8413be2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.525883328311274e+28]], [[2.8162963527365777e+28]], [[2.8812966578350645e+28]], [[2.4425463102781286e+28]], [[3.1385872398678445e+28]], [[3.0035268501028e+28]], [[2.6485623815132066e+28]], [[2.8105183012264625e+28]], [[2.683697496500729e+28]], [[2.2281503997191984e+28]], [[2.540410744322526e+28]], [[2.8349820485543204e+28]], [[3.009070436117041e+28]], [[2.686297631486197e+28]], [[2.9865478175319665e+28]], [[2.9745744934329747e+28]], [[2.831993262807312e+28]], [[2.7763814945134185e+28]], [[2.9743386112271554e+28]], [[3.142320270572553e+28]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ea32e1a74d453c0eac9b88baafa14b97(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fd62e34de40628c4a98f4de802faae99(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 360, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9b9fde159d356366a7118507e87e2da7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_102bab94e6c5131693993cf258ba1cd7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6af0effdfa975472eab4d129bb5d7c96(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_336b8da4c393d4ad70de5105e4fd4c62(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 1200, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3892eff1da8f341a325b526594f8d631(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d466ead746e3ac0bca6e59352446b298(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_13e7355d01b273bc100e4a98fb400739(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 56, 56], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_26345792f25d3358c8278363fd2505fe(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 28, 28], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0ec7834994122efcdb1cf52ead3c89d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_87d1450e307fb6326f8cf16e542c8b76(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 28, 28], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ed56d4b998c9b7f7e336fb63133841c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9fc8f81c8a5d24ea255f936ebb79a489(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0329c1af0b33b5634954bc90d1e4c662(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 10, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_180c694c1cff47afefa8f8fcf9ce11f3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0fd9abadf5c41225203ca33bb9ba251d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9bca7df20372a0bf42e81677b3b2202e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 360, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a8e45f6b10e91abe33bd1e77bc911745(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5e3213994895aae475a0251b62868228(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a6839c744977e1f68d39cf67e532c612(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ba3a14e12f4e3cc69f40990908c98e13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 1200, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5c85162b49e61d0c783efce084a789f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3ddf95edf39225de98ae5a876f7e3573(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 96, 96], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9297b7b1790c9b03bc3699fa42dca87f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.34375]], [[2.0859375]], [[2.3125]], [[2.07421875]], [[1.994140625]], [[2.078125]], [[2.15625]], [[1.9072265625]]]], dtype='float16').reshape([1, 8, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d37865472d01391a0a3a19a6f1112a38(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 96, 96], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a24b8d07330381128716201acaf71728(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 48, 48], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d296c8a4c0e102f8e36477ce1b70df14(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[109.1875]], [[89.0625]], [[108.1875]], [[90.5]]]], dtype='float16').reshape([1, 4, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6a2f0bf13735541651d5d169451e8ad8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 48, 48], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_068f83e04dabefeab060c89ae68c7ac4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[9880.0]], [[11872.0]], [[13104.0]], [[14544.0]], [[10264.0]], [[11248.0]]]], dtype='float16').reshape([1, 6, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_416c24a7d4e294d513fa88c0d034b97e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 24, 24], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e48f2900a50260824924a6c4d55f6987(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0ea7438f7e93d13c73aa8f638b2b0abc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 24, 24], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_73ee42af3a26591981763da69204fef6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 12, 12], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5d96dbf7e2bd105af259bbb110cb426c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 12, 12], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a84935f3400b3bfc930e99007c77574b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 12, 12], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e76edc0387c1d1013aea9829934a563f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 28, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ec3a0ee43bbfe5a5c6448bc9f3f09d5b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 6, 6], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e9479fb731a02e5003d885f416b958ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 1152, 6, 6], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e365650093283f1589b499bb9a665c2a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_04c83d3bd47dcd3591d968f25b5187e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 6, 6], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b6a841ef81dce952f91499dfdc10aca9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 112, 112], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_14c4e71078e6507a54e9685f98057f4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 112, 112], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cf0cee8921bf34a59fb0b90958587320(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 162, 56, 56], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_76ab5156587107c28908969da80e84b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 228, 56, 56], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_03709c2f9acc93fc68c204fa3a663e02(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 28, 28], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9b518df91e7c09e5ca00f9fb21cc8039(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 366, 28, 28], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c109e26d46030ad36ddff55763a4c869(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 432, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f5e045d51bd38888c824fce34b805520(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 504, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e265e537ff02ea129c13eecfa9279097(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 570, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6700b898a1f72bc3d9af422c40c61356(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 636, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_197dec4f49a5100a42191edf2ff7f043(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 702, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_05f11788c018776b11db7fad519360aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_06f48e9f4bb92e959dee3202e0b31d95(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 840, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_42915f81b4f0091b31dbdab3a6de2816(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 906, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9bef1b4450cb30012a6f398535e64245(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 972, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5d02c869b8c62cbd5728a19ea5287b40(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 1044, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b2f49e34dd1f2b0e7eb383499efa6a79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_956371fd17edbeecaefacbda81a8da35(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.020218372344971]], [[5.779989719390869]], [[5.272001266479492]], [[6.338002681732178]], [[5.078469276428223]], [[4.759707927703857]], [[6.7526044845581055]], [[5.370635509490967]]]], dtype='float32').reshape([1, 8, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_219417035c760190f5a04040e8bdc37f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6f3c41f2738dde80643499e94364b4fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[715.2088012695312]], [[646.95166015625]], [[684.4810791015625]], [[567.3255615234375]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8d0b7f03b92dbf1090564ec20999be9d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ed5f52938367721eac554835755f9724(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[25508.236328125]], [[25183.271484375]], [[26768.939453125]], [[31446.046875]], [[23898.638671875]], [[32278.58203125]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9094273acae911448a3e559f65fdc1a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_70937f8712b88032448c04e1c513d71a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[17797448.0]], [[17554908.0]], [[16649942.0]], [[18681426.0]], [[15140606.0]], [[15624559.0]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8aa9c05f7d059f6773210dd4e4324cd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[10291249152.0]], [[10166941696.0]], [[10204102656.0]], [[10139587584.0]], [[10066112512.0]], [[10081742848.0]], [[10857426944.0]], [[10663124992.0]], [[9413766144.0]], [[8894158848.0]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_13b5c3d238b82b048928dd8ac75186d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6818450898944.0]], [[7250532368384.0]], [[7546705805312.0]], [[7084813844480.0]], [[6699838603264.0]], [[7219813810176.0]], [[7028518944768.0]], [[7132170682368.0]], [[6570401333248.0]], [[7165505437696.0]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b7807f23c519fa942e262272976aa120(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6531123765051392.0]], [[7142669125943296.0]], [[7095445960523776.0]], [[6605274227933184.0]], [[6777022453907456.0]], [[6994374642630656.0]], [[6674554197901312.0]], [[7101333286944768.0]], [[6636092228894720.0]], [[6957819102232576.0]], [[6728381949280256.0]], [[6979635925483520.0]], [[6489485667729408.0]], [[6680534939860992.0]], [[6601847917772800.0]], [[7085154145140736.0]], [[7319664191340544.0]], [[6287095534452736.0]], [[6285706112532480.0]], [[6858460939419648.0]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cd67adbe5f9bee86131fcecb7e88759a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.395463970395259e+18]], [[9.493493456497017e+18]], [[8.395826259476611e+18]], [[8.607407481032802e+18]], [[9.047475465178055e+18]], [[8.0678051072715e+18]], [[8.880835132139766e+18]], [[9.054044497398202e+18]], [[7.843933544740028e+18]], [[8.459766708922483e+18]], [[8.272340658316902e+18]], [[8.689773546336944e+18]], [[9.060855422176461e+18]], [[8.943138408771879e+18]], [[8.190451231303401e+18]], [[9.229058710993633e+18]], [[8.714407554601452e+18]], [[8.913166821310333e+18]], [[9.229562287319155e+18]], [[8.176931636328268e+18]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fd4ef5d9b0f01185cab08feeee43623e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.9019669278501798e+22]], [[2.5641613773603363e+22]], [[2.8760322738560227e+22]], [[3.033274554125738e+22]], [[3.0006432726656624e+22]], [[2.933711900903589e+22]], [[2.7823495203073684e+22]], [[3.175701567881274e+22]], [[2.7040900192225886e+22]], [[2.7544042342595718e+22]], [[2.737521590336448e+22]], [[2.5323718186706162e+22]], [[3.0144308177448758e+22]], [[2.935909207161783e+22]], [[2.631587919901439e+22]], [[2.7230533261735575e+22]], [[2.9178290560977414e+22]], [[2.7512346008418284e+22]], [[2.8170029180001335e+22]], [[2.8096375059895504e+22]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_54cc471dd99a6c285db42bedd85713e0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_98b5e6e10a19562ffb92c960d493dfca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.4477805337740055e+26]], [[1.617800300085131e+26]], [[1.66807782339523e+26]], [[1.6584626425142497e+26]], [[1.6625373438126914e+26]], [[1.5483328133825926e+26]], [[1.5240972965511122e+26]], [[1.6539834041182715e+26]], [[1.4677878723963509e+26]], [[1.644693992737633e+26]], [[1.558735116833198e+26]], [[1.4568405598920876e+26]], [[1.3781457351027186e+26]], [[1.5635762803479025e+26]], [[1.5452214932934004e+26]], [[1.5204014913759445e+26]], [[1.5220516448670582e+26]], [[1.4656622540767373e+26]], [[1.5049836104454177e+26]], [[1.562080802805847e+26]], [[1.5175447363549695e+26]], [[1.4995752095504468e+26]], [[1.6197503054011628e+26]], [[1.6250906378105017e+26]], [[1.6204479612620305e+26]], [[1.4511868173009363e+26]], [[1.3681237112811926e+26]], [[1.6216560385314177e+26]]]], dtype='float32').reshape([1, 28, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_53e6a55b4d6b1224728de9c68e0b591c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.2131591756034647e+30]], [[1.250661565685185e+30]], [[1.2212379735087669e+30]], [[1.187104179665539e+30]], [[1.280835598564129e+30]], [[1.194373677292473e+30]], [[1.2478533821219477e+30]], [[1.2433682673311774e+30]], [[1.1825945841269216e+30]], [[1.2473470688771203e+30]], [[1.1833245486483776e+30]], [[1.1905373778775171e+30]], [[1.1557524263646993e+30]], [[1.2314163733312848e+30]], [[1.1649871084692806e+30]], [[1.1837959541601636e+30]], [[1.2203029449451587e+30]], [[1.2320648109177806e+30]], [[1.2233648518147877e+30]], [[1.2497246481749837e+30]], [[1.19152847037601e+30]], [[1.2204810348299607e+30]], [[1.2190449065141223e+30]], [[1.2439414492854022e+30]], [[1.2545820365603315e+30]], [[1.1796272001448137e+30]], [[1.2065575339340041e+30]], [[1.301488433919243e+30]]]], dtype='float32').reshape([1, 28, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f28cbdd5289d2bdcd721bc80dc86bfaf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_30bc514b6d0c1cd12c3e90ff767eec63(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.9691154585608501e+34]], [[1.8652579837889874e+34]], [[1.929610316027175e+34]], [[1.852574792910493e+34]], [[1.7480766841683001e+34]], [[1.913484413899428e+34]], [[2.0427533314457096e+34]], [[1.820106101560116e+34]], [[1.6903744420271496e+34]], [[1.8102308062787287e+34]], [[1.8400894248502764e+34]], [[1.795956119685729e+34]], [[1.7846258734761696e+34]], [[1.8624232248930279e+34]], [[1.8319388276316214e+34]], [[1.769947370842355e+34]], [[1.905243942233921e+34]], [[1.9601901584656104e+34]], [[1.8708710515151152e+34]], [[1.8222624693145473e+34]], [[1.938899075317949e+34]], [[1.7215445342763162e+34]], [[1.8520508966858674e+34]], [[1.8999314463493317e+34]], [[1.771561768447587e+34]], [[1.858814382090503e+34]], [[1.7961267078231425e+34]], [[1.8316959437959136e+34]]]], dtype='float32').reshape([1, 28, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7573d57f6043cb7ed3818fb2532d72ef(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cb60f9c200b2237364d4f861ea777605
    def get_inputs(self):
        return [
            paddle.uniform([1, 1152, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_15ba73761a3c19e8a19b5033cf29fa23(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.73046875]], [[4.796875]], [[5.6171875]], [[4.68359375]], [[5.72265625]], [[3.890625]], [[3.013671875]], [[4.1328125]]]], dtype='float16').reshape([1, 8, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4cd744e78bb1d9c9b3f7f3d4aee5e9f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 56, 56], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ddab715217ea4b16f8f7dbfea3025fd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[579.5]], [[656.0]], [[689.5]], [[645.0]]]], dtype='float16').reshape([1, 4, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1a3ffb4c72827fcf7a0a94c1f17c93cb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 56, 56], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0cbfeba32f7407aee502d1ca69c8c560(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 28, 28], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_164b84a3177d009f46793f85d26b8f46(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_32aa854ae4b34196f5d5d4b9e1c2722b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_caf72d9b1a3dbbe94f1dbc2829a7cb87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 1152, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_29fb4c5dc6dc3c263a2cf62f917669b6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 320, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7bbba99488ffc1abc0c8015c8e4519c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 320, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ce84fe13913165f1951e29ffdc2636fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 160, 160], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_01fd8b59b3ff87468c134b1ffa76dc17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 160, 160], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e1cc232f60da72bfaf8ecabef8fef83b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 160, 160], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cd5c2569656e783d0b7291238ce1a230(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 80, 80], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_2aeef4a9ca87c23856f250680b4ed183(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80, 80], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7bae603c02e442a85de29342119501e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 80, 80], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3c982643560eddcf9eb87e453d78bfee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ec01b0c56a60ccf9f59bebe7c4b305e5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dd2966ce08f91cdd7f73e0d4b4ea4227(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_80d66bccef3f141a0564ea3060732436(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_872196f136918023d71f0dfc40116de7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3f6dce2a64c9f00c6e6e2bdf1154ef31(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d466ead746e3ac0bca6e59352446b298
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_30cee020f0c23037ff589bb4984c284d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 96, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e2684c983afd6a4b818be4bee21daaf9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_30cee020f0c23037ff589bb4984c284d
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 96, 96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_29007ee7b2e66f3b636fcfe61bbce4a2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a3d28acbe8014ea0bd0c50b2319711b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29007ee7b2e66f3b636fcfe61bbce4a2
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4.084073543548584]], [[2.1860599517822266]], [[4.302435398101807]], [[5.035457134246826]], [[3.3365373611450195]], [[6.1132659912109375]], [[3.9774508476257324]], [[2.044682025909424]]]], dtype='float32').reshape([1, 8, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e3d9939d54ec203c7b968960029317e8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 96, 96], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_569a3a7a5478c8ccf559c03f80b5b6b1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e3d9939d54ec203c7b968960029317e8
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 96, 96], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ac77e68e6c5ab49d396aa9c9e2f8174b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 48, 48], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_38e9b4e2c477a1db7bab666948b9f349(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac77e68e6c5ab49d396aa9c9e2f8174b
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 48, 48], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_10b285d1685c7e1ec0629a08f4f1c2b5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3d39200d777bccc8a2c4f83234fd49a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10b285d1685c7e1ec0629a08f4f1c2b5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[120.87162780761719]], [[93.590087890625]], [[121.81399536132812]], [[102.97189331054688]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d1c69918082585ed516f637b12ba849a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 48, 48], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3c77c5770ebec4b6d59f2959c1851b19(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1c69918082585ed516f637b12ba849a
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 48, 48], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f8d07bd6ef969a87be053df3f69b43d8(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 6, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f05335d2712408151709906ce713ca82(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8d07bd6ef969a87be053df3f69b43d8
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[38169.98046875]], [[29501.55859375]], [[29287.126953125]], [[28427.654296875]], [[33067.99609375]], [[28659.333984375]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2a658ede0cce07de6004443e6b0407f7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 24, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c3dd45dd374a3b0b44a08a068338ba93(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a658ede0cce07de6004443e6b0407f7
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 24, 24], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fa8aa209505b618038b8e4239f8ededc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8d07bd6ef969a87be053df3f69b43d8
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[12704672.0]], [[13112854.0]], [[11213443.0]], [[10963389.0]], [[11479708.0]], [[12300290.0]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b56e6ce950d861d165eed15e3874aac4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, 24, 24], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_754b816cb64078a5c0ea52a1ff5119b0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b56e6ce950d861d165eed15e3874aac4
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 24, 24], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ff33d5afbedb70bb059ced37dbdb332d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 10, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c2ae274608417fa99d1b9cd38615ec03(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff33d5afbedb70bb059ced37dbdb332d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[20794421248.0]], [[21243885568.0]], [[14570157056.0]], [[21341642752.0]], [[17447108608.0]], [[17596098560.0]], [[17947897856.0]], [[17306509312.0]], [[20433899520.0]], [[17335146496.0]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f334fd59e8671ada5bd37c30d1b7aa3e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, 12, 12], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_def7db6db61a6350ad7333905c31bb48(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f334fd59e8671ada5bd37c30d1b7aa3e
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 12, 12], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cdc3037304effa3100805ba847b2e188(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff33d5afbedb70bb059ced37dbdb332d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[4136647524352.0]], [[4266075881472.0]], [[4329928916992.0]], [[4151151689728.0]], [[4188881813504.0]], [[3968265617408.0]], [[4221484138496.0]], [[4134500564992.0]], [[4134837682176.0]], [[4516995923968.0]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_40f5dd261d5360f4326d884ef14215a0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480, 12, 12], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a30aa871ea2708bc05d3fd98238b117e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40f5dd261d5360f4326d884ef14215a0
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 12, 12], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4f85ff14ecb1201f01ba8ba107f90885(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 20, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_87a1fa4b34d521a3276cbd3f34060b2b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f85ff14ecb1201f01ba8ba107f90885
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[9250891404148736.0]], [[9848080836853760.0]], [[9425116752510976.0]], [[9266629238063104.0]], [[9590617948553216.0]], [[7944887412457472.0]], [[9306087102611456.0]], [[8054471355531264.0]], [[8635626601578496.0]], [[7865451455447040.0]], [[9412944815194112.0]], [[8805285325963264.0]], [[9036523143954432.0]], [[8954717438738432.0]], [[1.0360265350578176e+16]], [[9684357757272064.0]], [[9426586705068032.0]], [[8379586421194752.0]], [[9036406106095616.0]], [[8308818781929472.0]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b87edfad345f60e3f63e263939392fb3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f85ff14ecb1201f01ba8ba107f90885
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.2921134094670103e+19]], [[1.2300644498744017e+19]], [[1.3194345141986394e+19]], [[1.2899814564207526e+19]], [[1.1796499725711049e+19]], [[1.2174788900270637e+19]], [[1.2061600775260865e+19]], [[1.0648668959940805e+19]], [[1.2359384807946322e+19]], [[1.1939772687880028e+19]], [[1.2545779516645048e+19]], [[1.2074128610747744e+19]], [[1.2337226350111752e+19]], [[1.266942949479311e+19]], [[1.2902584233997894e+19]], [[1.0900813364468908e+19]], [[1.3282225807859646e+19]], [[1.1602068686524908e+19]], [[1.173952633169458e+19]], [[1.2873751740582724e+19]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ed052f173999b31251273af626c8d23f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f85ff14ecb1201f01ba8ba107f90885
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.939725129306423e+22]], [[1.838752849400907e+22]], [[2.083941198013932e+22]], [[1.9013116762847665e+22]], [[1.6683789733377048e+22]], [[1.8566440744605805e+22]], [[1.7757571733531927e+22]], [[1.9442102641753214e+22]], [[1.7698620740309554e+22]], [[1.835695805973848e+22]], [[1.704057165125678e+22]], [[1.9874858033546873e+22]], [[1.8094821539827753e+22]], [[1.8804735208089232e+22]], [[1.730277122156229e+22]], [[1.9713656188484773e+22]], [[1.898762638895675e+22]], [[1.5995058745962905e+22]], [[1.838375560342124e+22]], [[1.9749617431509327e+22]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7781bcae466dafd9a7c7fae6600e00ab(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f85ff14ecb1201f01ba8ba107f90885
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.488690339274873e+26]], [[1.3495483934012892e+26]], [[1.535659346801632e+26]], [[1.47407424607551e+26]], [[1.5288283330036965e+26]], [[1.453183308412034e+26]], [[1.4048603098702648e+26]], [[1.3664479168158164e+26]], [[1.540907629958043e+26]], [[1.4500199685045538e+26]], [[1.5328436357862209e+26]], [[1.5949385122848198e+26]], [[1.3089760710832502e+26]], [[1.5025457809823566e+26]], [[1.556153126065201e+26]], [[1.5752117486398355e+26]], [[1.845169149047012e+26]], [[1.3481031832368345e+26]], [[1.2154525543690687e+26]], [[1.7846199252343493e+26]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_366ef5a4fa4734e954e72d04b6ee3ad6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 672, 12, 12], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eaf0b509dd9b1ef2f9d4cd21336e7058(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_366ef5a4fa4734e954e72d04b6ee3ad6
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 12, 12], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ebb9deba38bc6d7f056f2f423928fa79(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 28, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a293c8ddd875d4073ce0bcefbdde343f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebb9deba38bc6d7f056f2f423928fa79
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.1985650231090769e+30]], [[1.395382680014162e+30]], [[1.3322687430967235e+30]], [[1.3293154884351324e+30]], [[1.1862243083424508e+30]], [[1.310513669625576e+30]], [[1.2080782114993519e+30]], [[1.3944915505693786e+30]], [[1.3278696131548733e+30]], [[1.2736695396526357e+30]], [[1.310833430504864e+30]], [[1.24965785502345e+30]], [[1.2040855828643471e+30]], [[1.284046354425298e+30]], [[1.1335086446309725e+30]], [[1.2995287651656476e+30]], [[1.2816282005546137e+30]], [[1.3003801511741112e+30]], [[1.2974355101089848e+30]], [[1.2641011940218409e+30]], [[1.2597174778841908e+30]], [[1.2594038371918645e+30]], [[1.2292064569330293e+30]], [[1.2706612788662522e+30]], [[1.2443006513695552e+30]], [[1.3341079726155397e+30]], [[1.2467568864035572e+30]], [[1.3736818570891697e+30]]]], dtype='float32').reshape([1, 28, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b2f53e979f07927f3b927371e9b8b744(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebb9deba38bc6d7f056f2f423928fa79
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.1391893147635776e+34]], [[1.157838633867404e+34]], [[1.1935227506758208e+34]], [[1.1442386245958148e+34]], [[1.1695650208895348e+34]], [[1.0834960017181794e+34]], [[1.1631392455276162e+34]], [[1.0310266559170922e+34]], [[1.0896489350954396e+34]], [[1.116867894121219e+34]], [[1.1071952496242627e+34]], [[1.067699886816898e+34]], [[1.1161365191460092e+34]], [[1.120800210656009e+34]], [[1.1540729440619018e+34]], [[1.2425662212121897e+34]], [[1.0848501843271537e+34]], [[1.1433559733478044e+34]], [[1.041805276045146e+34]], [[1.123234000773244e+34]], [[1.1930481244647588e+34]], [[1.2710680575006806e+34]], [[1.1998231227117598e+34]], [[1.1085177409682313e+34]], [[1.1112477701368673e+34]], [[1.239194567721192e+34]], [[1.1546212277053013e+34]], [[1.0397557425161052e+34]]]], dtype='float32').reshape([1, 28, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dcf16e376f3a9e027cd363956030e2ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebb9deba38bc6d7f056f2f423928fa79
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.1169612211333231e+38]], [[1.2678051521894092e+38]], [[1.1245902466816406e+38]], [[1.1046493928557145e+38]], [[1.1644399081263192e+38]], [[1.0964417101614047e+38]], [[1.0214675802374103e+38]], [[1.1379036203454776e+38]], [[1.271786082134366e+38]], [[1.1065979239463373e+38]], [[1.1815876713257265e+38]], [[1.3380350236830135e+38]], [[1.3105300480195015e+38]], [[1.219681383158969e+38]], [[1.1979884333794313e+38]], [[1.1445890067869852e+38]], [[1.2567041837651386e+38]], [[1.221420295546338e+38]], [[1.2002837936742766e+38]], [[1.3002880396419455e+38]], [[1.3507934706200945e+38]], [[1.2858157290933559e+38]], [[1.2282487743875994e+38]], [[1.2222447754967265e+38]], [[1.3418792501872416e+38]], [[1.2436865276333229e+38]], [[1.2064914279575702e+38]], [[1.1935161620618261e+38]]]], dtype='float32').reshape([1, 28, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_fbc77978c5f6f0e2055f03f4b99dda49(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 672, 6, 6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0454c17e4f4e7802787940fc1d12999b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbc77978c5f6f0e2055f03f4b99dda49
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 6, 6], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e63ccd8f8948421ad70fd4518ee0b6bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebb9deba38bc6d7f056f2f423928fa79
    def get_inputs(self):
        return [
            paddle.uniform([1, 28, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_50430e09fdeada4ef8c6191366df4f94(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1152, 6, 6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7ce3f9055f48e0e422a4e80825471fd2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_50430e09fdeada4ef8c6191366df4f94
    def get_inputs(self):
        return [
            paddle.uniform([1, 1152, 6, 6], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6805d17f42c4d352974081cacfd8806e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_de1f4ede93b0f394aa640f3d898e3cae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6805d17f42c4d352974081cacfd8806e
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_41d01f866774ba6c371b2411c5f52100(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1280, 6, 6], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_40178c37dcacea518cbfcb6f83c6563f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_41d01f866774ba6c371b2411c5f52100
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 6, 6], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_40f4200a073923ec2a3ef43b22c01bfc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 112, 112], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_285d500fde5405bf86629bf46b2a54bc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_40f4200a073923ec2a3ef43b22c01bfc
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 112, 112], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a3da64f7f5990f021b0124c25fcad265(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 112, 112], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_70d77a35828941e4f30bea330f8cb830(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a3da64f7f5990f021b0124c25fcad265
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 112, 112], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_04e2928e420437008a197f98d9d4f9f6(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 162, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bf57f168e6c625d023f6ece6cb342719(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_04e2928e420437008a197f98d9d4f9f6
    def get_inputs(self):
        return [
            paddle.uniform([1, 162, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c9e6aee3a43236ad20011d8978b33a37(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 228, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d36af4c076a33442dcf3c661ea03141f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9e6aee3a43236ad20011d8978b33a37
    def get_inputs(self):
        return [
            paddle.uniform([1, 228, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6134b1a00aee304c40df2ef4b4425fe4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 300, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d2345d43700b4575b29acc39bdf2a208(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6134b1a00aee304c40df2ef4b4425fe4
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c5cd048b470a6dd93d1d5c1b5873c79f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 366, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f6a194e3b7d9a7e38520fa214631fa04(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c5cd048b470a6dd93d1d5c1b5873c79f
    def get_inputs(self):
        return [
            paddle.uniform([1, 366, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ed562f7044e097c4ac5d0bd95f66247f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 432, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6f1c90b4b223cb991f638be062c7d87c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ed562f7044e097c4ac5d0bd95f66247f
    def get_inputs(self):
        return [
            paddle.uniform([1, 432, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6b165432cf776f255b330b71d404f345(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 504, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ceced35de16de6b5eda434e12a935800(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6b165432cf776f255b330b71d404f345
    def get_inputs(self):
        return [
            paddle.uniform([1, 504, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2a8893f4941ff7d9afa6b44e2189fe87(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 570, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_72e3e8d1cb9d87b5e994ef10e0106b5d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2a8893f4941ff7d9afa6b44e2189fe87
    def get_inputs(self):
        return [
            paddle.uniform([1, 570, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_43ae63baeea2f9cccf3276595c82b59c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 636, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_43148a4009db2476120c458186db808b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43ae63baeea2f9cccf3276595c82b59c
    def get_inputs(self):
        return [
            paddle.uniform([1, 636, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f4c3b91cd70e06416abcf0ff480b2587(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 702, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d8ccd996afa03e54cfb68ab96f275f50(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f4c3b91cd70e06416abcf0ff480b2587
    def get_inputs(self):
        return [
            paddle.uniform([1, 702, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b013f7d7c7264e41a7e80ec327e129e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 768, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_683d4ad058d9291aae7fed795327d295(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b013f7d7c7264e41a7e80ec327e129e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4830d7c05595785e2d992b497b0193ef(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 840, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_379a4e13f3de2b7cbddd6ccb63c5a295(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4830d7c05595785e2d992b497b0193ef
    def get_inputs(self):
        return [
            paddle.uniform([1, 840, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2bd86e7a202faaaa3918c2b406fb0dab(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 906, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_776c74c43e938b034b0c33515748db84(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2bd86e7a202faaaa3918c2b406fb0dab
    def get_inputs(self):
        return [
            paddle.uniform([1, 906, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_cff4f1e7509703145e1b67b575b96bf3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 972, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e5fd5b16952556b1bb97164d6eea2a87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cff4f1e7509703145e1b67b575b96bf3
    def get_inputs(self):
        return [
            paddle.uniform([1, 972, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_cbdf8a41926997b38a692b1dabc664ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1044, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e9643ffca5ef067664a67be7c6eca293(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cbdf8a41926997b38a692b1dabc664ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 1044, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_43c992dda1e50146f05ff6eaf038a5c3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1280, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0eeebe3af6c1f80d2be5c4b2e4721f6e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_43c992dda1e50146f05ff6eaf038a5c3
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_da95b1327d5d72d400dca0b8837463b7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 320, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d995fc2c2085781ffae20b54e6253c87(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_da95b1327d5d72d400dca0b8837463b7
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 320, 320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1d50f3e8e3a8daf81418fc1b7d8fcafa(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 320, 320], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e4adec966b8e68a1e39b9f9e4be647f1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1d50f3e8e3a8daf81418fc1b7d8fcafa
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 320, 320], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_87c5ff624678b32608ff367aeb778ec0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, 160, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0327a9fa811f70c7fdd5962c9cfa61be(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_87c5ff624678b32608ff367aeb778ec0
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 160, 160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2427b689f1bea3cecd1a3bd025ed6c17(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, 160, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c20eccb8436bed89b21792bd08c22273(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2427b689f1bea3cecd1a3bd025ed6c17
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 160, 160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c477ff89aea86b3e5038a58ed32ce2cd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 160, 160], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7c0d0ea14e047cc96a32e7624068c0d0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c477ff89aea86b3e5038a58ed32ce2cd
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 160, 160], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_cfbedc54ac592b5370ed7b4c068339c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 80, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ccc69dc5146554b1f2506c1e20843ad5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cfbedc54ac592b5370ed7b4c068339c7
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 80, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_845b9c06b06340eaea9f4b50d80ef120(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, 80, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a2ebbe27674abbb11026ac79e30b820d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_845b9c06b06340eaea9f4b50d80ef120
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e132e99c4aa1d07ef3fa902aa92ee36e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 80, 80], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4f0fc8e8e24f23ea21fa9a8d0b25d640(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e132e99c4aa1d07ef3fa902aa92ee36e
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 80, 80], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b54937ef3c7e83902191005023bce200(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, 40, 40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_dcc3ea35f514fb8060abefd891ba26c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b54937ef3c7e83902191005023bce200
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_02d5971b1c2f8d67d98dc903a2317518(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 40, 40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_82597985dd67217116b097679b9af2bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_02d5971b1c2f8d67d98dc903a2317518
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_af22232e04bc30a6c224b2a2d09c70c7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 40, 40], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3c27c3e2aa59698cba7b42e79f1ef117(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_af22232e04bc30a6c224b2a2d09c70c7
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 40], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_fbd7beab8444ace11dfa1d654b86c029(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, 20, 20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c03c93a908b5760057097962dae5abbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fbd7beab8444ace11dfa1d654b86c029
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8d23b79ee053ef6c8084cd2ab03c8ca3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, 20, 20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4fdd929e2415bafcd4588b871d3c2ef8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8d23b79ee053ef6c8084cd2ab03c8ca3
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_59dcd9551ab05cf152b82fa68af8b078(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 20, 20], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_ddbd033a2bc5fae326d508ac546bfe5f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_59dcd9551ab05cf152b82fa68af8b078
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 20, 20], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2baf39f918140533c69b3494d5b51fcf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_46cd651425149752513b68ccbf8e6c67(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2baf39f918140533c69b3494d5b51fcf
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1e6ea80e7ca41c32de8ab46e875c3923(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d47f91714db8ea76cca30824e439b7c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1e6ea80e7ca41c32de8ab46e875c3923
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_aacbfd2635e3191f2daceb0b4ef2f514(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bd1706b4087d43d9213fa2be399b30f4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aacbfd2635e3191f2daceb0b4ef2f514
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[13980526.0]], [[16786818.0]], [[13421100.0]], [[14328887.0]], [[15565914.0]], [[14893876.0]], [[17302300.0]], [[13505882.0]], [[13839678.0]], [[16651823.0]], [[16340616.0]], [[14594916.0]], [[14825827.0]], [[15898333.0]], [[13663712.0]], [[13155664.0]]]], dtype='float32').reshape([1, 16, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_658c822848e4a0a114189e2a7849cd2e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_19375d3009ca1f3fe9d7d271ae578cac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_658c822848e4a0a114189e2a7849cd2e
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eb08911385f82c955ad25fb5eefe85d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f85ff14ecb1201f01ba8ba107f90885
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6589385728.0]], [[7747439616.0]], [[6090580992.0]], [[6799010304.0]], [[7744267264.0]], [[6030663680.0]], [[6990911488.0]], [[7763949056.0]], [[7313573888.0]], [[6446460928.0]], [[5292515840.0]], [[8529415168.0]], [[7481018880.0]], [[8095347712.0]], [[9393774592.0]], [[8455483392.0]], [[7120087040.0]], [[5546908672.0]], [[8256373760.0]], [[7469214720.0]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_024e9008d8290a213c120dfe3d9ea64c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f85ff14ecb1201f01ba8ba107f90885
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1721964691456.0]], [[1814429827072.0]], [[1740761464832.0]], [[2057639559168.0]], [[2237306765312.0]], [[2160493592576.0]], [[2041806585856.0]], [[2195535167488.0]], [[2102716530688.0]], [[2022248677376.0]], [[1794602827776.0]], [[1983108349952.0]], [[2021920210944.0]], [[2177277624320.0]], [[2015597953024.0]], [[1641698951168.0]], [[2085865127936.0]], [[2026762928128.0]], [[1806169276416.0]], [[1769575809024.0]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3fc3e9097d5d2c6f08a44e725576e1bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f85ff14ecb1201f01ba8ba107f90885
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[679419131723776.0]], [[771216407265280.0]], [[628409482870784.0]], [[745873751408640.0]], [[826519412801536.0]], [[682559960776704.0]], [[530898692866048.0]], [[573434639679488.0]], [[435650075557888.0]], [[632708678025216.0]], [[588598424371200.0]], [[632598418161664.0]], [[607354412335104.0]], [[593554481086464.0]], [[897616489480192.0]], [[717797751521280.0]], [[742576088940544.0]], [[853886675976192.0]], [[502872286429184.0]], [[766375979122688.0]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f75ed0e773b4981be911642e4c02622c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a41fa467fc39e26287877ba9e43e9e85(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f75ed0e773b4981be911642e4c02622c
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7ea83f4b890cae01bbe946830f6978b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff33d5afbedb70bb059ced37dbdb332d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.1746728054882304e+17]], [[4.908796806399263e+17]], [[4.6199805279889e+17]], [[4.765287517949788e+17]], [[4.533135945666396e+17]], [[4.772737396422738e+17]], [[4.879870029862011e+17]], [[4.719863256632525e+17]], [[4.7954024542398054e+17]], [[4.660497875069829e+17]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c228375dcbdafc61b7dcced4e0ada903(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7bdfff3a6a50f1a17ab6f76be967bbfa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c228375dcbdafc61b7dcced4e0ada903
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_042952ca54ad2fe296673d5fb2c63e12(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f85ff14ecb1201f01ba8ba107f90885
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[3.0448465533682665e+21]], [[2.5647132934946706e+21]], [[3.3636206236677817e+21]], [[3.7591757195894063e+21]], [[3.0866759867072837e+21]], [[3.1825041425284265e+21]], [[2.6460874277867445e+21]], [[2.8330625290913553e+21]], [[3.354648327310153e+21]], [[3.096355066731433e+21]], [[3.383155268526478e+21]], [[3.679187005132702e+21]], [[3.2077096637429124e+21]], [[3.2808312331928066e+21]], [[3.4659677399495433e+21]], [[2.705366902306939e+21]], [[3.4397756489416867e+21]], [[2.646977170188127e+21]], [[3.266049856265823e+21]], [[2.8648283875880364e+21]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_14a38ac95be34e2fb498b33eb6210b37(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f85ff14ecb1201f01ba8ba107f90885
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.3341877434942511e+25]], [[1.3362224193655813e+25]], [[1.1710643385719422e+25]], [[8.747087481165137e+24]], [[1.3129383625349928e+25]], [[1.327094278645007e+25]], [[7.299406605925989e+24]], [[9.356594999015858e+24]], [[1.2165826019110242e+25]], [[7.734146788118625e+24]], [[1.4094151799169451e+25]], [[8.013293872196292e+24]], [[1.224833484658743e+25]], [[1.2853865361708011e+25]], [[1.2648768696488985e+25]], [[1.2101833110997039e+25]], [[7.842301201544289e+24]], [[8.735102862124748e+24]], [[1.1806317423857716e+25]], [[1.3840458339609742e+25]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e987cdffc5f1a198c28dd204668a0eb0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f85ff14ecb1201f01ba8ba107f90885
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.525883328311274e+28]], [[2.8162963527365777e+28]], [[2.8812966578350645e+28]], [[2.4425463102781286e+28]], [[3.1385872398678445e+28]], [[3.0035268501028e+28]], [[2.6485623815132066e+28]], [[2.8105183012264625e+28]], [[2.683697496500729e+28]], [[2.2281503997191984e+28]], [[2.540410744322526e+28]], [[2.8349820485543204e+28]], [[3.009070436117041e+28]], [[2.686297631486197e+28]], [[2.9865478175319665e+28]], [[2.9745744934329747e+28]], [[2.831993262807312e+28]], [[2.7763814945134185e+28]], [[2.9743386112271554e+28]], [[3.142320270572553e+28]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_acda317bf8f0c83d825cd2096de6e66c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 40, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e47f67e278a7126efd75b8434ff15b6a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_acda317bf8f0c83d825cd2096de6e66c
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d1fe3058386f1e38d1d7a6b6e66c621b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 360, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3f0961295a564b75074eebfa313393f5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d1fe3058386f1e38d1d7a6b6e66c621b
    def get_inputs(self):
        return [
            paddle.uniform([1, 360, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ae8cb7dc893aa2a83f250ad4fcd1a043(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 60, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_47884db3a642357d62cf859c633a5149(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae8cb7dc893aa2a83f250ad4fcd1a043
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d3cd38e84d2f8529181c6b46d71d3a6e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 720, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5b804b7c4f58b40930a211a6b406f6bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d3cd38e84d2f8529181c6b46d71d3a6e
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1df830ca8155d2262d6656b51f0144d2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 720, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d8489f35711798390a90ed316c7ce962(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1df830ca8155d2262d6656b51f0144d2
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_275ff2047afc54ceb8f38bf4994a181d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1200, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b552ddfba6eb79fb2e81a30c47d773a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_275ff2047afc54ceb8f38bf4994a181d
    def get_inputs(self):
        return [
            paddle.uniform([1, 1200, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5afd4a66990530436ac42996497c5a64(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 100, 1, 1], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_9824554a24de892014c97e9ef81b7cec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5afd4a66990530436ac42996497c5a64
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2e3bb683e6ccc14fddac1dbf09004b7f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, 56, 56], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f659e7b1bfd59a45ff068b940db695da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2e3bb683e6ccc14fddac1dbf09004b7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 56, 56], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_37b4b67af87b4b9580d2713d21293397(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, 28, 28], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_837d93d8813247262f1717d7a284dc4a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37b4b67af87b4b9580d2713d21293397
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 28, 28], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d352af92189b382a88389c75c594c6db(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e67f2d41127652d3aed74eefa5ac8fd8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d352af92189b382a88389c75c594c6db
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ae9fa94ee4d35468e86e2f27340af165(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, 28, 28], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_16ba0c936b0990075560bcd646d032aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ae9fa94ee4d35468e86e2f27340af165
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 28, 28], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_e1909e352c3b6a762606ec8811cfe583(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 20, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7865ef717e856719a25c7b9d79ba19b4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e1909e352c3b6a762606ec8811cfe583
    def get_inputs(self):
        return [
            paddle.uniform([1, 20, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ec9acf295dcacfa02c1fdbb1bf76cad3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, 14, 14], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d69f55d5a8fb19734207919a708034ca(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ec9acf295dcacfa02c1fdbb1bf76cad3
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9a9ac243f3ec436b52c11bcb7067ffaf(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 10, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6844f5265e0172fc0dbf513415e564db(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9a9ac243f3ec436b52c11bcb7067ffaf
    def get_inputs(self):
        return [
            paddle.uniform([1, 10, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_18fcfb6ef3a3b74a196942d57f8efc69(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480, 14, 14], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d0f11a199316e565b3aab0e22a1816cd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18fcfb6ef3a3b74a196942d57f8efc69
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_fc083c25b3600ce2e7b58b5b17b84064(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 40, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_840a1d6aa5a1ba83a544ac51fcc9c662(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_fc083c25b3600ce2e7b58b5b17b84064
    def get_inputs(self):
        return [
            paddle.uniform([1, 40, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_081a72bb4186f6278afe9eb95474b4ee(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 360, 14, 14], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1d241cd6a2fa187d526ae6a2bc9d92f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_081a72bb4186f6278afe9eb95474b4ee
    def get_inputs(self):
        return [
            paddle.uniform([1, 360, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_10374c9c4941da6564584e11fb23ad1c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 60, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c459b2b43fb4c6a8b6c19242e96a9131(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10374c9c4941da6564584e11fb23ad1c
    def get_inputs(self):
        return [
            paddle.uniform([1, 60, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b27125bd890f82a6dfc2016f696db45d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 720, 14, 14], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_bbd7dc02ba6fb464962ecb91f525863b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b27125bd890f82a6dfc2016f696db45d
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b578e75678d0bb8bc513c8a5edc7ae9d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 720, 7, 7], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d72eb845f865066a08c0fa42dd3770e9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b578e75678d0bb8bc513c8a5edc7ae9d
    def get_inputs(self):
        return [
            paddle.uniform([1, 720, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5a244ba2a55fe16221f6029331507315(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1200, 7, 7], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5940b1fee86bb03a52a6d67a96ad0bcb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a244ba2a55fe16221f6029331507315
    def get_inputs(self):
        return [
            paddle.uniform([1, 1200, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_bb9216b68d2aa6e73c4862f2c0e499da(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 100, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_78fd66cdca87fcf0c1ee6d447a9fd17e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_bb9216b68d2aa6e73c4862f2c0e499da
    def get_inputs(self):
        return [
            paddle.uniform([1, 100, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3dc1690e533876ca19cdfde4781b1365(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 96, 96], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_f23beaa623ae9ba8d9993d71cfe903f7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3dc1690e533876ca19cdfde4781b1365
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 96, 96], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a017166d18a8d46098cd9fd671652a44(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 8, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e6a19450a26ce190c2b0b5f79e82e619(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a017166d18a8d46098cd9fd671652a44
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.34375]], [[2.0859375]], [[2.3125]], [[2.07421875]], [[1.994140625]], [[2.078125]], [[2.15625]], [[1.9072265625]]]], dtype='float16').reshape([1, 8, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_cf1f6ce76d1420c61e9792b569a92c04(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 96, 96], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a605c46bec7168bd56a6d745acf01253(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_cf1f6ce76d1420c61e9792b569a92c04
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 96, 96], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8b8473edeeaadd655a47801c1a239bec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 48, 48], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6637fe18677d00757f13332d2b4a1b5c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8b8473edeeaadd655a47801c1a239bec
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 48, 48], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_b4dfc9c024095308c1ae3e63ee7fbddb(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_aad43b124bd5dced26ca19ab3c6f5f98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4dfc9c024095308c1ae3e63ee7fbddb
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[109.1875]], [[89.0625]], [[108.1875]], [[90.5]]]], dtype='float16').reshape([1, 4, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1823451a12818f79746b8556a29ceb3c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 48, 48], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c252ee679cf62146b214c678fb07723d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1823451a12818f79746b8556a29ceb3c
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 48, 48], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1afc2dc807288bbd8add4e189dadfb7f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 6, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8da46374b604ff90a4ff540ebb444825(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1afc2dc807288bbd8add4e189dadfb7f
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[9880.0]], [[11872.0]], [[13104.0]], [[14544.0]], [[10264.0]], [[11248.0]]]], dtype='float16').reshape([1, 6, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8fe23ba84c21455bf7f8ffb82917ea5d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 24, 24], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_90306acb8e11c86b88d7f63913174544(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8fe23ba84c21455bf7f8ffb82917ea5d
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 24, 24], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_89139e4074a2b74ca4b32b6de5ccd052(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1afc2dc807288bbd8add4e189dadfb7f
    def get_inputs(self):
        return [
            paddle.uniform([1, 6, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_89e15e710e38e63b6aec83dc5a970d14(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, 24, 24], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8291fbc92880c818bdb0d49d7e146922(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_89e15e710e38e63b6aec83dc5a970d14
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 24, 24], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_85367fcc6fee992391e8ac9fc972716d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 240, 12, 12], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_edd798efe1ad68536f8d56e3602dc6c3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_85367fcc6fee992391e8ac9fc972716d
    def get_inputs(self):
        return [
            paddle.uniform([1, 240, 12, 12], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a6476c6ddd439b173d9aa06762d0d1ad(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 480, 12, 12], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e0db06bbc18244740b894c6cc44f975f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a6476c6ddd439b173d9aa06762d0d1ad
    def get_inputs(self):
        return [
            paddle.uniform([1, 480, 12, 12], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_18b613424751ba165a1c96cd8b69c826(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 672, 12, 12], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e72068bc0fbd25af157cd685736fc4c2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_18b613424751ba165a1c96cd8b69c826
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 12, 12], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_49f2efb639a1b925bbc25b8290798a2c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 28, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_37034a2b69e185284fb8cb76371616d5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_49f2efb639a1b925bbc25b8290798a2c
    def get_inputs(self):
        return [
            paddle.uniform([1, 28, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7df7b4344cb169e4c95073d50e7cbf6e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 672, 6, 6], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8e95a5b43b6a9621ee989f054c62ef47(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7df7b4344cb169e4c95073d50e7cbf6e
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 6, 6], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_75ad211c5894b03d736a81b791ee7199(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1152, 6, 6], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_36bc4b21b99a48e59bb1604b53c703c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_75ad211c5894b03d736a81b791ee7199
    def get_inputs(self):
        return [
            paddle.uniform([1, 1152, 6, 6], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ca6af9a970a1cdede5fd8d0a47175acc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, 1, 1], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_384022f6aa2035e9c307cd9f914f12ce(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ca6af9a970a1cdede5fd8d0a47175acc
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 1, 1], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5e91723fa5c596957f17a6b1f54ee101(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1280, 6, 6], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_abfde652a068f8edfb0c9a7c9c251435(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5e91723fa5c596957f17a6b1f54ee101
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 6, 6], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_aa3bf7ef09e34a67b0d0f1be04108b50(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 112, 112], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_d7083222986325faf0f34e5105e97614(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_aa3bf7ef09e34a67b0d0f1be04108b50
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 112, 112], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_73ae3bc953ccc3aaa7fbd882482dce6a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 112, 112], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3f29f8c9b15a678626ac3d501462690c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_73ae3bc953ccc3aaa7fbd882482dce6a
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 112, 112], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_9bbea6faafdf76b3b2011c2c75aa9ac0(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 162, 56, 56], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a8034f66faa0735332771866c5fe9ab1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_9bbea6faafdf76b3b2011c2c75aa9ac0
    def get_inputs(self):
        return [
            paddle.uniform([1, 162, 56, 56], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1249b5d423a5387298ac8b2b4d0c7acc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 228, 56, 56], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8f4790cba8803d2bcec2b637efc14844(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1249b5d423a5387298ac8b2b4d0c7acc
    def get_inputs(self):
        return [
            paddle.uniform([1, 228, 56, 56], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5cb9484bb86b90704334db7557a2b4dd(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 300, 28, 28], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7009ec1d56b9713db8b8a54ad4a98f41(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5cb9484bb86b90704334db7557a2b4dd
    def get_inputs(self):
        return [
            paddle.uniform([1, 300, 28, 28], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2cc7fcc0f0022ba2ade0301aa5a1b337(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 366, 28, 28], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5b33c08d552008447cd7717f8b12a38c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2cc7fcc0f0022ba2ade0301aa5a1b337
    def get_inputs(self):
        return [
            paddle.uniform([1, 366, 28, 28], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_86c87ffae2fc445b840f8fdb054fac9d(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 432, 14, 14], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fbb4534ad3273f1352339a7d45e48cbc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_86c87ffae2fc445b840f8fdb054fac9d
    def get_inputs(self):
        return [
            paddle.uniform([1, 432, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_5a40fcf1d915d4d6370418640ce67267(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 504, 14, 14], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1b2b63978ccd5941fef449b84ed62e10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_5a40fcf1d915d4d6370418640ce67267
    def get_inputs(self):
        return [
            paddle.uniform([1, 504, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2611edb453a5cac5a04c794f519b6cde(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 570, 14, 14], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_eb08b6592959a1f717e92601cc0b2e8d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2611edb453a5cac5a04c794f519b6cde
    def get_inputs(self):
        return [
            paddle.uniform([1, 570, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d442a37eb726e5d2f0df50e40b51cae4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 636, 14, 14], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fae29476ca36b70c9f8aa957801e4b30(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d442a37eb726e5d2f0df50e40b51cae4
    def get_inputs(self):
        return [
            paddle.uniform([1, 636, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0e7359142b465b76c672037b66ca9fcc(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 702, 14, 14], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0c907230318ad3ce097d14bf3ecd3335(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0e7359142b465b76c672037b66ca9fcc
    def get_inputs(self):
        return [
            paddle.uniform([1, 702, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_dcd1037552173753bb2edc6cb5694eff(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 768, 14, 14], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c8e28c9a431f44130c0d00c4dc43f0c8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dcd1037552173753bb2edc6cb5694eff
    def get_inputs(self):
        return [
            paddle.uniform([1, 768, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a72b0d966f09d8c40e61223cbdeede84(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 840, 7, 7], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e906e317e5bc126f62d04afa863f1545(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a72b0d966f09d8c40e61223cbdeede84
    def get_inputs(self):
        return [
            paddle.uniform([1, 840, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_38f117cc1566c69a250fea18ff27f7b3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 906, 7, 7], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e55de75c2e15c39b1b914244e62523d9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_38f117cc1566c69a250fea18ff27f7b3
    def get_inputs(self):
        return [
            paddle.uniform([1, 906, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_83bbce45b225a7a2a66e3ff2e778e6fe(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 972, 7, 7], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a8d96016ddf1b17593e5bdc4164a0a79(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_83bbce45b225a7a2a66e3ff2e778e6fe
    def get_inputs(self):
        return [
            paddle.uniform([1, 972, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ac3eac5ed5fbc3ed098b68b8f353650c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1044, 7, 7], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b225c746c5d611cba307f6e39375c167(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ac3eac5ed5fbc3ed098b68b8f353650c
    def get_inputs(self):
        return [
            paddle.uniform([1, 1044, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4b6400392149d6fb405cb9bb9539fc8f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1280, 7, 7], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b292993fa19500f034e5f40752db60fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4b6400392149d6fb405cb9bb9539fc8f
    def get_inputs(self):
        return [
            paddle.uniform([1, 1280, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_92f3cacb0c6f3d8c3e389fa110a398bb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_29007ee7b2e66f3b636fcfe61bbce4a2
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.020218372344971]], [[5.779989719390869]], [[5.272001266479492]], [[6.338002681732178]], [[5.078469276428223]], [[4.759707927703857]], [[6.7526044845581055]], [[5.370635509490967]]]], dtype='float32').reshape([1, 8, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_a7532207870f6a96f69dc32ec3988d86(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e33bf8458bc23297cd797190848b89a6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a7532207870f6a96f69dc32ec3988d86
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0685ac0f8d266dec37162a8c43028aec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10b285d1685c7e1ec0629a08f4f1c2b5
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[715.2088012695312]], [[646.95166015625]], [[684.4810791015625]], [[567.3255615234375]]]], dtype='float32').reshape([1, 4, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_7e0a868c572be307a632e284b2ab8672(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 56, 56], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_e97baec600c5ad0e8d792d6056390159(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_7e0a868c572be307a632e284b2ab8672
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 56, 56], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_de1437cb150951eeb1c42449fb289b0e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8d07bd6ef969a87be053df3f69b43d8
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[25508.236328125]], [[25183.271484375]], [[26768.939453125]], [[31446.046875]], [[23898.638671875]], [[32278.58203125]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f3acdd14a5031f31eda60d1272074fec(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 28, 28], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3cef37a4834c0b5117a96d44a98849fc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f3acdd14a5031f31eda60d1272074fec
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 28, 28], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_6f243b43269881eb311b36cd38f33a17(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f8d07bd6ef969a87be053df3f69b43d8
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[17797448.0]], [[17554908.0]], [[16649942.0]], [[18681426.0]], [[15140606.0]], [[15624559.0]]]], dtype='float32').reshape([1, 6, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_cae297017c8964ad615fa9a4b829c100(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff33d5afbedb70bb059ced37dbdb332d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[10291249152.0]], [[10166941696.0]], [[10204102656.0]], [[10139587584.0]], [[10066112512.0]], [[10081742848.0]], [[10857426944.0]], [[10663124992.0]], [[9413766144.0]], [[8894158848.0]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a0d63d08417bfac661789d31b08b3d86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ff33d5afbedb70bb059ced37dbdb332d
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6818450898944.0]], [[7250532368384.0]], [[7546705805312.0]], [[7084813844480.0]], [[6699838603264.0]], [[7219813810176.0]], [[7028518944768.0]], [[7132170682368.0]], [[6570401333248.0]], [[7165505437696.0]]]], dtype='float32').reshape([1, 10, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0abf1be48f30b0ba9a673008985b41aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f85ff14ecb1201f01ba8ba107f90885
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[6531123765051392.0]], [[7142669125943296.0]], [[7095445960523776.0]], [[6605274227933184.0]], [[6777022453907456.0]], [[6994374642630656.0]], [[6674554197901312.0]], [[7101333286944768.0]], [[6636092228894720.0]], [[6957819102232576.0]], [[6728381949280256.0]], [[6979635925483520.0]], [[6489485667729408.0]], [[6680534939860992.0]], [[6601847917772800.0]], [[7085154145140736.0]], [[7319664191340544.0]], [[6287095534452736.0]], [[6285706112532480.0]], [[6858460939419648.0]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3681a369c9a8a5b82ba98db294c1becc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f85ff14ecb1201f01ba8ba107f90885
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[8.395463970395259e+18]], [[9.493493456497017e+18]], [[8.395826259476611e+18]], [[8.607407481032802e+18]], [[9.047475465178055e+18]], [[8.0678051072715e+18]], [[8.880835132139766e+18]], [[9.054044497398202e+18]], [[7.843933544740028e+18]], [[8.459766708922483e+18]], [[8.272340658316902e+18]], [[8.689773546336944e+18]], [[9.060855422176461e+18]], [[8.943138408771879e+18]], [[8.190451231303401e+18]], [[9.229058710993633e+18]], [[8.714407554601452e+18]], [[8.913166821310333e+18]], [[9.229562287319155e+18]], [[8.176931636328268e+18]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a74f03ac59b4d9c8711c165b2ff1bb81(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4f85ff14ecb1201f01ba8ba107f90885
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[2.9019669278501798e+22]], [[2.5641613773603363e+22]], [[2.8760322738560227e+22]], [[3.033274554125738e+22]], [[3.0006432726656624e+22]], [[2.933711900903589e+22]], [[2.7823495203073684e+22]], [[3.175701567881274e+22]], [[2.7040900192225886e+22]], [[2.7544042342595718e+22]], [[2.737521590336448e+22]], [[2.5323718186706162e+22]], [[3.0144308177448758e+22]], [[2.935909207161783e+22]], [[2.631587919901439e+22]], [[2.7230533261735575e+22]], [[2.9178290560977414e+22]], [[2.7512346008418284e+22]], [[2.8170029180001335e+22]], [[2.8096375059895504e+22]]]], dtype='float32').reshape([1, 20, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_3b031f294275ad91c6827d1eb6f8bd78(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 672, 14, 14], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7232aada3855960d4344a45dd474de72(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3b031f294275ad91c6827d1eb6f8bd78
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 14, 14], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0e90e9cd46671e962218c77d2cf0c2ba(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebb9deba38bc6d7f056f2f423928fa79
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.4477805337740055e+26]], [[1.617800300085131e+26]], [[1.66807782339523e+26]], [[1.6584626425142497e+26]], [[1.6625373438126914e+26]], [[1.5483328133825926e+26]], [[1.5240972965511122e+26]], [[1.6539834041182715e+26]], [[1.4677878723963509e+26]], [[1.644693992737633e+26]], [[1.558735116833198e+26]], [[1.4568405598920876e+26]], [[1.3781457351027186e+26]], [[1.5635762803479025e+26]], [[1.5452214932934004e+26]], [[1.5204014913759445e+26]], [[1.5220516448670582e+26]], [[1.4656622540767373e+26]], [[1.5049836104454177e+26]], [[1.562080802805847e+26]], [[1.5175447363549695e+26]], [[1.4995752095504468e+26]], [[1.6197503054011628e+26]], [[1.6250906378105017e+26]], [[1.6204479612620305e+26]], [[1.4511868173009363e+26]], [[1.3681237112811926e+26]], [[1.6216560385314177e+26]]]], dtype='float32').reshape([1, 28, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8f3bdd5ab0f4c5152a155cc6b9ad6477(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebb9deba38bc6d7f056f2f423928fa79
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.2131591756034647e+30]], [[1.250661565685185e+30]], [[1.2212379735087669e+30]], [[1.187104179665539e+30]], [[1.280835598564129e+30]], [[1.194373677292473e+30]], [[1.2478533821219477e+30]], [[1.2433682673311774e+30]], [[1.1825945841269216e+30]], [[1.2473470688771203e+30]], [[1.1833245486483776e+30]], [[1.1905373778775171e+30]], [[1.1557524263646993e+30]], [[1.2314163733312848e+30]], [[1.1649871084692806e+30]], [[1.1837959541601636e+30]], [[1.2203029449451587e+30]], [[1.2320648109177806e+30]], [[1.2233648518147877e+30]], [[1.2497246481749837e+30]], [[1.19152847037601e+30]], [[1.2204810348299607e+30]], [[1.2190449065141223e+30]], [[1.2439414492854022e+30]], [[1.2545820365603315e+30]], [[1.1796272001448137e+30]], [[1.2065575339340041e+30]], [[1.301488433919243e+30]]]], dtype='float32').reshape([1, 28, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_ea19380a1a88097bcfcef18bb98f25c3(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 672, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_7d3cb2652a0aab1b9dfb3a652932fcd5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ea19380a1a88097bcfcef18bb98f25c3
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_46fbc316889436a427ebd1ac53241091(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_ebb9deba38bc6d7f056f2f423928fa79
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[1.9691154585608501e+34]], [[1.8652579837889874e+34]], [[1.929610316027175e+34]], [[1.852574792910493e+34]], [[1.7480766841683001e+34]], [[1.913484413899428e+34]], [[2.0427533314457096e+34]], [[1.820106101560116e+34]], [[1.6903744420271496e+34]], [[1.8102308062787287e+34]], [[1.8400894248502764e+34]], [[1.795956119685729e+34]], [[1.7846258734761696e+34]], [[1.8624232248930279e+34]], [[1.8319388276316214e+34]], [[1.769947370842355e+34]], [[1.905243942233921e+34]], [[1.9601901584656104e+34]], [[1.8708710515151152e+34]], [[1.8222624693145473e+34]], [[1.938899075317949e+34]], [[1.7215445342763162e+34]], [[1.8520508966858674e+34]], [[1.8999314463493317e+34]], [[1.771561768447587e+34]], [[1.858814382090503e+34]], [[1.7961267078231425e+34]], [[1.8316959437959136e+34]]]], dtype='float32').reshape([1, 28, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_1844f5acfa6f3aeb58bc5253857258b4(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1152, 7, 7], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_87d42ae023ef31cfbda41b642a011ba5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_1844f5acfa6f3aeb58bc5253857258b4
    def get_inputs(self):
        return [
            paddle.uniform([1, 1152, 7, 7], dtype='float32', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_616cd58935445e91df7790c3a02c1d2c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a017166d18a8d46098cd9fd671652a44
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[5.73046875]], [[4.796875]], [[5.6171875]], [[4.68359375]], [[5.72265625]], [[3.890625]], [[3.013671875]], [[4.1328125]]]], dtype='float16').reshape([1, 8, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_2f2ade257627fdddb333c7239a36d060(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 56, 56], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_5f476d07592221575deb75729af8582e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_2f2ade257627fdddb333c7239a36d060
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 56, 56], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_78b22487821afa5e69276b434894aa22(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_b4dfc9c024095308c1ae3e63ee7fbddb
    def get_inputs(self):
        return [
            paddle.to_tensor([[[[579.5]], [[656.0]], [[689.5]], [[645.0]]]], dtype='float16').reshape([1, 4, 1, 1]),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_72b67a94e27adde99ffab51d31fb929c(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 56, 56], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_772d3ef812ab4aaa6639ab4e80630f10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_72b67a94e27adde99ffab51d31fb929c
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 56, 56], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_60042b0175ca2892a7c9c0f316421c14(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 144, 28, 28], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8b69a3a3ed27df2c7700d98c9ab9c2c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60042b0175ca2892a7c9c0f316421c14
    def get_inputs(self):
        return [
            paddle.uniform([1, 144, 28, 28], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_135a20185eb5dfca5393a18b92112a33(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 672, 14, 14], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_172ff2a7600ba5d43f0daa06b08a8804(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_135a20185eb5dfca5393a18b92112a33
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 14, 14], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6a2b8e83b1279960ef1a750b5067862e(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 672, 7, 7], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_116dff2c31cea264c00283ef0829f686(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6a2b8e83b1279960ef1a750b5067862e
    def get_inputs(self):
        return [
            paddle.uniform([1, 672, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_d679dd56903a9c0bd61452eac78ccd24(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1152, 7, 7], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_3e20cbe5fdb61ffb9a004ae658aa69c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d679dd56903a9c0bd61452eac78ccd24
    def get_inputs(self):
        return [
            paddle.uniform([1, 1152, 7, 7], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_826aad6eea83e34f72af7c9af1cc0a59(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 16, 320, 320], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_a7c222ebd8ee39f5e40a32ffcf3bd4e4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_826aad6eea83e34f72af7c9af1cc0a59
    def get_inputs(self):
        return [
            paddle.uniform([1, 16, 320, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0c7f25aae5380cd99500ed9661aec94b(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 32, 320, 320], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_0a23876ce00aa1964f7007b3cf41f8ae(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0c7f25aae5380cd99500ed9661aec94b
    def get_inputs(self):
        return [
            paddle.uniform([1, 32, 320, 320], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_f5f8297eae2b837ffe74d20f24216d2a(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, 160, 160], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_c06c6e473b3d7b570eca587ff0a7da0a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f5f8297eae2b837ffe74d20f24216d2a
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 160, 160], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c9b065d4dda3f0656cfac7f62f8a4018(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 24, 160, 160], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fdbdb8bbb7a91521e2191e63f6c254c1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c9b065d4dda3f0656cfac7f62f8a4018
    def get_inputs(self):
        return [
            paddle.uniform([1, 24, 160, 160], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_0a048a9d0b01e19683a755a063e940c1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 64, 160, 160], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_718991193deca6edc47e5bb6cbea76d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0a048a9d0b01e19683a755a063e940c1
    def get_inputs(self):
        return [
            paddle.uniform([1, 64, 160, 160], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_639512479f39e970c08a355e30a94987(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 80, 80], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_afaf2e0be6805efa41c20e57bbc1b649(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_639512479f39e970c08a355e30a94987
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 80, 80], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_dd69da155ea481490d1d920111b805ea(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 48, 80, 80], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_4c11bb29bdbad0bee730caa370e8d56a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_dd69da155ea481490d1d920111b805ea
    def get_inputs(self):
        return [
            paddle.uniform([1, 48, 80, 80], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_6bd0577118b4f850fdcfc9f1c87ba8d5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 128, 80, 80], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_479844d2eab09613e773eb773167fea5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_6bd0577118b4f850fdcfc9f1c87ba8d5
    def get_inputs(self):
        return [
            paddle.uniform([1, 128, 80, 80], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_8f3b225abe3644db4dff03829dea62df(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, 40, 40], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_fe4beb95b9a60dce366709c247830a4d(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_8f3b225abe3644db4dff03829dea62df
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_37553785549b288b6902ded01c6ecd18(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 96, 40, 40], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_1a9dd60ddeb0e46f21cda9c27a9ebc13(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_37553785549b288b6902ded01c6ecd18
    def get_inputs(self):
        return [
            paddle.uniform([1, 96, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_10b576186e33c7ee4912df9c8580b3e2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 256, 40, 40], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_8dabd859ebb660ef173e7d68a7d8b759(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_10b576186e33c7ee4912df9c8580b3e2
    def get_inputs(self):
        return [
            paddle.uniform([1, 256, 40, 40], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_60bc5851a9ec1dff038511f36a0e0715(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 384, 20, 20], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_30e44eecb949839ec44bdf29f803ed9a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_60bc5851a9ec1dff038511f36a0e0715
    def get_inputs(self):
        return [
            paddle.uniform([1, 384, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_4134c34347f20da852c7c9cefdf0bac2(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 192, 20, 20], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_b273bd38a66c3fb7cbabacbb6ac3ad98(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_4134c34347f20da852c7c9cefdf0bac2
    def get_inputs(self):
        return [
            paddle.uniform([1, 192, 20, 20], dtype='float16', min=0, max=0.5),
        ]


    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        return self._test_entry()

class PrimitiveOp_c1cd154dcb474e82e2332591e88b9552(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.swish(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 512, 20, 20], dtype='float16'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None



@unittest.skipIf(need_skip, skip_message)
class TestPrimitiveOp_79d08719bc65356f580fe162b09e07ed(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_c1cd154dcb474e82e2332591e88b9552
    def get_inputs(self):
        return [
            paddle.uniform([1, 512, 20, 20], dtype='float16', min=0, max=0.5),
        ]


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