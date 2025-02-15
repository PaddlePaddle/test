# api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.meshgrid||method:__sub__||method:__sub__||method:__add__||method:__add__||api:paddle.tensor.manipulation.stack||method:astype||api:paddle.tensor.manipulation.stack||method:astype||method:reshape||method:reshape||api:paddle.tensor.creation.full||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.meshgrid||method:__sub__||method:__sub__||method:__add__||method:__add__||api:paddle.tensor.manipulation.stack||method:astype||api:paddle.tensor.manipulation.stack||method:astype||method:reshape||method:reshape||api:paddle.tensor.creation.full||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.meshgrid||method:__sub__||method:__sub__||method:__add__||method:__add__||api:paddle.tensor.manipulation.stack||method:astype||api:paddle.tensor.manipulation.stack||method:astype||method:reshape||method:reshape||api:paddle.tensor.creation.full||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.meshgrid||method:__sub__||method:__sub__||method:__add__||method:__add__||api:paddle.tensor.manipulation.stack||method:astype||api:paddle.tensor.manipulation.stack||method:astype||method:reshape||method:reshape||api:paddle.tensor.creation.full||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.arange||method:__add__||method:__mul__||api:paddle.tensor.creation.meshgrid||method:__sub__||method:__sub__||method:__add__||method:__add__||api:paddle.tensor.manipulation.stack||method:astype||api:paddle.tensor.manipulation.stack||method:astype||method:reshape||method:reshape||api:paddle.tensor.creation.full||api:paddle.tensor.manipulation.concat||api:paddle.tensor.manipulation.concat||api:paddle.tensor.manipulation.concat||method:__truediv__||api:paddle.tensor.manipulation.split
import paddle
import unittest
import numpy as np


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(
        self,
    ):
        var_0 = paddle.tensor.creation.arange(end=152)
        var_1 = var_0.__add__(0.5)
        var_2 = var_1.__mul__(8.0)
        var_3 = paddle.tensor.creation.arange(end=100)
        var_4 = var_3.__add__(0.5)
        var_5 = var_4.__mul__(8.0)
        out = paddle.tensor.creation.meshgrid(var_5, var_2)
        var_6 = out[0]
        var_7 = out[1]
        var_8 = var_7.__sub__(32.0)
        var_9 = var_6.__sub__(32.0)
        var_10 = var_7.__add__(32.0)
        var_11 = var_6.__add__(32.0)
        var_12 = paddle.tensor.manipulation.stack([var_8, var_9, var_10, var_11], axis=-1)
        var_13 = var_12.astype('float32')
        var_14 = paddle.tensor.manipulation.stack([var_7, var_6], axis=-1)
        var_15 = var_14.astype('float32')
        var_16 = var_13.reshape([-1, 4])
        var_17 = var_15.reshape([-1, 2])
        var_18 = paddle.tensor.creation.full([15200, 1], 8.0, dtype='float32')
        var_19 = paddle.tensor.creation.arange(end=76)
        var_20 = var_19.__add__(0.5)
        var_21 = var_20.__mul__(16.0)
        var_22 = paddle.tensor.creation.arange(end=50)
        var_23 = var_22.__add__(0.5)
        var_24 = var_23.__mul__(16.0)
        out = paddle.tensor.creation.meshgrid(var_24, var_21)
        var_25 = out[0]
        var_26 = out[1]
        var_27 = var_26.__sub__(64.0)
        var_28 = var_25.__sub__(64.0)
        var_29 = var_26.__add__(64.0)
        var_30 = var_25.__add__(64.0)
        var_31 = paddle.tensor.manipulation.stack([var_27, var_28, var_29, var_30], axis=-1)
        var_32 = var_31.astype('float32')
        var_33 = paddle.tensor.manipulation.stack([var_26, var_25], axis=-1)
        var_34 = var_33.astype('float32')
        var_35 = var_32.reshape([-1, 4])
        var_36 = var_34.reshape([-1, 2])
        var_37 = paddle.tensor.creation.full([3800, 1], 16.0, dtype='float32')
        var_38 = paddle.tensor.creation.arange(end=38)
        var_39 = var_38.__add__(0.5)
        var_40 = var_39.__mul__(32.0)
        var_41 = paddle.tensor.creation.arange(end=25)
        var_42 = var_41.__add__(0.5)
        var_43 = var_42.__mul__(32.0)
        out = paddle.tensor.creation.meshgrid(var_43, var_40)
        var_44 = out[0]
        var_45 = out[1]
        var_46 = var_45.__sub__(128.0)
        var_47 = var_44.__sub__(128.0)
        var_48 = var_45.__add__(128.0)
        var_49 = var_44.__add__(128.0)
        var_50 = paddle.tensor.manipulation.stack([var_46, var_47, var_48, var_49], axis=-1)
        var_51 = var_50.astype('float32')
        var_52 = paddle.tensor.manipulation.stack([var_45, var_44], axis=-1)
        var_53 = var_52.astype('float32')
        var_54 = var_51.reshape([-1, 4])
        var_55 = var_53.reshape([-1, 2])
        var_56 = paddle.tensor.creation.full([950, 1], 32.0, dtype='float32')
        var_57 = paddle.tensor.creation.arange(end=19)
        var_58 = var_57.__add__(0.5)
        var_59 = var_58.__mul__(64.0)
        var_60 = paddle.tensor.creation.arange(end=13)
        var_61 = var_60.__add__(0.5)
        var_62 = var_61.__mul__(64.0)
        out = paddle.tensor.creation.meshgrid(var_62, var_59)
        var_63 = out[0]
        var_64 = out[1]
        var_65 = var_64.__sub__(256.0)
        var_66 = var_63.__sub__(256.0)
        var_67 = var_64.__add__(256.0)
        var_68 = var_63.__add__(256.0)
        var_69 = paddle.tensor.manipulation.stack([var_65, var_66, var_67, var_68], axis=-1)
        var_70 = var_69.astype('float32')
        var_71 = paddle.tensor.manipulation.stack([var_64, var_63], axis=-1)
        var_72 = var_71.astype('float32')
        var_73 = var_70.reshape([-1, 4])
        var_74 = var_72.reshape([-1, 2])
        var_75 = paddle.tensor.creation.full([247, 1], 64.0, dtype='float32')
        var_76 = paddle.tensor.creation.arange(end=10)
        var_77 = var_76.__add__(0.5)
        var_78 = var_77.__mul__(128.0)
        var_79 = paddle.tensor.creation.arange(end=7)
        var_80 = var_79.__add__(0.5)
        var_81 = var_80.__mul__(128.0)
        out = paddle.tensor.creation.meshgrid(var_81, var_78)
        var_82 = out[0]
        var_83 = out[1]
        var_84 = var_83.__sub__(512.0)
        var_85 = var_82.__sub__(512.0)
        var_86 = var_83.__add__(512.0)
        var_87 = var_82.__add__(512.0)
        var_88 = paddle.tensor.manipulation.stack([var_84, var_85, var_86, var_87], axis=-1)
        var_89 = var_88.astype('float32')
        var_90 = paddle.tensor.manipulation.stack([var_83, var_82], axis=-1)
        var_91 = var_90.astype('float32')
        var_92 = var_89.reshape([-1, 4])
        var_93 = var_91.reshape([-1, 2])
        var_94 = paddle.tensor.creation.full([70, 1], 128.0, dtype='float32')
        var_95 = paddle.tensor.manipulation.concat([var_16, var_35, var_54, var_73, var_92])
        var_96 = paddle.tensor.manipulation.concat([var_17, var_36, var_55, var_74, var_93])
        var_97 = paddle.tensor.manipulation.concat([var_18, var_37, var_56, var_75, var_94])
        var_98 = var_96.__truediv__(var_97)
        out = paddle.tensor.manipulation.split(var_98, [15200, 3800, 950, 247, 70])
        var_99 = out[0]
        var_100 = out[1]
        var_101 = out[2]
        var_102 = out[3]
        var_103 = out[4]
        return var_98, var_95, var_16, var_35, var_54, var_73, var_92, var_97



def create_inputspec(): 
    inputspec = ( 
    )
    return inputspec

def create_tensor_inputs():
    inputs = (
    )
    return inputs


def create_numpy_inputs():
    inputs = (
    )
    return inputs


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = create_tensor_inputs()
        self.net = LayerCase()
    def train(self, net, to_static, with_prim=False, with_cinn=False):
        if to_static:
            paddle.set_flags({'FLAGS_prim_all': with_prim})
            if with_cinn:
                build_strategy = paddle.static.BuildStrategy()
                build_strategy.build_cinn_pass = True
                net = paddle.jit.to_static(net, build_strategy=build_strategy, full_graph=True)
            else:
                net = paddle.jit.to_static(net, full_graph=True)
        paddle.seed(123)
        outs = net(*self.inputs)
        return outs
    def test_ast_prim_cinn(self):
        st_out = self.train(self.net, to_static=True)
        cinn_out = self.train(self.net, to_static=True, with_prim=True, with_cinn=True)
        for st, cinn in zip(paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()