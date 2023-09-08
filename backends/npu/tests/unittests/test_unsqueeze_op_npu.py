# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import numpy as np
import unittest

from tests.op_test import OpTest
import paddle

paddle.enable_static()


class TestUnsqueeze2Op(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "unsqueeze2"
        self.place = paddle.CustomPlace("npu", 0)
        self.init_test_case()
        self.x = np.random.random(self.ori_shape).astype("float32")
        self.inputs = {"X": OpTest.np_dtype_to_base_dtype(self.x)}
        self.init_attrs()
        self.outputs = {
            "Out": self.x.reshape(self.new_shape),
            "XShape": np.random.random(self.ori_shape).astype("float32"),
        }

    def set_npu(self):
        self.__class__.use_custom_device = True

    def test_check_output(self):
        self.check_output_with_place(self.place, no_check_set=["XShape"])

    @unittest.skip("skip check_grad because unstable.")
    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["X"], "Out")

    def init_test_case(self):
        self.ori_shape = (3, 40)
        self.axes = (0, 2)
        self.new_shape = (1, 3, 1, 40)

    def init_attrs(self):
        self.attrs = {"axes": self.axes}


# Correct: There is mins axis.
class TestUnsqueeze2Op1(TestUnsqueeze2Op):
    def init_test_case(self):
        self.ori_shape = (20, 5)
        self.axes = (0, -2)
        self.new_shape = (1, 20, 1, 5)


# Correct: No axes input.
class TestUnsqueeze2Op2(TestUnsqueeze2Op):
    def init_test_case(self):
        self.ori_shape = (20, 5)
        self.axes = ()
        self.new_shape = (20, 5)


# Correct: Just part of axes be squeezed.
class TestUnsqueeze2Op3(TestUnsqueeze2Op):
    def init_test_case(self):
        self.ori_shape = (6, 5, 1, 4)
        self.axes = (1, -1)
        self.new_shape = (6, 1, 5, 1, 4, 1)


if __name__ == "__main__":
    unittest.main()
