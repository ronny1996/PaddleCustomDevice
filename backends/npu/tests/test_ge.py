# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle_npu._ge as ge

paddle.set_device("cpu")

ge.ge_initialize(
    {
        "ge.exec.deviceId": "0",
        "ge.graphRunMode": "1",
        "ge.exec.precision_mode": "allow_fp32_to_fp16",
    }
)

g = ge.Graph("")
x = ge.Op.create_op("Data", "data_1")
x.set_attr("index", 0)
weight = ge.Op.create_op("Data", "data_2")
weight.set_attr("index", 1)
add = ge.Op.create_op("Add", "add_1")
add.set_input("x1", x, 0)
add.set_input("x2", weight, 0)
g.set_inputs([x, weight])
g.set_outputs([(add, [0])])

x_desc = ge.TensorDesc([1, 2, 3], ge.Format.FORMAT_ND, ge.DataType.DT_FLOAT)
y_desc = ge.TensorDesc([4, 2, 3], ge.Format.FORMAT_ND, ge.DataType.DT_FLOAT)
x = paddle.rand([1, 2, 3]).astype("float32")
y = paddle.rand([4, 2, 3]).astype("float32")
x_tensor = ge.Tensor(x_desc, x.data_ptr(), x.size * 4, True)
y_tensor = ge.Tensor(y_desc, y.data_ptr(), y.size * 4, True)

sess = ge.Session({})
sess.add_graph(0, g, {})
outputs = sess.run_graph(0, [x_tensor, y_tensor])
print([t.desc().shape() for t in outputs])

ge.dump_graph(g, "1.txt")

ge.ge_finalize()
