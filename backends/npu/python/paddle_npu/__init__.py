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

from . import _ge


def to_ge_tensor(tensor):
    array = tensor.numpy()
    desc = _ge.TensorDesc(tensor.shape, _ge.Format.FORMAT_ND, _ge.DataType.DT_FLOAT)
    return _ge.Tensor(desc, array.__array_interface__["data"][0], array.nbytes)


def to_pd_tensor(tensor):
    pass


def ge_executor():
    pass
