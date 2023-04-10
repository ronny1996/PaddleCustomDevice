// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>

// NOLINT
#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "all_ops.h"  // NOLINT
#include "ge/ge_api.h"
#include "ge/ge_api_types.h"
#include "ge/ge_error_codes.h"
#include "ge/ge_ir_build.h"
#include "graph/graph.h"
#include "graph/tensor.h"
#include "graph/types.h"

PYBIND11_MODULE(_ge, m) {
  namespace py = pybind11;
  m.def("ge_initialize", [](const std::map<std::string, std::string>& options) {
    py::gil_scoped_release gil;
    std::map<ge::AscendString, ge::AscendString> options_;
    for (auto& item : options) {
      options_.insert({ge::AscendString(item.first.c_str()),
                       ge::AscendString(item.second.c_str())});
    }

    auto ret = ge::GEInitialize(options_);
    if (ret != ge::SUCCESS) {
      std::cout << "Initialize ge failed." << std::endl;
      return;
    }
    std::cout << "Initialize ge success." << std::endl;
  });
  m.def("ge_finalize", []() { ge::GEFinalize(); });

  m.def("dump_graph", [](const ge::Graph& g, const std::string& file) {
    auto ret = ge::aclgrphDumpGraph(g, file.c_str(), file.size());
    if (ret != ge::SUCCESS) {
      std::cout << "Dump graph failed." << std::endl;
      return;
    }
    std::cout << "Dump graph success." << std::endl;
  });

  py::enum_<ge::Format>(m, "Format")
      .value("FORMAT_NCHW", ge::Format::FORMAT_NCHW)
      .value("FORMAT_NHWC", ge::Format::FORMAT_NHWC)
      .value("FORMAT_ND", ge::Format::FORMAT_ND)
      .value("FORMAT_NC1HWC0", ge::Format::FORMAT_NC1HWC0)
      .value("FORMAT_FRACTAL_Z", ge::Format::FORMAT_FRACTAL_Z)
      .value("FORMAT_NC1C0HWPAD", ge::Format::FORMAT_NC1C0HWPAD)
      .value("FORMAT_NHWC1C0", ge::Format::FORMAT_NHWC1C0)
      .value("FORMAT_FSR_NCHW", ge::Format::FORMAT_FSR_NCHW)
      .value("FORMAT_FRACTAL_DECONV", ge::Format::FORMAT_FRACTAL_DECONV)
      .value("FORMAT_C1HWNC0", ge::Format::FORMAT_C1HWNC0)
      .value("FORMAT_FRACTAL_DECONV_TRANSPOSE",
             ge::Format::FORMAT_FRACTAL_DECONV_TRANSPOSE)
      .value("FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS",
             ge::Format::FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS)
      .value("FORMAT_NC1HWC0_C04", ge::Format::FORMAT_NC1HWC0_C04)
      .value("FORMAT_FRACTAL_Z_C04", ge::Format::FORMAT_FRACTAL_Z_C04)
      .value("FORMAT_CHWN", ge::Format::FORMAT_CHWN)
      .value("FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS",
             ge::Format::FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS)
      .value("FORMAT_HWCN", ge::Format::FORMAT_HWCN)
      .value("FORMAT_NC1KHKWHWC0", ge::Format::FORMAT_NC1KHKWHWC0)
      .value("FORMAT_BN_WEIGHT", ge::Format::FORMAT_BN_WEIGHT)
      .value("FORMAT_FILTER_HWCK", ge::Format::FORMAT_FILTER_HWCK)
      .value("FORMAT_HASHTABLE_LOOKUP_LOOKUPS",
             ge::Format::FORMAT_HASHTABLE_LOOKUP_LOOKUPS)
      .value("FORMAT_HASHTABLE_LOOKUP_KEYS",
             ge::Format::FORMAT_HASHTABLE_LOOKUP_KEYS)
      .value("FORMAT_HASHTABLE_LOOKUP_VALUE",
             ge::Format::FORMAT_HASHTABLE_LOOKUP_VALUE)
      .value("FORMAT_HASHTABLE_LOOKUP_OUTPUT",
             ge::Format::FORMAT_HASHTABLE_LOOKUP_OUTPUT)
      .value("FORMAT_HASHTABLE_LOOKUP_HITS",
             ge::Format::FORMAT_HASHTABLE_LOOKUP_HITS)
      .value("FORMAT_C1HWNCoC0", ge::Format::FORMAT_C1HWNCoC0)
      .value("FORMAT_NDHWC", ge::Format::FORMAT_NDHWC)
      .value("FORMAT_FRACTAL_ZZ", ge::Format::FORMAT_FRACTAL_ZZ)
      .value("FORMAT_FRACTAL_NZ", ge::Format::FORMAT_FRACTAL_NZ)
      .value("FORMAT_NCDHW", ge::Format::FORMAT_NCDHW)
      .value("FORMAT_DHWCN", ge::Format::FORMAT_DHWCN)
      .value("FORMAT_NDC1HWC0", ge::Format::FORMAT_NDC1HWC0)
      .value("FORMAT_FRACTAL_Z_3D", ge::Format::FORMAT_FRACTAL_Z_3D)
      .value("FORMAT_CN", ge::Format::FORMAT_CN)
      .value("FORMAT_NC", ge::Format::FORMAT_NC)
      .value("FORMAT_DHWNC", ge::Format::FORMAT_DHWNC)
      .value("FORMAT_FRACTAL_Z_3D_TRANSPOSE",
             ge::Format::FORMAT_FRACTAL_Z_3D_TRANSPOSE)
      .value("FORMAT_FRACTAL_ZN_LSTM", ge::Format::FORMAT_FRACTAL_ZN_LSTM)
      .value("FORMAT_FRACTAL_Z_G", ge::Format::FORMAT_FRACTAL_Z_G)
      .value("FORMAT_ALL", ge::Format::FORMAT_ALL)
      .value("FORMAT_NULL", ge::Format::FORMAT_NULL)
      .value("FORMAT_ND_RNN_BIAS", ge::Format::FORMAT_ND_RNN_BIAS)
      .value("FORMAT_FRACTAL_ZN_RNN", ge::Format::FORMAT_FRACTAL_ZN_RNN)
      //   .value("FORMAT_YUV", ge::Format::FORMAT_YUV)
      //   .value("FORMAT_YUV_A", ge::Format::FORMAT_YUV_A)
      .export_values();

  py::enum_<ge::DataType>(m, "DataType")
      .value("DT_FLOAT", ge::DataType::DT_FLOAT)
      .value("DT_FLOAT16", ge::DataType::DT_FLOAT16)
      .value("DT_INT8", ge::DataType::DT_INT8)
      .value("DT_INT16", ge::DataType::DT_INT16)
      .value("DT_UINT16", ge::DataType::DT_UINT16)
      .value("DT_UINT8", ge::DataType::DT_UINT8)
      .value("DT_INT32", ge::DataType::DT_INT32)
      .value("DT_INT64", ge::DataType::DT_INT64)
      .value("DT_UINT32", ge::DataType::DT_UINT32)
      .value("DT_UINT64", ge::DataType::DT_UINT64)
      .value("DT_BOOL", ge::DataType::DT_BOOL)
      .value("DT_DOUBLE", ge::DataType::DT_DOUBLE)
      .value("DT_STRING", ge::DataType::DT_STRING)
      .value("DT_DUAL_SUB_INT8", ge::DataType::DT_DUAL_SUB_INT8)
      .value("DT_DUAL_SUB_UINT8", ge::DataType::DT_DUAL_SUB_UINT8)
      .value("DT_COMPLEX64", ge::DataType::DT_COMPLEX64)
      .value("DT_COMPLEX128", ge::DataType::DT_COMPLEX128)
      .value("DT_QINT8", ge::DataType::DT_QINT8)
      .value("DT_QINT16", ge::DataType::DT_QINT16)
      .value("DT_QINT32", ge::DataType::DT_QINT32)
      .value("DT_QUINT8", ge::DataType::DT_QUINT8)
      .value("DT_QUINT16", ge::DataType::DT_QUINT16)
      .value("DT_RESOURCE", ge::DataType::DT_RESOURCE)
      .value("DT_STRING_REF", ge::DataType::DT_STRING_REF)
      .value("DT_DUAL", ge::DataType::DT_DUAL)
      .value("DT_VARIANT", ge::DataType::DT_VARIANT)
      .value("DT_BF16", ge::DataType::DT_BF16)
      .value("DT_UNDEFINED", ge::DataType::DT_UNDEFINED)
      .value("DT_INT4", ge::DataType::DT_INT4)
      .export_values();

  py::class_<ge::TensorDesc, std::shared_ptr<ge::TensorDesc>>(m, "TensorDesc")
      .def(py::init<>())
      .def("__init__",
           [](ge::TensorDesc& self,
              const std::vector<int64_t>& shape,
              ge::Format format,
              ge::DataType dtype) {
             ge::Shape shape_(shape);
             new (&self) ge::TensorDesc(shape_, format, dtype);
           })
      .def("update",
           [](ge::TensorDesc& self,
              const std::vector<int64_t>& shape,
              ge::Format format,
              ge::DataType dtype) {
             ge::Shape shape_(shape);
             self.Update(shape_, format, dtype);
           })
      .def("dtype", [](ge::TensorDesc& self) { return self.GetDataType(); })
      .def("format", [](ge::TensorDesc& self) { return self.GetFormat(); })
      .def("shape",
           [](ge::TensorDesc& self) {
             auto shape = self.GetShape();
             return shape.GetDims();
           })
      .def("orig_format",
           [](ge::TensorDesc& self) { return self.GetOriginFormat(); })
      .def("orig_shape",
           [](ge::TensorDesc& self) {
             auto shape = self.GetOriginShape();
             return shape.GetDims();
           })
      .def("set_dtype",
           [](ge::TensorDesc& self, ge::DataType dtype) {
             self.SetDataType(dtype);
           })
      .def("set_format",
           [](ge::TensorDesc& self, ge::Format format) {
             self.SetFormat(format);
           })
      .def("set_orig_format",
           [](ge::TensorDesc& self, ge::Format format) {
             self.SetOriginFormat(format);
           })
      .def("set_shape",
           [](ge::TensorDesc& self, const std::vector<int64_t>& shape) {
             ge::Shape shape_(shape);
             self.SetShape(shape_);
           })
      .def("set_orig_shape",
           [](ge::TensorDesc& self, const std::vector<int64_t>& shape) {
             ge::Shape shape_(shape);
             self.SetOriginShape(shape_);
           })
      .def("set_real_dim_cnt", [](ge::TensorDesc& self, int64_t dim_cnt) {
        self.SetRealDimCnt(dim_cnt);
      });

  py::class_<ge::Tensor, std::shared_ptr<ge::Tensor>>(m, "Tensor")
      .def(py::init<>())
      .def("__init__",
           [](ge::Tensor& self, ge::TensorDesc desc) {
             new (&self) ge::Tensor(desc);
           })
      .def(
          "__init__",
          [](ge::Tensor& self,
             ge::TensorDesc desc,
             uint64_t data_ptr,
             uint64_t size,
             bool device) {
            new (&self) ge::Tensor(desc);
            if (device) {
              self.SetData(reinterpret_cast<uint8_t*>(data_ptr),
                           size,
                           [](uint8_t*) { return; });
            } else {
              self.SetData(reinterpret_cast<uint8_t*>(data_ptr), size);
            }
          },
          py::arg("desc"),
          py::arg("data_ptr"),
          py::arg("size"),
          py::arg("device") = false)
      .def("is_valid", [](ge::Tensor& self) { return self.IsValid(); })
      .def("data", [](ge::Tensor& self) { return self.GetData(); })
      .def("desc", [](ge::Tensor& self) { return self.GetTensorDesc(); })
      .def("size", [](ge::Tensor& self) { return self.GetSize(); })
      .def(
          "set_data",
          [](ge::Tensor& self, uint64_t data_ptr, uint64_t size, bool device) {
            if (device) {
              self.SetData(reinterpret_cast<uint8_t*>(data_ptr),
                           size,
                           [](uint8_t*) { return; });
            } else {
              self.SetData(reinterpret_cast<uint8_t*>(data_ptr), size);
            }
          },
          py::arg("data_ptr"),
          py::arg("size"),
          py::arg("device") = false)
      .def("set_desc", [](ge::Tensor& self, ge::TensorDesc& desc) {
        self.SetTensorDesc(desc);
      });

  py::class_<ge::Session, std::shared_ptr<ge::Session>>(m, "Session")
      .def("__init__",
           [](ge::Session& self,
              const std::map<std::string, std::string>& options) {
             std::map<ge::AscendString, ge::AscendString> options_;
             for (auto& item : options) {
               options_.insert({ge::AscendString(item.first.c_str()),
                                ge::AscendString(item.second.c_str())});
             }
             new (&self) ge::Session(options_);
           })
      .def("add_graph",
           [](ge::Session& self,
              uint32_t graphId,
              const ge::Graph& graph,
              const std::map<std::string, std::string>& options) {
             std::map<ge::AscendString, ge::AscendString> options_;
             for (auto& item : options) {
               options_.insert({ge::AscendString(item.first.c_str()),
                                ge::AscendString(item.second.c_str())});
             }
             auto ret = self.AddGraph(graphId, graph, options_);
             if (ret != ge::SUCCESS) {
               std::cout << "AddGraph failed." << std::endl;
               return;
             }
             std::cout << "AddGraph success." << std::endl;
           })
      .def("add_graph_with_copy",
           [](ge::Session& self,
              uint32_t graphId,
              const ge::Graph& graph,
              const std::map<std::string, std::string>& options) {
             std::map<ge::AscendString, ge::AscendString> options_;
             for (auto& item : options) {
               options_.insert({ge::AscendString(item.first.c_str()),
                                ge::AscendString(item.second.c_str())});
             }
             auto ret = self.AddGraphWithCopy(graphId, graph, options_);
             if (ret != ge::SUCCESS) {
               std::cout << "AddGraph failed." << std::endl;
               return;
             }
             std::cout << "AddGraph success." << std::endl;
           })
      .def("remove_graph",
           [](ge::Session& self, uint32_t graph_id) {
             self.RemoveGraph(graph_id);
           })
      .def("run_graph",
           [](ge::Session& self,
              uint32_t graph_id,
              const std::vector<ge::Tensor>& inputs) {
             py::gil_scoped_release gil;

             std::vector<ge::Tensor> outputs;
             auto ret = self.RunGraph(graph_id, inputs, outputs);
             if (ret != ge::SUCCESS) {
               std::cout << "RunGraph failed." << std::endl;
               return outputs;
             }
             std::cout << "RunGraph success." << std::endl;
             return outputs;
           })
      .def("run_graph_with_device_inputs",
           [](ge::Session& self,
              uint32_t graph_id,
              const std::vector<ge::Tensor>& inputs,
              std::vector<ge::Tensor>& outputs) {
             py::gil_scoped_release gil;
             aclrtStream stream;
             aclrtCreateStream(&stream);
             auto ret = self.RunGraphWithStreamAsync(
                 graph_id, reinterpret_cast<void*>(stream), inputs, outputs);
             if (ret != ge::SUCCESS) {
               std::cout << "RunGraph failed." << std::endl;
               return;
             }
             std::cout << "RunGraph success." << std::endl;
             aclrtSynchronizeStream(stream);
             aclrtDestroyStream(stream);
             return;
           });

  py::class_<ge::Graph, std::shared_ptr<ge::Graph>>(m, "Graph")
      .def("__init__",
           [](ge::Graph& self, const std::string& name) {
             new (&self) ge::Graph(name.c_str());
           })
      .def("set_inputs",
           [](ge::Graph& self, const std::vector<ge::Operator>& inputs) {
             self.SetInputs(inputs);
           })
      .def("set_outputs",
           [](ge::Graph& self,
              const std::vector<std::pair<ge::Operator, std::vector<size_t>>>&
                  outputs) { self.SetOutputs(outputs); })
      .def(
          "set_outputs",
          [](ge::Graph& self,
             const std::vector<std::pair<ge::Operator, std::string>>& outputs) {
            std::vector<std::pair<ge::Operator, ge::AscendString>> outputs_;
            for (auto& item : outputs) {
              outputs_.push_back(std::make_pair(
                  item.first, ge::AscendString(item.second.c_str())));
            }
            self.SetOutputs(outputs_);
          })
      .def("set_targets",
           [](ge::Graph& self, const std::vector<ge::Operator>& targets) {
             self.SetTargets(targets);
           })
      .def("add_op", [](ge::Graph& self, ge::Operator& op) { self.AddOp(op); });

  py::class_<ge::Operator, std::shared_ptr<ge::Operator>>(m, "Op")
      .def_static(
          "create_op",
          [](const std::string& op_type, const std::string& op_name) {
            return ge::OperatorFactory::CreateOperator(op_name.c_str(),
                                                       op_type.c_str());
          },
          py::arg("op_name"),
          py::arg("op_type") = "")
      .def(py::init<>())
      .def("is_empty", [](ge::Operator& self) { return self.IsEmpty(); })
      .def("type", [](ge::Operator& self) { return self.GetOpType(); })
      .def(
          "set_input",
          [](ge::Operator& self,
             const std::string& in_name,
             ge::Operator& out_op,
             const std::string& out_name) {
            self.SetInput(in_name.c_str(), out_op, out_name.c_str());
          },
          py::arg("in_name"),
          py::arg("out_op"),
          py::arg("out_name"))
      .def(
          "set_input",
          [](ge::Operator& self,
             const std::string& in_name,
             ge::Operator& out_op,
             int out_index) {
            self.SetInput(in_name.c_str(), out_op, out_index);
          },
          py::arg("in_name"),
          py::arg("out_op"),
          py::arg("out_index"))
      .def(
          "set_input",
          [](ge::Operator& self,
             int in_index,
             ge::Operator& out_op,
             int out_index) { self.SetInput(in_index, out_op, out_index); },
          py::arg("in_name"),
          py::arg("out_op"),
          py::arg("out_index"))
      .def("add_control_input",
           [](ge::Operator& self, ge::Operator& other) {
             self.AddControlInput(other);
           })
      .def("set_attr",
           [](ge::Operator& self, const std::string& attr, int64_t value) {
             self.SetAttr(attr.c_str(), value);
           })
      .def("set_attr",
           [](ge::Operator& self, const std::string& attr, int32_t value) {
             self.SetAttr(attr.c_str(), value);
           })
      .def("set_attr",
           [](ge::Operator& self, const std::string& attr, uint32_t value) {
             self.SetAttr(attr.c_str(), value);
           })
      .def("set_attr",
           [](ge::Operator& self, const std::string& attr, float value) {
             self.SetAttr(attr.c_str(), value);
           })
      .def("set_attr",
           [](ge::Operator& self, const std::string& attr, bool value) {
             self.SetAttr(attr.c_str(), value);
           })
      .def("set_attr",
           [](ge::Operator& self,
              const std::string& attr,
              const std::string& value) {
             self.SetAttr(attr.c_str(), value.c_str());
           })
      .def("set_attr",
           [](ge::Operator& self,
              const std::string& attr,
              const std::vector<int64_t>& value) {
             self.SetAttr(attr.c_str(), value);
           })
      .def("set_attr",
           [](ge::Operator& self,
              const std::string& attr,
              const std::vector<int32_t>& value) {
             self.SetAttr(attr.c_str(), value);
           })
      .def("set_attr",
           [](ge::Operator& self,
              const std::string& attr,
              const std::vector<uint32_t>& value) {
             self.SetAttr(attr.c_str(), value);
           })
      .def("set_attr",
           [](ge::Operator& self,
              const std::string& attr,
              const std::vector<float>& value) {
             self.SetAttr(attr.c_str(), value);
           })
      .def("set_attr",
           [](ge::Operator& self,
              const std::string& attr,
              const std::vector<std::string>& value) {
             self.SetAttr(attr.c_str(), value);
           })
      .def(
          "update_input_desc",
          [](ge::Operator& self, const std::string& name, ge::TensorDesc desc) {
            self.UpdateInputDesc(name.c_str(), desc);
          })
      .def(
          "update_output_desc",
          [](ge::Operator& self, const std::string& name, ge::TensorDesc desc) {
            self.UpdateOutputDesc(name.c_str(), desc);
          });
};
