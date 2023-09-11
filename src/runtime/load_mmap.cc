/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include <dmlc/json.h>
#include <dmlc/memory_io.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/serializer.h>

#include <fstream>
#include <unordered_map>
#include <vector>

#include "./file_utils.h"

namespace tvm {
namespace runtime {

constexpr uint64_t kDataAlignedTVMNDArrayListMagic = 0x5049CB7F7E58D4F0;

/*!
 * \brief A seekable dmlc stream which wraps standard file operations.
 */
struct SeekBinaryFileStream : public dmlc::SeekStream {
 public:
  SeekBinaryFileStream(const std::string& path, std::string mode) {
    const char* fname = path.c_str();

    CHECK(mode == "wb" || mode == "rb") << "Only allowed modes are 'wb' and 'rb'";
    read_ = mode == "rb";
    fp_ = std::fopen(fname, mode.c_str());
    CHECK(fp_ != nullptr) << "Unable to open file " << path;
  }
  virtual ~SeekBinaryFileStream(void) { this->Close(); }
  virtual size_t Read(void* ptr, size_t size) {
    CHECK(read_) << "File opened in write-mode, cannot read.";
    CHECK(fp_ != nullptr) << "File is closed";
    return std::fread(ptr, 1, size, fp_);
  }
  virtual void Write(const void* ptr, size_t size) {
    CHECK(!read_) << "File opened in read-mode, cannot write.";
    CHECK(fp_ != nullptr) << "File is closed";
    CHECK(std::fwrite(ptr, 1, size, fp_) == size) << "SimpleBinaryFileStream.Write incomplete";
  }
  inline void Close(void) {
    if (fp_ != nullptr) {
      std::fclose(fp_);
      fp_ = nullptr;
    }
  }

  virtual void Seek(size_t pos) {
#ifndef _MSC_VER
    CHECK(!std::fseek(fp_, static_cast<long>(pos), SEEK_SET));  // NOLINT(*)
#else                                                           // _MSC_VER
    CHECK(!_fseeki64(fp_, pos, SEEK_SET));
#endif                                                          // _MSC_VER
  }
  virtual size_t Tell(void) {
#ifndef _MSC_VER
    return std::ftell(fp_);
#else   // _MSC_VER
    return _ftelli64(fp_);
#endif  // _MSC_VER
  }
  virtual bool AtEnd(void) const { return std::feof(fp_) != 0; }

 private:
  std::FILE* fp_ = nullptr;
  bool read_;
};  // class SimpleBinaryFileStream

/*!
 * \brief An effectiveless writable stream to count data length.
 */
struct PreCountWriteStreamWrapper : public dmlc::Stream {
 public:
  size_t Read(void* ptr, size_t size) final {
    LOG(FATAL) << "Do not support read";
    return 0;
  }
  void Write(const void* ptr, size_t size) { offset_ += size; }
  size_t offset_ = 0;
};

size_t GetTensorHeaderSize(const DLTensor* tensor) {
  PreCountWriteStreamWrapper pre_cnt;
  {
    dmlc::Stream* strm = &pre_cnt;
    uint64_t header = kTVMNDArrayMagic, reserved = 0;
    strm->Write(header);
    strm->Write(reserved);
    strm->Write(tensor->device);
    strm->Write(tensor->ndim);
    strm->Write(tensor->dtype);
    int ndim = tensor->ndim;
    strm->WriteArray(tensor->shape, ndim);
    int type_bytes = (tensor->dtype.bits + 7) / 8;
    int64_t num_elems = 1;
    for (int i = 0; i < ndim; ++i) {
      num_elems *= tensor->shape[i];
    }
    int64_t data_byte_size = type_bytes * num_elems;
    strm->Write(data_byte_size);
  }
  return pre_cnt.offset_;
}

NDArray LoadNDArray(dmlc::SeekStream* strm, const char* begin) {
  DLTensor dl_tensor;
  std::vector<int64_t> shape;
  ICHECK(tvm::runtime::LoadDLTensorMetaData(strm, &dl_tensor, &shape)) << "Invalid DLTensor file format";
 
  int64_t num_elems = 1;
  int elem_bytes = (dl_tensor.dtype.bits + 7) / 8;
  for (int i = 0; i < dl_tensor.ndim; ++i) {
    num_elems *= dl_tensor.shape[i];
  }
  int64_t data_byte_size;
  ICHECK(strm->Read(&data_byte_size)) << "Invalid DLTensor file format";
  ICHECK_GT(data_byte_size, num_elems * elem_bytes) << "Invalid DLTensor file format";
  int64_t pad = data_byte_size - num_elems * elem_bytes;
  if (pad > 0) {
    strm->Seek(strm->Tell() + pad);
  }
  dl_tensor.data = const_cast<char*>(begin + strm->Tell());
  dl_tensor.shape = (new std::vector<int64_t>(shape))->data();
  strm->Seek(strm->Tell() + data_byte_size - pad);
  return NDArray::FromExternalDLTensor(dl_tensor);
}

Map<String, NDArray> LoadParamsMMap(const std::string& filepath) {
  auto* fp = fopen(filepath.c_str(), "r");
  fseek(fp, 0L, SEEK_END);
  size_t filesize = ftell(fp);
  auto addr = mmap(nullptr, filesize, PROT_READ, MAP_PRIVATE, fileno(fp), 0);
  dmlc::MemoryFixedSizeStream mstrm(addr, filesize);
  dmlc::SeekStream* strm = &mstrm;

  Map<String, NDArray> params;
  uint64_t header, reserved;

  ICHECK(strm->Read(&header)) << "Invalid parameters file format";
  ICHECK(header == kTVMNDArrayListMagic) << "Invalid parameters file format";
  ICHECK(strm->Read(&reserved)) << "Invalid parameters file format";

  std::vector<std::string> names;
  ICHECK(strm->Read(&names)) << "Invalid parameters file format";
  uint64_t sz;
  strm->Read(&sz);
  size_t size = static_cast<size_t>(sz);
  ICHECK(size == names.size()) << "Invalid parameters file format";
  for (size_t i = 0; i < size; ++i) {
    NDArray temp = LoadNDArray(strm, reinterpret_cast<char*>(addr));
    params.Set(names[i], temp);
  }
  return params;
}

void SaveParamsMMap(const std::string& filepath, const Map<String, NDArray>& params) {
  SeekBinaryFileStream file_strm(filepath, "wb");

  std::vector<std::string> names;
  std::vector<const DLTensor*> arrays;
  for (auto& p : params) {
    names.push_back(p.first);
    arrays.push_back(p.second.operator->());
  }
  uint64_t header = kTVMNDArrayListMagic;
  uint64_t reserved = 0;

  {
    dmlc::SeekStream* strm = &file_strm;
    strm->Write(header);
    strm->Write(reserved);
    strm->Write(names);
    uint64_t sz = static_cast<uint64_t>(arrays.size());
    strm->Write(sz);

    for (size_t i = 0; i < sz; ++i) {
      size_t tensor_meta_size = GetTensorHeaderSize(arrays[i]);
      size_t data_offset = strm->Tell() + tensor_meta_size;
      const size_t align = runtime::kAllocAlignment;
      uint32_t pad = (align - (sizeof(uint32_t) + data_offset) % align) % align;
      tvm::runtime::SaveDLTensor(strm, arrays[i], pad);
    }
  }
}

TVM_REGISTER_GLOBAL("runtime.LoadParamsMMap").set_body_typed([](const std::string& filepath) {
  return LoadParamsMMap(filepath);
});

TVM_REGISTER_GLOBAL("runtime.SaveParamsMMap")
    .set_body_typed([](const Map<String, NDArray>& params, const std::string& filepath) {
      return SaveParamsMMap(filepath, params);
    });

}  // namespace runtime
}  // namespace tvm

