//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <cstdarg>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <typeinfo>

#include <grid/models/llama.h>
#include <grid/util/demangle.h>

#include "ggml.h"

namespace libai {

const char* TensorNames[] =
{
  "token_embd.weight",          // kEmbeddings      [dim]
  "blk.%d.attn_norm.weight",    // kAttentionRms    [dim]
  "blk.%d.attn_q.weight",       // kAttentionQuery  [dim,dim]
  "blk.%d.attn_k.weight",       // kAttentionKey    [kvdim,dim]
  "blk.%d.attn_v.weight",       // kAttentionValue  [kvdim,dim]
  "blk.%d.attn_output.weight",  // kFeedForwardWo   [dim,dim]
  "blk.%d.ffn_gate.weight",     // kFeedForwardW1   [hidden,dim]
  "blk.%d.ffn_down.weight",     // kFeedForwardW2   [dim,hidden]
  "blk.%d.ffn_up.weight",       // kFeedForwardW3   [hidden,dim]
  "blk.%d.ffn_norm.weight",     // kFeedForwardRms  [dim]
  "output_norm.weight",         // kFinalRms        [dim]
  "output.weight"               // kOutput          [vocab,dim]
};


// ifstream_helper is a helper class to use template deduction to read data from an ifstream.
struct ifstream_helper
{
  ifstream_helper(std::ifstream& ifs) : ifs_(ifs) {}

  // read is a generic read of len bytes to destination dest
  void read(char* dest, size_t len)
  {
    ifs_.read(dest, len);
  }

  // read reads and returns a specific type, which requires to specify the type: read<TYPE>()
  template <typename T>
  T read()
  {
    T value;
    read(value);
    return value;
  }

  // read reads a specific type into the provided variable; the template is deduced: read(VARIABLE)
  template <typename T>
  void read(T& value)
  {
    ifs_.read(reinterpret_cast<char*>(&value), sizeof(value));
  }

  // readstring reads and returns a string converted to std::string; the string is stored as <len><string>
  std::string readstring()
  {
    std::string str;
    uint64_t size;
    read(size);
    str.resize(size);
    ifs_.read(&str[0], size);
    return str;
  }

  template <typename T>
  auto readarray(size_t size)
  {
    if constexpr (std::is_same<T, std::string>::value)
    {
      std::vector<std::string> array(size);
      for (size_t i = 0; i < size; i++)
        array[i] = readstring();
      return array;
    }
    else if constexpr (std::is_same<T, bool>::value)
    {
      std::vector<T> array;
      for (size_t i = 0; i < size; i++)
        array[i] = read<bool>();
      return array;
    }
    else
    {
      std::vector<T> array;
      array.resize(size);
      ifs_.read(reinterpret_cast<char*>(&array[0]), sizeof(T) * size);
      return array;
    }
  }

  std::ifstream& ifs_;
};


GgmlFile::GgmlValue GgmlFile::ReadArray(ifstream_helper& is)
{
  auto type = is.read<GgmlType>();
  auto size = is.read<uint64_t>();

  switch(type)
  {
    case kGgufTypeU8:       return GgmlValue{type, size, is.readarray<uint8_t>(size)}; break;
    case kGgufTypeI8:       return GgmlValue{type, size, is.readarray<int8_t>(size)}; break;
    case kGgufTypeU16:      return GgmlValue{type, size, is.readarray<uint16_t>(size)}; break;
    case kGgufTypeI16:      return GgmlValue{type, size, is.readarray<int16_t>(size)}; break;
    case kGgufTypeU32:      return GgmlValue{type, size, is.readarray<uint32_t>(size)}; break;
    case kGgufTypeI32:      return GgmlValue{type, size, is.readarray<int32_t>(size)}; break;
    case kGgufTypeU64:      return GgmlValue{type, size, is.readarray<uint64_t>(size)}; break;
    case kGgufTypeI64:      return GgmlValue{type, size, is.readarray<int64_t>(size)}; break;
    case kGgufTypeFloat32:  return GgmlValue{type, size, is.readarray<float>(size)}; break;
    case kGgufTypeFloat64:  return GgmlValue{type, size, is.readarray<double>(size)}; break;
    case kGgufTypeBool:     return GgmlValue{type, size, is.readarray<bool>(size)}; break;
    case kGgufTypeString:   return GgmlValue{type, size, is.readarray<std::string>(size)}; break;
    default: throw std::runtime_error("invalid value type in key-value table " + std::to_string(type));
  }
}


std::tuple<std::string, GgmlFile::GgmlValue> GgmlFile::ReadKeyValue(ifstream_helper& is)
{
  std::string key = is.readstring();
  auto type = is.read<GgmlType>();
  switch (type)
  {
    case kGgufTypeU8:       return std::make_tuple(key, GgmlValue{type, -1, is.read<uint8_t>()}); break;
    case kGgufTypeI8:       return std::make_tuple(key, GgmlValue{type, -1, is.read<int8_t>()}); break;
    case kGgufTypeU16:      return std::make_tuple(key, GgmlValue{type, -1, is.read<uint16_t>()}); break;
    case kGgufTypeI16:      return std::make_tuple(key, GgmlValue{type, -1, is.read<int16_t>()}); break;
    case kGgufTypeU32:      return std::make_tuple(key, GgmlValue{type, -1, is.read<uint32_t>()}); break;
    case kGgufTypeI32:      return std::make_tuple(key, GgmlValue{type, -1, is.read<int32_t>()}); break;
    case kGgufTypeU64:      return std::make_tuple(key, GgmlValue{type, -1, is.read<uint64_t>()}); break;
    case kGgufTypeI64:      return std::make_tuple(key, GgmlValue{type, -1, is.read<int64_t>()}); break;
    case kGgufTypeFloat32:  return std::make_tuple(key, GgmlValue{type, -1, is.read<float>()}); break;
    case kGgufTypeFloat64:  return std::make_tuple(key, GgmlValue{type, -1, is.read<double>()}); break;
    case kGgufTypeBool:     return std::make_tuple(key, GgmlValue{type, -1, is.read<bool>()}); break;
    case kGgufTypeString:   return std::make_tuple(key, GgmlValue{type, -1, is.readstring()}); break;
    case kGgufTypeArray:    return std::make_tuple(key, ReadArray(is)); break;
    default: throw std::runtime_error("invalid value type in key-value table" + std::to_string(type));
  }
}


template <typename T>
const T& GgmlFile::GetValue(std::string key) const
{
  auto it = kv_map_.find(key);
  if (it == kv_map_.end())
    throw std::runtime_error(key + " missing in model kv table");
  try {
    return std::any_cast<const T&>(it->second.value);
  } catch (...) {
    throw std::runtime_error("failed to cast " + key + " to " + Demangle(typeid(T).name()));
  }
}


const GgmlFile::GgmlTensor& GgmlFile::GetTensor(std::string key) const
{
  auto it = tensor_map_.find(key);
  if (it == tensor_map_.end())
    throw std::runtime_error("no such tensor: " + key);
  return it->second;
}

//
// GgmlFile
//

void GgmlFile::Load()
{
  std::ifstream ifs(path_, std::ios::in | std::ios::binary);
  if (!ifs)
    throw std::runtime_error("failed to open file");

  ifstream_helper is(ifs);
  is.read(magic_);

  if (magic_ != kGgmlMagicGGUF)
    throw std::runtime_error("not a 'gguf' file");

  is.read(version_);
  if (version_ == 1)
    throw std::runtime_error("gguf version 1 not supported");

  is.read(n_tensors_);
  is.read(n_kv_);

  // read key-value information
  for (uint64_t i = 0; i < n_kv_; ++i)
  {
    auto [key, value] = ReadKeyValue(is);
    kv_map_[key] = value;
  }

  model_arch_ = GetValue<std::string>("general.architecture");
  ftype_ = static_cast<GgmlDataType>(GetValue<uint32_t>("general.file_type"));

  //
  // Tokens
  //

  for (uint64_t i = 0; i < n_tensors_; ++i)
  {
    std::string name = is.readstring();
    auto rank = is.read<int>();
    auto dims = is.readarray<int64_t>(rank); // note that dims are inverse ordered
    auto type = is.read<GgmlType>();
    auto offset = is.read<size_t>();

    size_t size = GgmlFileTypeSize[ftype_];
    for (int i = 0; i < rank; i++)
      size *= dims[i];

    tensor_map_[name] = GgmlTensor{type, offset, size};
  }

  // TODO: check for updated alignment in KV
  size_t offset = ifs.tellg();
  data_offset_ = (offset + 31) & -32;

  ifs.seekg(0, ifs.end);
  file_size_ = ifs.tellg();
  ifs.close();

  if (!ifs.good())
    throw std::runtime_error("failed to read file");

  parameters_.dim_ = GetValue<uint32_t>(model_arch_ + ".embedding_length");
  parameters_.vocab_size_ = GetValue<std::vector<std::string>>("tokenizer.ggml.tokens").size();
  parameters_.hidden_dim_ = GetValue<uint32_t>(model_arch_ + ".feed_forward_length");
  parameters_.num_layers_ = GetValue<uint32_t>(model_arch_ + ".block_count");
  parameters_.num_heads_ = GetValue<uint32_t>(model_arch_ + ".attention.head_count");
  parameters_.num_kv_heads_ = GetValue<uint32_t>(model_arch_ + ".attention.head_count_kv");
  parameters_.max_seq_len_ = GetValue<uint32_t>(model_arch_ + ".context_length");

  //llama.rope.dimension_count: 128
  //llama.attention.layer_norm_rms_epsilon: 1e-05
  //llama.rope.freq_base: 10000
}


const std::type_info& GgmlFile::DataType() const
{
  switch (ftype_)
  {
    case kGgmlDataTypeF32:  return typeid(float);
    case kGgmlDataTypeF16:  return typeid(libai::float16_t);
    default: throw std::runtime_error("DataType not supported");
  }
}


void GgmlFile::GetTokenizer(LLaMAVocab& vocab) const
{
  vocab.scores_.resize(parameters_.vocab_size_);
  const auto tokens = GetValue<std::vector<std::string>>("tokenizer.ggml.tokens");
  const auto scores = GetValue<std::vector<float>>("tokenizer.ggml.scores");
  vocab.bos_token_ = GetValue<uint32_t>("tokenizer.ggml.bos_token_id");
  vocab.eos_token_ = GetValue<uint32_t>("tokenizer.ggml.eos_token_id");
  vocab.add_bos_token_ = GetValue<bool>("tokenizer.ggml.add_bos_token");
  vocab.add_eos_token_ = GetValue<bool>("tokenizer.ggml.add_eos_token");

  for (size_t i = 0; i < parameters_.vocab_size_; i++)
  {
    vocab.tokens_[tokens[i]] = i;
    vocab.scores_[i] = LLaMAVocab::TokenScore{tokens[i], scores[i]};
  }
}


MMap* GgmlFile::MapTensors() const
{
  return MMap::MMapFile(path_);
}


std::tuple<size_t, size_t> GgmlFile::GetTensorOffset(TensorType type, size_t nargs, ...) const
{
  va_list va;
  va_start(va, nargs);
  std::string name = TensorNames[type];

  if (nargs > 0)
  {
    size_t len = name.size() + nargs * 100;
    name.resize(len);
    int cnt = vsnprintf(&name[0], len, TensorNames[type], va);
    if (cnt < 0 || static_cast<size_t>(cnt) == len)
      throw std::runtime_error("");
    name.resize(cnt);
  }

  const GgmlTensor& tensor = GetTensor(name);
  return std::make_tuple(tensor.offset + data_offset_, tensor.size);
}

} // end of namespace libai
