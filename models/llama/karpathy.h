//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef _KARPATHY_H
#define _KARPATHY_H

#include <libai/models/llama.h>
#include <libai/tensor/mmap.h>
#include <libai/tensor/tensor.h>

#include "llama_vocab.h"

namespace libai {

class LLaMAFile;

/// KarpathyFile are snapshots created by the Karpathy LLaMA2 implementation.
/// For more details, see: https://github.com/karpathy/llama2.c
class KarpathyFile : public LLaMAFile
{
 public:

  /// Constructor
  ///
  /// @param path  Path for the model
  /// @param tokenizer_path  Path for the tokenizer model
  /// @throws system_error if any of the files does not exist.
  KarpathyFile(std::string_view path, std::string_view tokenizer_path = "tokenizer.bin")
   : tokenizer_path_(tokenizer_path),
     path_(path)
  {}

  virtual ~KarpathyFile() = default;

  // LLaMAFile::
  virtual void Load();
  virtual const std::type_info& DataType() const              { return typeid(float); }
  virtual void GetParameters(LLaMAModel::Parameters& p) const { p = parameters_; }
  virtual void GetTokenizer(LLaMAVocab&) const;
  virtual MMap* MapTensors() const;

 protected:
  // LLaMAFile::
  virtual std::tuple<size_t, size_t> GetTensorOffset(LLaMAFile::TensorType, size_t nargs, ...) const;

 private:
  // Data is stored as floats
  size_t OffsetOf(size_t offset) const                        { return sizeof(float) * offset; }

 private:
  LLaMAModel::Parameters  parameters_;

  std::string tokenizer_path_;
  std::string path_;
  size_t      file_size_;

  // tensor offsets into the file.
  size_t embeddings_;
  size_t final_rms_;
  size_t attention_rms_;
  size_t feed_forward_rms_;
  size_t wq_;
  size_t wk_;
  size_t wv_;
  size_t wo_;
  size_t w1_;
  size_t w2_;
  size_t w3_;
};

} // end of namespace libai

#endif // _KARPATHY_H
