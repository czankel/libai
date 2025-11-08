//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef LIBAI_MODELS_LLAMA_H
#define LIBAI_MODELS_LLAMA_H

#include <iostream>

#include <grid/tensor/tensor.h>
#include <grid/tensor/mmap.h>

namespace libai {

class LLaMAFile;

// TODO: rename to LLaMATokenizer?
struct LLaMAVocab;

/// LLaMAModel is an interface for providing a LLaMA base class without any templated paramters.
class LLaMAModel
{
 public:
  struct Parameters
  {
    size_t vocab_size_;   // size of the vocabulary table
    size_t dim_;          // same as model size or num_heads * query_size
    size_t hidden_dim_;   // hidden dimension
    size_t num_layers_;   // number of encoder/decoder layers
    size_t num_heads_;    // number of heads (multi-head attention)
    size_t num_kv_heads_; // number of key-value heads (typically same as num_heads_)
    size_t max_seq_len_;  // max sequence length
  };

  // default stream start and end markers.
  static constexpr uint32_t kBOS = 1;
  static constexpr uint32_t kEOS = 2;

 public:
  virtual ~LLaMAModel() = default;

  /// Predict predicts the next words from the input prompt.
  virtual void Predict(std::string_view prompt, size_t steps) = 0;

  /// Load creates and loads the model from the provided file.
  ///
  /// @param file   LLaMA file.
  /// @param mmap   Enable to map the tensors into memory (mmap)
  /// @returns LLaMA model
  static LLaMAModel* Load(LLaMAFile& file, bool mmap = true);

  /// Load creates and loads the model from the provided file.
  ///
  /// @param file   LLaMA file.
  /// @param device Accelerator.
  /// @param mmap   Enable to map the tensors into memory (mmap)
  /// @returns LLaMA model
  static LLaMAModel* Load(LLaMAFile& file, std::string_view device_name, bool mmap = true);
};


/// LLaMAFile is an interface for managing LLaMA files.
class LLaMAFile
{
 public:

  /// Type lists the support model file types.
  enum Type
  {
    kKarpathy,    ///> https://github.com/karpathy/llama2.c
    kGgml,        ///> https://github.com/ggerganov/ggml
  };

  // TensorType enumerates all possible tensors for the model
  enum TensorType
  {
    kEmbeddings,
    kAttentionRms,
    kAttentionQuery,
    kAttentionKey,
    kAttentionValue,
    kFeedForwardWo,
    kFeedForwardW1,
    kFeedForwardW2,
    kFeedForwardW3,
    kFeedForwardRms,
    kFinalRms,
    kOutput,
  };

 protected:
  LLaMAFile() = default;

 public:
  virtual ~LLaMAFile() = default;

  /// Load loads the model parameters and information from the file and initializes internal
  /// data structures.
  virtual void Load() = 0;

  /// DataType returns the dominant data type of the file.
  virtual const std::type_info& DataType() const = 0;

  /// GetParameters returns a reference to the model paramters.
  virtual void GetParameters(LLaMAModel::Parameters&) const = 0;

  /// GetTokenizer loads and/or returns the Vocab information.
  virtual void GetTokenizer(LLaMAVocab&) const = 0;

  /// MmapTensors maps the tensors into memory (mmap).
  virtual MMap* MapTensors() const = 0;

  /// PrintModelParameters prints the model informatio and parameter
  std::ostream& PrintModelInfo(std::ostream&) const;

  /// GetTensor returns the address of the specified tensor and optional indices.
  template <typename T, typename... Index>
  auto GetTensor(char* base, TensorType type, Index... indices) const
  {
    auto [offset, size] = GetTensorOffset(type, sizeof...(indices), indices...);
    return std::make_tuple(reinterpret_cast<T*>(base + offset), size);
  }

  /// Open opens the specified model file.
  static LLaMAFile* Open(Type file_type, std::string_view model_path);

  /// Open opens the specified model.and tokenizer file
  static LLaMAFile* Open(Type file_type, std::string_view model_path, std::string_view tokenizer_path);

 protected:
  /// GetTensorOffset returns the offset from the beginning of the file for the specified tensor
  /// The optional additional arguments define additional indices, in this order:
  ///   - layer
  virtual std::tuple<size_t, size_t> GetTensorOffset(TensorType, size_t nargs, ...) const = 0;
};

} // end namespace libai

#endif // LIBAI_MODELS_LLAMA_H
