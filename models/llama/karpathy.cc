//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <cstdarg>
#include <fstream>
#include <fcntl.h>
#include <sys/stat.h>

#include <libai/models/llama.h>

#include "karpathy.h"

namespace libai {

void KarpathyFile::Load()
{
  struct FileParameters
  {
    int dim;          // transformer dimension
    int hidden_dim;   // number of ffn dimensions
    int n_layers;     // number of layers
    int n_heads;      // number of query heads
    int n_kv_heads;   // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size;   // vocabulary size, usually 256 (byte-level)
    int max_seq_len;  // max sequence length
  };

  std::ifstream ifs(path_, std::ios::in | std::ios::binary);
  if (!ifs)
    throw std::runtime_error("failed to open file");

  FileParameters p;
  ifs.read(reinterpret_cast<char*>(&p), sizeof(p));
  parameters_.vocab_size_ = p.vocab_size;
  parameters_.dim_ = p.dim;
  parameters_.hidden_dim_ = p.hidden_dim;
  parameters_.num_layers_ = p.n_layers;
  parameters_.num_heads_ = p.n_heads;
  parameters_.num_kv_heads_ = p.n_kv_heads;
  parameters_.max_seq_len_ = p.max_seq_len;

  ifs.seekg(0, ifs.end);
  file_size_ = ifs.tellg();
  ifs.close();

  if (!ifs.good())
    throw std::runtime_error("failed to read file");

  // Note that Karpathy combines the weights for each tensor instead of separating tensors by layer
  int head_size = p.dim/p.n_heads;
  size_t offset = sizeof(FileParameters);
  embeddings_ = offset;
  offset += OffsetOf(p.vocab_size * p.dim);
  attention_rms_ = offset;
  offset += OffsetOf(p.n_layers * p.dim);
  wq_ = offset;
  offset += OffsetOf(p.n_layers * p.dim * p.n_heads * head_size);
  wk_ = offset;
  offset += OffsetOf(p.n_layers * p.dim * p.n_kv_heads * head_size);
  wv_ = offset;
  offset += OffsetOf(p.n_layers * p.dim * p.n_kv_heads * head_size);
  wo_ = offset;
  offset += OffsetOf(p.n_layers * p.dim * p.n_heads * head_size);
  feed_forward_rms_ = offset;
  offset += OffsetOf(p.n_layers * p.dim);
  w1_ = offset;
  offset += OffsetOf(p.n_layers * p.hidden_dim * p.dim);
  w2_ = offset;
  offset += OffsetOf(p.n_layers * p.hidden_dim * p.dim);
  w3_ = offset;
  offset += OffsetOf(p.n_layers * p.hidden_dim * p.dim);
  final_rms_ = offset;
  offset += OffsetOf(p.dim);
  // offset += seq_len * head_size * sizeof(value_type);  // skip freq_cis_{real|imag}
  // wcls_weights_ = shared_weights ? w->token_embedding_table : offset;
  // offset += ...
}


void KarpathyFile::GetTokenizer(LLaMAVocab& vocab) const
{
  std::ifstream ifs(tokenizer_path_, std::ios::in | std::ios::binary);
  if (!ifs)
    throw std::runtime_error("failed to open tokenizer file");

  int max_token_length = 0;
  ifs.read(reinterpret_cast<char*>(&max_token_length), sizeof(max_token_length));

  vocab.max_token_length_ = max_token_length;
  vocab.scores_.resize(parameters_.vocab_size_);
  for (size_t i = 0; i < parameters_.vocab_size_; i++)
  {
    float score = 0.0f;
    ifs.read(reinterpret_cast<char*>(&score), sizeof(score));
    int len = 0;
    ifs.read(reinterpret_cast<char*>(&len), sizeof(len));
    if (len > max_token_length)
      throw std::runtime_error("token length exceeds max token length");

    std::string symbol(len, '\0');
    ifs.read(&symbol[0], len);

    vocab.tokens_[symbol] = i;
    vocab.scores_[i] = LLaMAVocab::TokenScore{symbol, score};
  }

  ifs.close();
}


MMap* KarpathyFile::MapTensors() const
{
  return MMap::MMapFile(path_);
}


std::tuple<size_t, size_t> KarpathyFile::GetTensorOffset(TensorType type, size_t nargs, ...) const
{
  va_list va;
  va_start(va, nargs);

  size_t layer = 0;
  if (nargs > 0)
    layer = va_arg(va, size_t);

  int dim = parameters_.dim_;
  int hidden_dim = parameters_.hidden_dim_;
  int n_heads = parameters_.num_heads_;
  int n_kv_heads = parameters_.num_kv_heads_;
  int head_size = dim / n_heads;

  size_t size, offset;
  switch (type)
  {
    case kEmbeddings:     // fall through
    case kOutput:         size = dim * parameters_.vocab_size_;
                          offset = embeddings_; break;
    case kAttentionQuery: size = dim * n_heads * head_size;
                          offset = wq_ + OffsetOf(layer * size); break;
    case kAttentionKey:   size = dim * n_kv_heads * head_size;
                          offset = wk_ + OffsetOf(layer * size); break;
    case kAttentionValue: size = dim * n_kv_heads * head_size;
                          offset = wv_ + OffsetOf(layer * size); break;
    case kFeedForwardWo:  size = n_kv_heads * head_size * dim;
                          offset = wo_ + OffsetOf(layer * size); break;
    case kFeedForwardW1:  size = hidden_dim * dim;
                          offset = w1_ + OffsetOf(layer * size); break;
    case kFeedForwardW2:  size = dim * hidden_dim;
                          offset = w2_ + OffsetOf(layer * size); break;
    case kFeedForwardW3:  size = hidden_dim * dim;
                          offset = w3_ + OffsetOf(layer * size); break;
    case kFeedForwardRms: size = dim;
                          offset = feed_forward_rms_ + OffsetOf(layer * size); break;
    case kAttentionRms:   size = dim;
                          offset = attention_rms_ + OffsetOf(layer * size); break;
    case kFinalRms:       size = dim;
                          offset = final_rms_; break;
    default: throw std::runtime_error("invalid tensor type " + std::to_string(type));
  }

  return std::make_tuple(offset, size * sizeof(float));
}

} // end of namespace libai
