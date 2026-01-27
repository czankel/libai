//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef _LLAMA_H
#define _LLAMA_H

#include <algorithm>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <vector>

#include <libai/models/llama.h>

#include <libai/tensor/mmap.h>
#include <libai/tensor/tensor.h>

#include "llama_vocab.h"

using libai::view::Slice;
using libai::view::Extent;

namespace libai {

/// LLaMAModelT is the templated version of the LLaMAModel class for data type and backend.
template <typename T, typename TDevice>
class LLaMAModelT : public LLaMAModel
{
  friend class KarpathyFile;
  friend class GgmlFile;

  /// Using two tensor types, a dynamically allocated default tensor and a memory-mapped file tensors.
  using Tensor1D = Tensor<T, 1, DeviceMemory<TDevice>>;
  using Tensor2D = Tensor<T, 2, DeviceMemory<TDevice>>;

 protected:
  LLaMAModelT() = default;

 public:
  virtual ~LLaMAModelT() = default;

  // LLaMAModel::
  virtual void Predict(std::string_view prompt, size_t steps);

  /// Load loads the LLaMA model from the provided file.
  static LLaMAModelT<T, TDevice>* Load(LLaMAFile& file);

 protected:
  // EncodeBPE encodes the prompt into a token vector using byte-pair encoding
  void EncodeBPE(std::string_view prompt, std::vector<uint32_t>& token_ids);

  // Decode decodes the provided current token.
  std::string Decode(LLaMAVocab::token , LLaMAVocab::token);

  /// Forward runs a single forward run through the model (seq len = 1)
  void Forward(LLaMAVocab::token token, size_t);

  /// Sample samples the current logits to a word.
  LLaMAVocab::token Sample();
  LLaMAVocab::token SampleArgMax();

 private:
  LLaMAModel::Parameters parameters_;
  std::shared_ptr<MMap>  mmap_;
  LLaMAVocab             vocab_;
  size_t                 max_token_length_;

  struct LLaMALayer
  {
    // (note that dim = n_heads * head_size and n_kv_heads = n_heads for this implementation)
    Tensor2D  wq_;              // {dim, n_heads * head_size}
    Tensor2D  wk_;              // {dim, n_kv_heads * head_size}
    Tensor2D  wv_;              // {dim, n_kv_heads * head_size}
    Tensor2D  wo_;              // {n_heads * head_size, dim}

    // Weights for FFN
    Tensor2D  w1_;              // {hidden_dim, dim}
    Tensor2D  w2_;              // {dim, hidden_dim}
    Tensor2D  w3_;              // {hidden_dim, dim}

    Tensor1D      att_norm_;        // {dim}
    Tensor1D  ffn_norm_;        // {dim}

    // Runtime tensors
    Tensor2D      key_cache_;       // {kv_dim, max_sequence_length}
    Tensor2D      value_cache_;     // {kv_dim, max_sequence_length}
    Tensor1D      q_;
  };

  Tensor2D  embeddings_;
  Tensor1D  output_norm_;       // {dim}
  Tensor2D  output_;            // {vocab_size, dim}

  // Runtime tensors
  Tensor1D      x_;                 // {dim}
  Tensor1D      xb_;
  Tensor1D      logits_;            // output {vocab_size}
  Tensor1D      scores_;            // {n_heads * head_size}

  std::vector<LLaMALayer> layers_;
};


template <typename T, typename TDevice>
LLaMAModelT<T, TDevice>* LLaMAModelT<T, TDevice>::Load(LLaMAFile& file)
{
  auto* model = new LLaMAModelT<T, TDevice>();

  file.GetParameters(model->parameters_);
  file.GetTokenizer(model->vocab_);

  model->mmap_ = std::shared_ptr<MMap>(file.MapTensors());
  char *base = static_cast<char*>(model->mmap_->Address());

  auto& params = model->parameters_;
  size_t n_layers =   params.num_layers_;
  size_t hidden_dim = params.hidden_dim_;
  size_t dim =        params.dim_;
  size_t kv_dim =     dim * params.num_kv_heads_ / params.num_heads_;

  model->layers_.resize(n_layers);
  for (size_t i = 0; i < n_layers; i++)
  {
    auto& layer = model->layers_[i];
    layer.att_norm_ =   Tensor({dim}, file.GetTensor<T>(base, LLaMAFile::kAttentionRms, i));
    layer.wq_ =         Tensor({dim, dim}, file.GetTensor<T>(base, LLaMAFile::kAttentionQuery, i));
    layer.wk_ =         Tensor({kv_dim, dim}, file.GetTensor<T>(base, LLaMAFile::kAttentionKey, i));
    layer.wv_ =         Tensor({kv_dim, dim}, file.GetTensor<T>(base, LLaMAFile::kAttentionValue, i));
    layer.wo_ =         Tensor({dim, dim}, file.GetTensor<T>(base, LLaMAFile::kFeedForwardWo, i));
    layer.ffn_norm_ =   Tensor({dim}, file.GetTensor<T>(base, LLaMAFile::kFeedForwardRms, i));
    layer.w1_ =         Tensor({hidden_dim, dim}, file.GetTensor<T>(base, LLaMAFile::kFeedForwardW1, i));
    layer.w2_ =         Tensor({dim, hidden_dim}, file.GetTensor<T>(base, LLaMAFile::kFeedForwardW2, i));
    layer.w3_ =         Tensor({hidden_dim, dim}, file.GetTensor<T>(base, LLaMAFile::kFeedForwardW3, i));
  }

  model->embeddings_ =  Tensor({params.vocab_size_, dim}, file.GetTensor<T>(base, LLaMAFile::kEmbeddings));
  model->output_norm_=  Tensor({dim}, file.GetTensor<T>(base, LLaMAFile::kFinalRms));
  model->output_     =  Tensor({params.vocab_size_, dim}, file.GetTensor<T>(base, LLaMAFile::kOutput));

  // Initialize runtime tensors
  model->x_ =           Tensor({dim}, std::type_identity<T>{});
  model->xb_ =          Tensor({dim}, std::type_identity<T>{});
  model->logits_ =      Tensor({params.vocab_size_}, std::type_identity<T>{});
  model->scores_ =      Tensor({dim}, std::type_identity<T>{});

  for (size_t i = 0; i < n_layers; i++)
  {
    auto& layer =        model->layers_[i];
    layer.key_cache_ =   Tensor({params.max_seq_len_, kv_dim}, T{});
    layer.value_cache_ = Tensor({params.max_seq_len_, kv_dim}, T{});
    layer.q_ =           Tensor({dim}, std::type_identity<T>{});
  }

  return model;
}


// Byte-Pair Encoding
template <typename T, typename TDevice>
void LLaMAModelT<T, TDevice>::EncodeBPE(std::string_view prompt, std::vector<LLaMAVocab::token>& tokens)
{
  if (vocab_.add_bos_token_)
    tokens.push_back(vocab_.bos_token_);

  // TODO: SentencePiece uses a special 'LOWER ONE EIGHTH BLOCK' (underscore) character as a separator.
  auto sep = vocab_.tokens_.find(std::string("\u2581"));

  // split text into characters; handle utf-8 characters
  std::string symbol;
  for (size_t i = 0, utf_idx = 0; i < prompt.size(); i++)
  {
    char c = prompt[i];

    if (c  == ' ' && sep != vocab_.tokens_.end())
    {
      tokens.push_back(sep->second);
      continue;
    }

    symbol.push_back(c);
    if (c < 0 && utf_idx++ < 4)
      continue;

    auto it = vocab_.tokens_.find(symbol);
    if (it != vocab_.tokens_.end())
      tokens.push_back(it->second);
    else for (size_t j = 0; j < utf_idx; j++)
      tokens.push_back(symbol[j] + 3);

    symbol.clear();
    utf_idx = 0;
  }

  while (1)
  {
    LLaMAVocab::token best_token;
    float best_score = std::numeric_limits<float>::lowest();
    int   best_index = -1;

    for (size_t i = 0; i < tokens.size() - 1; i++)
    {
      auto symbol = vocab_.scores_[tokens[i]].text + vocab_.scores_[tokens[i + 1]].text;
      auto it = vocab_.tokens_.find(symbol);

      float score;
      if (it != vocab_.tokens_.end() && (score = vocab_.scores_[it->second].score) > best_score)
      {
        best_score = score;
        best_token = it->second;
        best_index = i;
      }
    }

    if (best_index == -1)
      break;

    tokens[best_index] = best_token;
    tokens.erase(tokens.begin() + best_index + 1);
  }

  if (tokens.size() < 2)
    throw std::runtime_error("expected at least 1 prompt token");

  if (vocab_.add_eos_token_)
    tokens.push_back(vocab_.eos_token_);
}


template <typename T, typename TDevice>
std::string LLaMAModelT<T, TDevice>::Decode(LLaMAVocab::token prev, LLaMAVocab::token token)
{
  std::string symbol = vocab_.scores_[token].text;

  // if first token after <BOS> drop any space
  if (prev == kBOS && symbol[0] == ' ')
    symbol.erase(0,1);

  // onvert raw bytes, e.g. <0x01> to actual bytes
  if (symbol[0] == '<')
  {
    if (symbol == "<unk>")
      throw std::runtime_error("Failed to find symbol in vocab: " + std::to_string(token));

    auto end = symbol.find('>');
    if (end != std::string::npos)
    {
      symbol = symbol.substr(1, end - 1);
      wchar_t x = std::stoul(symbol, nullptr, 16);
      symbol = x;
    }
  }
  else
  {
    std::string sep("\u2581");
    std::string::size_type pos = 0;
    while ((pos = symbol.find(sep, pos)) != std::string::npos)
      symbol.replace(pos, sep.size(), " "); // FIXME: doesn't work with gcc
  }

  return symbol;
}

// Note that this is a "lower-rank" implementation going through the calculation for each
// token vector instead of combining a sequence into a matrix and using higher-rank tensors.
template <typename T, typename TDevice>
void LLaMAModelT<T, TDevice>::Forward(LLaMAVocab::token token, size_t pos)
{
  using namespace libai;

  size_t dim = parameters_.dim_;
  size_t n_heads = parameters_.num_heads_;
  size_t n_kv_heads = parameters_.num_kv_heads_;
  size_t head_size = dim / n_heads;
  size_t kv_dim = parameters_.num_kv_heads_ * head_size;

  x_ = embeddings_.View(token);

  for (auto& l: layers_)
  {
    // normalize input and element-multiply with weight.
    xb_ = RmsNorm(x_) * l.att_norm_;                      // (dim) * (dim) -> (dim)

    // Insert Weight(xb) vectors into the key and value caches at row "pos"
    l.key_cache_.View(pos) = Matmul(l.wk_, xb_);          // (kv_dim, dim) @ (dim) -> (kv_dim)
    l.value_cache_.View(pos) = Matmul(l.wv_, xb_);        // (kv_dim, dim) @ (dim) -> (kv_dim)
    l.q_ = Matmul(l.wq_, xb_);                            // (dim, dim) @ (dim)    -> (dim)

    // RoPE, rotate for each 'head'
    auto q = l.q_.Data();
    auto k = l.key_cache_.View(pos).Data();
    for (size_t i = 0; i < dim; i+=2)
    {
      float rot = (float) pos / powf(10000.0f, (float)(i % head_size) / (float)head_size);
      float fcr = cosf(rot);
      float fci = sinf(rot);

      float v0 = q[i];
      float v1 = q[i+1];
      q[i]   = v0 * fcr - v1 * fci;
      q[i+1] = v0 * fci + v1 * fcr;

      if (i < kv_dim)
      {
        float v0 = k[i];
        float v1 = k[i+1];
        k[i]   = v0 * fcr - v1 * fci;
        k[i+1] = v0 * fci + v1 * fcr;
      }
    }

    // MultiHead(Q,K,V) = concat(head_1, ..., head_h) W_0, with head = Attention(Q_head,K_head,V_head)
    for (size_t head = 0; head < n_heads; head++)
    {
      size_t head_offset = head * head_size;
      size_t kv_head_offset = (head / (n_heads/n_kv_heads)) * head_size;

      // Attention(Q,K,V) = softmax(Q * K^T / sqrt(kv_dim)) * V
      //
      // For a single token (seq = 1) at position pos, and looping through the head count, this
      // reduces to:
      // scores [head_offset:head_offset + head_size] =
      //   softmax(K [:pos+1, head:head+head_size] @ q [head:head+head_size] @ V [:pos+1, head:head+head_size]
      scores_.View(Extent(head_offset, head_size)) =
        Matmul(
          SoftMax(
            Matmul(
              l.key_cache_.View(Extent(pos + 1), Extent(kv_head_offset, head_size)),
              l.q_.View(Extent(head_offset, head_size))) / sqrt(static_cast<T>(head_size))),
          l.value_cache_.View(Extent(pos + 1), Extent(kv_head_offset, head_size)));
    }

    // bring it all together
    // (dim, dim) @ (dim = n_heads * head_size) -> (dim)
    x_ += Matmul(l.wo_, scores_);

    // (dim) * (dim) -> (dim)
    xb_ = RmsNorm(x_) * l.ffn_norm_;

    // self.w2(F.silu(self.w1(x)) * self.w3(x))
    // w1(x), w3(x)         -> (hidden_dim, dim) @ (dim)        -> (hidden_dim)
    // silu(w1(x)) * w3(x)  -> (hidden_dim) * (hiddem_dim)      -> (hidden_dim)
    // w2(...)              -> (dim, hidden_dim) @ (hidden_dim) -> (dim)
    x_ += Matmul(l.w2_, Silu(Matmul(l.w1_, xb_)) * Matmul(l.w3_, xb_));
  }

  // Final RMS norm and classified into logits
  // (vocab_size, dim) @ ((dim, dim) * (dim)) -> (vocab_size)
  logits_ = Matmul(output_, RmsNorm(x_) * output_norm_);
}

template <typename T, typename TDevice>
LLaMAVocab::token LLaMAModelT<T, TDevice>::SampleArgMax()
{
  float max_p = std::numeric_limits<float>::lowest();
  int max_i = 0;
  auto data = logits_.Data();

  for (LLaMAVocab::token i = 0; i < logits_.Dimensions()[0]; i++)
  {
    if (data[i] > max_p)
    {
      max_p = data[i];
      max_i = i;
    }
  }

  return max_i;
}

template <typename T, typename TDevice>
LLaMAVocab::token LLaMAModelT<T, TDevice>::Sample()
{
  // greedy argmax sampling: return the token with the highest probability
  // TODO: implement entropy sampling: if (temperature_ == value_type(0))
  return SampleArgMax();
}

template <typename T, typename TDevice>
void LLaMAModelT<T, TDevice>::Predict(std::string_view prompt, size_t steps)
{
  using token = LLaMAVocab::token;

  std::vector<token> prompt_tokens;
  EncodeBPE(prompt, prompt_tokens);

  size_t pos = 0;
  size_t prompt_token_size = prompt_tokens.size();

  for (token curr = prompt_tokens[0]; pos < steps; pos++)
  {
    Forward(curr, pos);
    token prev = curr;

    curr = (pos < prompt_token_size - 1) ? prompt_tokens[pos + 1] : Sample();
    if (curr == kBOS)
      break;

    std::string symbol = Decode(prev, curr);
    std::cout << symbol << std::flush;
  }
  std::cout << std::endl;
}

} // end of namespace libai

#endif  // _LLAMA_H
