//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef _LLAMA_VOCAB_H
#define _LLAMA_VOCAB_H

namespace libai {

/// LLaMAVocab contains the 'vocabs' (vocabularies).
struct LLaMAVocab
{
  using symbol = std::string;
  using token = uint32_t;

  struct TokenScore
  {
    symbol text;
    float score;
  };

  // TODO: are both members requiers or can it be combined?
  std::unordered_map<symbol, token> tokens_;
  std::vector<TokenScore> scores_;
  size_t max_token_length_;
  uint32_t bos_token_;
  uint32_t eos_token_;
  bool add_bos_token_;
  bool add_eos_token_;
};

} // end of namespace libai

#endif  // define _LLAMA_VOCAB_H
