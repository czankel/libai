# LibAI

> **_NOTE:_**  This repository is still incomplete and serves for educational purposes
>  at this time.

## Introduction

LibAI implements a C++ tensor library for machine learning applications.

Tensors are defined as multi-dimensional arrays for storing data and supporting
various arithmetic operations, such as matrix and vector operations, but also many
machine-learning operations, including norms, 'neuron' (weights,biases).
The definition, thus, differs from the more mathematical or physical usage of tensors.

The implementation relies heavily on C++ templates and C++20 features. These features
provide an easy abstraction for defining ML models, which can be mostly agnostic to
the optimizations implmeneted for CPUs and accelerators.

The following example demonstrates the usage of tensors for a simple vector addition
and vector scaling operation. Data type and rank are automatically deduced from the
tensor arguments. Changing the example to use an accelerator will (in future) be as
simple as adding a template parameter to the first line but leaving the rest of the
code as is.

```
using Tensor = libai::Tensor;

Tensor t1{1.0, 2.0, 3.0};
Tensor t2(3, 4, 3.3);
Tensor res = t1 + t2 * 3;
```

## Building LibAI

Note that LibAI requires cmake and a more recent version gcc or clang.

Assuming LibAI should be built in the current (build) directory and having
SRC pointing to the git repository, use the following commands:

```
cmake $SRC
make
```

This builds the library and the simple program ```llama``` for running a LLaMA 2 model.


## Using the llama program

LibAI comes with a tool for running a LLaMA 2 large language model.

The current implementation only supports float32 and requires that the model parameters
are stored in a GGUF file format. It also doesn't (yet) support entropy sampling,
i.e. specifying a different temperature than 0.

Refer to the following documents for more information:
* [Tutorial: How to convert HuggingFace model to GGUF format](https://github.com/ggerganov/llama.cpp/discussions/2948)
for details on converting a model file to the GGUF format. Use f32 for float-32 paramters.
* [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) is a freely downloadable model.

```
llama -m $MODEL PROMPT
```


## Building unit tests

LibAI comes with a set of unit tests. They can be built setting the BUILD_TEST variable:

```
cmake -B $BUILD_DIRECTORY -DBUILD_TEST=1
./libai_test
```
