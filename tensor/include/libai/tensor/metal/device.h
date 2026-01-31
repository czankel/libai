//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef LIBAI_TENSOR_METAL_DEVICE_H
#define LIBAI_TENSOR_METAL_DEVICE_H

#include <Metal/Metal.hpp>

#include "../device.h"

namespace libai {
  template <typename> class MetalAllocator;
}

namespace libai::device {

class CommandEncoder
{
 public:
  CommandEncoder(const CommandEncoder&) = delete;
  CommandEncoder& operator=(const CommandEncoder&) = delete;

  CommandEncoder(MTL::CommandBuffer* command_buffer)
    : command_buffer_(command_buffer),
      encoder_(command_buffer->computeCommandEncoder(MTL::DispatchTypeConcurrent))
  {
    encoder_->retain();
    command_buffer_->retain();
  }


  ~CommandEncoder()
  {
    encoder_->endEncoding();
    command_buffer_->commit();
    command_buffer_->waitUntilCompleted();

    // TODO: do we need to actually check (when callbacks are implemented)?
#if 0
    MTL::CommandBufferStatus status = command_buffer_->status();
    std::cout << "status: " << status << std::endl;
#endif

    command_buffer_->release();
    encoder_->release();
  }

  /// @brief Return the Metal encoder instance.
  MTL::ComputeCommandEncoder* operator->()            { return encoder_; }

  /// @brief Dispatch threadsgroups.
  void DispatchThreadgroups(MTL::Size grid_dims, MTL::Size group_dims);

  /// @brief Dispatch threads.
  void DispatchThreads(MTL::Size grid_dims, MTL::Size group_dims);

 private:
  MTL::CommandBuffer*          command_buffer_;
  MTL::ComputeCommandEncoder*  encoder_;
};


/// Metal is a Device that implements a singleton for managing the GPU devices.
class Metal : public Device
{
  // Private constructor
  Metal();

  Metal(Metal&) = delete;
  Metal& operator=(Metal&) = delete;

 public:
  template <typename T> using allocator_type = MetalAllocator<T>;

  ~Metal()
  {
    for (auto& k : kernels_) {
      k.second->release();
    }
  }

  /// @brief Return the default device (singleton)
  static Metal& GetDevice();

  /// @brief Create a new metal device buffer.
  // TODO use smart ptr?
  MTL::Buffer* NewBuffer(size_t length, MTL::ResourceOptions options)
  {
    return mtl_device_->newBuffer(length, options);
  }

  /// @brief Find the kernel in the library.
  MTL::ComputePipelineState* GetKernel(const std::string& name);

  // TODO: make it multi-stream
  /// @brief Return the default encoder (singleton)
  CommandEncoder& Encoder();

  // TODO Sync queues, wait for completion, etc... set callback...
  void Wait();

  MTL::Size max_threads_per_threadgroup_;
  static const size_t max_thread_execution_width_ = 32; ///< Maximum SIMD width

 private:
  static Metal*       g_device_;

  MTL::Device*        mtl_device_;

  MTL::Library*       library_;
  MTL::CommandQueue*  queue_;

  std::unique_ptr<CommandEncoder> command_encoder_;

  std::unordered_map<std::string, MTL::ComputePipelineState*> kernels_;
};


} // end of namespace libai::device

#endif  // LIBAI_TENSOR_METAL_DEVICE_H
