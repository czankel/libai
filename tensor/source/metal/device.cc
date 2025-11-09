//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <iostream>

#include <libai/tensor/metal/device.h>

using namespace libai::device;

//
// CommandEncoder
//

// TODO: could make these inline
void CommandEncoder::DispatchThreadgroups(MTL::Size grid_dims, MTL::Size group_dims)
{
  encoder_->dispatchThreadgroups(grid_dims, group_dims);
}


void CommandEncoder::DispatchThreads(MTL::Size grid_dims, MTL::Size group_dims)
{
  encoder_->dispatchThreads(grid_dims, group_dims);
}

//
// Metal
//

// TODO: placeholder, resetting command_encoder_ waits and releases the encoder
void Metal::Wait()
{
  command_encoder_ = nullptr;
}


Metal::Metal()
{
  auto devices = MTL::CopyAllDevices();
  mtl_device_ = static_cast<MTL::Device*>(devices->object(0)) ?: MTL::CreateSystemDefaultDevice();
  if (!mtl_device_)
    throw std::runtime_error("Failed to load device");

  auto library = NS::String::string(METAL_PATH, NS::UTF8StringEncoding);
  NS::Error* error = nullptr;
  library_ = mtl_device_->newLibrary(library, &error);
  if (error != nullptr)
    throw std::runtime_error("failed to create metal library");

  queue_ = mtl_device_->newCommandQueue();

  max_threads_per_threadgroup_ = mtl_device_->maxThreadsPerThreadgroup();
}


Metal& Metal::GetDevice()
{
  if (g_device_ == nullptr)
    g_device_ = new Metal();

  return *g_device_;
}


MTL::ComputePipelineState* Metal::GetKernel(const std::string& name)
{
  if (auto it = kernels_.find(name); it != kernels_.end())
    return it->second;

  auto function_name = NS::String::string(name.c_str(), NS::ASCIIStringEncoding);
  auto mtl_function = library_->newFunction(function_name);
  if (!mtl_function)
    throw std::runtime_error("Failed to find metal function: " + name);

  NS::Error* error = nullptr;
  auto kernel = mtl_device_->newComputePipelineState(mtl_function, &error);
  if (error)
    throw std::runtime_error(error->localizedDescription()->utf8String());

  kernels_.insert({name, kernel});

  return kernel;
}


CommandEncoder& Metal::Encoder()
{
  if (command_encoder_ == nullptr)
  {
    MTL::CommandBuffer* command_buffer = queue_->commandBufferWithUnretainedReferences();

    if (!command_buffer)
      throw std::runtime_error("failed to create command buffer");

    command_buffer->retain();
    command_encoder_ = std::make_unique<CommandEncoder>(command_buffer);
  }
  return *command_encoder_;
}

libai::device::Metal* libai::device::Metal::g_device_;
