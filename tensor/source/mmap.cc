//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <grid/tensor/mmap.h>

namespace libai {

MMap* MMap::MMapFile(const std::string& name)
{
  int fd = open(name.c_str(), O_RDONLY);
  if (fd < 0)
    throw("no such file: " + name);

  auto mmap = MMapFile(fd, lseek(fd, 0, SEEK_END));
  close(fd);

  return mmap;
}

MMap* MMap::MMapFile(int fd, size_t file_size)
{
  void* addr = mmap(NULL, file_size, PROT_READ, MAP_FILE | MAP_SHARED, fd, 0);
  if (addr == MAP_FAILED)
    throw("mmap failed");

  return new MMap(reinterpret_cast<char*>(addr), file_size);
}


} // end of namespace libai
