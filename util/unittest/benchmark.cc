#include <gtest/gtest.h>


#include <algorithm>
#include <chrono>
#include <cmath>
#include <execution>
#include <iostream>
#include <numeric>
#include <vector>

#include <grid/util/worker.h>

int test_stl(const std::vector<double>& X)
{
    std::vector<double> Y(X.size());
    const auto start = std::chrono::high_resolution_clock::now();
    std::transform(X.cbegin(), X.cend(), Y.begin(), [](double x){
        volatile double y = std::sin(x);
        return y;
    });
    const auto stop = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    return diff.count();
}

template <class ExecutionPolicy>
int test_stl(const std::vector<double>& X, ExecutionPolicy&& policy)
{
    std::vector<double> Y(X.size());
    const auto start = std::chrono::high_resolution_clock::now();
    std::transform(policy, X.cbegin(), X.cend(), Y.begin(), [](double x){
        volatile double y = std::sin(x);
        return y;
    });
    const auto stop = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    return diff.count();
}

struct size3
{
  size_t operator[](const size_t i)
  {
    //static_assert(i < 3);
    return reinterpret_cast<size_t*>(this)[i];
  }
  size_t x;
  size_t y;
  size_t z;
};

template <typename T>
bool SinusJob(size3 position, size3 dimensions, T* d, const T* x)
{
  for (size_t i = 0; i < dimensions.x; i++)
    d[position.x + i] = std::sin(x[position.x + i]);
  return false;
}


int test_worker(const std::vector<double>& X)
{
  std::vector<double> Y(X.size());
  std::chrono::time_point<std::chrono::high_resolution_clock> start;

  std::mutex              wait_mutex;
  std::condition_variable wait_cond;

  {
    libai::Worker worker(100);

    const double* x = &*X.cbegin();
    double* d = &*Y.begin();

    size_t ncpus = worker.GetConcurrentThreadCount();
    start = std::chrono::high_resolution_clock::now();

    libai::Job main = worker.PostBlocked([&]() -> bool {
        wait_cond.notify_all();
        return false;
    });

    size_t width = X.size();
    size_t w = ((((width + ncpus - 1) / ncpus) + 7) & ~7);

    for (size_t i = 0; i < ncpus; i++)
    {
      size3 pos{ i * w, 0, 0 };
      size3 dim{ i * w > width ? width - i * w : w, 0, 0 };
      worker.PostRunBefore(main, SinusJob<double>, pos, dim, d, x);
    }
    worker.ReleaseBlock(main);

    {
      std::unique_lock lock(wait_mutex);
      wait_cond.wait(lock);
    }
  }

  const auto stop = std::chrono::high_resolution_clock::now();

  std::vector<double> Z(X.size());
  std::transform(X.cbegin(), X.cend(), Z.begin(), [](double x){
      volatile double z = std::sin(x);
      return z;
  });

  size_t i = 0;
  for (; i < Z.size() && Y[i] == Z[i]; i++)
    ;
  EXPECT_EQ(i, X.size());

  auto diff = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  return diff.count();
}

#if 0
int test_openmp(const std::vector<double>& X)
{
    std::vector<double> Y(X.size());
    const auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (size_t i = 0; i < X.size(); ++i) {
        volatile double y = std::sin(X[i]);
        Y[i] = y;
    }
    const auto stop = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    return diff.count();
}
#endif

TEST(Worker, Benchmark)
{
    const size_t N = 100000000;
    std::vector<double> data(N);
    std::iota(data.begin(), data.end(), 1);
    //std::cout << "OpenMP:        " << test_openmp(data) << std::endl;
    std::cout << "Tensor Worker  " << test_worker(data) << std::endl;

#if !APPLE && __clang_major__ > 17
    std::cout << "STL seq:       " << test_stl(data, std::execution::seq) << std::endl;
    std::cout << "STL par:       " << test_stl(data, std::execution::par) << std::endl;
    std::cout << "STL par_unseq: " << test_stl(data, std::execution::par_unseq) << std::endl;
    std::cout << "STL unseq:     " << test_stl(data, std::execution::unseq) << std::endl;
#endif

    std::cout << "STL default:   " << test_stl(data) << std::endl;
}
