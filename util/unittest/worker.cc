//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <gtest/gtest.h>

#include <libai/util/worker.h>

using namespace libai;

// Make sure we can create and destroy a worker; also check default thread count
TEST(Worker, SimpleCreate)
{
  Worker worker;
  EXPECT_EQ(worker.GetThreadCount(), worker.GetConcurrentThreadCount());
}


// Make sure we can post a static function, member function, and 'bound' fct.
// use locks to make helgrind happy
static std::mutex SimplePost_Mutex;
static volatile bool SimplePost_Complete;

static bool SimplePost_IsComplete()
{
  std::unique_lock lock(SimplePost_Mutex);
  return SimplePost_Complete;
}

static void SimplePost_Reset()
{
  std::unique_lock lock(SimplePost_Mutex);
  SimplePost_Complete = false;
}

static bool SimplePost_StaticFuncVoid()
{
  SimplePost_Complete = true;
  return false;
}

static bool SimplePost_StaticFuncArgs(char a, int i)
{
  SimplePost_Complete = (a == 'a' && i == 1);
  return false;
}

class SimplePost_Class
{
 public:
  SimplePost_Class() : complete_(false) {}
  void reset()
  {
    std::unique_lock lock(mutex_);
    complete_ = false;
  }
  bool MemberFunc()
  {
    std::unique_lock lock(mutex_);
    complete_ = true;
    return false;
  }
  bool IsComplete()
  {
    std::unique_lock lock(mutex_);
    return complete_;
  }
 private:
  volatile bool complete_;
  std::mutex mutex_;
};

TEST(Worker, SimplePost)
{
  Worker worker;
  {
    SimplePost_Reset();
    Job job = worker.Post(SimplePost_StaticFuncVoid);

    // Wait up to 1 second for completion
    int loop;
    for (loop = 0; !SimplePost_IsComplete() && loop < 100; loop++)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    EXPECT_GT(100, loop);
  }

  SimplePost_Class klass;
  {
    klass.reset();
    Job job;
    job = worker.Post(&SimplePost_Class::MemberFunc, klass);

    EXPECT_NE(Job::kInvalid, job.GetId());

    // Wait up to 1 second for completion
    int loop;
    for (loop = 0; !klass.IsComplete() && loop < 100; loop++)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    EXPECT_GT(100, loop);
  }

  // post a 'bound' function with arguments
  {
    SimplePost_Reset();
    Job job = worker.Post(SimplePost_StaticFuncArgs, 'a', 1);
    // Wait up to 1 second for completion
    int loop;
    for (loop = 0; !SimplePost_IsComplete() && loop < 100; loop++)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    EXPECT_GT(100, loop);
  }
}

TEST(Worker, PostLambda)
{
  Worker worker;
  bool is_complete = false;
  Job job = worker.Post([&]() -> bool {
      is_complete = true;
      return false;
  });

  int loop;
  for (loop = 0; !is_complete && loop < 100; loop++)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_GT(100, loop);
}


TEST(Worker, JobGroup)
{
  Worker worker;
  std::atomic<int> count;
  std::atomic<bool> woken = false;

  Job main = worker.PostBlocked([&]() -> bool {
      woken = true;
      return false;
  });

  srand(time(NULL));

  std::list<Job> jobs;
  for (int i = 0; i < 20; i++)
    jobs.push_back(worker.PostRunBefore(main, [&count]() -> bool {
          usleep(1 + rand()/((RAND_MAX + 1u)/1000));
          count++;
          return false;
    }));

  worker.ReleaseBlock(main);

  int loop;
  for (loop = 0; (count != 20 || !woken) && loop < 200; loop++)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_GT(200, loop);
}
