//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#include <grid/util/worker.h>

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>


#define TAG "Worker"

namespace grid {

//
// CurrentJob::
//

__thread Job::Id CurrentJob::tls_current_job_id_ = Job::kInvalid;

//
// Worker::
//

Worker::Worker(unsigned int job_capacity, unsigned int thread_count)
    : thread_count_adjust_(0),
      thread_count_(0),
      job_capacity_(job_capacity),
      job_alloc_bitmap_(nullptr),
      jobs_(nullptr),
      queue_needs_reschedule_(false),
      queue_ready_(nullptr),
      queue_scheduled_(&queue_ready_),
      queue_sleeping_(&queue_ready_)
{
  if (job_capacity == 0)
    throw std::runtime_error("invalid job capacity");

  job_alloc_bitmap_ = new uint32_t[(job_capacity_ + 31) / 32];
  memset(job_alloc_bitmap_, 0, (job_capacity_ + 31) / 32 * sizeof(uint32_t));

  jobs_ = new WorkerJob[job_capacity_];
  for (unsigned int i = 0; i < job_capacity_; i++)
    new (&jobs_[i]) Job();

  unsigned int max_threads = GetConcurrentThreadCount();
  thread_count_adjust_ = std::min(thread_count, max_threads);
  UpdateThreadCount(Duration::zero());
}

Worker::~Worker()
{
  thread_count_adjust_ = 0;

  for (unsigned int i = 0; i < job_capacity_; i++)
    if (jobs_[i].refcount_ > 0)
      KillWorkerJob(jobs_[i]);

  UpdateThreadCount(Duration::max());
  for (auto& t: thread_pool_)
    t.join();

  for (unsigned int i = 0; i < job_capacity_; i++)
  {
    if (job_alloc_bitmap_[i] != 0)
    {
      if (--jobs_[i].refcount_ > 0)
      {
        std::cerr << "jobs still referenced" << std::endl;
        exit(-1);
      }
    }
  }

  if (job_alloc_bitmap_ != nullptr)
    delete[] job_alloc_bitmap_;
  if (jobs_ != nullptr)
    delete[] jobs_;
}


// 'no-thread' Run
bool Worker::Run()
{
  if (thread_count_ != 0)
    throw std::runtime_error("cannot use Run() with threads");

  WorkerRun();
  return true;
}

// Thread Functions (public)

// TODO: all threads created will remain in the thread_pool_ until "joined" in the worker destructor
// TODO: simply delete all threads and re-create them; reducing threds is not relaly a use case
bool Worker::UpdateThreadCount(Duration timeout)
{
  std::scoped_lock lock(thread_mutex_);

  if (thread_count_adjust_ > thread_count_)
  {
    return CreateThreadsLocked();
  }
  else
  {
    if (timeout <= Duration::zero())
      return false;

    while (thread_count_ > thread_count_adjust_)
    {
      for (unsigned int i = thread_count_adjust_; i < thread_count_; i++)
        queue_wait_cond_.notify_one();

      usleep(1000);
      timeout -= std::chrono::milliseconds(1);
    }
  }

  return timeout > Duration::zero();
}

bool Worker::CreateThreadsLocked()
{
  while (thread_count_ < thread_count_adjust_)
  {
    thread_count_++;
    std::thread thread(std::bind(std::mem_fn(&Worker::WorkerRun), this));
    thread_pool_.push_back(std::move(thread));
  }

  return true;
}

//
// Worker public functions
//

// returns false if job has been killed
bool Worker::Wake(const Job& job)
{
  Job::Id id = job.GetId();
  if (id == Job::kInvalid)
    throw std::runtime_error("invalid job");

  WorkerJob& wjob = *reinterpret_cast<WorkerJob*>(id);
  std::unique_lock lock(queue_mutex_);
  return WakeJobLocked(wjob);
}

void Worker::WakeBlocked(const Job& job)
{
  Job::Id id = job.GetId();
  if (id == Job::kInvalid)
    throw std::runtime_error("invalid job");

  WorkerJob& wjob = *reinterpret_cast<WorkerJob*>(id);
  std::lock_guard lock(queue_mutex_);
  WakeBlockedLocked(wjob);
}

void Worker::KillJob(const Job& job)
{
  Job::Id id = job.GetId();
  if (id == Job::kInvalid)
    throw std::runtime_error("invalid job");

  WorkerJob& wjob = *reinterpret_cast<WorkerJob*>(id);

  KillWorkerJob(wjob);
}

void Worker::KillWorkerJob(WorkerJob& wjob)
{
  // modify reschedule, once set to kKill, it cannot change
  wjob.reschedule_ = WorkerJob::kKill;

  {
    std::lock_guard lock(queue_mutex_);
    if (wjob.is_queued_)
    {
      DequeueWorkerJobLocked(wjob);
      ReleaseWorkerJobLocked(wjob);
    }
  }
}

void Worker::Block(Job& job)
{
  Job::Id id = job.GetId();
  if (id == Job::kInvalid)
    throw std::runtime_error("invalid job");

  std::lock_guard lock(queue_mutex_);
  AddRefBlockedLocked(*reinterpret_cast<WorkerJob*>(id));
}

void Worker::ReleaseBlock(Job& job)
{
  Job::Id id = job.GetId();
  if (id == Job::kInvalid)
    throw std::runtime_error("invalid job");

  std::lock_guard lock(queue_mutex_);
  ReleaseBlockedLocked(*reinterpret_cast<WorkerJob*>(id));
}


bool Worker::IsRescheduled(const Job& job)
{
  Job::Id id = job.GetId();
  if (id == Job::kInvalid)
    throw std::runtime_error("invalid job");

  return reinterpret_cast<WorkerJob*>(id)->reschedule_.load() == WorkerJob::kAgain;
}

bool Worker::IsRescheduledWaiting(const Job& job)
{
  Job::Id id = job.GetId();
  if (id == Job::kInvalid)
    throw std::runtime_error("invalid job");

  TimePoint now = SteadyClock::now();
  WorkerJob* wjob = reinterpret_cast<WorkerJob*>(id);

  return wjob->scheduled_time_ > now;;
}


bool Worker::WaitForJob(const Job& job)
{
  bool completed = false;

  Job::Id id = job.GetId();
  if (id == Job::kInvalid)
    throw std::runtime_error("invalid job");

  WorkerJob* wjob = reinterpret_cast<WorkerJob*>(id);
  std::unique_lock lock(wjob->wait_mutex_);

  while (true)
  {
    Job::Status status = GetJobStatus(id) ;
    completed = (status == Job::kDone || status == Job::kError);
    if (completed)
      break;

    wjob->wait_cond_.wait(lock);
  }

  return completed;
}


bool Worker::WaitForJobFor(const Job& job, Duration timeout)
{
  bool completed = false;

  Job::Id id = job.GetId();
  if (id == Job::kInvalid)
    throw std::runtime_error("invalid job");

  WorkerJob* wjob = reinterpret_cast<WorkerJob*>(id);
  std::unique_lock lock(wjob->wait_mutex_);

  std::cv_status s;
  do
  {
    Job::Status status = GetJobStatus(id) ;
    completed = (status == Job::kDone || status == Job::kError);
  } while (!completed && (s = wjob->wait_cond_.wait_for(lock, timeout)) == std::cv_status::no_timeout);

  return completed;
}


bool Worker::WaitForJobUntil(const Job& job, TimePoint time)
{
  bool completed = false;

  Job::Id id = job.GetId();
  if (id == Job::kInvalid)
    throw std::runtime_error("invalid job");

  WorkerJob* wjob = reinterpret_cast<WorkerJob*>(id);
  std::unique_lock lock(wjob->wait_mutex_);

  std::cv_status s;
  do
  {
    Job::Status status = GetJobStatus(id) ;
    completed = (status == Job::kDone || status == Job::kError);
  } while (!completed && (s = wjob->wait_cond_.wait_until(lock, time)) == std::cv_status::no_timeout);

  return completed;
}


//
// Worker Protected Functions
//

Job::Id Worker::AllocateJob(size_t size)
{
  if (size > WorkerJob::kMaxFunctorSize)
    throw std::runtime_error("size of job function too large");

  int slot = AllocateSlot();
  if (slot < 0)
    return Job::kInvalid;

  WorkerJob& wjob = jobs_[slot];

  wjob.refcount_ = 2;
  wjob.block_refcount_ = 0;

  // reset important fields
  wjob.next_ = nullptr;
  wjob.prev_ = nullptr;
  wjob.block_ = nullptr;

  wjob.worker_ = this;
  wjob.is_queued_ = false;
  wjob.result_ = false;
  wjob.running_ = false;

  wjob.reschedule_ = WorkerJob::kOnce;

  return reinterpret_cast<Job::Id>(&wjob);
}

void* Worker::GetFunctionPointer(Job::Id id)
{
  if (id == Job::kInvalid)
    throw std::runtime_error("invalid job");

  return reinterpret_cast<WorkerJob*>(id)->function_buffer_;
}

void Worker::AddRefJob(Job::Id id)
{
  if (id == Job::kInvalid)
    throw std::runtime_error("invalid job");

  ++reinterpret_cast<WorkerJob*>(id)->refcount_;
}

void Worker::ReleaseJob(Job::Id id)
{
  if (id == Job::kInvalid)
    throw std::runtime_error("invalid job");

  std::lock_guard lock (queue_mutex_);
  ReleaseWorkerJobLocked(*reinterpret_cast<WorkerJob*>(id));
}

Job::Status Worker::GetJobStatus(Job::Id id) const
{
  if (id == Job::kInvalid)
    throw std::runtime_error("invalid job");

  WorkerJob* wjob = reinterpret_cast<WorkerJob*>(id);
  return wjob->is_queued_ ? Job::kWaiting : (
           wjob->reschedule_ != WorkerJob::kKill && wjob->running_ ? Job::kRunning : (
             wjob->result_ ? Job::kDone : Job::kError));
}

bool Worker::NeedsReschedule(Job::Id id) const
{
  if (id == Job::kInvalid)
    throw std::runtime_error("invalid job");

  return queue_needs_reschedule_.load();
}

// Rescheduling Jobs

// Note that the current job is not in the queue, so we can change its fields
bool Worker::Reschedule(Job::Id id)
{
  if (id == Job::kInvalid)
    throw std::runtime_error("invalid job");

  return RescheduleWorkerJob(*reinterpret_cast<WorkerJob*>(id), kScheduleNormal);
}

bool Worker::RescheduleDelayedMsec(Job::Id id, int msec)
{
  if (id == Job::kInvalid)
    throw std::runtime_error("invalid job");

  return RescheduleWorkerJob(*reinterpret_cast<WorkerJob*>(id),
                             SteadyClock::now() + std::chrono::milliseconds(msec));
}

bool Worker::RescheduleAtTime(Job::Id id, TimePoint time)
{
  if (id == Job::kInvalid)
    throw std::runtime_error("invalid job");

  return RescheduleWorkerJob(*reinterpret_cast<WorkerJob*>(id), time);
}

bool Worker::RescheduleAfterJob(Job::Id id, Job::Id yield, bool immediate)
{
  if (id == Job::kInvalid || yield == Job::kInvalid)
    throw std::runtime_error("invalid job");

  // note that if yield is invalid, it will just reschedule the job normally
  TimePoint time = immediate ? kScheduleImmediate : kScheduleNormal;
  return RescheduleWorkerJobRunAfter(*reinterpret_cast<WorkerJob*>(id), time,
                                     *reinterpret_cast<WorkerJob*>(yield));
}

//
// Threads
//

// WorkerRun is a "thread runner" until it's killed or if queue is empty in no-thread mode
void Worker::WorkerRun()
{
  WorkerJob* wjob = nullptr;

  while (1)
  {
    unsigned int count = thread_count_;
    if (thread_count_adjust_ < count)
    {
      if (thread_count_.compare_exchange_strong(count, count - 1))
        return;

      continue;
    }

    // Debugging: DumpQueue();

    // get next job
    if (wjob == nullptr)
    {
      TimePoint timeout;
      std::unique_lock lock(queue_mutex_);
      wjob = DequeueNextWorkerJobLocked(timeout);

      // no pending job, sleep and restart loop
      if (wjob == nullptr)
      {
        // special case, one-shot, no running threads
        if (thread_count_ == 0 && queue_ready_ == 0)
          return;

        if (timeout != kScheduleImmediate && timeout != kScheduleSleeping)
          queue_wait_cond_.wait_until(lock, timeout);
        else
          queue_wait_cond_.wait(lock);

        continue;
      }
    }

    // set running in the loop to ensure we won't get a false-positive
    // 'killed' job status. (might get a false-positive run status)
    WorkerJob::Reschedule reschedule;
    do
    {
      reschedule = wjob->reschedule_.load();
      wjob->running_ = reschedule != WorkerJob::kKill;
    }
    while (reschedule != WorkerJob::kKill &&
        !wjob->reschedule_.compare_exchange_strong(reschedule, WorkerJob::kOnce));

    if (reschedule != WorkerJob::kKill)
    {
      // execute job; returns false to quit (kill) job
      const std::function<bool()>& functor =
        reinterpret_cast<std::function<bool()>&>(*wjob->function_buffer_);

      CurrentJob::tls_current_job_id_ = reinterpret_cast<Job::Id>(wjob);
      wjob->result_ = functor();
      wjob->running_ = false;

      if (!wjob->result_)
        wjob->reschedule_ = WorkerJob::kKill;
      reschedule = wjob->reschedule_.load();
    }

    if (reschedule == WorkerJob::kAgain)
    {
      if (queue_needs_reschedule_.load())
      {
        std::lock_guard lock (queue_mutex_);
        // don't add to main queue if already added to a yielding job
        if (!wjob->is_queued_ && !QueueWorkerJobLocked(*wjob))
        {
          WakeBlockedLocked(*wjob);
          ReleaseWorkerJobLocked(*wjob);
        }
        wjob = nullptr;
      }
    }
    else
    {
      // Job has completed
      std::lock_guard lock (queue_mutex_);
      WakeBlockedLocked(*wjob);
      ReleaseWorkerJobLocked(*wjob);

      wjob->wait_mutex_.lock();
      wjob->wait_mutex_.unlock();

      wjob->wait_cond_.notify_all();
      wjob = nullptr;
    }
  }
}

//
// Slot (Bitmap Allocator)
//

// Find a free slot from the allocation bitmap
int Worker::AllocateSlot()
{
  std::scoped_lock lock(job_mutex_);

  for (unsigned int i = 0; i < job_capacity_ / 32; i++)
  {
    if (job_alloc_bitmap_[i] != ~0U)
    {
      uint32_t b = job_alloc_bitmap_[i];
      int index = 0;
      b = ~b & (b+1);

      if ((b & 0xffff0000) != 0)
        index += 16;
      if ((b & 0xff00ff00) != 0)
        index += 8;
      if ((b & 0xf0f0f0f0) != 0)
        index += 4;
      if ((b & 0xcccccccc) != 0)
        index += 2;
      if ((b & 0xaaaaaaaa) != 0)
        index += 1;
      if (index + i * 32 < job_capacity_)
      {
        job_alloc_bitmap_[i] |= 1 << index;
        return index + i * 32;
      }
    }
  }
  // no space left
  return -1;
}

void Worker::FreeSlot(int index)
{
  std::scoped_lock lock(job_mutex_);
  job_alloc_bitmap_[index/32] &= ~ (1 << (index & 31));
}

//
// Queue
//

inline static WorkerJob* ThisFromNext(WorkerJob** next)
{
  return (WorkerJob*)((uintptr_t)next - (uintptr_t)(&(((WorkerJob*)0))->next_));
}

// Managing the Queue (queue has to be locked for these functions)
//
// Notes
//  * For performance reasons, most fields of the WorkerJob are only
//    initialized when they leave the locked area.
//    These fields are  next_, prev_, reschedule_
//
//  * scheduled_ and sleeping_ point to the pointers for the first
//    scheduled or sleeping job. This is usually the next_ pointer of the
//    last job of the previous category, or ready_ if no job is in any
//    of the previous categories.
//
//  * The 'locked' versions can assume that queue and job are non-null.
//
// Returns the pointer to the job where next points to &job->next_
bool Worker::QueueWorkerJobLocked(WorkerJob& wjob)
{
  if (wjob.is_queued_)
    throw std::runtime_error("already queued");

  if (wjob.reschedule_.load() == WorkerJob::kKill)
    return false;

  TimePoint time = wjob.scheduled_time_;
  if (time < SteadyClock::now())
    time = kScheduleImmediate;

  // add job as sleeping
  if (time == kScheduleSleeping)
  {
    WorkerJob** sleeping = queue_sleeping_;
    wjob.prev_ = sleeping != &queue_ready_ ? ThisFromNext(sleeping) : nullptr;
    wjob.next_ = sleeping != &queue_ready_ ? *sleeping : nullptr;
    if (*sleeping != nullptr)
      (*sleeping)->prev_ = &wjob;
    *sleeping = &wjob;
  }

  // insert job before the job that is scheduled later (note O(n)!)
  else if (time > kScheduleTime)
  {
    WorkerJob* prev = queue_sleeping_ != &queue_ready_ ?  ThisFromNext(queue_sleeping_) : nullptr;
    WorkerJob* next = *queue_sleeping_;

    while (prev != nullptr && time < prev->scheduled_time_ )
      next = prev, prev = prev->prev_;

    wjob.prev_ = prev;
    wjob.next_ = next;

    // next == 0 -> prev: last scheduled job or nullptr if none is scheduled
    if (next != nullptr)
      next->prev_ = &wjob;

    if (next == *queue_sleeping_)
      queue_sleeping_ = &wjob.next_;

    // prev == 0 -> no jobs ready, scheduled_ points to ready_
    if (prev != nullptr)
      prev->next_ = &wjob;
    else
      queue_ready_ = &wjob;
  }

  // insert job at ready_ before any other jobs
  else if (time == kScheduleImmediate)
  {
    // insert job before all other jobs
    wjob.prev_ = nullptr;
    wjob.next_ = queue_ready_;
    queue_ready_ = &wjob;
    if (wjob.next_ != nullptr)
      wjob.next_->prev_ = &wjob;

    // push scheduled and sleeping pointers, if necessary
    if (queue_scheduled_ == &queue_ready_)
      queue_scheduled_ = &wjob.next_;
    if (queue_sleeping_ == &queue_ready_)
      queue_sleeping_ = &wjob.next_;

    queue_needs_reschedule_ = true;
  }

  // insert job at the end of any ready job and before any 'scheduled' job
  else /* kScheduleNormal */
  {
    WorkerJob** scheduled = queue_scheduled_;
    wjob.prev_ = scheduled != &queue_ready_ ? ThisFromNext(scheduled) : nullptr;
    wjob.next_ = *scheduled;
    if (*scheduled != nullptr)
      (*scheduled)->prev_ = &wjob;
    *scheduled = &wjob;  // updates ready_ if scheduled_ pointed there

    // move scheduled and sleeping pointers
    queue_scheduled_ = &wjob.next_;
    if (queue_sleeping_ == scheduled)
      queue_sleeping_ = &wjob.next_;
  }
  wjob.is_queued_ = true;
  wjob.refcount_++;

  // ensure a worker thread is taking up the job
  if (time != kScheduleSleeping)
  {
    queue_mutex_.unlock();
    queue_wait_cond_.notify_one();
    queue_mutex_.lock();
  }

  return true;
}

// note that it returns the current job reference count
void Worker::AddRefBlockedLocked(WorkerJob& job)
{
  if (job.block_refcount_++ == 0)
    job.refcount_++;
}

bool Worker::ReleaseBlockedLocked(WorkerJob& job)
{
  bool unblocked = --job.block_refcount_ == 0;
  if (unblocked)
  {
    ReleaseWorkerJobLocked(job);        // release as a blocked reference
    DequeueWorkerJobLocked(job);

    // TODO: support other options than immediate
    job.scheduled_time_ = kScheduleImmediate;
    job.is_queued_ = false;
    job.prev_ = nullptr;
    job.next_ = nullptr;
    if (job.reschedule_.load() != WorkerJob::kKill)
      QueueWorkerJobLocked(job);
    else
      ReleaseWorkerJobLocked(job);  // release the worker reference
  }
  return unblocked;
}


// TODO: currently only one of two blocking options is supported
// TODO:  1. *one* job blocks *multiple* jobs (wyield->block_ != nullptr)
// TODO:  2. *multiple* jobs blocks *one* job (wjob->block_refcount_ >= 1)
// TODO: multiple jobs blocking multiple jobs (n x m) is not supported, yet
bool Worker::QueueWorkerJobBlockedLocked(WorkerJob& wblock, WorkerJob& wyield)
{
  if (wblock.reschedule_.load() == WorkerJob::kKill)// || wblock.is_queued_)
    return false;

  if (wyield.reschedule_.load() == WorkerJob::kKill)
    return false;

  if (wyield.block_ != nullptr && wblock.block_refcount_ > 0)
    throw std::runtime_error("multiple jobs blocking multiple jobs unsupported");

  AddRefBlockedLocked(wblock);

  // insert the job to the blocked queue of the yielding job
  wblock.prev_ = &wyield;
  wblock.next_ = wyield.block_;
  if (wblock.next_ != nullptr)
    wblock.next_->prev_ = &wblock;
  wyield.block_ = &wblock;

  wblock.is_queued_ = true;
  queue_needs_reschedule_ = true;

  return true;
}


// Dequeue (can be from the main queue or from a 'blocked' queue)
void Worker::DequeueWorkerJobLocked(WorkerJob& wjob)
{
  if (wjob.block_refcount_ != 0)
    throw std::runtime_error("cannot dequeue blocked job");

  WorkerJob* prev = wjob.prev_;
  WorkerJob* next = wjob.next_;

  if (prev != nullptr)
    prev->next_ = wjob.next_;
  if (next != nullptr)
    next->prev_ = wjob.prev_;

  // adjust main queue pointers if removed from there
  if (wjob.prev_ == nullptr)
  {
    if (queue_sleeping_ == &wjob.next_)
      queue_sleeping_ = prev != nullptr ? &prev->next_ : &queue_ready_;
    if (queue_scheduled_ == &wjob.next_)
      queue_scheduled_ = prev != nullptr ? &prev->next_ : &queue_ready_;
    if (queue_ready_ == &wjob)
      queue_ready_ = next;
  }

  // remove job from linked list
  wjob.next_ = nullptr;
  wjob.prev_ = nullptr;
  wjob.scheduled_time_ = kScheduleNormal;
  wjob.is_queued_ = false;
  if (--wjob.refcount_ == 0)
    throw std::runtime_error("internal error, refcount == 0 in dequeue");
}

WorkerJob* Worker::DequeueNextWorkerJobLocked(TimePoint& timeout)
{
  // we are 'rescheduling' right now, so clear flag
  queue_needs_reschedule_ = false;

again:
  // scheduled_ can never be nullptr, just security check
  WorkerJob* wjob = *queue_scheduled_ != nullptr ? *queue_scheduled_ : nullptr;
  TimePoint next_scheduled_time = wjob != nullptr ? wjob->scheduled_time_ : kScheduleImmediate;

  if (wjob == nullptr || wjob->scheduled_time_ > SteadyClock::now())
    wjob = (wjob != queue_ready_ ? queue_ready_ : nullptr);

  if (wjob != nullptr)
  {
    DequeueWorkerJobLocked(*wjob);
    if (wjob->reschedule_.load() == WorkerJob::kKill)
    {
      ReleaseWorkerJobLocked(*wjob);
      goto again;
    }
  }

  timeout = next_scheduled_time;

  return wjob;
}

//
// Wake
//

bool Worker::WakeJobLocked(WorkerJob& wjob)
{
  bool woken = false;

  // is job is already queued, we need to re-queue it; cannot wake blocked job (yet)
  if (wjob.is_queued_ && wjob.block_refcount_ == 0)
  {
    DequeueWorkerJobLocked(wjob);

    wjob.scheduled_time_ = kScheduleNormal;
    woken = QueueWorkerJobLocked(wjob);
    if (!woken)
      ReleaseWorkerJobLocked(wjob);
  }

  return woken;
}

void Worker::WakeBlockedLocked(WorkerJob& wjob)
{
  WorkerJob* next = wjob.block_;
  while (next != nullptr)
  {
    WorkerJob* wblocked = next;
    next = wblocked->next_;
    ReleaseBlockedLocked(*wblocked);
  }
}

//
// Post Job
//

bool Worker::PostWorkerJob(WorkerJob& wjob, TimePoint time)
{
  wjob.reschedule_ = WorkerJob::kOnce;
  wjob.scheduled_time_ = time;

  std::lock_guard thread_lock(queue_mutex_);
  return QueueWorkerJobLocked(wjob);
}

// Note that when the yield job is not alive, this job will automatically be
// scheduled after the yield job is released.
bool Worker::PostWorkerJobBlocked(WorkerJob& wjob, WorkerJob& wyield)
{
  wjob.reschedule_ = WorkerJob::kOnce;
  wjob.scheduled_time_ = kScheduleSleeping;

  std::lock_guard lock(queue_mutex_);
  return QueueWorkerJobBlockedLocked(wjob, wyield);
}

// Note that when the yield job is not alive, this job will automatically be
// scheduled after the yield job is released.
bool Worker::PostWorkerJobBlocking(WorkerJob& wjob, TimePoint time, WorkerJob& wblock)
{
  wjob.reschedule_ = WorkerJob::kOnce;
  wjob.scheduled_time_ = time;

  {
    std::lock_guard lock(queue_mutex_);
    if (!QueueWorkerJobBlockedLocked(wblock, wjob))
      return false;
    if (!QueueWorkerJobLocked(wjob))
      return false;
  }

  return true;
}

//
// Reschedule
//

bool Worker::RescheduleWorkerJob(WorkerJob& wjob, TimePoint time)
{
  // set reschedule flag to kAgain unless it has been killed
  WorkerJob::Reschedule reschedule;
  wjob.scheduled_time_ = time;
  do
  {
    reschedule = wjob.reschedule_.load();
    if (reschedule == WorkerJob::kKill)
      return false;
  }
  while (!wjob.reschedule_.compare_exchange_strong(reschedule, WorkerJob::kAgain));

  /// don't force a reschedule if the job is currently running and not
  /// scheduled for a later time. (job is normally running at this point)
  if (wjob.is_queued_ || time > kScheduleNormal)
    queue_needs_reschedule_ = true; // FIMXE: wake job??

  return true;
}

bool Worker::RescheduleWorkerJobRunAfter(WorkerJob& wjob, TimePoint time, WorkerJob& wyield)
{
  std::lock_guard lock (queue_mutex_);

  // set reschedule flag to kAgain unless it has been killed
  WorkerJob::Reschedule reschedule;
  do
  {
    reschedule = wjob.reschedule_.load();
    if (reschedule == WorkerJob::kKill)
      return false;
  }
  while (!wjob.reschedule_.compare_exchange_strong(reschedule, WorkerJob::kAgain));

  if (wyield.reschedule_.load() != WorkerJob::kKill)
  {
    wjob.scheduled_time_ = time;
    return QueueWorkerJobBlockedLocked(wjob, wyield);
  }

  return true;
}

//
// Release
//

void Worker::ReleaseWorkerJobLocked(WorkerJob& wjob)
{
  if (wjob.refcount_.load() == 0)
    throw std::runtime_error("job already released");

  if (--wjob.refcount_ == 0)
  {
    if (wjob.is_queued_)
      throw std::runtime_error("trying to release queued job");
    FreeSlot(&wjob - jobs_);
  }
}

//
// Debug helpers
//

void Worker::DumpQueue()
{
  std::lock_guard lock(queue_mutex_);

  printf("-------------------\n");
  WorkerJob* wjob;
  if (queue_scheduled_ == &queue_ready_)
      printf("-- scheduled --\n");
  if (queue_sleeping_ == &queue_ready_)
      printf("-- sleeping --\n");
  for (wjob = queue_ready_; wjob != nullptr; wjob = wjob->next_)
  {
    printf("Job %p time %ld ref %d next %p block %p (ref: %d)\n", wjob,
           static_cast<long unsigned int>(wjob->scheduled_time_.time_since_epoch().count()),
           wjob->refcount_.load(),
           wjob->next_,
           wjob->block_,
           wjob->block_refcount_);
    if (wjob->next_ == *queue_scheduled_)
      printf("-- scheduled --\n");
    if (wjob->next_ == *queue_sleeping_)
      printf("-- sleeping --\n");
    if (wjob == wjob->next_)
    {
      printf("ERROR: looping queue\n");
      break;
    }
  }
}


} // end of namespace grid
