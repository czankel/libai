//
// Copyright (C) Chris Zankel. All rights reserved.
// This code is subject to U.S. and other copyright laws and
// intellectual property protections.
//
// The contents of this file are confidential and proprietary to Chris Zankel.
//

#ifndef GRID_UTIL_WORKER_H
#define GRID_UTIL_WORKER_H

#include <stdint.h>
#include <time.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <limits>
#include <list>
#include <mutex>
#include <new>
#include <thread>

namespace grid {


using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;
using Duration = std::chrono::duration<float>;
using SteadyClock = std::chrono::steady_clock;


// When a job is added to the queue, it can be scheduled to run at a specific
// time or with these values:
//
//  - kScheduleImmediate: the job is inserted to the front of the queue. O(1)
//  - kScheduleNormal:    the jobs is inserted at the end of the 'ready' queue. O(1)
//  - <time>:             the job is inserted in the scheduled queue at the specified time. O(n)
//  - kScheduleSleeping:  the jobs is added to the end of the scheduled queue

static const TimePoint kScheduleImmediate(SteadyClock::duration(0));
static const TimePoint kScheduleNormal(SteadyClock::duration(1));
static const TimePoint kScheduleTime(SteadyClock::duration(2));
static const TimePoint kScheduleBlocked(TimePoint::max() - std::chrono::hours(1));
static const TimePoint kScheduleSleeping(TimePoint::max());

class Worker;
struct WorkerJob;

/// Job is a scheduling entity for a worker.
///
/// The job describes a function that is executed when it is scheduled
/// by the worker. The function are not interrupted and are rescheduled
/// after they return.
///
/// The job includes many function that allow to reschedule with a variety
/// of options (delayed, at-time, waiting for another job)
class Job
{
 public:

  /// Type of the job dentifier type.
  using Id = uintptr_t;

  /// Status represents the state of a job.
  enum Status
  {
    kInvalid = 0,   //< Job is invalid
    kRunning,       //< Job is running
    kWaiting,       //< Job is waiting
    kDone,          //< Job has completed successfully
    kError          //< Job has encountered an error
  };

  inline Job() : id_(kInvalid) {}

  /// Copy constructor.
  ///
  /// @param[in] job  Other job.
  Job(const Job& other);

  /// Move constructor
  ///
  /// @param[in] job  Other job.
  Job(Job&& other) : id_(other.id_)          {}

  ~Job();

  /// Assign operator.
  ///
  /// @param[in] other  Other job.
  /// @returns This job.
  Job& operator=(const Job& other);
  Job& operator=(Job&& other)                     { id_ = other.id_; return *this; }


  // Comparison operator
  bool operator==(const Job& other) const         { return id_ == other.id_; }

  /// not-operator to check if job is valid
  bool operator!() const                          { return id_ == kInvalid; }

  /// IsValid returns if the given job is valid.
  ///
  /// @returns True if job is valid or false, otherwise.
  bool IsValid() const                            { return id_ != kInvalid; }

  /// GetStatus returns the current status of the job.
  ///
  /// Note that there's a race condition, by the time the caller processes
  /// the information, the status might have changed.
  ///
  /// @returns Current status of the job.
  Status GetStatus() const;

  /// GetId returns the job identifier.
  ///
  /// @returns Job identifier
  inline Id GetId() const                         { return id_; }

  /// GetWorker returns the Worker for the job.
  ///
  /// Worker is only valid while the Job is valid.
  ///
  /// @returns Worker
  /// @throws runtime_error if invalid job
  Worker& GetWorker();
  Worker& GetWorker() const;

  /// GetContext returns the job's context id.
  ///
  /// The context id is an unsigned long value that is inherited when
  /// creating a new job.
  ///
  /// @returns Context id.
  uint64_t GetContext() const;

  /// SetContext tests the job's context id.
  ///
  /// Note that this will not change the current context id if the job is
  /// running. Use 'CurrentJob::SetContext()' for this.
  ///
  /// @param[in] context  Context for the job
  void SetContext(uint64_t context);


  /// Wake wakes up the job.
  ///
  /// This function is a no-op if the job is already running.
  /// Note that the job might still be blocked and will be scheduled after these jobs
  ///
  /// @returns false if invalid or killed.
  bool Wake();

  /// Kill kills the job.
  ///
  /// The job will be removed if it is queued and currently not running,
  /// or won't be scheduled once the currently running function returns.
  ///
  /// Note that the job cannot be rescheduled once it is 'killed'.
  void Kill();


  /// Wait waits for the job to complete; returns true if it was completed and not killed.
  bool Wait();

  /// Wait waits for the job to complete; returns true if it was completed and not killed or timeout.
  bool WaitFor(Duration);

  /// Wait waits for the job to complete; returns true if it was completed and not killed or timeout.
  bool WaitUntil(TimePoint);

  /// ChangeScheduledTime changes the reschedule time for the job.
  ///
  /// @param[in] time  New scheduled time for the job.
  void ChangeScheduledTime(TimePoint time);

 private:
  friend class Worker;
  friend class CurrentJob;

  Job(const Id id);

  Id      id_;
};

/// The CurrentJob describes the current job that has been set for the current
/// thread. If the thread is not scheduled by the worker, the current job
/// is invalid.
///
/// To get the current job, use:
///
///   Job job = CurrentJob::GetJob();
class CurrentJob
{
  friend class Worker;

  CurrentJob();

 public:

  /// GetWorker is a staic function to return the current worker of the thread
  static Worker& GetWorker();

  /// GetJob is a static function to return the current job for the thread
  /// of the caller.
  ///
  /// @returns Current job or kInvalid if none was set.
  static Job GetJob();

  /// IsValid returns if the current thread is a valid job
  static bool IsValid();

  /// SetContext sets the context id of the current job.
  ///
  /// @param[in] context  New context for the job.
  static void SetContext(uint64_t context);

  /// GetCOntext returns the context id of the current job.
  ///
  /// @returns  Context of the current job.
  static uint64_t GetContext();


  /// Reschedule reschedules the current job to run after all scheduled job.
  ///
  /// @returns True for success or false if job is invalid or has been killed.
  static bool Reschedule();

  /// RescheduleDelayedMsec reschedules the current job to run after the
  /// specified delay from now.
  ///
  /// Setting the delay to kInifiniteTime set the job to sleep.
  ///
  /// @params[in] mdelay  Delay in miliseconds
  /// @returns True for success or false if job is invalid or has been killed.
  static bool RescheduleDelayedMsec(int msec);

  /// RescheduleAtTime reschedules the current job to run at a later  time.
  ///
  /// @params[in] time  Time when this job should run again.
  /// @returns True for success or false if job is invalid or has been killed.
  static bool RescheduleAtTime(TimePoint time);

  /// RescheduleAfterJob reschedules the current job to run after the specified
  /// job has completed.
  ///
  /// The specified job won't change its priority (and might continue to sleep),
  /// unless inherit_priority is set, in which case the specified job will
  /// inherit the priority of the current job.
  ///
  /// There is currently no mechanism to wait for multiple jobs.
  ///
  /// @param[in] function  Function to execute
  /// @returns True for success or false if job is invalid or has been killed.
  static bool RescheduleAfterJob(const Job& job, bool inherit_priority = false);

  /// RescheduleSleeping sets job to sleep.
  ///
  /// @returns True for success or false if job is invalid or has been killed.
  static bool RescheduleSleeping();

  /// IsRescheduled returns if the current job is still alive and hasn't been
  /// killed.
  ///
  /// @returns true if the current job is still alive.
  static bool IsRescheduled();

  /// IsRescheduledWaiting returns if the specified job is rescheduled
  /// for a later time or is blocked by another job.
  ///
  /// @returns true if the job is waiting.
  static bool IsRescheduledWaiting();

  /// NeedsReschedule checks if another higher priority job is waiting in the
  /// same queue of the current job.
  ///
  /// @param[in] id  Job id.
  /// @returns True if the worker would need to reschedule the queue.
  static bool NeedsReschedule();


  /// Kill kills the current job.
  ///
  /// It won't be scheduled again once the current function returns.
  ///
  /// Note that the job cannot be rescheduled once it is 'killed'.
  static void Kill();

  /// WakeBlocked wakes all blocked jobs.
  static void WakeBlocked();

 private:
  static inline Job GetCurrentJob()
  {
    return Job(tls_current_job_id_);
  }

  static __thread Job::Id tls_current_job_id_;
};

/// Worker is an interface for managing work queue of uninterruptible and
/// cooperative jobs.
///
/// Jobs consist of a function that is called when the job is ready to run.
/// They can be scheduled or rescheduled as follows:
///
///  - immediately or when any currently running job returns
///  - next after the current job gets rescheduled
///  - after all currently available jobs
///  - at a specific time
///  - after another job has completed.
///
/// Jobs can be in different queues. Each queue is managed by a queue of
/// threads.
class Worker
{
 public:
  static const unsigned int kDefaultJobCapacity = 100;
  static const unsigned int kMaxThreadCount = std::numeric_limits<unsigned int>::max();

  Worker (const Worker&) = delete;
  Worker(Worker&&) = delete;
  Worker& operator= (const Worker&) = delete;

  Worker(unsigned int job_capacity = kDefaultJobCapacity,
         unsigned int thread_count = kMaxThreadCount);

  ~Worker();

  /// Run runs a no-thread worker until the last job exits.
  bool Run();


  /// UpdateThreadCount updates the thread count to align with thread_count_adjust_
  ///
  /// @returns True if thread count was changed
  bool UpdateThreadCount(Duration timeout);

  /// GetThreadCount returns the number of concurrent threads for the worker.
  ///
  /// @returns max concurrent thread count
  unsigned int GetThreadCount() const             { return thread_count_; }

  /// GetConcurrentThreadCount returns the maximum number of concurrent threads.
  static unsigned int GetConcurrentThreadCount()
  {
    return std::thread::hardware_concurrency();
  }


  /// Wake wakes the specified job.
  ///
  /// @param[in] job  Job to wake.
  /// @returns True for success or false if job is invalid or has been killed.
  bool Wake(const Job& job);

  /// WakeBlocked Wakes all jobst that are blocked on the specified job.
  ///
  /// @param[in] job Job that is  blocking other jobs.
  void WakeBlocked(const Job& job);


  /// KillJob kills the specified job.
  ///
  /// Note that the job might still be running when this function returns.
  /// Use WaitForJob() to wait for completion.
  /// Also note that the job cannot be rescheduled once it has been 'killed'.
  ///
  /// @param[in] job  Job to kill.
  void KillJob(const Job& job);


  /// Block increments the block count and a job will remain sleeping.
  ///
  /// If the job is running, it will go to sleep if it's rescheduled.
  ///
  /// @param[in] id  Job id.
  void Block(Job&);

  /// Release releases a blocked job; the job will resume unless there are other pending blocks.
  ///
  /// @param[in] id  Job id.
  void ReleaseBlock(Job&);


  /// IsRescheduled returns if the current job is still alive and hasn't been
  /// killed.
  ///
  /// @returns true if job is still alive.
  bool IsRescheduled(const Job& job);

  /// IsRescheduledWaiting returns if the specified job is rescheduled
  /// for a later time or is blocked by another job.
  ///
  /// @param[in] job  Job.
  /// @returns True if job has been rescheduled or false for errors.
  bool IsRescheduledWaiting(const Job& job);


  /// Wait for a job to complete
  /// Note that jobs that are rescheduled are not complete
  /// @param[in] job to wait for
  /// @returns true if job has completed
  bool WaitForJob(const Job& job);

  /// Wait for a job to complete for the specific time
  ///
  /// @returns true if job has completed
  bool WaitForJobFor(const Job&, Duration);

  /// Wait for a job to complete
  ///
  /// @returns true if job has completed
  bool WaitForJobUntil(const Job&, TimePoint);


  /// Post posts a new job to the current queue.
  ///
  /// @param[in] function  Function to post; see function.h
  /// @returns Job identification; use IsValid() to check if it was successful
  template <typename F, typename... Args> Job Post(F&&, Args&&...);

  /// PostImmediate posts a job to run immediately.
  ///
  /// @param[in] function  Function to post.
  /// @returns Job identification; use IsValid() to check if it was successful
  template <typename F> Job PostImmediate(F&& function);

  /// PostNext posts a job to run after the current job.
  ///
  /// @param[in] function  Function to post; see function.h
  /// @returns Job identification; use IsValid() to check if it was successful
  template <typename F> Job PostNext(F&& function);

  /// PostDelayedMsec posts a job to run after the specified delay.
  ///
  /// @param[in] function  Function to post; see function.h
  /// @param[in] mdealy  Delay in [ms] or kInifinteTime
  /// @returns Job identification; use IsValid() to check if it was successful
  template <typename F> Job PostDelayedMsec(F&& function, int msec);

  /// PostAtTime posta a job to run at the specific time.
  ///
  /// @param[in] function  Function to post; see function.h
  /// @param[in] time  Time when the job should run.
  /// @returns Job identification; use IsValid() to check if it was successful
  template <typename F> Job PostAtTime(F&& function, TimePoint time);

  /// PostRunAfter posts a job to run after the specified job has completed.
  ///
  /// The created job is scheduled as "sleeping" until the dependent job has finished.
  ///
  /// @param[in] function  Function that should be run
  /// @param[in] job  Dependent job that must complete
  /// @returns Job identification; use IsValid() to check if it was successful
  template <typename F, typename... Args> Job PostRunAfter(Job& job, F&& function, Args&&...);

  /// PostRunBefore posts a job to run before the specified job has completed.
  ///
  /// The depentent job has to be in the "sleeping" state and will be woken when all
  /// jobs created job is scheduled as "sleeping" until the dependent job has finished.
  ///
  /// @param[in] function  Function that should be run
  /// @param[in] job  Dependent job that must complete
  /// @returns Job identification; use IsValid() to check if it was successful
  template <typename F, typename... Args> Job PostRunBefore(Job& job, F&& function, Args&&...);

  /// PostBlocked posts a job as 'blocked'; the job is sleeping until unblocked (ReleaseBlock)
  ///
  /// @param[in] function  Function that should be run
  /// @returns Job identification; use IsValid() to check if it was successful
  template <typename F, typename... Args> Job PostBlocked(F&& function, Args&&...);

  /// PostSleeping posts a job as 'sleeping', waiting for a wake-up event
  ///
  /// @param[in] function  Function that should be run.
  template <typename F> Job PostSleeping(F&& function);

 protected:

  friend class Job;
  friend class CurrentJob;

  /// AllocateJob allocates a new job.
  ///
  /// @param[in] size  Size required for the job function.
  /// @returns New job id or kInvalid for errors.
  Job::Id AllocateJob(size_t size);

  /// Return the 'function' pointer for the specific id.
  ///
  /// The function pointer is used to 're-initialize' a job and it may point
  /// to a pre-allocated array.
  ///
  /// @param[in] id  Job id.
  /// @returns Pointer to the job function or nullptr for errors.
  void* GetFunctionPointer(const Job::Id id);

  /// AddRefJob increments the reference counter for the job.
  ///
  /// @param[in] id  Job id.
  void AddRefJob(Job::Id id);

  /// ReleaseJob releases the job and decrements the reference counter. The
  /// job is deleted when the count goes to 0.
  ///
  /// @param[in] id  Job id.
  void ReleaseJob(Job::Id id);


  /// SetContext sets the context id for the provided job.
  ///
  /// @param[in] id  Job id.
  /// @param[in] context  Context.
  void SetContext(Job::Id id, uint64_t context);

  /// GetContext returns the context id for the provided job.
  ///
  /// @param[in] id  Job id.
  /// @returns Context of the job.
  uint64_t GetContext(Job::Id id) const;

  /// GetJobStatus returns the status of the specified job.
  ///
  /// @param[in] id  Job id.
  /// @returns Status of the job.
  Job::Status GetJobStatus(Job::Id id) const;


  /// NeedsReschedule returns if the worker would need to reschedule the queue.
  ///
  /// NeedsReschedule checks if another higher priority job is waiting in the
  /// same queue of the specified job.
  ///
  /// @param[in] id  Job id.
  /// @returns True if the worker would need to reschedule the queue.
  bool NeedsReschedule(Job::Id id) const;

 protected:

  /// Reschedule reschedules a job to run after all scheduled jobs.
  ///
  /// @param[in] id  Job id.
  /// @returns True for success or false if jobs was killed.
  bool Reschedule(Job::Id id);

  /// RescheduleDelayedMsec reschedules the job to run after the specified
  /// delay from now.
  ///
  /// @param[in] id  Job id.
  /// @returns True for success or false if jobs was killed.
  bool RescheduleDelayedMsec(Job::Id id, int msec);

  /// RescheduleAtTime reschedules the job to run at a specific time.
  ///
  /// @param[in] id  Job id.
  /// @returns True for success or false if jobs was killed.
  bool RescheduleAtTime(Job::Id id, TimePoint time);

  /// RescheduleAfterJob reeschedule the job to run after the specified job
  /// has finished.
  ///
  /// @param[in] id  Job id.
  /// @param[in] yield  Yield job id.
  /// @param[in] immediate  Schedule yield job immedidately.
  /// @returns True for success or false if jobs was killed.
  bool RescheduleAfterJob(Job::Id id, Job::Id yield, bool inherit_priority = false);

 private:
  inline Job::Id AllocateJob(std::function<bool()>&&);

  template <typename F, typename... Args>
  inline Job::Id AllocateJob(F&&, Args&&...);

  template <typename R, typename... Args>
  Job::Id AllocateJob(R(&&function)(Args&&...), Args&&...);

  template <typename R, typename C, typename... Args>
  Job::Id AllocateJob(R(C::*)(Args...), C&, Args&&...);

  bool CreateThreadsLocked();

  // returns false if timeout
  void WorkerRun();

  // Slot
  int AllocateSlot();
  void FreeSlot(int index);

  // manage blocking jobs

  // Queue
  // Returns false if job has been killed or if already queued
  bool QueueWorkerJobLocked(WorkerJob& wjob);
  bool QueueWorkerJobBlockedLocked(WorkerJob& wjob, WorkerJob& wyield);


  WorkerJob* DequeueNextWorkerJobLocked(TimePoint&);
  void DequeueWorkerJobLocked(WorkerJob& wjob);
  void RemoveWorkerJobLocked(WorkerJob& wjob);

  // Release the job; this will wake any blocked jobs and remove it from the table if unreferenced
  // The job must not be queued or blocked
  void ReleaseWorkerJobLocked(WorkerJob& wjob);

  void AddRefBlockedLocked(WorkerJob& wjob);
  bool ReleaseBlockedLocked(WorkerJob& wjob);

  bool WakeJobLocked(WorkerJob&);
  void WakeBlockedLocked(WorkerJob&);

  bool PostWorkerJob(WorkerJob& wjob, TimePoint time);
  bool PostWorkerJobBlocked(WorkerJob& wjob, WorkerJob& wyield);
  bool PostWorkerJobBlocking(WorkerJob& wjob, TimePoint time, WorkerJob& wblock);
  bool RescheduleWorkerJob(WorkerJob& wjob, TimePoint time);
  bool RescheduleWorkerJobRunAfter(WorkerJob& wjob, TimePoint time, WorkerJob& wyield);


  void DumpQueue();

 private:
  mutable std::mutex        thread_mutex_;
  unsigned int              thread_count_adjust_;
  std::atomic<unsigned int> thread_count_;
  std::list<std::thread>    thread_pool_;

  mutable std::mutex        job_mutex_;
  unsigned int              job_capacity_;
  uint32_t*                 job_alloc_bitmap_;
  WorkerJob*                jobs_;

  // The Worker implementation uses a single queue for all jobs.
  // The queue is divided into three sections and three pointers pointing to those sections:
  //
  //  - ready:     immediately ready to run
  //  - scheduled: scheduled at a later time
  //  - sleeping:  un-scheduled (sleeping or blocked)

  mutable std::mutex        queue_mutex_;
  std::condition_variable   queue_wait_cond_;
  std::atomic<bool>         queue_needs_reschedule_;
  WorkerJob*                queue_ready_;
  WorkerJob**               queue_scheduled_;
  WorkerJob**               queue_sleeping_;
};

//
// WorkerJob
//

struct WorkerJob
{
  static const int kMaxFunctorSize = 200;

  enum Reschedule
  {
    kKill = -1,
    kOnce = 0,      // single run
    kAgain = 1,     // reschedule
  };

  // start of mutex-protected section
  WorkerJob*              next_;
  WorkerJob*              prev_;
  WorkerJob*              block_;

  Worker*                 worker_;

  TimePoint               scheduled_time_;
  int                     block_refcount_;

  bool                    is_queued_;
  bool                    result_;
  bool                    running_;
  // end of mutex-protected section

  std::atomic<Reschedule> reschedule_;
  std::atomic<int>        refcount_;

  std::mutex              wait_mutex_;
  std::condition_variable wait_cond_;

  char                    function_buffer_[kMaxFunctorSize];
};

//
// Worker Implementation
//

template <typename F, typename... Args>
inline Job::Id Worker::AllocateJob(F&& function, Args&&... args)
{
  auto f = std::bind(function, std::forward<Args>(args)...);
  Job::Id id = AllocateJob(sizeof(f));
  if (id != Job::kInvalid)
    new (reinterpret_cast<std::function<bool(Args...)>*>(GetFunctionPointer(id)))
      std::function<bool(Args...)>(std::move(f));
  return id;
}


template <typename R, typename... Args>
inline Job::Id Worker::AllocateJob(R(&&function)(Args&&...), Args&&... args)
{
  auto f = std::bind(&function, std::forward<Args>(args)...);

  Job::Id id = AllocateJob(sizeof(f));
  if (id != Job::kInvalid)
    new (reinterpret_cast<std::function<bool()>*>(GetFunctionPointer(id)))
      std::function<bool()>(std::move(f));
  return id;
}


template <typename R, typename C, typename... Args>
inline Job::Id Worker::AllocateJob(R(C::*function)(Args...), C& c, Args&&... args)
{
  auto f = std::bind(std::mem_fn(function), &c, args...);

  Job::Id id = AllocateJob(sizeof(f));
  if (id != Job::kInvalid)
    *reinterpret_cast<std::function<bool()>*>(GetFunctionPointer(id)) = std::move(f);
  return id;
}

// Post Jobs

template <typename F, typename... Args>
inline Job Worker::Post(F&& function, Args&&... args)
{
  Job::Id id = AllocateJob(std::forward<F>(function), std::forward<Args>(args)...);
  if (id != Job::kInvalid && !PostWorkerJob(*reinterpret_cast<WorkerJob*>(id), kScheduleNormal))
  {
    ReleaseJob(id);
    id = Job::kInvalid;
  }
  return Job(id);
}


template <typename F, typename... Args>
inline Job Worker::PostRunAfter(Job& job, F&& function, Args&&... args)
{
  Job::Id id = AllocateJob(std::forward<F>(function), std::forward<Args>(args)...);
  Job::Id yield_id = job.GetId();
  if (id != Job::kInvalid && yield_id != Job::kInvalid &&
      !PostWorkerJobBlocked(*reinterpret_cast<WorkerJob*>(id),
                            *reinterpret_cast<WorkerJob*>(yield_id)))
  {
    ReleaseJob(id);
    id = Job::kInvalid;
  }
  return Job(id);
}


template <typename F, typename... Args>
inline Job Worker::PostRunBefore(Job& job, F&& function, Args&&... args)
{
  Job::Id id = AllocateJob(std::forward<F>(function), std::forward<Args>(args)...);
  Job::Id blocked_id = job.GetId();
  if (id != Job::kInvalid && blocked_id != Job::kInvalid &&
      !PostWorkerJobBlocking(*reinterpret_cast<WorkerJob*>(id), kScheduleNormal,
                             *reinterpret_cast<WorkerJob*>(blocked_id)))
  {
    ReleaseJob(id);
    id = Job::kInvalid;
  }
  return Job(id);
}

template <typename F, typename... Args>
inline Job Worker::PostBlocked(F&& function, Args&&... args)
{
  Job::Id id = AllocateJob(std::forward<F>(function), std::forward<Args>(args)...);
  if (id != Job::kInvalid)
  {
    if (PostWorkerJob(*reinterpret_cast<WorkerJob*>(id), kScheduleSleeping))
    {
      std::lock_guard lock(queue_mutex_);
      AddRefBlockedLocked(*reinterpret_cast<WorkerJob*>(id));
    }
    else
    {
      ReleaseJob(id);
      id = Job::kInvalid;
    }
  }
  return Job(id);
}

  template <typename F>
inline Job Worker::PostNext(F&& function)
{
  Job::Id id = AllocateJob(std::forward<F>(function));
  if (id != Job::kInvalid &&
      !PostWorkerJob(*reinterpret_cast<WorkerJob*>(id), kScheduleNormal))
  {
    ReleaseJob(id);
    id = Job::kInvalid;
  }
  return Job(id);
}


template <typename F>
inline Job Worker::PostImmediate(F&& function)
{
  Job::Id id = AllocateJob(std::forward<F>(function));
  if (id != Job::kInvalid &&
      !PostWorkerJob(*reinterpret_cast<WorkerJob*>(id), kScheduleImmediate))
  {
    ReleaseJob(id);
    id = Job::kInvalid;
  }
  return Job(id);
}


template <typename F>
inline Job Worker::PostAtTime(F&& function, TimePoint time)
{
  Job::Id id = AllocateJob(std::forward<F>(function));
  if (id != Job::kInvalid && !PostWorkerJob(*reinterpret_cast<WorkerJob*>(id), time))
  {
    ReleaseJob(id);
    id = Job::kInvalid;
  }
  return Job(id);
}


template <typename F>
inline Job Worker::PostDelayedMsec(F&& function, int msec)
{
  return Worker::PostAtTime(std::forward<F>(function),
                            SteadyClock::now() + std::chrono::milliseconds(msec));
}


template <typename F>
inline Job Worker::PostSleeping(F&& function)
{
  return Worker::PostAtTime(std::forward<F>(function), kScheduleSleeping);
}


//
// Job:: implementations
//

inline Worker& Job::GetWorker()
{
  if (!IsValid())
    throw std::runtime_error("invalid job");
  return *reinterpret_cast<WorkerJob*>(id_)->worker_;
}

inline Worker& Job::GetWorker() const
{
  if (!IsValid())
    throw std::runtime_error("invalid job");
  return *reinterpret_cast<WorkerJob*>(id_)->worker_;
}


inline Job::Job(const Job& other) : id_(other.id_)
{
  if (id_ != kInvalid)
    GetWorker().AddRefJob(id_);
}

inline Job::~Job()
{
  if (id_ != kInvalid)
  {
    GetWorker().ReleaseJob(id_);
    id_ = kInvalid;
  }
}

inline Job& Job::operator=(const Job& other)
{
  id_ = other.id_;

  if (other.IsValid())
    GetWorker().AddRefJob(id_);

  return *this;
}

inline bool Job::Wake()
{
  return id_ != kInvalid && GetWorker().Wake(*this);
}

inline void Job::Kill()
{
  if (id_ != kInvalid)
    GetWorker().KillJob(*this);
}

inline bool Job::Wait()
{
  return id_ != kInvalid && GetWorker().WaitForJob(*this);
}

inline bool Job::WaitFor(Duration timeout)
{
  return id_ != kInvalid && GetWorker().WaitForJobFor(*this, timeout);
}

inline bool Job::WaitUntil(TimePoint time)
{
  return id_ != kInvalid && GetWorker().WaitForJobUntil(*this, time);
}

inline void Job::ChangeScheduledTime(TimePoint time)
{
  if (id_ != kInvalid)
    GetWorker().RescheduleAtTime(id_, time);
}

inline void Job::SetContext(uint64_t context)
{
  if (id_ != kInvalid)
    GetWorker().SetContext(id_, context);
}

inline Job::Status Job::GetStatus() const
{
  return id_ == kInvalid ? kInvalid : GetWorker().GetJobStatus(id_);
}

inline Job::Job(const Id id) : id_(id)
{
  if (id_ != kInvalid)
    GetWorker().AddRefJob(id_);
}

//
// CurrentJob:: implementations
//

inline bool CurrentJob::IsValid()
{
  return tls_current_job_id_ != Job::kInvalid;
}

inline Worker& CurrentJob::GetWorker()
{
  if (!IsValid())
    throw std::runtime_error("invalid job");
  return *reinterpret_cast<WorkerJob*>(tls_current_job_id_)->worker_;
}

inline Job CurrentJob::GetJob()
{
  return Job(tls_current_job_id_);
}

inline void CurrentJob::SetContext(uint64_t context)
{
  if (IsValid())
    GetWorker().SetContext(tls_current_job_id_, context);
}

inline bool CurrentJob::Reschedule()
{
  return IsValid() &&
    GetWorker().Reschedule(tls_current_job_id_);
}

inline bool CurrentJob::NeedsReschedule()
{
  return IsValid() &&
    GetWorker().NeedsReschedule(tls_current_job_id_);
}

inline bool CurrentJob::RescheduleAtTime(TimePoint time)
{
  return IsValid() &&
    GetWorker().RescheduleAtTime(tls_current_job_id_, time);
}

inline bool CurrentJob::RescheduleDelayedMsec(int msec)
{
  return IsValid() &&
    GetWorker().RescheduleDelayedMsec(tls_current_job_id_, msec);
}

inline bool CurrentJob::RescheduleAfterJob(const Job& job, bool inherit_priority)
{
  return IsValid() &&
    GetWorker().RescheduleAfterJob(tls_current_job_id_, job.id_, inherit_priority);
}

inline void CurrentJob::Kill()
{
  if (IsValid())
    GetWorker().KillJob(Job(tls_current_job_id_));
}

inline bool CurrentJob::IsRescheduled()
{
  return IsValid() && GetWorker().IsRescheduled(Job(tls_current_job_id_));
}

inline bool CurrentJob::IsRescheduledWaiting()
{
  return IsValid() && GetWorker().IsRescheduledWaiting(Job(tls_current_job_id_));
}

inline bool CurrentJob::RescheduleSleeping()
{
  return IsValid() && GetWorker().RescheduleAtTime(tls_current_job_id_, kScheduleSleeping);
}

inline void CurrentJob::WakeBlocked()
{
  if (IsValid())
    GetWorker().WakeBlocked(Job(tls_current_job_id_));
}


} // end of namespace grid

#endif  // GRID_UTIL_WORKER_H
