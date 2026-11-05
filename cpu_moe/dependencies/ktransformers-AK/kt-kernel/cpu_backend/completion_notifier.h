#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace kt::cpu_async {

// Native notification only: Python request owners stay in SimpleQueue and
// are never touched/destroyed by this GIL-free wait. One consumer is used by
// the completion worker; atomic permit acquisition also prevents underflow.
class CompletionNotifier {
 public:
  void notify() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (permits_.load(std::memory_order_relaxed) == std::numeric_limits<uint64_t>::max()) {
        throw std::overflow_error("completion notification counter overflow");
      }
      permits_.fetch_add(1, std::memory_order_release);
    }
    cv_.notify_one();
  }

  void wait(uint64_t spin_us = kDefaultSpinUs) {
    if (try_acquire()) return;
    // Keep the existing short hot window without periodic Python wakeups or
    // retaining GIL. Large accidental settings cannot create unbounded spin.
    if (spin_us != 0) {
      const auto deadline = std::chrono::steady_clock::now() +
          std::chrono::microseconds(std::min(spin_us, kMaximumSpinUs));
      unsigned until_clock_check = kClockCheckInterval;
      while (true) {
        pause();
        if (try_acquire()) return;
        if (--until_clock_check == 0) {
          if (std::chrono::steady_clock::now() >= deadline) break;
          until_clock_check = kClockCheckInterval;
        }
      }
    }
    std::unique_lock<std::mutex> lock(mutex_);
    // notify increments under the same mutex: both the spin-to-sleep race
    // and spurious wakeups recheck/consume a permit, so no notification is lost.
    cv_.wait(lock, [this] { return try_acquire(); });
  }

  uint64_t pending() const { return permits_.load(std::memory_order_acquire); }
  static constexpr uint64_t kDefaultSpinUs = 3000;

 private:
  bool try_acquire() {
    auto count = permits_.load(std::memory_order_acquire);
    while (count != 0) {
      if (permits_.compare_exchange_weak(count, count - 1,
              std::memory_order_acq_rel, std::memory_order_acquire)) return true;
    }
    return false;
  }
  static void pause() {
#if defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#else
    std::this_thread::yield();
#endif
  }
  static constexpr uint64_t kMaximumSpinUs = 1000000;
  static constexpr unsigned kClockCheckInterval = 64;
  std::atomic<uint64_t> permits_{0};
  std::mutex mutex_;
  std::condition_variable cv_;
};

}  // namespace kt::cpu_async
