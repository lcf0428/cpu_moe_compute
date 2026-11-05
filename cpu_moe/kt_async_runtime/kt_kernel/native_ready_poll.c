#include <stdint.h>
#include <stdatomic.h>
#include <time.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define PAUSE() _mm_pause()
#else
#include <arm_acle.h>
#define PAUSE() __asm__ volatile("yield")
#endif

#define N 2048

static struct {
  _Atomic uint32_t *ptr;
  uint32_t seen;
} slots[N];
static _Atomic unsigned count;
static _Atomic int stop_requested;

static uint64_t now_ns(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000000ULL + (uint64_t)t.tv_nsec;
}

void start_poll(void) {
  atomic_store_explicit(&stop_requested, 0, memory_order_release);
}

void stop_poll(void) {
  atomic_store_explicit(&stop_requested, 1, memory_order_release);
}

int register_flag(void *p) {
  unsigned i = atomic_load_explicit(&count, memory_order_acquire);
  if (i >= N) return -1;
  slots[i].ptr = (_Atomic uint32_t *)p;
  slots[i].seen = 0;
  atomic_store_explicit(&count, i + 1, memory_order_release);
  return (int)i;
}

/*
 * Scan continuously until a new generation is visible or the owner closes.
 * The old implementation returned -1 every 1 ms, creating a 0..1 ms blind
 * window between scans. The poll worker owns a dedicated CPU, so continuous
 * scanning removes that window without changing sequence or memory ordering.
 */
int poll_next(uint64_t *stamp, uint32_t *seq) {
  for (;;) {
    if (atomic_load_explicit(&stop_requested, memory_order_acquire)) return -1;
    unsigned n = atomic_load_explicit(&count, memory_order_acquire);
    for (unsigned i = 0; i < n; ++i) {
      if (atomic_load_explicit(&stop_requested, memory_order_acquire)) return -1;
      uint32_t v = atomic_load_explicit(slots[i].ptr, memory_order_acquire);
      if (v != slots[i].seen) {
        if (v != slots[i].seen + 1) return -2;
        slots[i].seen = v;
        atomic_thread_fence(memory_order_acquire);
        *stamp = now_ns();
        *seq = v;
        return (int)i;
      }
    }
    PAUSE();
  }
}
