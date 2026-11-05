#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stdexcept>

#include "cpu_backend/python_cpuinfer_bindings.h"

namespace py = pybind11;

// Uses the exact production method registration, with native waits instead of
// weights/NPU kernels. It can exercise GIL concurrency without an engine.
class NativeWaitFixture {
 public:
  explicit NativeWaitFixture(py::object owner) : owner_(std::move(owner)) {}
  ~NativeWaitFixture() { destructor_had_gil_.store(PyGILState_Check() != 0); }

  void submit(std::pair<intptr_t, intptr_t> args) {
    wait(submit_entered, submit_open_);
    if (args.second) throw std::runtime_error("native submit failure");
  }
  void sync(size_t allow_n_pending) {
    wait(sync_entered, sync_open_);
    if (allow_n_pending == 9) throw std::runtime_error("native sync failure");
  }
  void release_submit() { release(submit_open_); }
  void release_sync() { release(sync_open_); }
  static bool destructor_had_gil() { return destructor_had_gil_.load(); }
  std::atomic<bool> submit_entered{false}, sync_entered{false};

 private:
  void wait(std::atomic<bool>& entered, bool& open) {
    std::unique_lock<std::mutex> lock(mutex_);
    entered.store(true);
    if (!cv_.wait_for(lock, std::chrono::milliseconds(1000), [&] { return open; })) {
      throw std::runtime_error("native wait timed out (GIL progress failed)");
    }
  }
  void release(bool& open) {
    { std::lock_guard<std::mutex> lock(mutex_); open = true; }
    cv_.notify_all();
  }
  std::mutex mutex_;
  std::condition_variable cv_;
  bool submit_open_ = false, sync_open_ = false;
  py::object owner_;  // Must only be destroyed after the call reacquires GIL.
  inline static std::atomic<bool> destructor_had_gil_{false};
};

PYBIND11_MODULE(_cpuinfer_gil_fixture, m) {
  kt::python_bindings::bind_completion_notifier(m);
  auto cls = py::class_<NativeWaitFixture>(m, "NativeWaitFixture");
  cls.def(py::init<py::object>())
      .def_property_readonly("submit_entered", [](NativeWaitFixture& x) { return x.submit_entered.load(); })
      .def_property_readonly("sync_entered", [](NativeWaitFixture& x) { return x.sync_entered.load(); })
      .def("release_submit", &NativeWaitFixture::release_submit)
      .def("release_sync", &NativeWaitFixture::release_sync)
      .def_static("destructor_had_gil", &NativeWaitFixture::destructor_had_gil);
  kt::python_bindings::bind_cpuinfer_native_calls(cls);
}
