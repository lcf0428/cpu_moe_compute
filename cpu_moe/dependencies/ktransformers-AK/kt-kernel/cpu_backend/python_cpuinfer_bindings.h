#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "completion_notifier.h"

namespace kt::python_bindings {

inline void bind_completion_notifier(pybind11::module_& m) {
  pybind11::class_<kt::cpu_async::CompletionNotifier>(m, "CompletionNotifier")
      .def(pybind11::init<>())
      .def("notify", [](kt::cpu_async::CompletionNotifier& self) {
        pybind11::gil_scoped_release release;
        self.notify();
      })
      .def("wait", [](kt::cpu_async::CompletionNotifier& self, uint64_t spin_us) {
        pybind11::gil_scoped_release release;
        self.wait(spin_us);
      }, pybind11::arg("spin_us") = kt::cpu_async::CompletionNotifier::kDefaultSpinUs)
      .def_property_readonly("pending", &kt::cpu_async::CompletionNotifier::pending);
}

// Descriptors have already been converted from Python before these lambdas
// run. Only the native call is outside the GIL: the release guard reacquires
// it before pybind11 converts a return value or destroys argument owners,
// including when a native exception unwinds the call.
template <class CPUInferType>
void bind_cpuinfer_native_calls(pybind11::class_<CPUInferType>& cls) {
  cls.def("submit", [](CPUInferType& self, std::pair<intptr_t, intptr_t> params) {
       pybind11::gil_scoped_release release;
       self.submit(params);
     })
      .def("sync", [](CPUInferType& self, size_t allow_n_pending) {
        pybind11::gil_scoped_release release;
        self.sync(allow_n_pending);
      }, pybind11::arg("allow_n_pending") = 0);
}

}  // namespace kt::python_bindings
