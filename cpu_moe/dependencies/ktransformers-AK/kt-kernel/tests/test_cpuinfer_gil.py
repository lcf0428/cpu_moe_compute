"""Compile only a tiny fixture sharing the production submit/sync bindings."""
import gc
import importlib.util
from pathlib import Path
import subprocess
import sys
import sysconfig
import threading
import time
import weakref

import pytest


@pytest.fixture(scope="session")
def native_binding(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path_factory.mktemp("native_gil") / ("_cpuinfer_gil_fixture" + sysconfig.get_config_var("EXT_SUFFIX"))
    try:
        import pybind11
        includes = [pybind11.get_include()]
    except ImportError:
        candidates = [root / "third_party/pybind11/include", root / "extern/pybind11/include",
            Path(sysconfig.get_path("purelib")) / "torch/include",
            Path('/mnt/sdc/y50063808/prof_traces/20261026-pipeline-opt/native_build/_deps/pybind11-src/include')]
        includes = [str(p) for p in candidates if (p / "pybind11/pybind11.h").is_file()]
        if not includes:
            raise RuntimeError("No local pybind11 headers for native GIL fixture")
    command = ["c++", "-shared", "-fPIC", "-std=c++17", "-O2", "-pthread",
        "-I" + str(root), "-I" + sysconfig.get_path("include"), *["-I" + p for p in includes],
        str(root / "tests/cpuinfer_gil_fixture.cpp"), "-o", str(output)]
    subprocess.run(command, check=True)
    spec = importlib.util.spec_from_file_location("_cpuinfer_gil_fixture", output)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("method,fail", [("submit", False), ("submit", True), ("sync", False), ("sync", True)])
def test_native_wait_releases_gil_and_recovers_on_exception(native_binding, method, fail):
    class Owner:
        pass

    owner = Owner()
    owner_ref = weakref.ref(owner)
    native = native_binding.NativeWaitFixture(owner)
    errors = []

    def call():
        try:
            getattr(native, method)((0, int(fail)) if method == "submit" else (9 if fail else 0))
        except RuntimeError as exc:
            errors.append(str(exc))

    thread = threading.Thread(target=call)
    start = time.monotonic()
    thread.start()
    deadline = start + .5
    while not getattr(native, method + "_entered") and time.monotonic() < deadline:
        time.sleep(.001)
    # This Python thread must progress while the other is inside the native
    # condition wait. A missing release blocks Python until fixture timeout.
    assert time.monotonic() < deadline
    assert getattr(native, method + "_entered")
    assert owner_ref() is not None
    getattr(native, "release_" + method)()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert errors == (["native " + method + " failure"] if fail else [])
    del thread, native, owner
    gc.collect()
    assert owner_ref() is None
    assert native_binding.NativeWaitFixture.destructor_had_gil()


def test_argument_conversion_failure_happens_before_native_call(native_binding):
    native = native_binding.NativeWaitFixture(object())
    with pytest.raises(TypeError):
        native.submit(("not-a-native-pointer", 0))
    assert not native.submit_entered
    with pytest.raises(TypeError):
        native.sync(-1)
    assert not native.sync_entered


def test_notifier_counts_contiguous_permits(native_binding):
    notifier = native_binding.CompletionNotifier()
    for _ in range(20): notifier.notify()
    assert notifier.pending == 20
    for _ in range(20): notifier.wait(0)
    assert notifier.pending == 0


@pytest.mark.parametrize("spin_us", [0, 50, 3000])
def test_notifier_wake_and_spin_to_sleep_race(native_binding, spin_us):
    _run_notifier_case(native_binding, "race", spin_us)


def test_notifier_condition_wait_releases_gil(native_binding):
    _run_notifier_case(native_binding, "gil", 0)


def _run_notifier_case(native_binding, mode, spin_us):
    # An incorrect native wait could hold GIL forever. Bound the whole probe
    # with a subprocess timeout rather than allowing a regression to hang UT.
    script = r'''
import importlib.util,sys,threading,time
spec=importlib.util.spec_from_file_location('_cpuinfer_gil_fixture',sys.argv[1])
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
n=m.CompletionNotifier();mode=sys.argv[2];spin=int(sys.argv[3]);consumed=[]
count=500 if mode=='race' else 1
def consumer():
    for i in range(count):n.wait(spin);consumed.append(i)
t=threading.Thread(target=consumer,daemon=True);t.start()
for i in range(count):
    if i%20==0:time.sleep(.0001)
    n.notify()
t.join(1)
assert not t.is_alive(), 'lost completion notification'
assert consumed==list(range(count)) and n.pending==0
'''
    subprocess.run([sys.executable,"-c",script,native_binding.__file__,mode,str(spin_us)],
                   check=True,timeout=3)
