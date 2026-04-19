"""Thread-safe memoization task family."""

from __future__ import annotations

from data.task_templates.base import TaskSpec, TaskTemplate
from data.task_templates.utils import build_rng, choose_variant, dedent, make_task_id, render_test_module


class ConcurrencySafeMemoizationTemplate(TaskTemplate):
    """Generate concurrent memoization repair tasks."""

    family = "concurrency_safe_memoization"

    def generate_instance(self, seed: int, difficulty: str) -> TaskSpec:
        self._validate_difficulty(difficulty)
        rng = build_rng(seed, self.family, difficulty)
        decorator_name = choose_variant(rng, ["memoize_threadsafe", "concurrent_memoize", "safe_memoize"])
        prompt = choose_variant(
            rng,
            [
                f"Repair `{decorator_name}` in `memo.py`. It should memoize results across threads without "
                "duplicating work for the same key, while still allowing unrelated keys to compute in parallel "
                "and avoiding exception caching.",
                f"`memo.py` contains a thread-safe memoization decorator. Fix it so contention is key-scoped, "
                "in-flight calls share a single computation, and failures do not poison the cache forever.",
            ],
        )
        reference = dedent(
            f"""
            from __future__ import annotations

            import threading
            from functools import wraps
            from typing import Any, Callable, ParamSpec, TypeVar

            P = ParamSpec("P")
            T = TypeVar("T")


            def {decorator_name}(func: Callable[P, T]) -> Callable[P, T]:
                cache: dict[tuple[Any, ...], T] = {{}}
                in_flight: dict[tuple[Any, ...], threading.Event] = {{}}
                lock = threading.Lock()

                @wraps(func)
                def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                    key = (args, tuple(sorted(kwargs.items())))
                    while True:
                        with lock:
                            if key in cache:
                                return cache[key]
                            event = in_flight.get(key)
                            if event is None:
                                event = threading.Event()
                                in_flight[key] = event
                                leader = True
                                break
                        event.wait()
                    try:
                        result = func(*args, **kwargs)
                    except Exception:
                        with lock:
                            in_flight.pop(key).set()
                        raise
                    with lock:
                        cache[key] = result
                        in_flight.pop(key).set()
                    return result

                return wrapper
            """
        )

        if difficulty == "medium":
            buggy = dedent(
                f"""
                from __future__ import annotations

                import threading
                from functools import wraps
                from typing import Any, Callable, ParamSpec, TypeVar

                P = ParamSpec("P")
                T = TypeVar("T")


                def {decorator_name}(func: Callable[P, T]) -> Callable[P, T]:
                    cache: dict[tuple[Any, ...], T] = {{}}
                    lock = threading.Lock()

                    @wraps(func)
                    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                        key = (args, tuple(sorted(kwargs.items())))
                        with lock:
                            if key in cache:
                                return cache[key]
                        result = func(*args, **kwargs)
                        with lock:
                            cache[key] = result
                        return result

                    return wrapper
                """
            )
            bug_types = ["same-key calls race and compute duplicate results under contention"]
            strategy_traps = [
                "A global cache lock around lookups is not enough if the computation happens outside any in-flight guard",
                "The bug appears only when multiple threads hit the same key at once",
            ]
        else:
            buggy = dedent(
                f"""
                from __future__ import annotations

                import threading
                from functools import wraps
                from typing import Any, Callable, ParamSpec, TypeVar

                P = ParamSpec("P")
                T = TypeVar("T")


                def {decorator_name}(func: Callable[P, T]) -> Callable[P, T]:
                    cache: dict[tuple[Any, ...], Any] = {{}}
                    lock = threading.Lock()

                    @wraps(func)
                    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                        key = (args, tuple(sorted(kwargs.items())))
                        with lock:
                            if key in cache:
                                value = cache[key]
                                if isinstance(value, Exception):
                                    raise value
                                return value
                            try:
                                result = func(*args, **kwargs)
                            except Exception as exc:
                                cache[key] = exc
                                raise
                            cache[key] = result
                            return result

                    return wrapper
                """
            )
            bug_types = ["global lock serializes unrelated keys and exceptions are cached as permanent failures"]
            strategy_traps = [
                "A serialized decorator can pass correctness checks while destroying parallelism",
                "Exception caching looks convenient but violates retry-after-failure behavior",
            ]

        visible_tests = render_test_module(
            dedent(
                f"""
                import threading
                import time

                from memo import {decorator_name}


                class MemoVisibleTests(unittest.TestCase):
                    def test_reuses_cached_result_sequentially(self) -> None:
                        calls = 0

                        @{decorator_name}
                        def compute(value: int) -> int:
                            nonlocal calls
                            calls += 1
                            return value * 10

                        self.assertEqual(compute(3), 30)
                        self.assertEqual(compute(3), 30)
                        self.assertEqual(calls, 1)

                    def test_same_key_only_computes_once_under_contention(self) -> None:
                        entered = threading.Event()
                        release = threading.Event()
                        calls = 0
                        results: list[int] = []
                        errors: list[str] = []

                        @{decorator_name}
                        def compute(value: int) -> int:
                            nonlocal calls
                            calls += 1
                            entered.set()
                            release.wait(timeout=0.2)
                            return value * 2

                        def worker() -> None:
                            try:
                                results.append(compute(5))
                            except Exception as exc:  # pragma: no cover - test helper
                                errors.append(str(exc))

                        threads = [threading.Thread(target=worker) for _ in range(3)]
                        for thread in threads:
                            thread.start()
                        entered.wait(timeout=0.2)
                        time.sleep(0.05)
                        release.set()
                        for thread in threads:
                            thread.join()

                        self.assertEqual(errors, [])
                        self.assertEqual(results, [10, 10, 10])
                        self.assertEqual(calls, 1)
                """
            )
        )

        hidden_tests = render_test_module(
            dedent(
                f"""
                import threading
                import time

                from memo import {decorator_name}


                class MemoHiddenTests(unittest.TestCase):
                    def test_failures_are_not_cached(self) -> None:
                        calls = 0
                        should_fail = True

                        @{decorator_name}
                        def flaky(value: int) -> int:
                            nonlocal calls, should_fail
                            calls += 1
                            if should_fail:
                                should_fail = False
                                raise ValueError("transient")
                            return value * 3

                        with self.assertRaises(ValueError):
                            flaky(4)
                        self.assertEqual(flaky(4), 12)
                        self.assertEqual(calls, 2)

                    def test_different_keys_can_run_in_parallel(self) -> None:
                        started = threading.Event()
                        overlap = threading.Event()
                        active = 0
                        active_lock = threading.Lock()

                        @{decorator_name}
                        def compute(value: int) -> int:
                            nonlocal active
                            with active_lock:
                                active += 1
                                if active == 2:
                                    overlap.set()
                            started.set()
                            time.sleep(0.05)
                            with active_lock:
                                active -= 1
                            return value

                        threads = [
                            threading.Thread(target=lambda: compute(1)),
                            threading.Thread(target=lambda: compute(2)),
                        ]
                        began = time.monotonic()
                        for thread in threads:
                            thread.start()
                        started.wait(timeout=0.2)
                        for thread in threads:
                            thread.join()
                        elapsed = time.monotonic() - began
                        self.assertTrue(overlap.is_set())
                        self.assertLess(elapsed, 0.095)
                """
            )
        )

        return self.build_spec(
            seed=seed,
            difficulty=difficulty,
            prompt=prompt,
            files={
                "memo.py": buggy,
                "test_visible.py": visible_tests,
                "test_hidden.py": hidden_tests,
            },
            reference_files={"memo.py": reference},
            entrypoint="memo.py",
            visible_test_file="test_visible.py",
            hidden_test_file="test_hidden.py",
            task_id=make_task_id(self.family, seed),
            metadata={
                "bug_type": bug_types,
                "strategy_traps": strategy_traps,
                "target_files": ["memo.py"],
                "expected_skill_tags": ["threading", "memoization", "locks", "concurrency"],
                "niche_topic": "key-scoped concurrent memoization",
                "repairable": True,
            },
        )
