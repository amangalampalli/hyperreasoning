"""Async retry contract task family."""

from __future__ import annotations

from data.task_templates.base import TaskSpec, TaskTemplate
from data.task_templates.utils import build_rng, choose_variant, dedent, make_task_id, render_test_module


class AsyncRetryContractTemplate(TaskTemplate):
    """Generate async retry semantics tasks."""

    family = "async_retry_contract"

    def generate_instance(self, seed: int, difficulty: str) -> TaskSpec:
        self._validate_difficulty(difficulty)
        rng = build_rng(seed, self.family, difficulty)
        function_name = choose_variant(rng, ["run_with_retry", "invoke_with_retry", "call_with_retry"])
        prompt = choose_variant(
            rng,
            [
                f"Repair `{function_name}` in `retry.py`. The helper wraps an async operation with retry "
                "logic, but it must preserve cancellation semantics, respect retry classification, avoid "
                "retrying committed side effects, and apply deterministic exponential backoff.",
                f"`retry.py` contains an async retry helper named `{function_name}`. Fix its contract so "
                "retryable failures retry, non-retryable failures surface immediately, cancellation is never "
                "swallowed, and non-idempotent operations are not retried.",
            ],
        )
        reference = dedent(
            f"""
            from __future__ import annotations

            import asyncio
            from typing import Awaitable, Callable, TypeVar

            T = TypeVar("T")


            async def {function_name}(
                operation: Callable[[], Awaitable[T]],
                *,
                attempts: int = 3,
                retry_for: tuple[type[Exception], ...] = (Exception,),
                sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                base_delay: float = 0.0,
                is_retryable: Callable[[BaseException], bool] | None = None,
                idempotent: bool = True,
            ) -> T:
                if attempts < 1:
                    raise ValueError("attempts must be at least 1")
                for attempt in range(1, attempts + 1):
                    try:
                        return await operation()
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        if not isinstance(exc, retry_for):
                            raise
                        if is_retryable is not None and not is_retryable(exc):
                            raise
                        if getattr(exc, "committed", False):
                            raise
                        if not idempotent:
                            raise
                        if attempt == attempts:
                            raise
                        delay = base_delay * (2 ** (attempt - 1))
                        if delay:
                            await sleep(delay)
                raise RuntimeError("retry loop exited unexpectedly")
            """
        )

        buggy = reference.replace("except asyncio.CancelledError:\n                        raise\n", "")
        bug_types = ["cancellation is swallowed by a broad retry handler"]
        strategy_traps = [
            "Retry loops must not catch asyncio.CancelledError as a normal failure",
            "A fix that only changes retry counts can still violate idempotency and commit semantics",
        ]
        if difficulty == "hard":
            buggy = buggy.replace(
                "                        if getattr(exc, \"committed\", False):\n                            raise\n                        if not idempotent:\n                            raise\n",
                "",
            )
            bug_types.append("committed or non-idempotent operations are retried when they should surface")
            strategy_traps.append(
                "A plausible retry patch still fails if committed exceptions or non-idempotent calls are retried"
            )

        visible_tests = render_test_module(
            dedent(
                f"""
                import asyncio

                from retry import {function_name}


                class TransientError(Exception):
                    pass


                class FatalError(Exception):
                    pass


                class AsyncRetryVisibleTests(unittest.IsolatedAsyncioTestCase):
                    async def test_retries_until_success(self) -> None:
                        attempts = 0
                        delays: list[float] = []

                        async def operation() -> str:
                            nonlocal attempts
                            attempts += 1
                            if attempts < 3:
                                raise TransientError("retry me")
                            return "ok"

                        async def fake_sleep(delay: float) -> None:
                            delays.append(delay)

                        result = await {function_name}(
                            operation,
                            attempts=4,
                            retry_for=(TransientError,),
                            sleep=fake_sleep,
                            base_delay=0.5,
                        )

                        self.assertEqual(result, "ok")
                        self.assertEqual(delays, [0.5, 1.0])

                    async def test_non_retryable_exception_surfaces(self) -> None:
                        calls = 0

                        async def operation() -> str:
                            nonlocal calls
                            calls += 1
                            raise FatalError("boom")

                        with self.assertRaises(FatalError):
                            await {function_name}(operation, retry_for=(TransientError,))
                        self.assertEqual(calls, 1)
                """
            )
        )

        hidden_tests = render_test_module(
            dedent(
                f"""
                import asyncio

                from retry import {function_name}


                class RetryableError(Exception):
                    def __init__(self, message: str, *, committed: bool = False) -> None:
                        super().__init__(message)
                        self.committed = committed


                class AsyncRetryHiddenTests(unittest.IsolatedAsyncioTestCase):
                    async def test_cancelled_error_is_not_retried(self) -> None:
                        calls = 0

                        async def operation() -> None:
                            nonlocal calls
                            calls += 1
                            raise asyncio.CancelledError()

                        with self.assertRaises(asyncio.CancelledError):
                            await {function_name}(operation)
                        self.assertEqual(calls, 1)

                    async def test_committed_retryable_error_is_not_retried(self) -> None:
                        calls = 0

                        async def operation() -> None:
                            nonlocal calls
                            calls += 1
                            raise RetryableError("committed", committed=True)

                        with self.assertRaises(RetryableError):
                            await {function_name}(operation, retry_for=(RetryableError,), attempts=4)
                        self.assertEqual(calls, 1)

                    async def test_non_idempotent_operation_is_not_retried(self) -> None:
                        calls = 0

                        async def operation() -> None:
                            nonlocal calls
                            calls += 1
                            raise RetryableError("write failed")

                        with self.assertRaises(RetryableError):
                            await {function_name}(
                                operation,
                                retry_for=(RetryableError,),
                                attempts=5,
                                idempotent=False,
                            )
                        self.assertEqual(calls, 1)

                    async def test_predicate_can_block_retries(self) -> None:
                        calls = 0

                        async def operation() -> None:
                            nonlocal calls
                            calls += 1
                            raise RetryableError("do not retry")

                        with self.assertRaises(RetryableError):
                            await {function_name}(
                                operation,
                                retry_for=(RetryableError,),
                                is_retryable=lambda exc: False,
                            )
                        self.assertEqual(calls, 1)
                """
            )
        )

        return self.build_spec(
            seed=seed,
            difficulty=difficulty,
            prompt=prompt,
            files={
                "retry.py": buggy,
                "test_visible.py": visible_tests,
                "test_hidden.py": hidden_tests,
            },
            reference_files={"retry.py": reference},
            entrypoint="retry.py",
            visible_test_file="test_visible.py",
            hidden_test_file="test_hidden.py",
            task_id=make_task_id(self.family, seed),
            metadata={
                "bug_type": bug_types,
                "strategy_traps": strategy_traps,
                "target_files": ["retry.py"],
                "expected_skill_tags": ["asyncio", "retries", "cancellation", "control-flow"],
                "niche_topic": "async retry contract repair",
                "repairable": True,
            },
        )
