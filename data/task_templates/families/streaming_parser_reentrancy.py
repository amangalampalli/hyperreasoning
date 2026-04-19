"""Streaming parser task family."""

from __future__ import annotations

from data.task_templates.base import TaskSpec, TaskTemplate
from data.task_templates.utils import build_rng, choose_variant, dedent, make_task_id, render_test_module


class StreamingParserReentrancyTemplate(TaskTemplate):
    """Generate chunked parser repair tasks."""

    family = "streaming_parser_reentrancy"

    def generate_instance(self, seed: int, difficulty: str) -> TaskSpec:
        self._validate_difficulty(difficulty)
        rng = build_rng(seed, self.family, difficulty)
        class_name = choose_variant(rng, ["StreamingFrameParser", "ChunkParser", "MessageFrameParser"])
        error_name = choose_variant(rng, ["FrameParseError", "StreamProtocolError"])
        payload_a = choose_variant(rng, ["alpha", "delta", "status=ok"])
        payload_b = choose_variant(rng, ["beta!", "gamma?", "worker:42"])
        prompt = choose_variant(
            rng,
            [
                f"Repair `{class_name}` in `parser.py`. It parses newline-terminated frames encoded as "
                "`kind|length|payload\\n` across repeated `feed()` calls. The parser must preserve partial "
                "state across chunk boundaries, allow `reset()` for reuse, and reject truncated frames on "
                "`close()` without leaking state between sessions.",
                f"`parser.py` contains a stateful framed-stream parser named `{class_name}`. Fix it so chunked "
                "input, repeated parser reuse, and end-of-stream handling all follow the documented "
                "`kind|length|payload\\n` protocol.",
            ],
        )
        reference = dedent(
            f"""
            from __future__ import annotations

            from dataclasses import dataclass


            class {error_name}(ValueError):
                pass


            @dataclass(frozen=True, slots=True)
            class EventFrame:
                kind: str
                payload: str


            class {class_name}:
                def __init__(self) -> None:
                    self.reset()

                def reset(self) -> None:
                    self._buffer = ""
                    self._pending_kind: str | None = None
                    self._pending_size: int | None = None
                    self._closed = False

                def feed(self, chunk: str) -> list[EventFrame]:
                    if self._closed:
                        raise RuntimeError("parser is closed")
                    if not isinstance(chunk, str):
                        raise TypeError("chunk must be a string")
                    self._buffer += chunk
                    frames: list[EventFrame] = []
                    while True:
                        if self._pending_kind is None:
                            separator = self._buffer.find("|")
                            if separator == -1:
                                break
                            kind = self._buffer[:separator]
                            if not kind:
                                raise {error_name}("empty kind")
                            self._pending_kind = kind
                            self._buffer = self._buffer[separator + 1 :]
                        if self._pending_size is None:
                            separator = self._buffer.find("|")
                            if separator == -1:
                                break
                            raw_size = self._buffer[:separator]
                            if not raw_size.isdigit():
                                raise {error_name}("invalid size")
                            self._pending_size = int(raw_size)
                            self._buffer = self._buffer[separator + 1 :]
                        assert self._pending_kind is not None
                        assert self._pending_size is not None
                        if len(self._buffer) < self._pending_size + 1:
                            break
                        payload = self._buffer[: self._pending_size]
                        terminator = self._buffer[self._pending_size : self._pending_size + 1]
                        if terminator != "\\n":
                            raise {error_name}("missing frame terminator")
                        frames.append(EventFrame(self._pending_kind, payload))
                        self._buffer = self._buffer[self._pending_size + 1 :]
                        self._pending_kind = None
                        self._pending_size = None
                    return frames

                def close(self) -> list[EventFrame]:
                    frames = self.feed("")
                    self._closed = True
                    if (
                        self._buffer
                        or self._pending_kind is not None
                        or self._pending_size is not None
                    ):
                        raise {error_name}("truncated frame")
                    return frames
            """
        )

        buggy = reference
        bug_types = ["loses trailing buffered frames after a successful parse"]
        strategy_traps = [
            "Fixing only close() is insufficient because buffered multi-frame chunks still break",
            "A local patch that ignores reset() can leave stale parser state across sessions",
        ]
        buggy = buggy.replace(
            'self._buffer = self._buffer[self._pending_size + 1 :]\n                        self._pending_kind = None',
            'self._buffer = ""\n                        self._pending_kind = None',
        )
        if difficulty == "hard":
            bug_types.append("reset() leaves partial header state behind after parser reuse")
            buggy = buggy.replace(
                "    def reset(self) -> None:\n                    self._buffer = \"\"\n                    self._pending_kind: str | None = None\n                    self._pending_size: int | None = None\n                    self._closed = False\n",
                "    def reset(self) -> None:\n                    self._buffer = \"\"\n                    self._closed = False\n",
            )
            strategy_traps.append(
                "A patch that only preserves leftover buffers still fails when the same parser instance is reused"
            )

        visible_tests = render_test_module(
            dedent(
                f"""
                from parser import {class_name}, EventFrame


                class StreamingParserVisibleTests(unittest.TestCase):
                    def test_splits_headers_and_payloads_across_feeds(self) -> None:
                        parser = {class_name}()
                        self.assertEqual(parser.feed("evt|"), [])
                        self.assertEqual(parser.feed("5|he"), [])
                        frames = parser.feed("llo\\n")
                        self.assertEqual(frames, [EventFrame("evt", "hello")])

                    def test_multiple_frames_can_arrive_in_one_chunk(self) -> None:
                        parser = {class_name}()
                        chunk = "job|{len(payload_a)}|{payload_a}\\nack|{len(payload_b)}|{payload_b}\\n"
                        frames = parser.feed(chunk)
                        self.assertEqual(
                            frames,
                            [
                                EventFrame("job", "{payload_a}"),
                                EventFrame("ack", "{payload_b}"),
                            ],
                        )
                """
            )
        )

        hidden_tests = render_test_module(
            dedent(
                f"""
                from parser import {class_name}, EventFrame, {error_name}


                class StreamingParserHiddenTests(unittest.TestCase):
                    def test_reset_allows_clean_reuse_after_partial_frame(self) -> None:
                        parser = {class_name}()
                        parser.feed("part|4|te")
                        parser.reset()
                        frames = parser.feed("done|4|pass\\n")
                        self.assertEqual(frames, [EventFrame("done", "pass")])

                    def test_close_rejects_truncated_frame(self) -> None:
                        parser = {class_name}()
                        parser.feed("evt|5|hel")
                        with self.assertRaises({error_name}):
                            parser.close()

                    def test_reuse_after_close_requires_reset(self) -> None:
                        parser = {class_name}()
                        parser.feed("ok|2|hi\\n")
                        self.assertEqual(parser.close(), [])
                        with self.assertRaises(RuntimeError):
                            parser.feed("again|5|hello\\n")

                    def test_follow_up_frame_after_same_chunk_is_preserved(self) -> None:
                        parser = {class_name}()
                        frames = parser.feed("a|1|x\\nb|1|y\\n")
                        self.assertEqual(frames, [EventFrame("a", "x"), EventFrame("b", "y")])
                """
            )
        )

        return self.build_spec(
            seed=seed,
            difficulty=difficulty,
            prompt=prompt,
            files={
                "parser.py": buggy,
                "test_visible.py": visible_tests,
                "test_hidden.py": hidden_tests,
            },
            reference_files={"parser.py": reference},
            entrypoint="parser.py",
            visible_test_file="test_visible.py",
            hidden_test_file="test_hidden.py",
            task_id=make_task_id(self.family, seed),
            metadata={
                "bug_type": bug_types,
                "strategy_traps": strategy_traps,
                "target_files": ["parser.py"],
                "expected_skill_tags": ["state-machines", "streaming", "reentrancy", "tests"],
                "niche_topic": "chunked framed protocol parsing",
                "repairable": True,
            },
        )
