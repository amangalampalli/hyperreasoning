"""Serializer round-trip task family."""

from __future__ import annotations

from data.task_templates.base import TaskSpec, TaskTemplate
from data.task_templates.utils import build_rng, choose_variant, dedent, make_task_id, render_test_module


class SerializerRoundtripEscapeTemplate(TaskTemplate):
    """Generate serializer/decoder round-trip tasks."""

    family = "serializer_roundtrip_escape"

    def generate_instance(self, seed: int, difficulty: str) -> TaskSpec:
        self._validate_difficulty(difficulty)
        rng = build_rng(seed, self.family, difficulty)
        encode_name = choose_variant(rng, ["encode_fields", "serialize_fields"])
        decode_name = choose_variant(rng, ["decode_fields", "deserialize_fields"])
        prompt = choose_variant(
            rng,
            [
                f"Repair `{encode_name}` and `{decode_name}` in `codec.py`. The codec prefixes the number of "
                "fields and then escapes separators. It must round-trip arbitrary text fields, preserve empty "
                "strings, and reject malformed trailing escapes or count mismatches.",
                f"`codec.py` contains a tiny custom field codec. Fix escaping and decoding so serialized data "
                "round-trips exactly, even when delimiters, backslashes, and empty fields appear inside values.",
            ],
        )
        reference = dedent(
            f"""
            from __future__ import annotations


            _ESCAPE_MAP = {{
                "\\\\": "\\\\\\\\",
                "|": "\\\\|",
                "#": "\\\\#",
                "\\n": "\\\\n",
                "\\t": "\\\\t",
            }}
            _UNESCAPE_MAP = {{
                "\\\\": "\\\\",
                "|": "|",
                "#": "#",
                "n": "\\n",
                "t": "\\t",
            }}


            def {encode_name}(fields: list[str]) -> str:
                encoded_fields: list[str] = []
                for field in fields:
                    buffer: list[str] = []
                    for character in field:
                        buffer.append(_ESCAPE_MAP.get(character, character))
                    encoded_fields.append("".join(buffer))
                return f"{{len(fields)}}#" + "|".join(encoded_fields)


            def {decode_name}(payload: str) -> list[str]:
                raw_count, separator, body = payload.partition("#")
                if separator != "#" or not raw_count.isdigit():
                    raise ValueError("missing field count prefix")
                expected_count = int(raw_count)
                if expected_count == 0:
                    if body:
                        raise ValueError("unexpected data for empty payload")
                    return []
                fields: list[str] = []
                current: list[str] = []
                escaped = False
                for character in body:
                    if escaped:
                        if character not in _UNESCAPE_MAP:
                            raise ValueError("unknown escape sequence")
                        current.append(_UNESCAPE_MAP[character])
                        escaped = False
                    elif character == "\\\\":
                        escaped = True
                    elif character == "|":
                        fields.append("".join(current))
                        current = []
                    else:
                        current.append(character)
                if escaped:
                    raise ValueError("dangling escape")
                fields.append("".join(current))
                if len(fields) != expected_count:
                    raise ValueError("field count mismatch")
                return fields
            """
        )

        if difficulty == "medium":
            buggy = dedent(
                f"""
                from __future__ import annotations


                _ESCAPE_MAP = {{
                    "\\\\": "\\\\\\\\",
                    "|": "\\\\|",
                    "#": "\\\\#",
                    "\\n": "\\\\n",
                    "\\t": "\\\\t",
                }}


                def {encode_name}(fields: list[str]) -> str:
                    encoded_fields: list[str] = []
                    for field in fields:
                        buffer: list[str] = []
                        for character in field:
                            buffer.append(_ESCAPE_MAP.get(character, character))
                        encoded_fields.append("".join(buffer))
                    return f"{{len(fields)}}#" + "|".join(encoded_fields)


                def {decode_name}(payload: str) -> list[str]:
                    raw_count, separator, body = payload.partition("#")
                    if separator != "#" or not raw_count.isdigit():
                        raise ValueError("missing field count prefix")
                    if not body:
                        return []
                    return body.split("|")
                """
            )
            bug_types = ["decoder ignores escaping rules and uses naive string splitting"]
            strategy_traps = [
                "Round-trip issues appear in delimiter-containing fields, not just in the count prefix",
                "A patch that only changes encode() still fails because decode() is structurally wrong",
            ]
        else:
            buggy = reference.replace(
                '                if escaped:\n                    raise ValueError("dangling escape")\n                fields.append("".join(current))\n                if len(fields) != expected_count:\n                    raise ValueError("field count mismatch")\n',
                '                if escaped:\n                    current.append("\\\\")\n                fields.append("".join(current))\n',
            )
            bug_types = ["decoder silently accepts malformed trailing escapes and count drift"]
            strategy_traps = [
                "Most happy-path round trips pass even when malformed payload handling is wrong",
                "Hidden edge cases check decoder validation, not just ordinary escape sequences",
            ]

        visible_tests = render_test_module(
            dedent(
                f"""
                from codec import {decode_name}, {encode_name}


                class CodecVisibleTests(unittest.TestCase):
                    def test_round_trip_preserves_delimiters_and_backslashes(self) -> None:
                        fields = ["alpha|beta", "path\\\\file", "literal#tag"]
                        encoded = {encode_name}(fields)
                        self.assertEqual({decode_name}(encoded), fields)

                    def test_round_trip_preserves_empty_fields(self) -> None:
                        fields = ["", "middle", ""]
                        encoded = {encode_name}(fields)
                        self.assertEqual({decode_name}(encoded), fields)
                """
            )
        )

        hidden_tests = render_test_module(
            dedent(
                f"""
                from codec import {decode_name}, {encode_name}


                class CodecHiddenTests(unittest.TestCase):
                    def test_empty_list_round_trip(self) -> None:
                        self.assertEqual({decode_name}({encode_name}([])), [])

                    def test_unicode_like_payload_and_control_escapes(self) -> None:
                        fields = ["snowman: ☃", "line\\nbreak", "tab\\tstop"]
                        encoded = {encode_name}(fields)
                        self.assertEqual({decode_name}(encoded), fields)

                    def test_dangling_escape_is_rejected(self) -> None:
                        with self.assertRaises(ValueError):
                            {decode_name}("1#abc\\\\")

                    def test_count_mismatch_is_rejected(self) -> None:
                        with self.assertRaises(ValueError):
                            {decode_name}("2#only-one")
                """
            )
        )

        return self.build_spec(
            seed=seed,
            difficulty=difficulty,
            prompt=prompt,
            files={
                "codec.py": buggy,
                "test_visible.py": visible_tests,
                "test_hidden.py": hidden_tests,
            },
            reference_files={"codec.py": reference},
            entrypoint="codec.py",
            visible_test_file="test_visible.py",
            hidden_test_file="test_hidden.py",
            task_id=make_task_id(self.family, seed),
            metadata={
                "bug_type": bug_types,
                "strategy_traps": strategy_traps,
                "target_files": ["codec.py"],
                "expected_skill_tags": ["parsing", "escaping", "roundtrip", "state-machines"],
                "niche_topic": "self-describing escaped field codec",
                "repairable": True,
            },
        )
