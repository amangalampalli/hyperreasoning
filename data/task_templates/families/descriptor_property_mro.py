"""Descriptor/property/MRO task family."""

from __future__ import annotations

from data.task_templates.base import TaskSpec, TaskTemplate
from data.task_templates.utils import build_rng, choose_variant, dedent, make_task_id, render_test_module


class DescriptorPropertyMROTemplate(TaskTemplate):
    """Generate descriptor precedence and MRO repair tasks."""

    family = "descriptor_property_mro"

    def generate_instance(self, seed: int, difficulty: str) -> TaskSpec:
        self._validate_difficulty(difficulty)
        rng = build_rng(seed, self.family, difficulty)
        descriptor_name = choose_variant(rng, ["ManagedAttribute", "StoredField", "TrackedAttribute"])
        prompt = choose_variant(
            rng,
            [
                f"Repair the descriptor utilities in `models.py`. `{descriptor_name}` instances declare managed "
                "fields that should be discovered across inheritance using normal Python MRO rules. Subclass "
                "overrides with properties or plain attributes must shadow inherited descriptors instead of "
                "leaking base fields back into exported state.",
                f"`models.py` contains a small descriptor-backed model layer. Fix managed field discovery and "
                "export so data descriptors, mixin order, and overriding properties behave like real Python "
                "attribute lookup.",
            ],
        )
        reference = dedent(
            f"""
            from __future__ import annotations


            class {descriptor_name}:
                def __init__(self, default: object) -> None:
                    self.default = default
                    self.storage_name = ""

                def __set_name__(self, owner: type, name: str) -> None:
                    self.storage_name = f"_managed_{{name}}"

                def __get__(self, instance: object, owner: type | None = None) -> object:
                    if instance is None:
                        return self
                    return getattr(instance, self.storage_name, self.default)

                def __set__(self, instance: object, value: object) -> None:
                    setattr(instance, self.storage_name, value)


            def collect_managed_attributes(cls: type) -> dict[str, {descriptor_name}]:
                collected: dict[str, {descriptor_name}] = {{}}
                shadowed: set[str] = set()
                for base in cls.__mro__:
                    for name, value in base.__dict__.items():
                        if name in collected or name in shadowed:
                            continue
                        if isinstance(value, {descriptor_name}):
                            collected[name] = value
                        else:
                            shadowed.add(name)
                return collected


            def export_managed_state(instance: object) -> dict[str, object]:
                cls = type(instance)
                return {{
                    name: getattr(instance, name)
                    for name in collect_managed_attributes(cls)
                }}
            """
        )

        if difficulty == "medium":
            buggy = reference.replace("for base in cls.__mro__:", "for base in reversed(cls.__mro__):")
            bug_types = ["managed attributes are collected in reverse MRO order"]
            strategy_traps = [
                "Subclass fields must win over base-class fields with the same name",
                "The bug is in discovery order, not in the descriptor __get__ implementation",
            ]
        else:
            buggy = reference.replace(
                "                for name, value in base.__dict__.items():\n                    if name in collected or name in shadowed:\n                        continue\n                    if isinstance(value, ManagedAttribute):\n                        collected[name] = value\n                    else:\n                        shadowed.add(name)\n",
                "                for name, value in base.__dict__.items():\n                    if name in collected:\n                        continue\n                    if isinstance(value, ManagedAttribute):\n                        collected[name] = value\n",
            ).replace("ManagedAttribute", descriptor_name)
            bug_types = ["properties and plain attributes do not shadow inherited descriptors during collection"]
            strategy_traps = [
                "Descriptor discovery must match real attribute lookup, including non-descriptor shadowing",
                "Patching export only hides the field list bug instead of fixing collection",
            ]

        visible_tests = render_test_module(
            dedent(
                f"""
                from models import {descriptor_name}, collect_managed_attributes, export_managed_state


                class Base:
                    name = {descriptor_name}("base")


                class Child(Base):
                    name = {descriptor_name}("child")
                    age = {descriptor_name}(10)


                class DescriptorVisibleTests(unittest.TestCase):
                    def test_subclass_descriptor_wins_for_same_name(self) -> None:
                        instance = Child()
                        self.assertEqual(collect_managed_attributes(Child)["name"].default, "child")
                        self.assertEqual(export_managed_state(instance), {{"name": "child", "age": 10}})

                    def test_data_descriptor_writes_to_instance_state(self) -> None:
                        instance = Child()
                        instance.age = 99
                        self.assertEqual(instance.age, 99)
                        self.assertEqual(export_managed_state(instance)["age"], 99)
                """
            )
        )

        hidden_tests = render_test_module(
            dedent(
                f"""
                from models import {descriptor_name}, collect_managed_attributes, export_managed_state


                class LeftMixin:
                    token = {descriptor_name}("left")


                class RightMixin:
                    token = {descriptor_name}("right")


                class Combined(LeftMixin, RightMixin):
                    pass


                class BaseWithDescriptor:
                    status = {descriptor_name}("pending")


                class PropertyOverride(BaseWithDescriptor):
                    @property
                    def status(self) -> str:
                        return "property-value"


                class DescriptorHiddenTests(unittest.TestCase):
                    def test_leftmost_mixin_wins(self) -> None:
                        self.assertEqual(collect_managed_attributes(Combined)["token"].default, "left")

                    def test_property_override_shadows_descriptor(self) -> None:
                        instance = PropertyOverride()
                        self.assertEqual(instance.status, "property-value")
                        self.assertNotIn("status", collect_managed_attributes(PropertyOverride))
                        self.assertEqual(export_managed_state(instance), {{}})
                """
            )
        )

        return self.build_spec(
            seed=seed,
            difficulty=difficulty,
            prompt=prompt,
            files={
                "models.py": buggy,
                "test_visible.py": visible_tests,
                "test_hidden.py": hidden_tests,
            },
            reference_files={"models.py": reference},
            entrypoint="models.py",
            visible_test_file="test_visible.py",
            hidden_test_file="test_hidden.py",
            task_id=make_task_id(self.family, seed),
            metadata={
                "bug_type": bug_types,
                "strategy_traps": strategy_traps,
                "target_files": ["models.py"],
                "expected_skill_tags": ["descriptors", "mro", "attribute-lookup", "inheritance"],
                "niche_topic": "descriptor-backed model discovery",
                "repairable": True,
            },
        )
