"""Registry semantics: naming, duplicates, freeze."""
from __future__ import annotations

import pytest

from agent_oo.core.capability import Capability, CapabilitySpec
from agent_oo.core.registry import CapabilityRegistry


class StubCap(Capability):
    def __init__(self, name: str = "stub"):
        self.spec = CapabilitySpec(name=name, description=f"{name} does stub things")

    def build(self):
        raise NotImplementedError


def test_register_and_lookup():
    reg = CapabilityRegistry([StubCap("alpha"), StubCap("beta")])
    assert reg.names() == ["alpha", "beta"]
    assert reg.get("alpha").spec.description == "alpha does stub things"
    assert "beta" in reg and "gamma" not in reg
    assert len(reg) == 2
    assert [s.name for s in reg.specs()] == ["alpha", "beta"]


def test_duplicate_name_rejected():
    reg = CapabilityRegistry([StubCap("alpha")])
    with pytest.raises(ValueError, match="already registered"):
        reg.register(StubCap("alpha"))


@pytest.mark.parametrize("bad", ["Alpha", "1abc", "with-dash", "", "with space"])
def test_invalid_name_rejected(bad):
    with pytest.raises(ValueError, match="invalid capability name"):
        CapabilitySpec(name=bad, description="x")


def test_freeze_blocks_registration():
    reg = CapabilityRegistry([StubCap("alpha")])
    reg.freeze()
    with pytest.raises(RuntimeError, match="frozen"):
        reg.register(StubCap("beta"))


def test_unknown_lookup_message():
    reg = CapabilityRegistry([StubCap("alpha")])
    with pytest.raises(KeyError, match="registered: \\['alpha'\\]"):
        reg.get("nope")
