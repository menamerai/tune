import pytest

from tune.prompt import BasePrompt
from tune.prompt.winogrande import (
    WinograndePrompt,
    winogrande_instructions_factory,
    winogrande_example_factory,
)


def test_base_prompt():
    with pytest.raises(ValueError):
        BasePrompt(1)

    assert str(BasePrompt("Hello")) == "Hello"
    assert repr(BasePrompt("Hello")) == "Hello"


def test_winogrande_prompt():
    with pytest.raises(ValueError):
        WinograndePrompt(instructions=1, examples=1, stop_words=1)

    assert (
        str(WinograndePrompt(examples=winogrande_example_factory(), stop_words=["."]))
        == winogrande_instructions_factory()
    )
