"""Dependency-light checks for batching boundaries and token accounting."""
from types import SimpleNamespace

import pytest

from home_observer.model import ModelConfig, encode_batch_messages, trim_generation_padding


class Tensor:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def sum(self, dim):
        return SimpleNamespace(tolist=lambda: [sum(row) for row in self.rows])


class Processor:
    def __init__(self, fail=False):
        self.tokenizer = SimpleNamespace(padding_side="right")
        self.fail = fail

    def apply_chat_template(self, conversations, **kwargs):
        assert self.tokenizer.padding_side == "left"
        assert kwargs["processor_kwargs"]["padding"] is True
        assert kwargs["add_generation_prompt"] is True
        if self.fail:
            raise ValueError("processor failure")
        return {"input_ids": Tensor([[0, 1, 2], [3, 4, 5]]),
                "attention_mask": Tensor([[0, 1, 1], [1, 1, 1]])}


def test_batch_left_padding_and_setting_restore():
    processor = Processor()
    encoded = encode_batch_messages(processor, [[], []], ModelConfig())
    assert encoded["attention_mask"].sum(-1).tolist() == [2, 3]
    assert processor.tokenizer.padding_side == "right"


def test_batch_padding_restored_after_failure():
    processor = Processor(fail=True)
    with pytest.raises(ValueError, match="processor failure"):
        encode_batch_messages(processor, [[], []], ModelConfig())
    assert processor.tokenizer.padding_side == "right"


def test_batch_bounds_reject_without_truncating():
    with pytest.raises(ValueError, match="1..4"):
        encode_batch_messages(Processor(), [[]] * 5, ModelConfig())
    with pytest.raises(ValueError, match="token budget"):
        encode_batch_messages(Processor(), [[], []], ModelConfig(max_input_tokens=2))


def test_completion_token_count_excludes_only_trailing_padding():
    assert trim_generation_padding([5, 6, 7, 0, 0], 0) == [5, 6, 7]
    assert trim_generation_padding([5, 0, 7, 0], 0) == [5, 0, 7]
    assert trim_generation_padding([5, 6, 7], None) == [5, 6, 7]


def test_adapter_merge_defaults_and_quantization_guard():
    assert ModelConfig(quantization="none").should_merge_adapter is False
    assert ModelConfig(quantization="none", merge_adapter=None).should_merge_adapter is False
    assert ModelConfig(quantization="none", merge_adapter=True).should_merge_adapter is True
    assert ModelConfig(quantization="none", merge_adapter=False).should_merge_adapter is False
    assert ModelConfig(quantization="4bit").should_merge_adapter is False
    with pytest.raises(ValueError, match="unquantized BF16"):
        ModelConfig(quantization="4bit", merge_adapter=True)
    # Existing callers may switch a reusable config before loading the adapter.
    config = ModelConfig(quantization="none")
    config.quantization = "4bit"
    assert config.should_merge_adapter is False
