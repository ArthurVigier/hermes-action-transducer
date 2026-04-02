from __future__ import annotations

from types import SimpleNamespace

from hermes_action_transducer.encoder import HermesHFConfig, HermesHFEncoder, SimpleHermesEncoder
from hermes_action_transducer.models import RobotObservation
from hermes_action_transducer.profiles import get_profile


def test_simple_encoder_returns_vectors():
    encoder = SimpleHermesEncoder()
    state = encoder.encode(RobotObservation(task="Pick up the mug"), get_profile("arm"))
    assert len(state.thought_vector) == 8
    assert len(state.intent_vector) == 8
    assert len(state.hidden_projection) >= 16
    assert "synthetic_input" in state.layer_projections


def test_hermes_hf_encoder_with_fake_backend():
    encoder = HermesHFEncoder(
        HermesHFConfig(
            model_id="fake/hermes",
            rich_projection_dim=24,
            layer_projection_dim=12,
            additional_layer_indices=(-2,),
        )
    )

    class FakeTensor:
        def __init__(self, data):
            self.data = data
            self.shape = (1, len(data[0]))

        def to(self, device):
            return self

        def sum(self):
            return SimpleNamespace(item=lambda: len(self.data[0]))

        def unsqueeze(self, dim):
            return self

    class FakeSlice:
        def __init__(self, values):
            self.values = values

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self.values

        def mean(self, dim=0):
            cols = len(self.values[0])
            out = []
            for idx in range(cols):
                out.append(sum(row[idx] for row in self.values) / len(self.values))
            return FakeSlice(out)

        def __getitem__(self, idx):
            if isinstance(idx, int):
                row = self.values[idx]
                if isinstance(row, list):
                    return FakeSlice(row)
            raise TypeError("unsupported index")

    class FakeHidden:
        def __init__(self):
            self.batch = [
                [0.1] * 16,
                [0.2] * 16,
                [0.3] * 16,
            ]

        def __getitem__(self, idx):
            if idx == 0:
                return FakeSlice(self.batch)
            raise IndexError(idx)

    class FakeTokenizer:
        def __call__(self, prompt, return_tensors, truncation, max_length):
            _ = (prompt, return_tensors, truncation, max_length)
            return {
                "input_ids": FakeTensor([[1, 2, 3]]),
                "attention_mask": FakeTensor([[1, 1, 1]]),
            }

    class FakeModel:
        device = "cpu"

        def eval(self):
            return self

        def __call__(self, **kwargs):
            _ = kwargs
            return SimpleNamespace(hidden_states=[FakeHidden(), FakeHidden()])

    encoder._tokenizer = FakeTokenizer()
    encoder._model = FakeModel()
    state = encoder.encode(
        RobotObservation(task="Pick up the mug", state_text="mug is left"),
        get_profile("arm"),
    )
    assert state.metadata["encoder_backend"] == "hermes_hf"
    assert state.metadata["model_id"] == "fake/hermes"
    assert len(state.thought_vector) == 8
    assert len(state.intent_vector) == 8
    assert len(state.hidden_projection) == 24
    assert "layer_-1" in state.layer_projections
    assert "layer_-2" in state.layer_projections
    assert len(state.layer_projections["layer_-1"]) == 12
