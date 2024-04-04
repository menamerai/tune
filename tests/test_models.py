import pytest

from tune.models import BaseModel


def test_base_model():
    with pytest.raises(NotImplementedError):
        BaseModel()

    with pytest.raises(NotImplementedError):
        BaseModel().predict()

    with pytest.raises(NotImplementedError):
        BaseModel().train()

    with pytest.raises(NotImplementedError):
        BaseModel().save()
