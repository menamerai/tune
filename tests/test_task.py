import pytest

from tune.tasks.base import BaseTask

def test_base_task():
    with pytest.raises(NotImplementedError):
        BaseTask()
    
    with pytest.raises(NotImplementedError):
        BaseTask().get_train_data()

    with pytest.raises(NotImplementedError):
        BaseTask().get_prompt()
