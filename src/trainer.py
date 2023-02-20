from src.model import BaseModel
from typing import Any
from pathlib import Path
import joblib


class Trainer:
    def __init__(
        self, model_config: BaseModel, eval_config: Any, hyperparam_space: Any
    ) -> None:
        self.model_config = model_config
        self.eval_config = eval_config
        self.hyperparam_space = hyperparam_space
        self.results = "results"

    def __repr__(self) -> str:
        return f"""
        {self.__class__.__name__} (model_config={self.model_config}, eval_config={self.eval_config}, hyperparam_space={self.hyperparam_space})
        """

    def evaluate(self):
        pass

    def export_best_model(self, dir_path: Path, name: str) -> None:
        """
        Exports a trained model to the given directory.

        Parameters
        ----------
        model : The trained model to be exported
        dir_path : Target directory
        name : File name to save the model as
        """

        path = f"{dir_path}/{name}"
        joblib.dump(self.model_config, path)

        ### To change
        print(f"Model saved to 'path'")

    def predict(self):
        pass


if __name__ == "__main__":
    train = Trainer(
        model_config=BaseModel(model_type="model"),
        eval_config="Eval Config",
        hyperparam_space="hyperparam",
    )
    print(repr(train))
