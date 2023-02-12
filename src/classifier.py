from typing import List

import pandas as pd

from exceptions import ModelNotFoud
from preprocessor import Preprocessor
from data_provider import DataProvider

IMPLEMENTED_MODELS = ["SVR", "LR", "RF"]


class AutoClassifier:
    def __init__(
        self, data_source: str, target: str, models: List[str] = ["all"]
    ) -> None:
        """Класс для решения задачи классификации а авто режиме.

        @param data_source[str]: Источник данных
        @param target[str]: Целевой параметр
        @param  models[List[str]]: Список используемых моеделей
                └─> default: ["all"]
        """
        self.data_source = data_source
        self.target = target
        self.models = models

        self.data_provider = DataProvider(data_source, target)

        # Validate models
        if len(self.models) == 1 and self.models[0] == "all":
            self.models = IMPLEMENTED_MODELS
        else:
            missing_models = set(self.models) - set(IMPLEMENTED_MODELS)
            if missing_models:
                raise ModelNotFoud(missing_models, IMPLEMENTED_MODELS)

    def preprocess(self, ignore_columns: List[str]):
        """Обработка данных для дальнейшего использования."""
        data = self.data_provider.get_data()
        self.preprocessor = Preprocessor(
            df=data, target=self.target, ignore_columns=ignore_columns
        )
        self.preprocessor.run()

    def fit_predict():
        """Будет отдавать готовую модель с лучшим скором."""
        pass
