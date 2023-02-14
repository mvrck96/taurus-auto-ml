from typing import List

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

from exceptions import ModelNotFoud
from preprocessor import Preprocessor
from data_provider import DataProvider


IMPLEMENTED_MODELS = {
    "SVC": SVC,
    "LR": LogisticRegression,
    "DT": DecisionTreeClassifier,
}


class AutoClassifier:
    def __init__(
        self,
        data_source: str,
        target: str,
        models: List[str] = ["all"],
        ignore_columns: List[str] = None,
    ) -> None:
        """Класс для решения задачи классификации а авто режиме.

        @param data_source[str]: Источник данных
        @param target[str]: Целевой параметр
        @param  models[List[str]]: Список используемых моеделей
                └─> default: ["all"]
        """
        self.data_source = data_source
        self.target = target
        self.ignore_columns = ignore_columns
        self.models = models

        self.data_provider = DataProvider(data_source, target)

        # Валидация списка моделей
        if len(self.models) == 1 and self.models[0] == "all":
            self.models = IMPLEMENTED_MODELS.keys()
        else:
            missing_models = set(self.models) - set(IMPLEMENTED_MODELS.keys())
            if missing_models:
                raise ModelNotFoud(missing_models, IMPLEMENTED_MODELS)

    def preprocess(self, ignore_columns: List[str] = None) -> pd.DataFrame:
        """Обработка данных для дальнейшего использования."""
        data = self.data_provider.get_data()
        self.preprocessor = Preprocessor(
            df=data, target=self.target, ignore_columns=ignore_columns
        )
        return self.preprocessor.run()  # Возвращает обработанный датафрейм

    def fit(self):
        """Будет отдавать готовую модель с лучшим скором."""
        fitted_models = {key: (None, None) for key in IMPLEMENTED_MODELS.keys()}

        data = self.preprocess(self.ignore_columns)
        X, y = data.drop(columns=[self.target]), data[self.target]

        # Луп обучения моделей и оценки качества CV
        for model in self.models:
            cls = IMPLEMENTED_MODELS[model]()
            fitted_cls = cls.fit(X, y)
            score = np.mean(
                cross_val_score(cls, X, y, cv=3, scoring="f1", n_jobs=-1)
            )
            fitted_models[model] = (fitted_cls, score)

        # Отдаем лучшую модель по скору на CV
        top_pick = sorted(fitted_models.items(), key=lambda x: x[1][1])[-1]
        return top_pick[1]
