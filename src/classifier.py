from typing import List

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import f1_score

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

        data = self.preprocess()
        X, y = data.drop(columns=[self.target]), data[self.target]
        for model in self.models:
            print(f"Fitting {model}")
            cls = IMPLEMENTED_MODELS[model]()
            fitted_cls = cls.fit(X, y)
            score = np.mean(
                cross_val_score(cls, X, y, cv=3, scoring="f1", n_jobs=-1)
            )
            # X_train, X_test, y_train, y_test = train_test_split(
            #     X, y, test_size=0.3, shuffle=True
            # )
            # fitted_cls = cls.fit(X_train, y_train)
            # pred = fitted_cls.predict(X_test)
            # score = f1_score(y_test, pred)
            fitted_models[model] = (fitted_cls, score)
        return fitted_models
