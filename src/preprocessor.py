from typing import List

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from exceptions import FeaturesNotFound


class Preprocessor:
    def __init__(
        self, df: pd.DataFrame, target: str, ignore_columns: List[str] = None
    ) -> None:
        """Класс отвечающий за препроцессинг данных.

        @param df[pd.DataFrame]: Оригинаьный датафрейм.
        @param target[str]: Целевая переменная.
        @param  ignore_columns[List[str]]: Параметры, которые не
                                            будут обрабатываться.
                └─> default: None
        """
        self.df = df  # Оригинальный датафрейм

        self.prep_df = df.copy()
        self.ignore_columns = ignore_columns
        self.target = target

        # Убираем таргет
        self.prep_df.drop(columns=[target], inplace=True)

        # Проверка фичей, которые надо проигнорить
        df_columns = list(df.columns)
        if ignore_columns:
            missing_features = set(self.ignore_columns) - set(df_columns)
            if missing_features:
                raise FeaturesNotFound(missing_features, df_columns)
            else:
                self.prep_df.drop(columns=ignore_columns, inplace=True)

        self.categorical_features = self.prep_df.select_dtypes(
            ["object", "bool", "category"]
        )
        self.numerical_features = self.prep_df.select_dtypes(
            ["int64", "float64"]
        )

    def fillna(self):
        """Заполнение пропускрвв данных, согласно типу данных."""
        cat_cols_with_na = self.categorical_features.columns[
            self.categorical_features.isna().any()
        ]
        num_cols_with_na = self.numerical_features.columns[
            self.numerical_features.isna().any()
        ]

        if list(cat_cols_with_na):
            for col in cat_cols_with_na:
                most_common = self.prep_df[col].value_counts().keys()[0]
                self.prep_df[col].fillna(most_common, inplace=True)

        if list(num_cols_with_na):
            for col in num_cols_with_na:
                mean = self.prep_df[col].mean()
                self.prep_df[col].fillna(mean, inplace=True)

    def encode(self):
        """Кодирование категориальных признаков."""
        encoder = LabelEncoder()
        res = pd.DataFrame()
        for col in self.categorical_features.columns:
            enc_feature = encoder.fit_transform(
                self.categorical_features[col].astype(str)
            )
            res[col] = list(map(int, enc_feature))

        # Удаляем из рабочего дф закодированные фичи
        self.prep_df.drop(
            columns=self.categorical_features.columns, inplace=True
        )
        # Добавляем обратно обработанные
        self.prep_df = self.prep_df.join(res).copy()

    def scale(self):
        """Скейлинг числовых параметров."""
        ss = StandardScaler()
        res = ss.fit_transform(self.prep_df)

        # Скейлим весь рабочий дф, остались только численные
        self.prep_df = pd.DataFrame(
            res,
            index=self.prep_df.index,
            columns=self.prep_df.columns,
        )

    def run(self) -> pd.DataFrame:
        """Запуск всего пайплайна обработки."""
        self.fillna()
        self.encode()
        self.scale()
        if self.ignore_columns:
            self.prep_df = self.prep_df.join(self.df[[*self.ignore_columns]])

        # Кодируем таргет после всех обработок
        self.prep_df = self.prep_df.join(self.df[self.target])
        encoder = LabelEncoder()
        self.prep_df[self.target] = encoder.fit_transform(
            self.prep_df[self.target]
        )
        return self.prep_df
