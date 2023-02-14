import pandas as pd

from exceptions import FileTypeNotSupported, TargetNotFound

SUPPORTED_FILE_TYPES = ["csv", "parquet"]
DATA_SOURCES = {
    "csv": pd.read_csv,
    "parquet": pd.read_parquet,
}


class DataProvider:
    def __init__(self, data_source: str, target: str) -> None:
        """Класс для валидации и получения файла данных.

        @param data_source[str]: Путь до файла с данными.
        @param target[str]: Целевая перменная.
        """
        self.data_source = data_source
        self.target = target

        # Валидация расширени файла
        filetype = self.data_source.split(".")[-1]
        if filetype not in SUPPORTED_FILE_TYPES:
            raise FileTypeNotSupported(filetype, SUPPORTED_FILE_TYPES)

        # Валидация самого файла
        self.df = DATA_SOURCES[filetype](data_source)
        df_columns = list(self.df.columns)

        # Проверка таргета в файле
        if self.target not in df_columns:
            raise TargetNotFound(self.target, df_columns)

    def get_data(self) -> pd.DataFrame:
        """Возвращает считанный в pd.DataFrame файл с данными."""
        return self.df
