from typing import List


class TargetNotFound(Exception):
    """Ошибка для отсутствия целевого параметра в файле данных."""

    def __init__(self, target: str, columns: List[str]) -> None:
        """Инициализация ошибки."""
        self.message = (
            f"Column {target} not found in data source columns: {columns}"
        )
        super().__init__(self.message)


class ModelNotFoud(Exception):
    """Ошибка для отсутствия выбранной модели."""

    def __init__(self, model: str, implemented_models: List[str]) -> None:
        """Инициализация ошибки."""
        self.message = (
            f"Model(s) {model} not found in implemented models !"
            f" Please use one of: {implemented_models}"
        )
        super().__init__(self.message)


class FileTypeNotSupported(Exception):
    """Ошибка для неподдерживаемого файла."""

    def __init__(self, filetype: str, supported_filetypes: List[str]) -> None:
        """Инициализация ошибки."""
        self.message = (
            f"Filetype {filetype} not supported !"
            f" Please use on from: {supported_filetypes}"
        )
        super().__init__(self.message)


class FeaturesNotFound(Exception):
    """Ошибка для отсутствия выбранных парамтров в файле данных."""

    def __init__(self, features: List[str], columns: List[str]) -> None:
        """Инициализация ошибки."""
        self.message = (
            f"Feature(s) {features} not found in dataframe: {columns}"
        )
        super().__init__(self.message)
