__all__ = ['SparkFiles']

class SparkFiles:
    def __init__(self) -> None: ...
    @classmethod
    def get(cls, filename: str) -> str: ...
    @classmethod
    def getRootDirectory(cls) -> str: ...
