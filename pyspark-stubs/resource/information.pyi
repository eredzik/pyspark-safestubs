class ResourceInformation:
    def __init__(self, name: str, addresses: list[str]) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def addresses(self) -> list[str]: ...