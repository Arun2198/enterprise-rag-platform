from abc import ABC
from abc import abstractmethod

from ingestion.contracts.document import Document
from ingestion.contracts.result import Result


class BaseParser(ABC):

    @abstractmethod
    def parse(
        self,
        file_path: str
    ) -> Result[Document]:
        pass