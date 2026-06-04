from pydantic import BaseModel


class ParserError(BaseModel):
    code: str
    message: str
    source: str | None = None