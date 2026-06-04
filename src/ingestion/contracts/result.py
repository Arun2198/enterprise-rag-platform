from typing import Generic, TypeVar

from pydantic import BaseModel


T = TypeVar("T")


class Error(BaseModel):
    code: str
    message: str


class Result(BaseModel, Generic[T]):
    success: bool

    data: T | None = None

    error: Error | None = None