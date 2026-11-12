from datetime import datetime
from typing import Annotated, Self
from uuid import UUID

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


def reject_nul(value: str) -> str:
    if "\x00" in value:
        raise ValueError("NUL characters are not allowed")
    return value


Title = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=200),
    AfterValidator(reject_nul),
]
Description = Annotated[str, StringConstraints(max_length=5000), AfterValidator(reject_nul)]
Password = Annotated[str, StringConstraints(min_length=12, max_length=128)]


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)


class RegisterRequest(StrictSchema):
    email: EmailStr
    password: Password = Field(repr=False)

    @field_validator("email", mode="before")
    @classmethod
    def strip_email(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        normalized = value.lower()
        if len(normalized) > 254:
            raise ValueError("Email must not exceed 254 characters")
        return normalized


class UserResponse(StrictSchema):
    id: UUID
    email: EmailStr
    created_at: datetime


class TokenResponse(StrictSchema):
    access_token: str = Field(repr=False)
    token_type: str = "bearer"
    expires_in: int


class ItemCreate(StrictSchema):
    title: Title
    description: Description = ""


class ItemPatch(StrictSchema):
    title: Title | None = None
    description: Description | None = None

    @field_validator("title", "description", mode="before")
    @classmethod
    def reject_null(cls, value: object) -> object:
        if value is None:
            raise ValueError("Field must not be null")
        return value

    @model_validator(mode="after")
    def require_fields(self) -> Self:
        if not self.model_fields_set:
            raise ValueError("At least one field must be supplied")
        return self


class ItemResponse(StrictSchema):
    id: UUID
    title: Title
    description: Description
    created_at: datetime
    updated_at: datetime


class ItemPage(StrictSchema):
    items: list[ItemResponse]
    next_cursor: str | None
