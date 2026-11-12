"""Account registration and OAuth2 password-form login."""

from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from sqlalchemy import insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.db import get_session
from app.models import User
from app.schemas import RegisterRequest, TokenResponse, UserResponse
from app.security import PasswordService, create_access_token

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])
Session = Annotated[AsyncSession, Depends(get_session)]


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(payload: RegisterRequest, request: Request, session: Session) -> UserResponse:
    passwords: PasswordService = request.app.state.passwords
    # Hashing happens before the first statement acquires a DB connection.
    password_hash = await passwords.hash(payload.password)
    try:
        result = await session.execute(
            insert(User)
            .values(email=str(payload.email), password_hash=password_hash)
            .returning(User.id, User.email, User.created_at)
        )
        user = result.one()
        await session.commit()
    except IntegrityError as error:
        await session.rollback()
        original = error.orig
        constraint_error = original.__cause__ or original
        if (
            getattr(original, "sqlstate", None) == "23505"
            and getattr(constraint_error, "constraint_name", None) == "uq_users_email"
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already registered",
            ) from error
        raise
    return UserResponse(id=user.id, email=user.email, created_at=user.created_at)


@router.post("/login", response_model=TokenResponse)
async def login(
    request: Request,
    session: Session,
    username: Annotated[str, Form()],
    password: Annotated[str, Form(min_length=12, max_length=128)],
) -> TokenResponse:
    # Reuse registration's exact email normalization and password bounds before
    # any hashing. The global request-body limit also bounds raw form parsing.
    try:
        credentials = RegisterRequest(email=username, password=password)
    except ValidationError as error:
        errors = []
        for entry in error.errors(include_input=False, include_context=False, include_url=False):
            field = "username" if entry["loc"] == ("email",) else "password"
            errors.append({**entry, "loc": ("body", field)})
        raise RequestValidationError(errors) from error

    result = await session.execute(
        select(User.id, User.password_hash).where(User.email == str(credentials.email))
    )
    user = result.one_or_none()
    # Neither the real nor the dummy verification holds an open transaction.
    await session.rollback()

    passwords: PasswordService = request.app.state.passwords
    password_hash = user.password_hash if user is not None else passwords.dummy_hash
    verified = await passwords.verify(credentials.password, password_hash)
    if not verified or user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    settings: Settings = request.app.state.settings
    return TokenResponse(
        access_token=create_access_token(user.id, settings),
        token_type="bearer",
        expires_in=settings.auth_token_ttl_seconds,
    )
