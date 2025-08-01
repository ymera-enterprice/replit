"""
YMERA Authentication Routes
User authentication and authorization endpoints
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database.connection import get_db, User
from config.settings import get_settings

logger = logging.getLogger(__name__)
security = HTTPBearer()
settings = get_settings()

router = APIRouter()

# Pydantic models
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    is_active: bool
    is_admin: bool
    created_at: datetime

# Password hashing (simplified for Phase 1-2)
import hashlib

def hash_password(password: str) -> str:
    """Simple password hashing for Phase 1-2"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    return hash_password(password) == hashed

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """Register a new user"""
    
    # Check if user exists
    result = await db.execute(
        select(User).where(
            (User.username == user_data.username) | 
            (User.email == user_data.email)
        )
    )
    existing_user = result.scalar_one_or_none()
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already registered"
        )
    
    # Create new user
    hashed_password = hash_password(user_data.password)
    new_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password,
        is_active=True,
        is_admin=False
    )
    
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    
    logger.info(f"New user registered: {user_data.username}")
    
    return UserResponse(
        id=new_user.id,
        username=new_user.username,
        email=new_user.email,
        is_active=new_user.is_active,
        is_admin=new_user.is_admin,
        created_at=new_user.created_at
    )

@router.post("/login", response_model=Token)
async def login_user(
    user_data: UserLogin,
    db: AsyncSession = Depends(get_db)
):
    """Authenticate user and return token"""
    
    # Find user
    result = await db.execute(
        select(User).where(User.username == user_data.username)
    )
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(user_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is inactive"
        )
    
    # Generate token (simplified for Phase 1-2)
    token_data = f"{user.id}:{user.username}:{datetime.utcnow().isoformat()}"
    access_token = hashlib.sha256(token_data.encode()).hexdigest()
    
    logger.info(f"User logged in: {user_data.username}")
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.security.access_token_expire_minutes * 60
    )

@router.get("/me", response_model=UserResponse)
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """Get current user information"""
    
    # For Phase 1-2, we'll implement a simplified token validation
    # In production, this would decode JWT tokens properly
    
    # For now, return a default user for development
    # This will be enhanced in Phase 3
    
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="User profile endpoint will be implemented in Phase 3"
    )

@router.post("/logout")
async def logout_user():
    """Logout user (token invalidation)"""
    
    # For Phase 1-2, logout is client-side token removal
    # Token blacklisting will be implemented in Phase 3
    
    return {"message": "Successfully logged out"}

@router.get("/status")
async def auth_status():
    """Authentication system status"""
    return {
        "status": "operational",
        "features": {
            "registration": "active",
            "login": "active",
            "token_auth": "basic",
            "user_profile": "phase_3",
            "jwt_tokens": "phase_3"
        },
        "phase": "1-2"
    }