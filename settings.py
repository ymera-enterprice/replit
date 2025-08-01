"""
YMERA Enterprise Platform - Configuration Settings
Centralized configuration management for Phase 1-2
"""

import os
from typing import List, Optional
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    url: str
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False


@dataclass
class SecurityConfig:
    """Security configuration settings"""
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    allowed_hosts: List[str] = None


@dataclass
class RedisConfig:
    """Redis configuration settings"""
    url: str
    max_connections: int = 100
    decode_responses: bool = True


@dataclass
class YMERASettings:
    """Main YMERA application settings"""
    
    # Environment
    environment: str
    debug: bool
    
    # Server
    host: str
    port: int
    
    # Database
    database: DatabaseConfig
    
    # Security
    security: SecurityConfig
    
    # Redis
    redis: RedisConfig
    
    # CORS
    cors_origins: List[str]
    
    # File upload
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    upload_directory: str = "uploads"


@lru_cache()
def get_settings() -> YMERASettings:
    """Get application settings (cached)"""
    
    # Database configuration
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        # Default for Replit
        database_url = "postgresql://user:password@localhost:5432/ymera"
    
    database_config = DatabaseConfig(
        url=database_url,
        pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
        max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
        echo=os.getenv("DB_ECHO", "false").lower() == "true"
    )
    
    # Security configuration
    secret_key = os.getenv("SECRET_KEY")
    if not secret_key:
        import hashlib
        secret_key = hashlib.sha256(os.urandom(32)).hexdigest()
    
    security_config = SecurityConfig(
        secret_key=secret_key,
        algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
        access_token_expire_minutes=int(os.getenv("ACCESS_TOKEN_EXPIRE", "30")),
        refresh_token_expire_days=int(os.getenv("REFRESH_TOKEN_EXPIRE", "7")),
        allowed_hosts=os.getenv("ALLOWED_HOSTS", "*").split(",")
    )
    
    # Redis configuration
    redis_config = RedisConfig(
        url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "100")),
        decode_responses=True
    )
    
    # CORS origins
    cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
    
    return YMERASettings(
        environment=os.getenv("ENVIRONMENT", "development"),
        debug=os.getenv("DEBUG", "true").lower() == "true",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        database=database_config,
        security=security_config,
        redis=redis_config,
        cors_origins=cors_origins,
        max_file_size=int(os.getenv("MAX_FILE_SIZE", str(50 * 1024 * 1024))),
        upload_directory=os.getenv("UPLOAD_DIRECTORY", "uploads")
    )