"""
YMERA Authentication Middleware
JWT-based authentication middleware for FastAPI
"""

import logging
from typing import Optional
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Basic authentication middleware for Phase 1-2"""
    
    def __init__(self, app, jwt_manager=None):
        super().__init__(app)
        self.jwt_manager = jwt_manager
        
        # Public endpoints that don't require authentication
        self.public_endpoints = {
            "/",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v1/auth/login",
            "/api/v1/auth/register"
        }
    
    async def dispatch(self, request: Request, call_next):
        """Process request for authentication"""
        
        # Skip authentication for public endpoints
        if request.url.path in self.public_endpoints:
            return await call_next(request)
        
        # For Phase 1-2, we'll implement basic authentication
        # In a production environment, this would validate JWT tokens
        
        # For now, allow all requests to pass through for Phase 1-2 development
        response = await call_next(request)
        return response