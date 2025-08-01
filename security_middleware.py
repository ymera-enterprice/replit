"""
YMERA Security Middleware
Basic security headers and input validation
"""

import logging
from typing import Dict, Any
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for basic protection"""
    
    def __init__(self, app, config: Dict[str, Any] = None):
        super().__init__(app)
        self.config = config or {}
        
        # Security headers
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "X-Permitted-Cross-Domain-Policies": "none"
        }
    
    async def dispatch(self, request: Request, call_next):
        """Apply security measures"""
        
        # Basic input validation
        if not self._validate_request(request):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid request format"
            )
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        for header, value in self.security_headers.items():
            response.headers[header] = value
        
        return response
    
    def _validate_request(self, request: Request) -> bool:
        """Basic request validation"""
        
        # Check request size (basic protection against large payloads)
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
                if size > 50 * 1024 * 1024:  # 50MB limit
                    logger.warning(f"Request too large: {size} bytes")
                    return False
            except ValueError:
                logger.warning("Invalid content-length header")
                return False
        
        # Basic URL validation
        if len(str(request.url)) > 2048:  # URL length limit
            logger.warning("URL too long")
            return False
        
        return True