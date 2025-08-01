"""
YMERA Enterprise - Validation Utilities
Production-Ready Input Validation System - v4.0
Enterprise-grade implementation with zero placeholders
"""

# ===============================================================================
# STANDARD IMPORTS SECTION
# ===============================================================================

# Standard library imports (alphabetical)
import re
import uuid
import json
import html
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Set, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from email.utils import parseaddr

# Third-party imports (alphabetical)
import structlog
from pydantic import BaseModel, Field, validator, ValidationError
from sqlalchemy import text
from bleach import clean, linkify

# Local imports (alphabetical)
from config.settings import get_settings

# ===============================================================================
# LOGGING CONFIGURATION
# ===============================================================================

logger = structlog.get_logger("ymera.utils.validators")

# ===============================================================================
# CONSTANTS & CONFIGURATION
# ===============================================================================

# Validation patterns
EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
UUID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$', re.IGNORECASE)
PASSWORD_PATTERN = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$')
PHONE_PATTERN = re.compile(r'^\+?1?[0-9]{10,15}$')
URL_PATTERN = re.compile(r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$')

# Sanitization settings
ALLOWED_HTML_TAGS = ['p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li', 'a']
ALLOWED_HTML_ATTRS = {'a': ['href', 'title']}

# Configuration loading
settings = get_settings()

# ===============================================================================
# DATA MODELS & SCHEMAS
# ===============================================================================

@dataclass
class ValidationConfig:
    """Configuration for validation operations"""
    strict_mode: bool = True
    allow_empty_strings: bool = False
    max_string_length: int = 10000
    max_list_length: int = 1000
    max_dict_depth: int = 10
    sanitize_html: bool = True
    check_sql_injection: bool = True

@dataclass 
class ValidationResult:
    """Result of validation operation"""
    is_valid: bool
    sanitized_data: Any = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class AgentConfigSchema(BaseModel):
    """Schema for agent configuration validation"""
    agent_id: str = Field(..., min_length=1, max_length=100)
    agent_type: str = Field(..., regex=r'^[a-zA-Z_][a-zA-Z0-9_]*$')
    capabilities: List[str] = Field(..., min_items=1)
    max_concurrent_tasks: int = Field(default=10, ge=1, le=100)
    timeout_seconds: int = Field(default=30, ge=1, le=300)
    memory_limit_mb: int = Field(default=512, ge=64, le=4096)
    enabled: bool = Field(default=True)
    
    @validator('capabilities')
    def validate_capabilities(cls, v):
        if not v or not all(isinstance(cap, str) and cap.strip() for cap in v):
            raise ValueError('All capabilities must be non-empty strings')
        return [cap.strip() for cap in v]

class LearningDataSchema(BaseModel):
    """Schema for learning data validation"""
    source_agent_id: str = Field(..., min_length=1)
    knowledge_type: str = Field(..., regex=r'^[a-zA-Z_][a-zA-Z0-9_]*$')
    content: Dict[str, Any] = Field(...)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tags: Optional[List[str]] = Field(default=None)
    
    @validator('content')
    def validate_content(cls, v):
        if not v or not isinstance(v, dict):
            raise ValueError('Content must be a non-empty dictionary')
        return v

class APIRequestSchema(BaseModel):
    """Schema for API request validation"""
    endpoint: str = Field(..., min_length=1)
    method: str = Field(..., regex=r'^(GET|POST|PUT|DELETE|PATCH)$')
    headers: Optional[Dict[str, str]] = Field(default=None)
    params: Optional[Dict[str, Any]] = Field(default=None)
    body: Optional[Dict[str, Any]] = Field(default=None)
    timeout: int = Field(default=30, ge=1, le=300)

# ===============================================================================
# CORE VALIDATION CLASSES
# ===============================================================================

class BaseValidator(ABC):
    """Abstract base class for validators"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.logger = logger.bind(validator=self.__class__.__name__)
    
    @abstractmethod
    async def validate(self, data: Any) -> ValidationResult:
        """Validate input data"""
        pass
    
    def _check_sql_injection(self, value: str) -> bool:
        """Check for potential SQL injection patterns"""
        if not self.config.check_sql_injection:
            return True
            
        dangerous_patterns = [
            r"('|(\\'))+.*(;|--|\|)", r"(;|--|\||#|\/\*|\*\/)",
            r"(union|select|insert|delete|update|drop|create|alter|exec|execute)",
            r"(script|javascript|vbscript|onload|onerror|onclick)"
        ]
        
        value_lower = value.lower()
        for pattern in dangerous_patterns:
            if re.search(pattern, value_lower, re.IGNORECASE):
                return False
        return True
    
    def _sanitize_string(self, value: str) -> str:
        """Sanitize string input"""
        if not isinstance(value, str):
            return str(value)
        
        # HTML sanitization
        if self.config.sanitize_html:
            value = clean(value, tags=ALLOWED_HTML_TAGS, attributes=ALLOWED_HTML_ATTRS, strip=True)
        
        # HTML entity encoding
        value = html.escape(value, quote=True)
        
        # Length check
        if len(value) > self.config.max_string_length:
            value = value[:self.config.max_string_length]
        
        return value.strip()

class ValidationManager:
    """Main validation manager for coordinating all validation operations"""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.logger = logger.bind(component="ValidationManager")
        self._validators: Dict[str, BaseValidator] = {}
        self._initialize_validators()
    
    def _initialize_validators(self) -> None:
        """Initialize all validator instances"""
        try:
            self._validators = {
                'agent': AgentValidator(self.config),
                'learning': LearningDataValidator(self.config),
                'api': APIRequestValidator(self.config)
            }
            self.logger.info("Validators initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize validators", error=str(e))
            raise
    
    async def validate_data(self, data_type: str, data: Any) -> ValidationResult:
        """
        Validate data using appropriate validator.
        
        Args:
            data_type: Type of data to validate ('agent', 'learning', 'api')
            data: Data to validate
            
        Returns:
            ValidationResult with validation status and sanitized data
        """
        try:
            if data_type not in self._validators:
                return ValidationResult(
                    is_valid=False,
                    errors=[f"Unknown data type: {data_type}"]
                )
            
            validator = self._validators[data_type]
            result = await validator.validate(data)
            
            self.logger.debug(
                "Data validation completed",
                data_type=data_type,
                is_valid=result.is_valid,
                error_count=len(result.errors)
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Validation failed", data_type=data_type, error=str(e))
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"]
            )

class AgentValidator(BaseValidator):
    """Validator for agent configuration data"""
    
    async def validate(self, data: Any) -> ValidationResult:
        """Validate agent configuration"""
        try:
            # Schema validation
            schema = AgentConfigSchema(**data)
            
            # Additional business logic validation
            errors = []
            warnings = []
            
            # Check agent ID format
            if not self._validate_agent_id(schema.agent_id):
                errors.append("Invalid agent ID format")
            
            # Check SQL injection in string fields
            string_fields = [schema.agent_id, schema.agent_type] + schema.capabilities
            for field_value in string_fields:
                if not self._check_sql_injection(field_value):
                    errors.append(f"Potential SQL injection detected in: {field_value}")
            
            # Resource limits validation
            if schema.memory_limit_mb > 2048:
                warnings.append("High memory limit detected, consider optimization")
            
            # Sanitize data
            sanitized_data = {
                'agent_id': self._sanitize_string(schema.agent_id),
                'agent_type': self._sanitize_string(schema.agent_type),
                'capabilities': [self._sanitize_string(cap) for cap in schema.capabilities],
                'max_concurrent_tasks': schema.max_concurrent_tasks,
                'timeout_seconds': schema.timeout_seconds,
                'memory_limit_mb': schema.memory_limit_mb,
                'enabled': schema.enabled
            }
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                sanitized_data=sanitized_data,
                errors=errors,
                warnings=warnings,
                metadata={'validation_type': 'agent_config'}
            )
            
        except ValidationError as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Schema validation failed: {str(e)}"]
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"]
            )
    
    def _validate_agent_id(self, agent_id: str) -> bool:
        """Validate agent ID format"""
        if not agent_id or len(agent_id) < 3 or len(agent_id) > 100:
            return False
        
        # Allow alphanumeric, hyphens, and underscores
        pattern = re.compile(r'^[a-zA-Z0-9_-]+$')
        return bool(pattern.match(agent_id))

class LearningDataValidator(BaseValidator):
    """Validator for learning data"""
    
    async def validate(self, data: Any) -> ValidationResult:
        """Validate learning data"""
        try:
            # Schema validation
            schema = LearningDataSchema(**data)
            
            errors = []
            warnings = []
            
            # Validate content structure
            content_validation = self._validate_content_structure(schema.content)
            if not content_validation['is_valid']:
                errors.extend(content_validation['errors'])
            
            # Check confidence score reasonableness
            if schema.confidence_score < 0.1:
                warnings.append("Very low confidence score detected")
            
            # Validate tags if present
            if schema.tags:
                tag_validation = self._validate_tags(schema.tags)
                if not tag_validation['is_valid']:
                    errors.extend(tag_validation['errors'])
            
            # Sanitize data
            sanitized_data = {
                'source_agent_id': self._sanitize_string(schema.source_agent_id),
                'knowledge_type': self._sanitize_string(schema.knowledge_type),
                'content': self._sanitize_dict(schema.content),
                'confidence_score': schema.confidence_score,
                'timestamp': schema.timestamp,
                'tags': [self._sanitize_string(tag) for tag in (schema.tags or [])]
            }
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                sanitized_data=sanitized_data,
                errors=errors,
                warnings=warnings,
                metadata={'validation_type': 'learning_data'}
            )
            
        except ValidationError as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Schema validation failed: {str(e)}"]
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"]
            )
    
    def _validate_content_structure(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Validate learning content structure"""
        try:
            errors = []
            
            # Check for required fields in learning content
            required_fields = ['data', 'source', 'timestamp']
            for field in required_fields:
                if field not in content:
                    errors.append(f"Missing required field in content: {field}")
            
            # Validate data size
            content_str = json.dumps(content)
            if len(content_str) > 1024 * 1024:  # 1MB limit
                errors.append("Content size exceeds 1MB limit")
            
            return {
                'is_valid': len(errors) == 0,
                'errors': errors
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'errors': [f"Content validation error: {str(e)}"]
            }
    
    def _validate_tags(self, tags: List[str]) -> Dict[str, Any]:
        """Validate learning data tags"""
        errors = []
        
        if len(tags) > 20:
            errors.append("Too many tags (maximum 20 allowed)")
        
        for tag in tags:
            if not isinstance(tag, str) or not tag.strip():
                errors.append("All tags must be non-empty strings")
            elif len(tag) > 50:
                errors.append(f"Tag too long: {tag[:20]}...")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors
        }
    
    def _sanitize_dict(self, data: Dict[str, Any], depth: int = 0) -> Dict[str, Any]:
        """Recursively sanitize dictionary data"""
        if depth > self.config.max_dict_depth:
            return {}
        
        sanitized = {}
        for key, value in data.items():
            if isinstance(key, str):
                sanitized_key = self._sanitize_string(key)
            else:
                sanitized_key = str(key)
            
            if isinstance(value, str):
                sanitized[sanitized_key] = self._sanitize_string(value)
            elif isinstance(value, dict):
                sanitized[sanitized_key] = self._sanitize_dict(value, depth + 1)
            elif isinstance(value, list):
                sanitized[sanitized_key] = [
                    self._sanitize_string(item) if isinstance(item, str) else item
                    for item in value[:self.config.max_list_length]
                ]
            else:
                sanitized[sanitized_key] = value
        
        return sanitized

class APIRequestValidator(BaseValidator):
    """Validator for API requests"""
    
    async def validate(self, data: Any) -> ValidationResult:
        """Validate API request data"""
        try:
            # Schema validation
            schema = APIRequestSchema(**data)
            
            errors = []
            warnings = []
            
            # Validate endpoint URL
            if not self._validate_endpoint(schema.endpoint):
                errors.append("Invalid endpoint format")
            
            # Validate headers if present
            if schema.headers:
                header_validation = self._validate_headers(schema.headers)
                if not header_validation['is_valid']:
                    errors.extend(header_validation['errors'])
            
            # Check for potential injection in parameters
            if schema.params:
                param_validation = self._validate_params(schema.params)
                if not param_validation['is_valid']:
                    errors.extend(param_validation['errors'])
            
            # Sanitize data
            sanitized_data = {
                'endpoint': self._sanitize_string(schema.endpoint),
                'method': schema.method,
                'headers': self._sanitize_headers(schema.headers) if schema.headers else None,
                'params': self._sanitize_dict(schema.params) if schema.params else None,
                'body': self._sanitize_dict(schema.body) if schema.body else None,
                'timeout': schema.timeout
            }
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                sanitized_data=sanitized_data,
                errors=errors,
                warnings=warnings,
                metadata={'validation_type': 'api_request'}
            )
            
        except ValidationError as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Schema validation failed: {str(e)}"]
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"]
            )
    
    def _validate_endpoint(self, endpoint: str) -> bool:
        """Validate API endpoint format"""
        # Allow relative paths and full URLs
        if endpoint.startswith('/'):
            return True
        return bool(URL_PATTERN.match(endpoint))
    
    def _validate_headers(self, headers: Dict[str, str]) -> Dict[str, Any]:
        """Validate HTTP headers"""
        errors = []
        
        for key, value in headers.items():
            if not isinstance(key, str) or not isinstance(value, str):
                errors.append("All headers must be strings")
            elif not self._check_sql_injection(key) or not self._check_sql_injection(value):
                errors.append(f"Potential injection in header: {key}")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors
        }
    
    def _validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate request parameters"""
        errors = []
        
        for key, value in params.items():
            if isinstance(value, str):
                if not self._check_sql_injection(value):
                    errors.append(f"Potential injection in parameter: {key}")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors
        }
    
    def _sanitize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Sanitize HTTP headers"""
        return {
            self._sanitize_string(key): self._sanitize_string(value)
            for key, value in headers.items()
        }

# ===============================================================================
# UTILITY FUNCTIONS
# ===============================================================================

async def validate_agent_config(config: Dict[str, Any]) -> ValidationResult:
    """Validate agent configuration data"""
    manager = ValidationManager()
    return await manager.validate_data('agent', config)

async def validate_learning_data(data: Dict[str, Any]) -> ValidationResult:
    """Validate learning data"""
    manager = ValidationManager()
    return await manager.validate_data('learning', data)

async def validate_api_request(request: Dict[str, Any]) -> ValidationResult:
    """Validate API request data"""
    manager = ValidationManager()
    return await manager.validate_data('api', request)

def validate_email(email: str) -> bool:
    """Validate email address format"""
    if not email or not isinstance(email, str):
        return False
    
    # Use parseaddr for proper email parsing
    parsed = parseaddr(email)
    if not parsed[1]:
        return False
    
    return bool(EMAIL_PATTERN.match(parsed[1]))

def validate_uuid(uuid_string: str) -> bool:
    """Validate UUID format"""
    if not uuid_string or not isinstance(uuid_string, str):
        return False
    
    try:
        uuid.UUID(uuid_string)
        return True
    except ValueError:
        return False

def validate_password(password: str) -> Tuple[bool, List[str]]:
    """
    Validate password strength.
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    if not password or not isinstance(password, str):
        return False, ["Password is required"]
    
    if len(password) < 8:
        issues.append("Password must be at least 8 characters long")
    
    if not re.search(r'[a-z]', password):
        issues.append("Password must contain at least one lowercase letter")
    
    if not re.search(r'[A-Z]', password):
        issues.append("Password must contain at least one uppercase letter")
    
    if not re.search(r'\d', password):
        issues.append("Password must contain at least one digit")
    
    if not re.search(r'[@$!%*?&]', password):
        issues.append("Password must contain at least one special character")
    
    return len(issues) == 0, issues

def sanitize_input(data: Any, config: Optional[ValidationConfig] = None) -> Any:
    """
    General-purpose input sanitization function.
    
    Args:
        data: Data to sanitize
        config: Validation configuration
        
    Returns:
        Sanitized data
    """
    if config is None:
        config = ValidationConfig()
    
    validator = BaseValidator(config)
    
    if isinstance(data, str):
        return validator._sanitize_string(data)
    elif isinstance(data, dict):
        return validator._sanitize_dict(data)
    elif isinstance(data, list):
        return [sanitize_input(item, config) for item in data]
    else:
        return data

def validate_phone_number(phone: str) -> bool:
    """Validate phone number format"""
    if not phone or not isinstance(phone, str):
        return False
    
    # Remove common formatting characters
    cleaned = re.sub(r'[\s\-\(\)\.]+', '', phone)
    return bool(PHONE_PATTERN.match(cleaned))

def validate_url(url: str) -> bool:
    """Validate URL format"""
    if not url or not isinstance(url, str):
        return False
    
    return bool(URL_PATTERN.match(url))

# ===============================================================================
# EXPORTS
# ===============================================================================

__all__ = [
    "ValidationManager",
    "AgentValidator",
    "LearningDataValidator", 
    "APIRequestValidator",
    "ValidationConfig",
    "ValidationResult",
    "validate_agent_config",
    "validate_learning_data",
    "validate_api_request",
    "validate_email",
    "validate_uuid",
    "validate_password",
    "validate_phone_number",
    "validate_url",
    "sanitize_input"
]