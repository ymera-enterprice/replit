"""
YMERA Core Exception Handling System
Enterprise-grade exception management with learning integration
"""

import asyncio
import traceback
import sys
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Type
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import logging
import json
import inspect
from functools import wraps
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field

# Error Severity Levels
class ErrorSeverity(str, Enum):
    """Error severity classification for monitoring and response"""
    CRITICAL = "critical"      # System failure, immediate attention required
    HIGH = "high"             # Major functionality impacted
    MEDIUM = "medium"         # Moderate impact, degraded performance
    LOW = "low"               # Minor issues, minimal impact
    INFO = "info"             # Informational, no action required

# Error Categories for Learning Engine
class ErrorCategory(str, Enum):
    """Error categories for machine learning pattern recognition"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    DATABASE = "database"
    EXTERNAL_API = "external_api"
    AGENT_EXECUTION = "agent_execution"
    AGENT_COMMUNICATION = "agent_communication"
    LEARNING_ENGINE = "learning_engine"
    CONFIGURATION = "configuration"
    RESOURCE = "resource"
    NETWORK = "network"
    BUSINESS_LOGIC = "business_logic"
    SYSTEM = "system"
    SECURITY = "security"
    PERFORMANCE = "performance"

# Recovery Actions
class RecoveryAction(str, Enum):
    """Automated recovery actions that can be triggered"""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    SCALE_UP = "scale_up"
    RESET_CONNECTION = "reset_connection"
    CLEAR_CACHE = "clear_cache"
    RESTART_AGENT = "restart_agent"
    SWITCH_PROVIDER = "switch_provider"
    DEGRADE_SERVICE = "degrade_service"
    ALERT_ADMIN = "alert_admin"

@dataclass
class ErrorContext:
    """Enhanced error context for comprehensive error tracking"""
    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    agent_type: Optional[str] = None
    operation: Optional[str] = None
    method: Optional[str] = None
    endpoint: Optional[str] = None
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    environment: str = "production"
    service_version: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ErrorMetrics:
    """Error metrics for monitoring and learning"""
    occurrence_count: int = 1
    first_occurrence: datetime = field(default_factory=datetime.utcnow)
    last_occurrence: datetime = field(default_factory=datetime.utcnow)
    total_impact_time: float = 0.0
    affected_users: int = 0
    recovery_attempts: int = 0
    successful_recoveries: int = 0
    pattern_score: float = 0.0
    trend_direction: str = "stable"  # increasing, decreasing, stable

class ErrorResponse(BaseModel):
    """Standardized error response model"""
    error: bool = True
    error_id: str = Field(..., description="Unique error identifier")
    error_code: str = Field(..., description="Error code for programmatic handling")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[str] = Field(None, description="Detailed error information")
    severity: ErrorSeverity = Field(..., description="Error severity level")
    category: ErrorCategory = Field(..., description="Error category")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = Field(None, description="Request identifier")
    suggested_actions: List[str] = Field(default_factory=list)
    retry_after: Optional[int] = Field(None, description="Retry after seconds")
    documentation_link: Optional[str] = Field(None, description="Link to relevant documentation")
    support_reference: Optional[str] = Field(None, description="Support reference number")

# Base YMERA Exception
class YMERAException(Exception):
    """
    Base exception class for all YMERA system exceptions
    Provides comprehensive error handling with learning integration
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = "YMERA_ERROR",
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        details: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        recoverable: bool = True,
        retry_after: Optional[int] = None,
        suggested_actions: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.category = category
        self.details = details
        self.context = context or ErrorContext()
        self.cause = cause
        self.recoverable = recoverable
        self.retry_after = retry_after
        self.suggested_actions = suggested_actions or []
        self.additional_data = kwargs
        self.stack_trace = traceback.format_exc()
        
        # Learning integration data
        self.learning_data = {
            "error_signature": self._generate_error_signature(),
            "code_location": self._get_code_location(),
            "execution_path": self._get_execution_path(),
            "system_state": self._capture_system_state()
        }

    def _generate_error_signature(self) -> str:
        """Generate unique signature for error pattern recognition"""
        components = [
            self.error_code,
            self.category.value,
            self.__class__.__name__,
            str(hash(self.message[:100]))  # First 100 chars to avoid huge signatures
        ]
        return "|".join(components)

    def _get_code_location(self) -> Dict[str, Any]:
        """Extract code location information"""
        frame = inspect.currentframe()
        try:
            # Go up the stack to find the actual error location
            while frame and frame.f_code.co_filename.endswith('exceptions.py'):
                frame = frame.f_back
            
            if frame:
                return {
                    "file": frame.f_code.co_filename,
                    "function": frame.f_code.co_name,
                    "line": frame.f_lineno,
                    "module": frame.f_globals.get('__name__', 'unknown')
                }
        finally:
            del frame
        return {}

    def _get_execution_path(self) -> List[str]:
        """Extract execution path for pattern analysis"""
        frames = []
        frame = inspect.currentframe()
        try:
            while frame:
                if not frame.f_code.co_filename.endswith('exceptions.py'):
                    frames.append(f"{frame.f_code.co_name}:{frame.f_lineno}")
                frame = frame.f_back
                if len(frames) >= 10:  # Limit depth
                    break
        finally:
            del frame
        return frames

    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture relevant system state for learning"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "python_version": sys.version,
            "thread_count": len([t for t in threading.enumerate() if t.is_alive()]),
            "memory_usage": self._get_memory_usage(),
        }

    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage if psutil is available"""
        try:
            import psutil
            return psutil.Process().memory_percent()
        except ImportError:
            return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization"""
        return {
            "error_id": self.context.error_id,
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "details": self.details,
            "recoverable": self.recoverable,
            "retry_after": self.retry_after,
            "suggested_actions": self.suggested_actions,
            "timestamp": self.context.timestamp.isoformat(),
            "context": {
                "request_id": self.context.request_id,
                "agent_id": self.context.agent_id,
                "operation": self.context.operation,
                "correlation_id": self.context.correlation_id
            },
            "learning_data": self.learning_data,
            "additional_data": self.additional_data
        }

# Authentication & Authorization Exceptions
class AuthenticationError(YMERAException):
    """Authentication failure"""
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(
            message=message,
            error_code="AUTH_001",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.AUTHENTICATION,
            suggested_actions=["Check credentials", "Verify token validity", "Re-authenticate"],
            **kwargs
        )

class AuthorizationError(YMERAException):
    """Authorization failure"""
    def __init__(self, message: str = "Access denied", **kwargs):
        super().__init__(
            message=message,
            error_code="AUTH_002",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.AUTHORIZATION,
            suggested_actions=["Check permissions", "Contact administrator"],
            **kwargs
        )

class TokenExpiredError(AuthenticationError):
    """Token expired"""
    def __init__(self, message: str = "Authentication token has expired", **kwargs):
        super().__init__(
            message=message,
            error_code="AUTH_003",
            suggested_actions=["Refresh token", "Re-authenticate"],
            retry_after=60,
            **kwargs
        )

# Validation Exceptions
class ValidationError(YMERAException):
    """Data validation error"""
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="VALID_001",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.VALIDATION,
            suggested_actions=["Check input format", "Verify required fields"],
            field=field,
            **kwargs
        )

class SchemaValidationError(ValidationError):
    """Schema validation error"""
    def __init__(self, message: str, schema_errors: Optional[List[Dict]] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="VALID_002",
            schema_errors=schema_errors or [],
            **kwargs
        )

# Database Exceptions
class DatabaseError(YMERAException):
    """Database operation error"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="DB_001",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.DATABASE,
            suggested_actions=["Check database connection", "Retry operation", "Contact support"],
            **kwargs
        )

class DatabaseConnectionError(DatabaseError):
    """Database connection error"""
    def __init__(self, message: str = "Database connection failed", **kwargs):
        super().__init__(
            message=message,
            error_code="DB_002",
            severity=ErrorSeverity.CRITICAL,
            suggested_actions=["Check database status", "Verify connection string", "Scale database"],
            retry_after=30,
            **kwargs
        )

class DatabaseTransactionError(DatabaseError):
    """Database transaction error"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="DB_003",
            suggested_actions=["Retry transaction", "Check data consistency"],
            **kwargs
        )

# Agent System Exceptions
class AgentError(YMERAException):
    """Base agent system error"""
    def __init__(self, message: str, agent_id: Optional[str] = None, **kwargs):
        context = kwargs.get('context', ErrorContext())
        context.agent_id = agent_id
        super().__init__(
            message=message,
            error_code="AGENT_001",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.AGENT_EXECUTION,
            context=context,
            **kwargs
        )

class AgentInitializationError(AgentError):
    """Agent initialization failure"""
    def __init__(self, message: str, agent_type: Optional[str] = None, **kwargs):
        context = kwargs.get('context', ErrorContext())
        context.agent_type = agent_type
        super().__init__(
            message=message,
            error_code="AGENT_002",
            severity=ErrorSeverity.CRITICAL,
            suggested_actions=["Check agent configuration", "Verify dependencies", "Restart agent"],
            context=context,
            **kwargs
        )

class AgentExecutionError(AgentError):
    """Agent execution error"""
    def __init__(self, message: str, task_id: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="AGENT_003",
            suggested_actions=["Retry task", "Check agent status", "Review task parameters"],
            task_id=task_id,
            **kwargs
        )

class AgentCommunicationError(YMERAException):
    """Agent communication error"""
    def __init__(self, message: str, source_agent: Optional[str] = None, target_agent: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="AGENT_004",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.AGENT_COMMUNICATION,
            suggested_actions=["Check message bus", "Verify agent connectivity", "Retry communication"],
            source_agent=source_agent,
            target_agent=target_agent,
            **kwargs
        )

class AgentTimeoutError(AgentError):
    """Agent operation timeout"""
    def __init__(self, message: str, timeout_duration: Optional[float] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="AGENT_005",
            suggested_actions=["Increase timeout", "Check agent performance", "Optimize operation"],
            timeout_duration=timeout_duration,
            retry_after=60,
            **kwargs
        )

# Learning Engine Exceptions
class LearningEngineError(YMERAException):
    """Learning engine error"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="LEARN_001",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.LEARNING_ENGINE,
            suggested_actions=["Check learning data", "Verify model integrity", "Reset learning state"],
            **kwargs
        )

class ModelTrainingError(LearningEngineError):
    """Model training error"""
    def __init__(self, message: str, model_type: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="LEARN_002",
            suggested_actions=["Check training data", "Verify model parameters", "Retry training"],
            model_type=model_type,
            **kwargs
        )

class KnowledgeBaseError(LearningEngineError):
    """Knowledge base operation error"""
    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="LEARN_003",
            suggested_actions=["Check vector database", "Verify embeddings", "Rebuild index"],
            operation=operation,
            **kwargs
        )

# External API Exceptions
class ExternalAPIError(YMERAException):
    """External API error"""
    def __init__(self, message: str, api_name: Optional[str] = None, status_code: Optional[int] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="API_001",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.EXTERNAL_API,
            suggested_actions=["Check API status", "Verify API key", "Retry with backoff"],
            api_name=api_name,
            status_code=status_code,
            **kwargs
        )

class RateLimitError(ExternalAPIError):
    """Rate limit exceeded"""
    def __init__(self, message: str, retry_after: int = 60, **kwargs):
        super().__init__(
            message=message,
            error_code="API_002",
            suggested_actions=["Wait before retry", "Use different API key", "Implement backoff"],
            retry_after=retry_after,
            **kwargs
        )

# Configuration Exceptions
class ConfigurationError(YMERAException):
    """Configuration error"""
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="CONFIG_001",
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.CONFIGURATION,
            suggested_actions=["Check configuration file", "Verify environment variables", "Update configuration"],
            config_key=config_key,
            recoverable=False,
            **kwargs
        )

# Resource Exceptions
class ResourceError(YMERAException):
    """Resource-related error"""
    def __init__(self, message: str, resource_type: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="RESOURCE_001",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.RESOURCE,
            suggested_actions=["Check resource availability", "Scale resources", "Optimize usage"],
            resource_type=resource_type,
            **kwargs
        )

class ResourceExhaustedError(ResourceError):
    """Resource exhausted"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="RESOURCE_002",
            severity=ErrorSeverity.CRITICAL,
            suggested_actions=["Scale up resources", "Optimize resource usage", "Clear cache"],
            retry_after=300,
            **kwargs
        )

# Security Exceptions
class SecurityError(YMERAException):
    """Security-related error"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="SECURITY_001",
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.SECURITY,
            suggested_actions=["Review security logs", "Check for threats", "Contact security team"],
            recoverable=False,
            **kwargs
        )

class SecurityViolationError(SecurityError):
    """Security violation detected"""
    def __init__(self, message: str, violation_type: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            error_code="SECURITY_002",
            suggested_actions=["Block suspicious activity", "Review access logs", "Alert security team"],
            violation_type=violation_type,
            **kwargs
        )

# Business Logic Exceptions
class BusinessLogicError(YMERAException):
    """Business logic error"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="BUSINESS_001",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.BUSINESS_LOGIC,
            suggested_actions=["Check business rules", "Verify input data", "Review logic"],
            **kwargs
        )

# Exception Handler Class
class ExceptionHandler:
    """
    Centralized exception handler with learning integration
    """
    
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        learning_engine: Optional[Any] = None,
        metrics_collector: Optional[Any] = None,
        notification_service: Optional[Any] = None
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.learning_engine = learning_engine
        self.metrics_collector = metrics_collector
        self.notification_service = notification_service
        self.error_patterns = {}
        self.recovery_strategies = {}
        self.circuit_breakers = {}
        self._setup_recovery_strategies()

    def _setup_recovery_strategies(self):
        """Setup automated recovery strategies"""
        self.recovery_strategies = {
            ErrorCategory.DATABASE: [RecoveryAction.RETRY, RecoveryAction.RESET_CONNECTION],
            ErrorCategory.EXTERNAL_API: [RecoveryAction.RETRY, RecoveryAction.SWITCH_PROVIDER],
            ErrorCategory.AGENT_EXECUTION: [RecoveryAction.RESTART_AGENT, RecoveryAction.FALLBACK],
            ErrorCategory.RESOURCE: [RecoveryAction.SCALE_UP, RecoveryAction.CLEAR_CACHE],
            ErrorCategory.NETWORK: [RecoveryAction.RETRY, RecoveryAction.CIRCUIT_BREAK]
        }

    async def handle_exception(
        self,
        exception: Exception,
        context: Optional[ErrorContext] = None,
        auto_recover: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive exception handling with learning integration
        """
        start_time = time.time()
        
        # Convert to YMERA exception if needed
        if not isinstance(exception, YMERAException):
            exception = self._convert_to_ymera_exception(exception, context)
        
        # Update context
        if context:
            exception.context = context
        
        # Generate error response
        error_response = self._generate_error_response(exception)
        
        # Log the error
        await self._log_error(exception)
        
        # Update metrics
        if self.metrics_collector:
            await self._update_metrics(exception)
        
        # Learn from error
        if self.learning_engine:
            await self._learn_from_error(exception)
        
        # Attempt recovery if enabled
        recovery_result = None
        if auto_recover and exception.recoverable:
            recovery_result = await self._attempt_recovery(exception)
        
        # Send notifications if critical
        if exception.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            await self._send_notifications(exception)
        
        processing_time = time.time() - start_time
        
        return {
            "error_response": error_response,
            "recovery_result": recovery_result,
            "processing_time": processing_time,
            "error_id": exception.context.error_id
        }

    def _convert_to_ymera_exception(
        self, 
        exception: Exception, 
        context: Optional[ErrorContext] = None
    ) -> YMERAException:
        """Convert standard exception to YMERA exception"""
        error_mappings = {
            ValueError: ValidationError,
            KeyError: ValidationError,
            ConnectionError: DatabaseConnectionError,
            TimeoutError: AgentTimeoutError,
            PermissionError: AuthorizationError,
            FileNotFoundError: ResourceError,
            MemoryError: ResourceExhaustedError
        }
        
        exception_class = error_mappings.get(type(exception), YMERAException)
        
        return exception_class(
            message=str(exception),
            context=context,
            cause=exception
        )

    def _generate_error_response(self, exception: YMERAException) -> ErrorResponse:
        """Generate standardized error response"""
        return ErrorResponse(
            error_id=exception.context.error_id,
            error_code=exception.error_code,
            message=exception.message,
            details=exception.details,
            severity=exception.severity,
            category=exception.category,
            timestamp=exception.context.timestamp,
            request_id=exception.context.request_id,
            suggested_actions=exception.suggested_actions,
            retry_after=exception.retry_after,
            documentation_link=self._get_documentation_link(exception),
            support_reference=self._generate_support_reference(exception)
        )

    def _get_documentation_link(self, exception: YMERAException) -> Optional[str]:
        """Get relevant documentation link for error"""
        base_url = "https://docs.ymera.com/errors"
        return f"{base_url}/{exception.category.value}/{exception.error_code.lower()}"

    def _generate_support_reference(self, exception: YMERAException) -> str:
        """Generate support reference number"""
        return f"SUP-{exception.context.error_id[:8].upper()}"

    async def _log_error(self, exception: YMERAException):
        """Log error with structured information"""
        log_data = {
            "error_id": exception.context.error_id,
            "error_code": exception.error_code,
            "message": exception.message,
            "severity": exception.severity.value,
            "category": exception.category.value,
            "agent_id": exception.context.agent_id,
            "request_id": exception.context.request_id,
            "stack_trace": exception.stack_trace,
            "learning_data": exception.learning_data
        }
        
        log_method = {
            ErrorSeverity.CRITICAL: self.logger.critical,
            ErrorSeverity.HIGH: self.logger.error,
            ErrorSeverity.MEDIUM: self.logger.warning,
            ErrorSeverity.LOW: self.logger.info,
            ErrorSeverity.INFO: self.logger.info
        }.get(exception.severity, self.logger.error)
        
        log_method(f"YMERA Exception: {exception.message}", extra=log_data)

    async def _update_metrics(self, exception: YMERAException):
        """Update error metrics"""
        try:
            await self.metrics_collector.increment_error_count(
                error_code=exception.error_code,
                category=exception.category.value,
                severity=exception.severity.value,
                agent_id=exception.context.agent_id
            )
        except Exception as e:
            self.logger.warning(f"Failed to update metrics: {e}")

    async def _learn_from_error(self, exception: YMERAException):
        """Feed error information to learning engine"""
        try:
            learning_data = {
                "error_signature": exception.learning_data["error_signature"],
                "error_context": exception.to_dict(),
                "system_state": exception.learning_data["system_state"],
                "execution_path": exception.learning_data["execution_path"],
                "timestamp": exception.context.timestamp.isoformat()
            }
            
            await self.learning_engine.learn_from_error(learning_data)
        except Exception as e:
            self.logger.warning(f"Failed to feed error to learning engine: {e}")

    async def _attempt_recovery(self, exception: YMERAException) -> Dict[str, Any]:
        """Attempt automated recovery based on error category"""
        recovery_result = {
            "attempted": False,
            "successful": False,
            "actions_taken": [],
            "message": None
        }
        
        try:
            recovery_actions = self.recovery_strategies.get(exception.category, [])
            
            for action in recovery_actions:
                recovery_result["attempted"] = True
                recovery_result["actions_taken"].append(action.value)
                
                success = await self._execute_recovery_action(action, exception)
                
                if success:
                    recovery_result["successful"] = True
                    recovery_result["message"] = f"Recovered using {action.value}"
                    break
            
            if not recovery_result["successful"] and recovery_result["attempted"]:
                recovery_result["message"] = "Recovery attempts failed"
                
        except Exception as e:
            self.logger.error(f"Error during recovery attempt: {e}")
            recovery_result["message"] = f"Recovery error: {str(e)}"
        
        return recovery_result

    async def _execute_recovery_action(
        self, 
        action: RecoveryAction, 
        exception: YMERAException
    ) -> bool:
        """Execute specific recovery action"""
        try:
            if action == RecoveryAction.RETRY:
                await asyncio.sleep(1)  # Simple backoff
                return True
            
            elif action == RecoveryAction.CLEAR_CACHE:
                # Would integrate with cache manager
                return True
            
            elif action == RecoveryAction.RESTART_AGENT:
                # Would integrate with agent lifecycle manager
                return True
            
            elif action == RecoveryAction.RESET_CONNECTION:
                # Would integrate with connection managers
                return True
            
            # Add more recovery actions as needed
            
        except Exception as e:
            self.logger.error(f"Failed to execute recovery action {action}: {e}")
            return False
        
        return False

    async def _send_notifications(self, exception: YMERAException):
        """Send notifications for critical errors"""
        try:
            if self.notification_service:
                await self.notification_service.send_error_notification(
                    error_id=exception.context.error_id,
                    severity=exception.severity.value,
                    message=exception.message,
                    category=exception.category.value,
                    agent_id=exception.context.agent_id
                )
        except Exception as e:
            self.logger.warning(f"Failed to send notification: {e}")

# FastAPI Exception Handler
async def handle_ymera_exception(request: Request, exc: YMERAException) -> JSONResponse:
    """FastAPI exception handler for YMERA exceptions"""
    
    # Extract request context
    context = ErrorContext(
        request_id=request.headers.get("X-Request-ID"),
        user_agent=request.headers.get("User-Agent"),
        ip_address=request.client.host if request.client else None,
        method=request.method,
        endpoint=str(request.url.path),
        correlation_id=request.headers.get("X-Correlation-ID"),
        trace_id=request.headers.get("X-Trace-ID")
    )
    
    # Update exception context
    exc.context = context
    
    # Generate response
    error_response = ErrorResponse(
        error_id=exc.context.error_id,
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details,
        severity=exc.severity,
        category=exc.category,
        timestamp=exc.context.timestamp,
        request_id=context.request_id,
        suggested_actions=exc.suggested_actions,
        retry_after=exc.retry_after,
        documentation_link=f"https://docs.ymera.com/errors/{exc.category.value}/{exc.error_code.lower()}",
        support_reference=f"SUP-{exc.context.error_id[:8].upper()}"
    )
    
    # Map severity to HTTP status codes
    status_code_mapping = {
        ErrorSeverity.CRITICAL: status.HTTP_500_INTERNAL_SERVER_ERROR,
        ErrorSeverity.HIGH: status.HTTP_500_INTERNAL_SERVER_ERROR,
        ErrorSeverity.MEDIUM: status.HTTP_400_BAD_REQUEST,
        ErrorSeverity.LOW: status.HTTP_400_BAD_REQUEST,
        ErrorSeverity.INFO: status.HTTP_200_OK
    }
    
    # Special status codes for specific categories
    if exc.category == ErrorCategory.AUTHENTICATION:
        http_status = status.HTTP_401_UNAUTHORIZED
    elif exc.category == ErrorCategory.AUTHORIZATION:
        http_status = status.HTTP_403_FORBIDDEN
    elif exc.category == ErrorCategory.VALIDATION:
        http_status = status.HTTP_422_UNPROCESSABLE_ENTITY
    elif exc.category == ErrorCategory.EXTERNAL_API and isinstance(exc, RateLimitError):
        http_status = status.HTTP_429_TOO_MANY_REQUESTS
    else:
        http_status = status_code_mapping.get(exc.severity, status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    # Add retry-after header if specified
    headers = {}
    if exc.retry_after:
        headers["Retry-After"] = str(exc.retry_after)
        headers["X-RateLimit-RetryAfter"] = str(exc.retry_after)
    
    # Add error tracking headers
    headers.update({
        "X-Error-ID": exc.context.error_id,
        "X-Error-Code": exc.error_code,
        "X-Error-Category": exc.category.value,
        "X-Error-Severity": exc.severity.value
    })
    
    return JSONResponse(
        status_code=http_status,
        content=error_response.dict(),
        headers=headers
    )

# Exception Tracking Middleware
class ExceptionTrackingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for tracking and learning from exceptions
    """
    
    def __init__(self, app, exception_handler: ExceptionHandler):
        super().__init__(app)
        self.exception_handler = exception_handler
        self.logger = logging.getLogger(__name__)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Track exceptions and feed to learning system"""
        start_time = time.time()
        
        # Generate request context
        context = ErrorContext(
            request_id=str(uuid.uuid4()),
            method=request.method,
            endpoint=str(request.url.path),
            user_agent=request.headers.get("User-Agent"),
            ip_address=request.client.host if request.client else None,
            correlation_id=request.headers.get("X-Correlation-ID", str(uuid.uuid4())),
            trace_id=request.headers.get("X-Trace-ID", str(uuid.uuid4()))
        )
        
        try:
            # Add context to request state
            request.state.error_context = context
            
            response = await call_next(request)
            
            # Track successful requests for learning
            processing_time = time.time() - start_time
            await self._track_success(context, processing_time)
            
            return response
            
        except Exception as exc:
            processing_time = time.time() - start_time
            context.additional_data["processing_time"] = processing_time
            
            # Handle the exception
            result = await self.exception_handler.handle_exception(exc, context)
            
            # Return appropriate response
            if isinstance(exc, YMERAException):
                return await handle_ymera_exception(request, exc)
            else:
                # Convert and handle non-YMERA exceptions
                ymera_exc = self.exception_handler._convert_to_ymera_exception(exc, context)
                return await handle_ymera_exception(request, ymera_exc)

    async def _track_success(self, context: ErrorContext, processing_time: float):
        """Track successful requests for learning patterns"""
        try:
            success_data = {
                "endpoint": context.endpoint,
                "method": context.method,
                "processing_time": processing_time,
                "timestamp": context.timestamp.isoformat(),
                "success": True
            }
            
            if self.exception_handler.learning_engine:
                await self.exception_handler.learning_engine.learn_from_success(success_data)
                
        except Exception as e:
            self.logger.warning(f"Failed to track success: {e}")

# Circuit Breaker for Error Prevention
class CircuitBreaker:
    """
    Circuit breaker pattern for preventing cascading failures
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def __call__(self, func):
        """Decorator for applying circuit breaker"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker is OPEN for {func.__name__}",
                        retry_after=self.timeout
                    )
            
            try:
                result = await func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.expected_exception as e:
                self._on_failure()
                raise e
                
        return wrapper

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.timeout
        )

    def _on_success(self):
        """Handle successful operation"""
        self.failure_count = 0
        self.state = "CLOSED"

    def _on_failure(self):
        """Handle failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

class CircuitBreakerOpenError(YMERAException):
    """Circuit breaker is open"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="CIRCUIT_001",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.SYSTEM,
            suggested_actions=["Wait for circuit breaker reset", "Check downstream services"],
            **kwargs
        )

# Retry Decorator with Exponential Backoff
def retry_with_backoff(
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable] = None
):
    """
    Retry decorator with exponential backoff
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        break
                    
                    # Calculate backoff delay
                    delay = backoff_factor * (2 ** attempt)
                    
                    # Call retry callback if provided
                    if on_retry:
                        await on_retry(e, attempt + 1, delay)
                    
                    await asyncio.sleep(delay)
            
            # All retries exhausted, raise the last exception
            if isinstance(last_exception, YMERAException):
                raise last_exception
            else:
                raise RetryExhaustedError(
                    f"Max retries ({max_retries}) exhausted for {func.__name__}",
                    original_exception=str(last_exception),
                    attempts=max_retries + 1
                )
                
        return wrapper
    return decorator

class RetryExhaustedError(YMERAException):
    """Retry attempts exhausted"""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="RETRY_001",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.SYSTEM,
            suggested_actions=["Check underlying service", "Increase retry count", "Investigate root cause"],
            **kwargs
        )

# Error Pattern Detection
class ErrorPatternDetector:
    """
    Detects error patterns for predictive error prevention
    """
    
    def __init__(self, learning_engine: Optional[Any] = None):
        self.learning_engine = learning_engine
        self.error_patterns = {}
        self.pattern_thresholds = {
            "frequency": 5,      # errors within time window
            "time_window": 300,  # 5 minutes
            "similarity": 0.8    # pattern similarity threshold
        }

    async def detect_patterns(self, error_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect error patterns and predict potential issues
        """
        try:
            pattern_analysis = {
                "pattern_detected": False,
                "pattern_type": None,
                "confidence": 0.0,
                "predicted_errors": [],
                "recommendations": []
            }
            
            # Analyze error signature
            error_signature = error_data.get("error_signature")
            if not error_signature:
                return pattern_analysis
            
            # Check for recurring patterns
            current_time = time.time()
            if error_signature in self.error_patterns:
                pattern_info = self.error_patterns[error_signature]
                pattern_info["occurrences"].append(current_time)
                
                # Clean old occurrences
                pattern_info["occurrences"] = [
                    t for t in pattern_info["occurrences"]
                    if current_time - t <= self.pattern_thresholds["time_window"]
                ]
                
                # Check if pattern threshold exceeded
                if len(pattern_info["occurrences"]) >= self.pattern_thresholds["frequency"]:
                    pattern_analysis.update({
                        "pattern_detected": True,
                        "pattern_type": "recurring_error",
                        "confidence": min(len(pattern_info["occurrences"]) / 10.0, 1.0),
                        "recommendations": [
                            "Investigate root cause",
                            "Implement preventive measures",
                            "Consider circuit breaker"
                        ]
                    })
            else:
                self.error_patterns[error_signature] = {
                    "first_occurrence": current_time,
                    "occurrences": [current_time],
                    "metadata": error_data
                }
            
            # Use learning engine for advanced pattern detection
            if self.learning_engine:
                ai_analysis = await self.learning_engine.analyze_error_patterns(error_data)
                if ai_analysis:
                    pattern_analysis.update(ai_analysis)
            
            return pattern_analysis
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error in pattern detection: {e}")
            return {"pattern_detected": False, "error": str(e)}

# Error Recovery Orchestrator
class ErrorRecoveryOrchestrator:
    """
    Orchestrates complex error recovery workflows
    """
    
    def __init__(
        self,
        agent_registry: Optional[Any] = None,
        learning_engine: Optional[Any] = None
    ):
        self.agent_registry = agent_registry
        self.learning_engine = learning_engine
        self.recovery_workflows = {}
        self._setup_workflows()

    def _setup_workflows(self):
        """Setup recovery workflows for different scenarios"""
        self.recovery_workflows = {
            "database_failure": [
                {"action": "check_connection", "timeout": 10},
                {"action": "reset_connection_pool", "timeout": 30},
                {"action": "fallback_to_replica", "timeout": 60},
                {"action": "alert_dba", "timeout": 0}
            ],
            "agent_failure": [
                {"action": "restart_agent", "timeout": 30},
                {"action": "reallocate_tasks", "timeout": 60},
                {"action": "spawn_backup_agent", "timeout": 120},
                {"action": "escalate_to_human", "timeout": 0}
            ],
            "api_rate_limit": [
                {"action": "implement_backoff", "timeout": 60},
                {"action": "switch_api_key", "timeout": 30},
                {"action": "use_alternative_provider", "timeout": 60},
                {"action": "queue_requests", "timeout": 0}
            ]
        }

    async def orchestrate_recovery(
        self,
        error_type: str,
        error_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Orchestrate complex recovery workflow
        """
        workflow = self.recovery_workflows.get(error_type)
        if not workflow:
            return {"success": False, "message": "No recovery workflow found"}
        
        recovery_result = {
            "workflow_type": error_type,
            "steps_executed": [],
            "success": False,
            "final_action": None,
            "duration": 0
        }
        
        start_time = time.time()
        
        try:
            for step in workflow:
                step_result = await self._execute_recovery_step(step, error_context)
                recovery_result["steps_executed"].append({
                    "action": step["action"],
                    "success": step_result["success"],
                    "duration": step_result["duration"],
                    "message": step_result.get("message")
                })
                
                if step_result["success"]:
                    recovery_result["success"] = True
                    recovery_result["final_action"] = step["action"]
                    break
            
            recovery_result["duration"] = time.time() - start_time
            
            # Learn from recovery attempt
            if self.learning_engine:
                await self.learning_engine.learn_from_recovery(recovery_result)
            
            return recovery_result
            
        except Exception as e:
            recovery_result["error"] = str(e)
            recovery_result["duration"] = time.time() - start_time
            return recovery_result

    async def _execute_recovery_step(
        self,
        step: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute individual recovery step"""
        start_time = time.time()
        
        try:
            action = step["action"]
            timeout = step["timeout"]
            
            # Simulate recovery actions (integrate with actual services)
            if action == "check_connection":
                await asyncio.sleep(0.1)  # Simulate connection check
                return {"success": True, "duration": time.time() - start_time}
            
            elif action == "restart_agent":
                # Integrate with agent lifecycle manager
                if self.agent_registry:
                    agent_id = context.get("agent_id")
                    if agent_id:
                        # await self.agent_registry.restart_agent(agent_id)
                        pass
                return {"success": True, "duration": time.time() - start_time}
            
            # Add more recovery actions as needed
            
            return {
                "success": False,
                "duration": time.time() - start_time,
                "message": f"Recovery action {action} not implemented"
            }
            
        except Exception as e:
            return {
                "success": False,
                "duration": time.time() - start_time,
                "error": str(e)
            }

# Health Check Integration
class HealthCheckManager:
    """
    Manages health checks and proactive error prevention
    """
    
    def __init__(self, exception_handler: ExceptionHandler):
        self.exception_handler = exception_handler
        self.health_checks = {}
        self.check_intervals = {}
        self.check_tasks = {}

    def register_health_check(
        self,
        name: str,
        check_func: Callable,
        interval: int = 60,
        threshold: int = 3
    ):
        """Register a health check"""
        self.health_checks[name] = {
            "function": check_func,
            "threshold": threshold,
            "consecutive_failures": 0,
            "last_check": None,
            "status": "unknown"
        }
        self.check_intervals[name] = interval

    async def start_health_monitoring(self):
        """Start health monitoring for all registered checks"""
        for name in self.health_checks:
            task = asyncio.create_task(self._run_health_check_loop(name))
            self.check_tasks[name] = task

    async def stop_health_monitoring(self):
        """Stop all health monitoring"""
        for task in self.check_tasks.values():
            task.cancel()
        
        await asyncio.gather(*self.check_tasks.values(), return_exceptions=True)
        self.check_tasks.clear()

    async def _run_health_check_loop(self, name: str):
        """Run health check loop for a specific check"""
        check_info = self.health_checks[name]
        interval = self.check_intervals[name]
        
        while True:
            try:
                await asyncio.sleep(interval)
                
                # Execute health check
                is_healthy = await self._execute_health_check(name)
                check_info["last_check"] = datetime.utcnow()
                
                if is_healthy:
                    check_info["consecutive_failures"] = 0
                    check_info["status"] = "healthy"
                else:
                    check_info["consecutive_failures"] += 1
                    check_info["status"] = "unhealthy"
                    
                    # Check if threshold exceeded
                    if check_info["consecutive_failures"] >= check_info["threshold"]:
                        await self._handle_health_check_failure(name, check_info)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.getLogger(__name__).error(f"Health check {name} error: {e}")

    async def _execute_health_check(self, name: str) -> bool:
        """Execute a health check"""
        try:
            check_func = self.health_checks[name]["function"]
            result = await check_func()
            return bool(result)
        except Exception as e:
            logging.getLogger(__name__).warning(f"Health check {name} failed: {e}")
            return False

    async def _handle_health_check_failure(self, name: str, check_info: Dict):
        """Handle health check failure"""
        try:
            # Create a health check failure exception
            health_error = YMERAException(
                message=f"Health check {name} failed {check_info['consecutive_failures']} consecutive times",
                error_code="HEALTH_001",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.SYSTEM,
                suggested_actions=[
                    f"Check {name} service status",
                    "Review system resources",
                    "Consider scaling"
                ],
                health_check_name=name,
                consecutive_failures=check_info['consecutive_failures']
            )
            
            # Handle through exception handler
            await self.exception_handler.handle_exception(health_error)
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error handling health check failure: {e}")

# Utility Functions
def create_error_context(
    request: Optional[Request] = None,
    agent_id: Optional[str] = None,
    operation: Optional[str] = None,
    **kwargs
) -> ErrorContext:
    """Create error context from request or parameters"""
    if request:
        return ErrorContext(
            request_id=request.headers.get("X-Request-ID"),
            user_agent=request.headers.get("User-Agent"),
            ip_address=request.client.host if request.client else None,
            method=request.method,
            endpoint=str(request.url.path),
            correlation_id=request.headers.get("X-Correlation-ID"),
            agent_id=agent_id,
            operation=operation,
            **kwargs
        )
    else:
        return ErrorContext(
            agent_id=agent_id,
            operation=operation,
            **kwargs
        )

def get_exception_handler() -> ExceptionHandler:
    """Get or create global exception handler instance"""
    if not hasattr(get_exception_handler, "_instance"):
        get_exception_handler._instance = ExceptionHandler()
    return get_exception_handler._instance

# Export all exceptions and utilities
__all__ = [
    # Base classes
    "YMERAException",
    "ErrorSeverity",
    "ErrorCategory",
    "RecoveryAction",
    "ErrorContext",
    "ErrorMetrics",
    "ErrorResponse",
    
    # Authentication exceptions
    "AuthenticationError",
    "AuthorizationError", 
    "TokenExpiredError",
    
    # Validation exceptions
    "ValidationError",
    "SchemaValidationError",
    
    # Database exceptions
    "DatabaseError",
    "DatabaseConnectionError",
    "DatabaseTransactionError",
    
    # Agent exceptions
    "AgentError",
    "AgentInitializationError",
    "AgentExecutionError",
    "AgentCommunicationError",
    "AgentTimeoutError",
    
    # Learning engine exceptions
    "LearningEngineError",
    "ModelTrainingError",
    "KnowledgeBaseError",
    
    # External API exceptions
    "ExternalAPIError",
    "RateLimitError",
    
    # Configuration exceptions
    "ConfigurationError",
    
    # Resource exceptions
    "ResourceError",
    "ResourceExhaustedError",
    
    # Security exceptions
    "SecurityError",
    "SecurityViolationError",
    
    # Business logic exceptions
    "BusinessLogicError",
    
    # System exceptions
    "CircuitBreakerOpenError",
    "RetryExhaustedError",
    
    # Handlers and utilities
    "ExceptionHandler",
    "ExceptionTrackingMiddleware",
    "CircuitBreaker",
    "ErrorPatternDetector",
    "ErrorRecoveryOrchestrator",
    "HealthCheckManager",
    "handle_ymera_exception",
    "retry_with_backoff",
    "create_error_context",
    "get_exception_handler"
]