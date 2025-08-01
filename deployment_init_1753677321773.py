"""
YMERA Enterprise - Deployment Package
Production-Ready Deployment Configuration Package - v4.0
Enterprise-grade implementation with zero placeholders
"""

# ===============================================================================
# STANDARD IMPORTS SECTION
# ===============================================================================

import asyncio
import logging
import os
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path

# Third-party imports
import structlog
from pydantic import BaseSettings, Field

# Local imports
from config.settings import get_settings

# ===============================================================================
# LOGGING CONFIGURATION
# ===============================================================================

logger = structlog.get_logger("ymera.deployment")

# ===============================================================================
# CONSTANTS & CONFIGURATION
# ===============================================================================

DEPLOYMENT_VERSION = "4.0"
SUPPORTED_PLATFORMS = ["replit", "docker", "production", "local"]

# ===============================================================================
# DEPLOYMENT CONFIGURATION
# ===============================================================================

@dataclass
class DeploymentConfig:
    """Base deployment configuration"""
    platform: str
    environment: str
    version: str = DEPLOYMENT_VERSION
    debug: bool = False
    auto_reload: bool = False
    workers: int = 1
    port: int = 8000
    host: str = "0.0.0.0"

class DeploymentSettings(BaseSettings):
    """Global deployment settings"""
    
    # Platform identification
    platform: str = Field(default="production", description="Deployment platform")
    environment: str = Field(default="production", description="Environment type")
    
    # Server configuration
    server_host: str = Field(default="0.0.0.0", description="Server bind host")
    server_port: int = Field(default=8000, ge=1000, le=65535, description="Server port")
    server_workers: int = Field(default=1, ge=1, le=16, description="Worker processes")
    
    # Feature flags
    debug_mode: bool = Field(default=False, description="Enable debug mode")
    auto_reload: bool = Field(default=False, description="Enable auto-reload")
    profiling_enabled: bool = Field(default=False, description="Enable profiling")
    
    # Resource limits
    max_memory_mb: int = Field(default=512, ge=128, le=8192, description="Memory limit MB")
    max_cpu_percent: int = Field(default=80, ge=10, le=100, description="CPU usage limit")
    
    # Security settings
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")
    api_rate_limit: int = Field(default=1000, ge=1, description="API rate limit per hour")
    
    class Config:
        env_prefix = "YMERA_DEPLOY_"
        case_sensitive = False

# ===============================================================================
# DEPLOYMENT DETECTION & UTILITIES
# ===============================================================================

def detect_deployment_platform() -> str:
    """
    Automatically detect the deployment platform based on environment variables
    and system characteristics.
    
    Returns:
        str: Detected platform name ('replit', 'docker', 'production', 'local')
    """
    try:
        # Replit detection
        if os.getenv("REPL_ID") or os.getenv("REPLIT_DB_URL"):
            logger.info("Detected Replit deployment platform")
            return "replit"
        
        # Docker detection
        if (os.path.exists("/.dockerenv") or 
            os.getenv("DOCKER_CONTAINER") or 
            os.path.exists("/proc/1/cgroup") and "docker" in open("/proc/1/cgroup").read()):
            logger.info("Detected Docker deployment platform")
            return "docker"
        
        # Production detection (common production indicators)
        if (os.getenv("PRODUCTION") or 
            os.getenv("NODE_ENV") == "production" or
            os.getenv("ENVIRONMENT") == "production"):
            logger.info("Detected production deployment platform")
            return "production"
        
        # Default to local development
        logger.info("Detected local development platform")
        return "local"
        
    except Exception as e:
        logger.warning("Failed to detect platform, defaulting to local", error=str(e))
        return "local"

def get_deployment_config(platform: Optional[str] = None) -> DeploymentConfig:
    """
    Get deployment configuration for specified or auto-detected platform.
    
    Args:
        platform: Target platform name, auto-detected if None
        
    Returns:
        DeploymentConfig: Platform-specific configuration
        
    Raises:
        ValueError: If platform is not supported
    """
    if platform is None:
        platform = detect_deployment_platform()
    
    if platform not in SUPPORTED_PLATFORMS:
        raise ValueError(f"Unsupported platform: {platform}. Supported: {SUPPORTED_PLATFORMS}")
    
    settings = DeploymentSettings()
    
    # Platform-specific defaults
    config_overrides = {
        "replit": {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 1,
            "debug": True,
            "auto_reload": True
        },
        "docker": {
            "host": "0.0.0.0",
            "port": int(os.getenv("PORT", 8000)),
            "workers": int(os.getenv("WORKERS", 1)),
            "debug": False,
            "auto_reload": False
        },
        "production": {
            "host": "0.0.0.0",
            "port": int(os.getenv("PORT", 8000)),
            "workers": int(os.getenv("WORKERS", 4)),
            "debug": False,
            "auto_reload": False
        },
        "local": {
            "host": "127.0.0.1",
            "port": 8000,
            "workers": 1,
            "debug": True,
            "auto_reload": True
        }
    }
    
    overrides = config_overrides.get(platform, {})
    environment = "development" if platform in ["local", "replit"] else "production"
    
    config = DeploymentConfig(
        platform=platform,
        environment=environment,
        host=overrides.get("host", settings.server_host),
        port=overrides.get("port", settings.server_port),
        workers=overrides.get("workers", settings.server_workers),
        debug=overrides.get("debug", settings.debug_mode),
        auto_reload=overrides.get("auto_reload", settings.auto_reload)
    )
    
    logger.info("Deployment configuration created", 
                platform=platform, 
                environment=environment,
                host=config.host,
                port=config.port,
                workers=config.workers)
    
    return config

def validate_deployment_environment() -> Dict[str, Any]:
    """
    Validate the current deployment environment and return status information.
    
    Returns:
        Dict[str, Any]: Environment validation results
    """
    validation_results = {
        "platform": detect_deployment_platform(),
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "environment_variables": {},
        "system_resources": {},
        "validation_passed": True,
        "warnings": [],
        "errors": []
    }
    
    try:
        # Check critical environment variables
        critical_env_vars = ["DATABASE_URL", "REDIS_URL", "SECRET_KEY"]
        for var in critical_env_vars:
            value = os.getenv(var)
            validation_results["environment_variables"][var] = "SET" if value else "MISSING"
            if not value:
                validation_results["warnings"].append(f"Missing environment variable: {var}")
        
        # Check system resources
        import psutil
        validation_results["system_resources"] = {
            "memory_mb": round(psutil.virtual_memory().total / (1024 * 1024)),
            "cpu_count": psutil.cpu_count(),
            "disk_free_gb": round(psutil.disk_usage('/').free / (1024**3), 2)
        }
        
        # Platform-specific validations
        platform = validation_results["platform"]
        if platform == "replit":
            if not os.getenv("REPL_ID"):
                validation_results["warnings"].append("REPL_ID not found, might not be running on Replit")
        
        elif platform == "docker":
            if not os.path.exists("/.dockerenv"):
                validation_results["warnings"].append("Docker environment file not found")
        
        # Check if any errors occurred
        if validation_results["errors"]:
            validation_results["validation_passed"] = False
        
        logger.info("Environment validation completed", 
                   platform=platform,
                   passed=validation_results["validation_passed"],
                   warnings_count=len(validation_results["warnings"]),
                   errors_count=len(validation_results["errors"]))
        
    except Exception as e:
        validation_results["errors"].append(f"Validation error: {str(e)}")
        validation_results["validation_passed"] = False
        logger.error("Environment validation failed", error=str(e))
    
    return validation_results

# ===============================================================================
# HEALTH CHECK UTILITIES
# ===============================================================================

async def deployment_health_check() -> Dict[str, Any]:
    """
    Comprehensive deployment health check for monitoring systems.
    
    Returns:
        Dict[str, Any]: Health check results
    """
    health_status = {
        "status": "healthy",
        "timestamp": asyncio.get_event_loop().time(),
        "deployment": {
            "platform": detect_deployment_platform(),
            "version": DEPLOYMENT_VERSION,
            "environment": "production" if detect_deployment_platform() in ["production", "docker"] else "development"
        },
        "checks": {},
        "metrics": {}
    }
    
    try:
        # System resource checks
        import psutil
        
        # Memory check
        memory = psutil.virtual_memory()
        memory_usage_percent = memory.percent
        health_status["checks"]["memory"] = "healthy" if memory_usage_percent < 80 else "warning"
        health_status["metrics"]["memory_usage_percent"] = memory_usage_percent
        
        # CPU check
        cpu_usage = psutil.cpu_percent(interval=1)
        health_status["checks"]["cpu"] = "healthy" if cpu_usage < 80 else "warning"
        health_status["metrics"]["cpu_usage_percent"] = cpu_usage
        
        # Disk space check
        disk = psutil.disk_usage('/')
        disk_usage_percent = (disk.used / disk.total) * 100
        health_status["checks"]["disk"] = "healthy" if disk_usage_percent < 80 else "warning"
        health_status["metrics"]["disk_usage_percent"] = disk_usage_percent
        
        # Process check
        current_process = psutil.Process()
        health_status["metrics"]["process_memory_mb"] = round(current_process.memory_info().rss / (1024 * 1024), 2)
        health_status["metrics"]["process_cpu_percent"] = current_process.cpu_percent()
        
        # Overall status determination
        warning_checks = [check for check in health_status["checks"].values() if check == "warning"]
        if warning_checks:
            health_status["status"] = "warning"
        
        logger.debug("Deployment health check completed", status=health_status["status"])
        
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["checks"]["system"] = f"error: {str(e)}"
        logger.error("Health check failed", error=str(e))
    
    return health_status

# ===============================================================================
# STARTUP UTILITIES
# ===============================================================================

def get_startup_banner(config: DeploymentConfig) -> str:
    """
    Generate startup banner with deployment information.
    
    Args:
        config: Deployment configuration
        
    Returns:
        str: Formatted startup banner
    """
    banner = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          YMERA ENTERPRISE PLATFORM                          ║
║                        Production-Ready Deployment v{DEPLOYMENT_VERSION}                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Platform: {config.platform.upper():<20} Environment: {config.environment.upper():<20} ║
║ Host: {config.host:<25} Port: {config.port:<25} ║
║ Workers: {config.workers:<23} Debug: {str(config.debug):<25} ║
║ Auto-reload: {str(config.auto_reload):<17} Version: {config.version:<25} ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    return banner

# ===============================================================================
# EXPORTS
# ===============================================================================

__all__ = [
    "DeploymentConfig",
    "DeploymentSettings", 
    "detect_deployment_platform",
    "get_deployment_config",
    "validate_deployment_environment",
    "deployment_health_check",
    "get_startup_banner",
    "DEPLOYMENT_VERSION",
    "SUPPORTED_PLATFORMS"
]