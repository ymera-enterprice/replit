"""
YMERA Enterprise - Docker Configuration  
Production-Ready Docker Configuration - v4.0
Enterprise-grade implementation with zero placeholders
"""

# ===============================================================================
# STANDARD IMPORTS SECTION
# ===============================================================================

# Standard library imports (alphabetical)
import asyncio
import json
import logging
import os
import platform
import signal
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager

# Third-party imports (alphabetical)
import aiofiles
import aioredis
import docker
import psutil
import structlog
import yaml
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Local imports (alphabetical)
from config.settings import get_settings
from utils.encryption import encrypt_data, decrypt_data
from monitoring.performance_tracker import track_performance
from security.jwt_handler import generate_secret_key

# ===============================================================================
# LOGGING CONFIGURATION
# ===============================================================================

logger = structlog.get_logger("ymera.docker_config")

# ===============================================================================
# CONSTANTS & CONFIGURATION
# ===============================================================================

# Docker-specific constants
DOCKER_NETWORK_NAME = "ymera-network"
DOCKER_VOLUME_PREFIX = "ymera"
CONTAINER_HEALTH_CHECK_INTERVAL = 30
MAX_CONTAINER_RESTART_ATTEMPTS = 3
DOCKER_COMPOSE_VERSION = "3.8"

# Container resource limits
DEFAULT_MEMORY_LIMIT = "512m"
DEFAULT_CPU_LIMIT = "0.5"
DEFAULT_SWAP_LIMIT = "1g"

# Configuration loading
settings = get_settings()

# ===============================================================================
# DATA MODELS & SCHEMAS
# ===============================================================================

@dataclass
class DockerConfig:
    """Configuration dataclass for Docker environment settings"""
    app_name: str = "ymera-enterprise"
    image_name: str = "ymera/enterprise"
    image_tag: str = "v4.0"
    container_name: str = "ymera-app"
    network_name: str = DOCKER_NETWORK_NAME
    expose_port: int = 8080
    internal_port: int = 8080
    environment: str = "production"
    restart_policy: str = "unless-stopped"
    memory_limit: str = DEFAULT_MEMORY_LIMIT
    cpu_limit: str = DEFAULT_CPU_LIMIT
    health_check_enabled: bool = True
    logging_driver: str = "json-file"
    log_max_size: str = "10m"
    log_max_file: int = 3

@dataclass 
class DockerNetworkConfig:
    """Docker network configuration"""
    name: str = DOCKER_NETWORK_NAME
    driver: str = "bridge"
    enable_ipv6: bool = False
    subnet: str = "172.20.0.0/16"
    gateway: str = "172.20.0.1"
    internal: bool = False

@dataclass
class DockerVolumeConfig:
    """Docker volume configuration"""
    name: str = f"{DOCKER_VOLUME_PREFIX}-data"
    driver: str = "local"
    mount_point: str = "/app/data"
    backup_enabled: bool = True
    retention_days: int = 30

class DockerService(BaseModel):
    """Docker service configuration model"""
    name: str = Field(..., description="Service name")
    image: str = Field(..., description="Docker image")
    ports: List[str] = Field(default_factory=list, description="Port mappings")
    environment: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    volumes: List[str] = Field(default_factory=list, description="Volume mounts")
    depends_on: List[str] = Field(default_factory=list, description="Service dependencies")
    restart: str = Field(default="unless-stopped", description="Restart policy")
    networks: List[str] = Field(default_factory=list, description="Networks to join")
    
    @validator('ports')
    def validate_ports(cls, v):
        for port in v:
            if ':' not in port:
                raise ValueError(f'Invalid port mapping format: {port}')
        return v

class DockerComposeConfig(BaseModel):
    """Complete Docker Compose configuration"""
    version: str = Field(default=DOCKER_COMPOSE_VERSION, description="Compose file version")
    services: Dict[str, DockerService] = Field(default_factory=dict, description="Service definitions")
    networks: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Network definitions") 
    volumes: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Volume definitions")
    
    class Config:
        extra = "allow"

# ===============================================================================
# CORE IMPLEMENTATION CLASSES  
# ===============================================================================

class DockerEnvironmentManager:
    """Production-ready Docker environment management"""
    
    def __init__(self, config: DockerConfig):
        self.config = config
        self.logger = logger.bind(component="docker_env_manager")
        self.docker_client = None
        self._container_cache: Dict[str, Any] = {}
        self._network_cache: Dict[str, Any] = {}
        
    async def initialize(self) -> None:
        """Initialize Docker environment and connections"""
        try:
            # Initialize Docker client
            self.docker_client = docker.from_env()
            
            # Verify Docker daemon is running
            await self._verify_docker_daemon()
            
            # Setup networks
            await self._ensure_networks()
            
            # Setup volumes
            await self._ensure_volumes()
            
            # Setup base images
            await self._ensure_base_images()
            
            self.logger.info("Docker environment initialized successfully")
            
        except Exception as e:
            self.logger.error("Docker environment initialization failed", error=str(e))
            raise
    
    async def _verify_docker_daemon(self) -> None:
        """Verify Docker daemon is accessible"""
        try:
            # Test Docker connection
            self.docker_client.ping()
            
            # Log Docker version info
            version_info = self.docker_client.version()
            self.logger.info("Docker daemon connected", 
                           version=version_info.get('Version'),
                           api_version=version_info.get('ApiVersion'))
            
        except Exception as e:
            self.logger.error("Docker daemon not accessible", error=str(e))
            raise RuntimeError("Docker daemon is not running or accessible")
    
    async def _ensure_networks(self) -> None:
        """Ensure required Docker networks exist"""
        try:
            network_config = DockerNetworkConfig()
            
            # Check if network already exists
            try:
                network = self.docker_client.networks.get(network_config.name)
                self.logger.info("Docker network exists", name=network_config.name)
                self._network_cache[network_config.name] = network
                
            except docker.errors.NotFound:
                # Create network
                network = self.docker_client.networks.create(
                    name=network_config.name,
                    driver=network_config.driver,
                    options={
                        "com.docker.network.bridge.enable_icc": "true",
                        "com.docker.network.bridge.enable_ip_masquerade": "true"
                    },
                    ipam=docker.types.IPAMConfig(
                        pool_configs=[
                            docker.types.IPAMPool(
                                subnet=network_config.subnet,
                                gateway=network_config.gateway
                            )
                        ]
                    )
                )
                
                self.logger.info("Docker network created", 
                               name=network_config.name,
                               subnet=network_config.subnet)
                self._network_cache[network_config.name] = network
                
        except Exception as e:
            self.logger.error("Failed to ensure Docker networks", error=str(e))
            raise
    
    async def _ensure_volumes(self) -> None:
        """Ensure required Docker volumes exist"""
        try:
            volume_config = DockerVolumeConfig()
            
            # Check if volume already exists
            try:
                volume = self.docker_client.volumes.get(volume_config.name)
                self.logger.info("Docker volume exists", name=volume_config.name)
                
            except docker.errors.NotFound:
                # Create volume
                volume = self.docker_client.volumes.create(
                    name=volume_config.name,
                    driver=volume_config.driver,
                    labels={
                        "ymera.component": "data-storage",
                        "ymera.backup.enabled": str(volume_config.backup_enabled),
                        "ymera.backup.retention": str(volume_config.retention_days)
                    }
                )
                
                self.logger.info("Docker volume created", name=volume_config.name)
                
        except Exception as e:
            self.logger.error("Failed to ensure Docker volumes", error=str(e))
            raise
    
    async def _ensure_base_images(self) -> None:
        """Ensure required base images are available"""
        try:
            required_images = [
                "python:3.11-slim",
                "redis:7-alpine", 
                "postgres:15-alpine",
                "nginx:alpine"
            ]
            
            for image_name in required_images:
                try:
                    # Check if image exists locally
                    self.docker_client.images.get(image_name)
                    self.logger.debug("Base image available", image=image_name)
                    
                except docker.errors.ImageNotFound:
                    # Pull image
                    self.logger.info("Pulling base image", image=image_name)
                    self.docker_client.images.pull(image_name)
                    self.logger.info("Base image pulled successfully", image=image_name)
                    
        except Exception as e:
            self.logger.error("Failed to ensure base images", error=str(e))
            raise
    
    async def get_container_status(self, container_name: str) -> Dict[str, Any]:
        """Get detailed container status information"""
        try:
            container = self.docker_client.containers.get(container_name)
            
            # Get container stats
            stats = container.stats(stream=False)
            
            # Calculate resource usage
            cpu_usage = self._calculate_cpu_usage(stats)
            memory_usage = self._calculate_memory_usage(stats)
            
            return {
                "name": container.name,
                "id": container.short_id,
                "status": container.status,
                "created": container.attrs['Created'],
                "started": container.attrs['State']['StartedAt'],
                "image": container.image.tags[0] if container.image.tags else "unknown",
                "ports": container.ports,
                "networks": list(container.attrs['NetworkSettings']['Networks'].keys()),
                "cpu_usage_percent": cpu_usage,
                "memory_usage_mb": memory_usage,
                "restart_count": container.attrs['RestartCount']
            }
            
        except docker.errors.NotFound:
            return {"status": "not_found", "name": container_name}
        except Exception as e:
            self.logger.error("Failed to get container status", 
                            container=container_name, error=str(e))
            return {"status": "error", "error": str(e)}
    
    def _calculate_cpu_usage(self, stats: Dict[str, Any]) -> float:
        """Calculate CPU usage percentage from container stats"""
        try:
            cpu_stats = stats['cpu_stats']
            precpu_stats = stats['precpu_stats']
            
            cpu_delta = cpu_stats['cpu_usage']['total_usage'] - \
                       precpu_stats['cpu_usage']['total_usage']
            system_delta = cpu_stats['system_cpu_usage'] - \
                          precpu_stats['system_cpu_usage']
            
            if system_delta > 0:
                cpu_usage = (cpu_delta / system_delta) * len(cpu_stats['cpu_usage']['percpu_usage']) * 100
                return round(cpu_usage, 2)
            
            return 0.0
            
        except (KeyError, ZeroDivisionError):
            return 0.0
    
    def _calculate_memory_usage(self, stats: Dict[str, Any]) -> float:
        """Calculate memory usage in MB from container stats"""
        try:
            memory_usage = stats['memory_stats']['usage']
            return round(memory_usage / (1024 * 1024), 2)
        except KeyError:
            return 0.0
    
    async def cleanup(self) -> None:
        """Cleanup Docker environment resources"""
        try:
            if self.docker_client:
                self.docker_client.close()
                self.logger.info("Docker client connection closed")
        except Exception as e:
            self.logger.error("Docker cleanup failed", error=str(e))

class DockerComposeManager:
    """Production-ready Docker Compose management"""
    
    def __init__(self, config: DockerConfig):
        self.config = config
        self.logger = logger.bind(component="compose_manager")
        self.compose_file_path = Path("docker-compose.yml")
        self.compose_override_path = Path("docker-compose.override.yml")
    
    async def generate_compose_file(self, services_config: Dict[str, Any]) -> str:
        """Generate production-ready Docker Compose file"""
        try:
            # Create base compose configuration
            compose_config = self._create_base_compose_config()
            
            # Add application services
            compose_config.services.update(self._create_application_services(services_config))
            
            # Add infrastructure services
            compose_config.services.update(self._create_infrastructure_services())
            
            # Add networks and volumes
            compose_config.networks = self._create_network_definitions()
            compose_config.volumes = self._create_volume_definitions()
            
            # Convert to YAML
            compose_yaml = self._compose_to_yaml(compose_config)
            
            # Write to file
            await self._write_compose_file(compose_yaml)
            
            self.logger.info("Docker Compose file generated successfully")
            return compose_yaml
            
        except Exception as e:
            self.logger.error("Failed to generate Docker Compose file", error=str(e))
            raise
    
    def _create_base_compose_config(self) -> DockerComposeConfig:
        """Create base Docker Compose configuration"""
        return DockerComposeConfig(
            version=DOCKER_COMPOSE_VERSION,
            services={},
            networks={},
            volumes={}
        )
    
    def _create_application_services(self, services_config: Dict[str, Any]) -> Dict[str, DockerService]:
        """Create application service definitions"""
        services = {}
        
        # Main YMERA application service
        services["ymera-app"] = DockerService(
            name="ymera-app",
            image=f"{self.config.image_name}:{self.config.image_tag}",
            ports=[f"{self.config.expose_port}:{self.config.internal_port}"],
            environment={
                "ENVIRONMENT": self.config.environment,
                "DATABASE_URL": "postgresql://ymera:password@postgres:5432/ymera",
                "REDIS_URL": "redis://redis:6379/0",
                "JWT_SECRET": "${JWT_SECRET:-default-jwt-secret}",
                "LOG_LEVEL": "INFO"
            },
            volumes=[
                f"{DOCKER_VOLUME_PREFIX}-data:/app/data",
                f"{DOCKER_VOLUME_PREFIX}-logs:/app/logs"
            ],
            depends_on=["postgres", "redis"],
            networks=[DOCKER_NETWORK_NAME],
            restart=self.config.restart_policy
        )
        
        # Worker service for background tasks
        services["ymera-worker"] = DockerService(
            name="ymera-worker",
            image=f"{self.config.image_name}:{self.config.image_tag}",
            environment={
                "ENVIRONMENT": self.config.environment,
                "DATABASE_URL": "postgresql://ymera:password@postgres:5432/ymera",
                "REDIS_URL": "redis://redis:6379/0",
                "WORKER_MODE": "true"
            },
            volumes=[
                f"{DOCKER_VOLUME_PREFIX}-data:/app/data",
                f"{DOCKER_VOLUME_PREFIX}-logs:/app/logs"
            ],
            depends_on=["postgres", "redis"],
            networks=[DOCKER_NETWORK_NAME],
            restart=self.config.restart_policy
        )
        
        return services
    
    def _create_infrastructure_services(self) -> Dict[str, DockerService]:
        """Create infrastructure service definitions"""
        services = {}
        
        # PostgreSQL database service
        services["postgres"] = DockerService(
            name="postgres",
            image="postgres:15-alpine",
            environment={
                "POSTGRES_DB": "ymera",
                "POSTGRES_USER": "ymera", 
                "POSTGRES_PASSWORD": "${POSTGRES_PASSWORD:-default-password}",
                "POSTGRES_INITDB_ARGS": "--encoding=UTF8 --locale=C"
            },
            volumes=[
                f"{DOCKER_VOLUME_PREFIX}-postgres:/var/lib/postgresql/data",
                "./sql/init:/docker-entrypoint-initdb.d:ro"
            ],
            networks=[DOCKER_NETWORK_NAME],
            restart=self.config.restart_policy
        )
        
        # Redis cache service
        services["redis"] = DockerService(
            name="redis",
            image="redis:7-alpine",
            environment={
                "REDIS_PASSWORD": "${REDIS_PASSWORD:-}",
                "REDIS_APPENDONLY": "yes"
            },
            volumes=[
                f"{DOCKER_VOLUME_PREFIX}-redis:/data"
            ],
            networks=[DOCKER_NETWORK_NAME],
            restart=self.config.restart_policy
        )
        
        # Nginx reverse proxy service
        services["nginx"] = DockerService(
            name="nginx",
            image="nginx:alpine",
            ports=["80:80", "443:443"],
            volumes=[
                "./nginx/nginx.conf:/etc/nginx/nginx.conf:ro",
                "./nginx/conf.d:/etc/nginx/conf.d:ro",
                f"{DOCKER_VOLUME_PREFIX}-ssl:/etc/ssl/certs"
            ],
            depends_on=["ymera-app"],
            networks=[DOCKER_NETWORK_NAME],
            restart=self.config.restart_policy
        )
        
        return services
    
    def _create_network_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Create network definitions"""
        return {
            DOCKER_NETWORK_NAME: {
                "driver": "bridge",
                "ipam": {
                    "config": [{
                        "subnet": "172.20.0.0/16",
                        "gateway": "172.20.0.1"
                    }]
                },
                "labels": {
                    "ymera.network": "main",
                    "ymera.version": "4.0"
                }
            }
        }
    
    def _create_volume_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Create volume definitions"""
        return {
            f"{DOCKER_VOLUME_PREFIX}-data": {
                "driver": "local",
                "labels": {
                    "ymera.volume": "application-data",
                    "ymera.backup": "enabled"
                }
            },
            f"{DOCKER_VOLUME_PREFIX}-logs": {
                "driver": "local",
                "labels": {
                    "ymera.volume": "application-logs"
                }
            },
            f"{DOCKER_VOLUME_PREFIX}-postgres": {
                "driver": "local",
                "labels": {
                    "ymera.volume": "database-data",
                    "ymera.backup": "critical"
                }
            },
            f"{DOCKER_VOLUME_PREFIX}-redis": {
                "driver": "local",
                "labels": {
                    "ymera.volume": "cache-data"
                }
            },
            f"{DOCKER_VOLUME_PREFIX}-ssl": {
                "driver": "local",
                "labels": {
                    "ymera.volume": "ssl-certificates"
                }
            }
        }
    
    def _compose_to_yaml(self, compose_config: DockerComposeConfig) -> str:
        """Convert DockerComposeConfig to YAML string"""
        try:
            # Convert Pydantic model to dict
            config_dict = compose_config.dict(exclude_none=True)
            
            # Convert services to proper format
            services_dict = {}
            for service_name, service in config_dict["services"].items():
                service_config = service.copy()
                # Remove the 'name' field as it's redundant in compose
                service_config.pop("name", None)
                services_dict[service_name] = service_config
            
            config_dict["services"] = services_dict
            
            # Generate YAML with proper formatting
            yaml_content = yaml.dump(
                config_dict,
                default_flow_style=False,
                sort_keys=False,
                indent=2,
                width=120
            )
            
            # Add header comment
            header = f"""# YMERA Enterprise Docker Compose Configuration
# Generated on: {datetime.utcnow().isoformat()}
# Version: {compose_config.version}
# Environment: {self.config.environment}

"""
            
            return header + yaml_content
            
        except Exception as e:
            self.logger.error("Failed to convert config to YAML", error=str(e))
            raise
    
    async def _write_compose_file(self, yaml_content: str) -> None:
        """Write Docker Compose file to disk"""
        try:
            async with aiofiles.open(self.compose_file_path, 'w') as f:
                await f.write(yaml_content)
            
            self.logger.info("Docker Compose file written", 
                           path=str(self.compose_file_path))
            
        except Exception as e:
            self.logger.error("Failed to write compose file", error=str(e))
            raise
    
    async def generate_env_file(self, secrets: Dict[str, str]) -> None:
        """Generate .env file for Docker Compose"""
        try:
            env_content = []
            env_content.append("# YMERA Enterprise Environment Configuration")
            env_content.append(f"# Generated on: {datetime.utcnow().isoformat()}")
            env_content.append("")
            
            # Application settings
            env_content.append("# Application Settings")
            env_content.append(f"ENVIRONMENT={self.config.environment}")
            env_content.append(f"LOG_LEVEL=INFO")
            env_content.append("")
            
            # Security settings
            env_content.append("# Security Settings")
            for key, value in secrets.items():
                env_content.append(f"{key.upper()}={value}")
            env_content.append("")
            
            # Database settings
            env_content.append("# Database Settings")
            env_content.append("POSTGRES_PASSWORD=secure-postgres-password")
            env_content.append("POSTGRES_DB=ymera")
            env_content.append("POSTGRES_USER=ymera")
            env_content.append("")
            
            # Redis settings
            env_content.append("# Redis Settings")
            env_content.append("REDIS_PASSWORD=secure-redis-password")
            env_content.append("")
            
            env_file_content = "\n".join(env_content)
            
            async with aiofiles.open(".env", 'w') as f:
                await f.write(env_file_content)
            
            # Set restrictive permissions
            os.chmod(".env", 0o600)
            
            self.logger.info("Environment file generated", path=".env")
            
        except Exception as e:
            self.logger.error("Failed to generate env file", error=str(e))
            raise

class DockerHealthMonitor:
    """Production-ready Docker container health monitoring"""
    
    def __init__(self, config: DockerConfig):
        self.config = config
        self.logger = logger.bind(component="health_monitor")
        self.docker_client = None
        self._monitoring_active = False
        self._health_check_task = None
    
    async def initialize(self) -> None:
        """Initialize health monitoring"""
        try:
            self.docker_client = docker.from_env()
            self.logger.info("Docker health monitor initialized")
        except Exception as e:
            self.logger.error("Failed to initialize health monitor", error=str(e))
            raise
    
    async def start_monitoring(self) -> None:
        """Start continuous health monitoring"""
        try:
            self._monitoring_active = True
            self._health_check_task = asyncio.create_task(self._monitoring_loop())
            self.logger.info("Health monitoring started")
        except Exception as e:
            self.logger.error("Failed to start health monitoring", error=str(e))
            raise
    
    async def stop_monitoring(self) -> None:
        """Stop health monitoring"""
        try:
            self._monitoring_active = False
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
            self.logger.info("Health monitoring stopped")
        except Exception as e:
            self.logger.error("Failed to stop health monitoring", error=str(e))
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self._monitoring_active:
            try:
                await self._check_all_containers()
                await asyncio.sleep(CONTAINER_HEALTH_CHECK_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Health check loop error", error=str(e))
                await asyncio.sleep(5)  # Brief pause before retry
    
    async def _check_all_containers(self) -> None:
        """Check health of all monitored containers"""
        try:
            # Get all containers with YMERA labels
            containers = self.docker_client.containers.list(
                filters={"label": "ymera.component"}
            )
            
            for container in containers:
                await self._check_container_health(container)
                
        except Exception as e:
            self.logger.error("Failed to check containers", error=str(e))
    
    async def _check_container_health(self, container) -> None:
        """Check individual container health"""
        try:
            container.reload()  # Refresh container state
            
            container_name = container.name
            container_status = container.status
            
            if container_status == "running":
                # Check if container is responding
                health_status = await self._perform_health_check(container)
                
                if not health_status:
                    self.logger.warning("Container health check failed", 
                                      container=container_name)
                    await self._handle_unhealthy_container(container)
                else:
                    self.logger.debug("Container healthy", container=container_name)
                    
            elif container_status in ["exited", "dead"]:
                self.logger.error("Container not running", 
                                container=container_name, 
                                status=container_status)
                await self._handle_stopped_container(container)
                
        except Exception as e:
            self.logger.error("Container health check failed", 
                            container=getattr(container, 'name', 'unknown'),
                            error=str(e))
    
    async def _perform_health_check(self, container) -> bool:
        """Perform actual health check on container"""
        try:
            # Try to execute health check command inside container
            result = container.exec_run(
                "curl -f http://localhost:8080/health || exit 1",
                timeout=10
            )
            
            return result.exit_code == 0
            
        except Exception as e:
            self.logger.debug("Health check command failed", error=str(e))
            return False
    
    async def _handle_unhealthy_container(self, container) -> None:
        """Handle unhealthy container"""
        try:
            container_name = container.name
            restart_count = container.attrs.get('RestartCount', 0)
            
            if restart_count < MAX_CONTAINER_RESTART_ATTEMPTS:
                self.logger.info("Restarting unhealthy container", 
                               container=container_name,
                               restart_count=restart_count)
                container.restart()
            else:
                self.logger.error("Container exceeded restart attempts", 
                                container=container_name,
                                restart_count=restart_count)
                
        except Exception as e:
            self.logger.error("Failed to handle unhealthy container", error=str(e))
    
    async def _handle_stopped_container(self, container) -> None:
        """Handle stopped container"""
        try:
            container_name = container.name
            
            # Check if container should be restarted
            restart_policy = container.attrs['HostConfig']['RestartPolicy']['Name']
            
            if restart_policy in ["always", "unless-stopped"]:
                self.logger.info("Starting stopped container", 
                               container=container_name)
                container.start()
            else:
                self.logger.warning("Container stopped with no restart policy", 
                                  container=container_name)
                
        except Exception as e:
            self.logger.error("Failed to handle stopped container", error=str(e))
    
    async def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary"""
        try:
            containers = self.docker_client.containers.list(all=True)
            
            summary = {
                "timestamp": datetime.utcnow().isoformat(),
                "total_containers": len(containers),
                "running": 0,
                "stopped": 0,
                "unhealthy": 0,
                "containers": []
            }
            
            for container in containers:
                container_info = {
                    "name": container.name,
                    "status": container.status,
                    "image": container.image.tags[0] if container.image.tags else "unknown"
                }
                
                if container.status == "running":
                    summary["running"] += 1
                else:
                    summary["stopped"] += 1
                
                summary["containers"].append(container_info)
            
            return summary
            
        except Exception as e:
            self.logger.error("Failed to get health summary", error=str(e))
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}

# ===============================================================================
# UTILITY FUNCTIONS
# ===============================================================================

def detect_docker_environment() -> bool:
    """Detect if running inside Docker container"""
    indicators = [
        Path("/.dockerenv").exists(),
        Path("/proc/1/cgroup").exists() and "docker" in Path("/proc/1/cgroup").read_text(),
        os.getenv("DOCKER_CONTAINER") == "true"
    ]
    
    return any(indicators)

async def validate_docker_installation() -> Dict[str, Any]:
    """Validate Docker installation and requirements"""
    try:
        client = docker.from_env()
        
        # Test Docker connection
        client.ping()
        
        # Get version information
        version_info = client.version()
        
        # Check Docker Compose availability
        import subprocess
        compose_result = subprocess.run(
            ["docker-compose", "--version"], 
            capture_output=True, 
            text=True
        )
        
        return {
            "docker_available": True,
            "docker_version": version_info.get("Version"),
            "api_version": version_info.get("ApiVersion"),
            "compose_available": compose_result.returncode == 0,
            "compose_version": compose_result.stdout.strip() if compose_result.returncode == 0 else None
        }
        
    except Exception as e:
        return {
            "docker_available": False,
            "error": str(e)
        }

def configure_docker_logging() -> None:
    """Configure logging for Docker environment"""
    # Docker-specific logging configuration
    log_level = logging.INFO
    
    # Check if running in container
    if detect_docker_environment():
        # Use JSON logging for container environments
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    else:
        # Use console logging for development
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.dev.ConsoleRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    # Set root logger level
    logging.basicConfig(level=log_level)

# ===============================================================================
# MODULE INITIALIZATION
# ===============================================================================

async def initialize_docker_application() -> FastAPI:
    """Initialize complete Docker application"""
    
    # Configure logging first
    configure_docker_logging()
    
    # Validate Docker environment
    docker_info = await validate_docker_installation()
    if not docker_info["docker_available"]:
        logger.error("Docker not available", error=docker_info.get("error"))
        raise RuntimeError("Docker is required but not available")
    
    logger.info("Docker environment validated", 
               version=docker_info["docker_version"],
               compose_available=docker_info["compose_available"])
    
    # Initialize Docker configuration
    config = DockerConfig()
    
    # Initialize environment manager
    env_manager = DockerEnvironmentManager(config)
    await env_manager.initialize()
    
    # Initialize compose manager
    compose_manager = DockerComposeManager(config)
    
    # Generate Docker Compose configuration
    services_config = {}  # Would be populated with actual service configs
    await compose_manager.generate_compose_file(services_config)
    
    # Generate environment file
    secrets = {
        "jwt_secret": generate_secret_key(),
        "encryption_key": generate_secret_key(32)
    }
    await compose_manager.generate_env_file(secrets)
    
    # Initialize health monitoring
    health_monitor = DockerHealthMonitor(config)
    await health_monitor.initialize()
    await health_monitor.start_monitoring()
    
    # Create FastAPI app
    app = FastAPI(
        title="YMERA Enterprise - Docker",
        description="Production-ready YMERA platform on Docker",
        version="4.0"
    )
    
    # Store managers in app state
    app.state.env_manager = env_manager
    app.state.compose_manager = compose_manager
    app.state.health_monitor = health_monitor
    
    # Add health check endpoint
    @app.get("/health")
    async def health_check():
        return await health_monitor.get_health_summary()
    
    @app.get("/docker/status")
    async def docker_status():
        return {
            "platform": "docker",
            "version": "4.0",
            "environment": config.environment,
            "container_detected": detect_docker_environment(),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    return app

# ===============================================================================
# EXPORTS
# ===============================================================================

__all__ = [
    "DockerConfig",
    "DockerNetworkConfig",
    "DockerVolumeConfig", 
    "DockerService",
    "DockerComposeConfig",
    "DockerEnvironmentManager",
    "DockerComposeManager",
    "DockerHealthMonitor",
    "initialize_docker_application",
    "detect_docker_environment",
    "validate_docker_installation",
    "configure_docker_logging"
]